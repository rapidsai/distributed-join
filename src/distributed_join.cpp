/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "distributed_join.hpp"

#include "comm.hpp"
#include "communicator.hpp"
#include "compression.hpp"
#include "error.hpp"
#include "strings_column.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/join.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <mpi.h>

#include <cuda_runtime.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

using cudf::column;
using cudf::table;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

/**
 * Local join thread used for merging incoming partitions and performing local joins.
 *
 * @param[in] communicated_left Left table after all-to-all communication.
 * @param[in] communicated_right Right table after all-to-all communication.
 * @param[out] batch_join_results Inner join result of each batch.
 * @param[in] left_on Column indices from the left table to join on. This argument will be
 * passed directly *cudf::inner_join*.
 * @param[in] right_on Column indices from the right table to join on. This argument will be
 * passed directly *cudf::inner_join*.
 * @param[in] columns_in_common Vector of pairs of column indices from the left and right table
 * that are in common and only one column will be produced in *batch_join_results*. This
 * argument will be passed directly *cudf::inner_join*.
 * @param[in] flags *flags[i]* is true if and only if the ith batch has finished the all-to-all
 *     communication.
 * @param[in] report_timing Whether to print the local join time to stderr.
 * @param[in] mr RMM memory resource.
 */
void inner_join_func(vector<std::unique_ptr<table>> &communicated_left,
                     vector<std::unique_ptr<table>> &communicated_right,
                     vector<std::unique_ptr<table>> &batch_join_results,
                     vector<cudf::size_type> const &left_on,
                     vector<cudf::size_type> const &right_on,
                     vector<std::pair<cudf::size_type, cudf::size_type>> const &columns_in_common,
                     vector<std::atomic<bool>> const &flags,
                     Communicator *communicator,
                     bool report_timing,
                     rmm::mr::device_memory_resource *mr)
{
  CUDA_RT_CALL(cudaSetDevice(communicator->current_device));
  rmm::mr::set_current_device_resource(mr);

  std::chrono::time_point<high_resolution_clock> start_time;
  std::chrono::time_point<high_resolution_clock> stop_time;

  for (size_t ibatch = 0; ibatch < flags.size(); ibatch++) {
    // busy waiting for all-to-all communication of ibatch to finish
    while (!flags[ibatch]) { ; }

    if (report_timing) { start_time = high_resolution_clock::now(); }

    if (communicated_left[ibatch]->num_rows() && communicated_right[ibatch]->num_rows()) {
      // Perform local join only when both left and right tables are not empty.
      // If either is empty, cuDF's inner join will return the other table, which is not desired.
      batch_join_results[ibatch] = cudf::inner_join(communicated_left[ibatch]->view(),
                                                    communicated_right[ibatch]->view(),
                                                    left_on,
                                                    right_on,
                                                    columns_in_common);
    } else {
      batch_join_results[ibatch] = std::make_unique<table>();
    }

    if (report_timing) {
      stop_time     = high_resolution_clock::now();
      auto duration = duration_cast<milliseconds>(stop_time - start_time);
      std::cout << "Rank " << communicator->mpi_rank << ": Local join on batch " << ibatch
                << " takes " << duration.count() << "ms" << std::endl;
    }
  }
}

/**
 * Allocate tables after all-to-all communication for all batches.
 *
 * @param[in] input_table Table that needs to be all-to-all communicated.
 * @param[in] recv_offsets Vector of size `(num_batches, mpi_size + 1)`, such that
 * `recv_offsets[i,j]` is the start row index to be sent to rank `j` in batch `i`.
 * @param[in] string_recv_offsets Vector with shape `(num_batches, num_columns, mpi_size + 1)`. The
 * output of `gather_string_offsets`.
 * @param[in] over_decom_factor Number of batches.
 * @param[out] communicated_table Vector of size `num_batches`, such that the ith element is the
 * allocated table of batch `i` after all-to-all communication.
 */
void allocate_communicated_table(cudf::table_view input_table,
                                 vector<vector<int64_t>> const &recv_offsets,
                                 vector<vector<vector<int64_t>>> const &string_recv_offsets,
                                 const int over_decom_factor,
                                 vector<std::unique_ptr<table>> &communicated_table)
{
  communicated_table.resize(over_decom_factor);

  for (int ibatch = 0; ibatch < over_decom_factor; ibatch++) {
    vector<std::unique_ptr<column>> communicated_columns;
    for (cudf::size_type icol = 0; icol < input_table.num_columns(); icol++) {
      cudf::data_type dtype = input_table.column(icol).type();

      if (dtype.id() == cudf::type_id::STRING) {
        std::unique_ptr<column> chars_column = cudf::strings::detail::create_chars_child_column(
          recv_offsets[ibatch].back(), 0, string_recv_offsets[ibatch][icol].back());
        std::unique_ptr<column> offset_column = cudf::make_numeric_column(
          input_table.column(icol).child(0).type(), recv_offsets[ibatch].back() + 1);

        communicated_columns.push_back(cudf::make_strings_column(
          recv_offsets[ibatch].back(), std::move(offset_column), std::move(chars_column), 0, {}));
      } else {
        communicated_columns.push_back(
          cudf::make_fixed_width_column(dtype, recv_offsets[ibatch].back()));
      }
    }
    communicated_table[ibatch] = std::make_unique<table>(std::move(communicated_columns));
  }
}

/**
 * Explicitly copy data destined to the current rank during all-to-all communication.
 *
 * This function can be used together with `all_to_all_comm` with `include_self = false` for a
 * complete all-to-all communication.
 *
 * @param[in] input_table Table to be all-to-all communicated.
 * @param[in] communicated_tables Table after all-to-all communication.
 * @param[in] send_offset Vector of size `(num_batches, mpi_size + 1)` such that `send_offset[i,j]`
 * represents the start index of `input_table` to be sent to rank `j` during batch `i`.
 * @param[in] recv_offset Vector of size `(num_batches, mpi_size + 1)` such that `recv_offset[i,j]`
 * represents the start index of `communicated_tables` to receive data from rank `j` during batch
 * `i`.
 * @param[in] string_send_offsets Vector with shape `(num_batches, num_columns, mpi_size + 1)`, such
 * that `string_send_offsets[i,j,k]` representing the start index in the char subcolumn of column
 * `j` that needs to be sent to rank `k`, for batch `i`.
 * @param[in] string_recv_offsets Vector with shape `(num_batches, num_columns, mpi_size + 1)`, such
 * that `string_recv_offsets[i,j,k]` representing the start index in the char subcolumn of column
 * `j` that receives data from rank `k`, for batch `i`.
 * @param[in] string_sizes_send String sizes of each row for all string columns.
 * @param[in] string_sizes_recv Vector of size `(num_batches, num_columns)`, representing the
 * receive buffers for string sizes. This argument needs to be preallocated.
 */
void copy_to_self(cudf::table_view input_table,
                  vector<std::unique_ptr<table>> &communicated_tables,
                  vector<cudf::size_type> const &send_offsets,
                  vector<vector<int64_t>> const &recv_offsets,
                  vector<vector<vector<cudf::size_type>>> const &string_send_offsets,
                  vector<vector<vector<int64_t>>> const &string_recv_offsets,
                  vector<rmm::device_uvector<cudf::size_type>> const &string_sizes_send,
                  vector<vector<rmm::device_uvector<cudf::size_type>>> &string_sizes_recv,
                  int over_decom_factor,
                  Communicator *communicator)
{
  int mpi_rank = communicator->mpi_rank;
  int mpi_size = communicator->mpi_size;

  for (int ibatch = 0; ibatch < over_decom_factor; ibatch++) {
    for (cudf::size_type icol = 0; icol < input_table.num_columns(); icol++) {
      cudf::data_type dtype = input_table.column(icol).type();
      if (dtype.id() != cudf::type_id::STRING) {
        // This is a fixed-width column
        cudf::size_type dtype_size = cudf::size_of(dtype);

        CUDA_RT_CALL(cudaMemcpy(
          static_cast<void *>(
            communicated_tables[ibatch]->mutable_view().column(icol).head<char>() +
            recv_offsets[ibatch][mpi_rank] * dtype_size),
          static_cast<const void *>(input_table.column(icol).head<char>() +
                                    send_offsets[ibatch * mpi_size + mpi_rank] * dtype_size),
          (recv_offsets[ibatch][mpi_rank + 1] - recv_offsets[ibatch][mpi_rank]) * dtype_size,
          cudaMemcpyDeviceToDevice));
      } else {
        // This is a string column
        CUDA_RT_CALL(
          cudaMemcpy(string_sizes_recv[ibatch][icol].data() + recv_offsets[ibatch][mpi_rank],
                     string_sizes_send[icol].data() + send_offsets[ibatch * mpi_size + mpi_rank],
                     (recv_offsets[ibatch][mpi_rank + 1] - recv_offsets[ibatch][mpi_rank]) *
                       sizeof(cudf::size_type),
                     cudaMemcpyDeviceToDevice));

        CUDA_RT_CALL(cudaMemcpy(
          communicated_tables[ibatch]->mutable_view().column(icol).child(1).head<int8_t>() +
            string_recv_offsets[ibatch][icol][mpi_rank],
          input_table.column(icol).child(1).head<int8_t>() +
            string_send_offsets[ibatch][icol][mpi_rank],
          string_send_offsets[ibatch][icol][mpi_rank + 1] -
            string_send_offsets[ibatch][icol][mpi_rank],
          cudaMemcpyDeviceToDevice));
      }
    }
  }
}

void append_to_all_to_all_comm_buffers(
  cudf::table_view input,
  cudf::mutable_table_view output,
  vector<cudf::size_type> const &send_offsets,
  vector<int64_t> const &recv_offsets,
  vector<vector<cudf::size_type>> const &string_send_offsets,
  vector<vector<int64_t>> const &string_recv_offsets,
  vector<rmm::device_uvector<cudf::size_type>> const &string_sizes_send,
  vector<rmm::device_uvector<cudf::size_type>> &string_sizes_recv,
  vector<AllToAllCommBuffer> &all_to_all_comm_buffers,
  vector<ColumnCompressionOptions> const &compression_options)
{
  for (cudf::size_type icol = 0; icol < input.num_columns(); icol++) {
    cudf::data_type dtype = input.column(icol).type();
    assert(dtype == output.column(icol).type());

    if (dtype.id() != cudf::type_id::STRING) {
      // This is a fixed-width column
      all_to_all_comm_buffers.emplace_back(
        input.column(icol).head(),
        output.column(icol).head(),
        vector<int64_t>(send_offsets.begin(), send_offsets.end()),
        recv_offsets,
        dtype,
        compression_options[icol].compression_method,
        compression_options[icol].cascaded_format);
    } else {
      // This is a string column
      all_to_all_comm_buffers.emplace_back(
        string_sizes_send[icol].data(),
        string_sizes_recv[icol].data(),
        vector<int64_t>(send_offsets.begin(), send_offsets.end()),
        recv_offsets,
        input.column(icol).child(0).type(),
        compression_options[icol].children_compression_options[0].compression_method,
        compression_options[icol].children_compression_options[0].cascaded_format);

      all_to_all_comm_buffers.emplace_back(
        input.column(icol).child(1).head(),
        output.column(icol).child(1).head(),
        vector<int64_t>(string_send_offsets[icol].begin(), string_send_offsets[icol].end()),
        string_recv_offsets[icol],
        input.column(icol).child(1).type(),
        compression_options[icol].children_compression_options[1].compression_method,
        compression_options[icol].children_compression_options[1].cascaded_format);
    }
  }
}

void append_to_all_to_all_comm_buffers(cudf::table_view input,
                                       cudf::mutable_table_view output,
                                       vector<cudf::size_type> const &send_offsets,
                                       vector<int64_t> const &recv_offsets,
                                       vector<AllToAllCommBuffer> &all_to_all_comm_buffers,
                                       vector<ColumnCompressionOptions> compression_options)
{
  // Without string columns, `string_sizes_recv` is not needed. This is only a placeholder passed to
  // `append_to_all_to_all_comm_buffers`.
  vector<rmm::device_uvector<cudf::size_type>> string_sizes_recv;

  append_to_all_to_all_comm_buffers(input,
                                    output,
                                    send_offsets,
                                    recv_offsets,
                                    vector<vector<cudf::size_type>>(),
                                    vector<vector<int64_t>>(),
                                    vector<rmm::device_uvector<cudf::size_type>>(),
                                    string_sizes_recv,
                                    all_to_all_comm_buffers,
                                    compression_options);
}

void all_to_all_comm(vector<AllToAllCommBuffer> &all_to_all_comm_buffers,
                     Communicator *communicator,
                     bool include_self,
                     bool report_timing,
                     void *preallocated_pinned_buffer)
{
  int mpi_rank = communicator->mpi_rank;
  int mpi_size = communicator->mpi_size;

  double start_time              = 0.0;
  double stop_time               = 0.0;
  double total_compression_time  = 0.0;
  double total_uncompressed_size = 0.0;
  double total_compressed_size   = 0.0;

  size_t *compressed_buffer_sizes_pinned    = static_cast<size_t *>(preallocated_pinned_buffer);
  bool alloc_compressed_buffer_sizes_pinned = false;

  vector<rmm::cuda_stream> compression_streams(mpi_size);

  vector<rmm::cuda_stream_view> compression_stream_views;
  compression_stream_views.reserve(mpi_size);

  for (const auto &compression_stream : compression_streams) {
    compression_stream_views.push_back(compression_stream.view());
  }

  for (auto &buffer : all_to_all_comm_buffers) {
    if (buffer.compression_method == CompressionMethod::none) {
      if (!communicator->group_by_batch()) communicator->start();

      send_data_by_offset(buffer.send_buffer,
                          buffer.send_offsets,
                          cudf::size_of(buffer.dtype),
                          communicator,
                          include_self);

      recv_data_by_offset(buffer.recv_buffer,
                          buffer.recv_offsets,
                          cudf::size_of(buffer.dtype),
                          communicator,
                          include_self);

      if (!communicator->group_by_batch()) communicator->stop();

      continue;
    }

    // If the code reaches here, the buffer will be compressed before communication
    assert(buffer.compression_method == CompressionMethod::cascaded);

    // General strategy of all-to-all with compression:
    // The all-to-all interface works on a single buffer with offsets. Since we don't know the
    // compressed size without actually doing the compression, we cannot pre-allocate this buffer
    // beforehand. Instead we compress each partition in the send buffer separately. Once the
    // compression is done, we can allocate the compressed send buffer and copy the compressed
    // data into the buffer. Then, all-to-all communication can reuse helper functions
    // `send_data_by_offset` and `recv_data_by_offset` functions.

    if (report_timing) { start_time = MPI_Wtime(); }

    if (compressed_buffer_sizes_pinned == nullptr) {
      CUDA_RT_CALL(cudaMallocHost(&compressed_buffer_sizes_pinned, mpi_size * sizeof(size_t)));
      alloc_compressed_buffer_sizes_pinned = true;
    }

    // Compress each partition in the send buffer separately and store the result in
    // `compressed_buffers`
    vector<const void *> uncompressed_data(mpi_size);
    vector<cudf::size_type> uncompressed_counts(mpi_size);

    for (int irank = 0; irank < mpi_size; irank++) {
      if (!include_self && irank == mpi_rank) {
        uncompressed_data[irank]   = nullptr;
        uncompressed_counts[irank] = 0;
        continue;
      }
      uncompressed_data[irank] = static_cast<const int8_t *>(buffer.send_buffer) +
                                 buffer.send_offsets[irank] * cudf::size_of(buffer.dtype);
      uncompressed_counts[irank] = buffer.send_offsets[irank + 1] - buffer.send_offsets[irank];
    }

    vector<rmm::device_buffer> compressed_buffers;
    vector<size_t> compressed_buffer_sizes(mpi_size);

    cudf::type_dispatcher(buffer.dtype,
                          compression_functor{},
                          uncompressed_data,
                          uncompressed_counts,
                          compressed_buffers,
                          compressed_buffer_sizes_pinned,
                          compression_stream_views,
                          buffer.cascaded_format);

    for (auto &stream : compression_streams) stream.synchronize();

    memcpy(
      compressed_buffer_sizes.data(), compressed_buffer_sizes_pinned, mpi_size * sizeof(size_t));

    // Calculate and communicate offsets for the compressed buffers
    buffer.compressed_send_offsets.resize(mpi_size + 1);
    buffer.compressed_send_offsets[0] = 0;
    for (int irank = 0; irank < mpi_size; irank++) {
      buffer.compressed_send_offsets[irank + 1] =
        buffer.compressed_send_offsets[irank] + compressed_buffer_sizes[irank];
    }

    if (report_timing) {
      stop_time = MPI_Wtime();
      total_compression_time += (stop_time - start_time);
      total_uncompressed_size +=
        ((buffer.send_offsets.back() - buffer.send_offsets[0]) * cudf::size_of(buffer.dtype));
      total_compressed_size += buffer.compressed_send_offsets.back();
    }

    communicate_sizes(buffer.compressed_send_offsets, buffer.compressed_recv_offsets, communicator);

    // Merge compressed data of all partitions in `compressed_buffers` into a single buffer
    buffer.compressed_send_buffer.resize(buffer.compressed_send_offsets.back());
    for (int irank = 0; irank < mpi_size; irank++) {
      if (!include_self && irank == mpi_rank) continue;

      CUDA_RT_CALL(cudaMemcpy(static_cast<int8_t *>(buffer.compressed_send_buffer.data()) +
                                buffer.compressed_send_offsets[irank],
                              compressed_buffers[irank].data(),
                              compressed_buffer_sizes[irank],
                              cudaMemcpyDeviceToDevice));
    }
    compressed_buffers.clear();

    // Allocate receive buffer and launch all-to-all communication on the compressed buffer
    buffer.compressed_recv_buffer.resize(buffer.compressed_recv_offsets.back());
    CUDA_RT_CALL(cudaStreamSynchronize(0));

    if (!communicator->group_by_batch()) communicator->start();

    send_data_by_offset(buffer.compressed_send_buffer.data(),
                        buffer.compressed_send_offsets,
                        1,
                        communicator,
                        include_self);

    recv_data_by_offset(buffer.compressed_recv_buffer.data(),
                        buffer.compressed_recv_offsets,
                        1,
                        communicator,
                        include_self);

    if (!communicator->group_by_batch()) communicator->stop();
  }

  if (alloc_compressed_buffer_sizes_pinned) {
    CUDA_RT_CALL(cudaFreeHost(compressed_buffer_sizes_pinned));
  }

  if (total_uncompressed_size && report_timing) {
    std::cout << "Rank " << mpi_rank << ": compression takes " << total_compression_time * 1e3
              << "ms"
              << " with compression ratio " << total_uncompressed_size / total_compressed_size
              << " and throughput " << total_uncompressed_size / total_compression_time / 1e9
              << "GB/s" << std::endl;
  }
}

void postprocess_all_to_all_comm(vector<AllToAllCommBuffer> &all_to_all_comm_buffers,
                                 Communicator *communicator,
                                 bool include_self,
                                 bool report_timing)
{
  int mpi_rank                   = communicator->mpi_rank;
  int mpi_size                   = communicator->mpi_size;
  double start_time              = 0.0;
  double stop_time               = 0.0;
  double total_uncompressed_size = 0.0;

  if (report_timing) { start_time = MPI_Wtime(); }

  vector<rmm::cuda_stream> decompression_streams(mpi_size);

  vector<rmm::cuda_stream_view> decompression_stream_views;
  decompression_stream_views.reserve(mpi_size);

  for (const auto &decompression_stream : decompression_streams) {
    decompression_stream_views.push_back(decompression_stream.view());
  }

  // Decompress compressed data into destination buffer
  for (auto &buffer : all_to_all_comm_buffers) {
    if (buffer.compression_method == CompressionMethod::none) continue;

    vector<const void *> compressed_data(mpi_size);
    vector<int64_t> compressed_sizes(mpi_size);
    vector<void *> outputs(mpi_size);
    vector<int64_t> expected_output_counts(mpi_size);

    for (int irank = 0; irank < mpi_size; irank++) {
      if (!include_self && irank == mpi_rank) {
        compressed_sizes[irank]       = 0;
        expected_output_counts[irank] = 0;
        continue;
      }
      compressed_data[irank] = static_cast<int8_t *>(buffer.compressed_recv_buffer.data()) +
                               buffer.compressed_recv_offsets[irank];
      compressed_sizes[irank] =
        buffer.compressed_recv_offsets[irank + 1] - buffer.compressed_recv_offsets[irank];
      outputs[irank] = static_cast<int8_t *>(buffer.recv_buffer) +
                       buffer.recv_offsets[irank] * cudf::size_of(buffer.dtype);
      expected_output_counts[irank] = buffer.recv_offsets[irank + 1] - buffer.recv_offsets[irank];
    }

    cudf::type_dispatcher(buffer.dtype,
                          decompression_functor{},
                          compressed_data,
                          compressed_sizes,
                          outputs,
                          expected_output_counts,
                          decompression_stream_views);

    for (auto &stream : decompression_streams) stream.synchronize();

    if (report_timing)
      total_uncompressed_size += (buffer.recv_offsets.back() * cudf::size_of(buffer.dtype));
  }

  if (total_uncompressed_size && report_timing) {
    stop_time       = MPI_Wtime();
    double duration = stop_time - start_time;
    std::cout << "Rank " << mpi_rank << ": decompression takes " << duration * 1e3 << "ms"
              << " with throughput " << total_uncompressed_size / duration / 1e9 << "GB/s"
              << std::endl;
  }
}

std::unique_ptr<table> distributed_inner_join(
  cudf::table_view const &left,
  cudf::table_view const &right,
  vector<cudf::size_type> const &left_on,
  vector<cudf::size_type> const &right_on,
  vector<std::pair<cudf::size_type, cudf::size_type>> const &columns_in_common,
  Communicator *communicator,
  vector<ColumnCompressionOptions> left_compression_options,
  vector<ColumnCompressionOptions> right_compression_options,
  int over_decom_factor,
  bool report_timing,
  void *preallocated_pinned_buffer)
{
  if (over_decom_factor == 1) {
    // @TODO: If over_decom_factor is 1, there is no opportunity for overlapping. Therefore,
    // we can get away with using just one thread.
  }

  int mpi_rank = communicator->mpi_rank;
  int mpi_size = communicator->mpi_size;
  std::chrono::time_point<high_resolution_clock> start_time;
  std::chrono::time_point<high_resolution_clock> stop_time;

  /* Hash partition */

  if (report_timing) { start_time = high_resolution_clock::now(); }

  std::unique_ptr<table> hashed_left;
  vector<cudf::size_type> left_offset;

  std::unique_ptr<table> hashed_right;
  vector<cudf::size_type> right_offset;

  std::tie(hashed_left, left_offset) =
    cudf::hash_partition(left, left_on, mpi_size * over_decom_factor, cudf::hash_id::HASH_IDENTITY);

  std::tie(hashed_right, right_offset) = cudf::hash_partition(
    right, right_on, mpi_size * over_decom_factor, cudf::hash_id::HASH_IDENTITY);

  CUDA_RT_CALL(cudaStreamSynchronize(0));

  left_offset.push_back(left.num_rows());
  right_offset.push_back(right.num_rows());

  if (report_timing) {
    stop_time     = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop_time - start_time);
    std::cout << "Rank " << mpi_rank << ": Hash partition takes " << duration.count() << "ms"
              << std::endl;
  }

  /* Communicate number of rows of each batch */

  // For batch `i`, `recv_offsets[i, j]` represents the start row index in the communicated table
  // to receive data from rank `j`.
  vector<vector<int64_t>> recv_offsets_left(over_decom_factor);
  vector<vector<int64_t>> recv_offsets_right(over_decom_factor);

  for (int ibatch = 0; ibatch < over_decom_factor; ibatch++) {
    size_t start_idx = ibatch * mpi_size;
    size_t end_idx   = (ibatch + 1) * mpi_size + 1;

    communicate_sizes(vector<cudf::size_type>(&left_offset[start_idx], &left_offset[end_idx]),
                      recv_offsets_left[ibatch],
                      communicator);

    communicate_sizes(vector<cudf::size_type>(&right_offset[start_idx], &right_offset[end_idx]),
                      recv_offsets_right[ibatch],
                      communicator);
  }

  // Note: General stategy for the string columns during all-to-all communication
  // Each string column in cuDF consists of two subcolumns: a char subcolumn and an offset
  // subcolumn. For the char subcolumn, we need to first gather the offsets in this string
  // subcolumn of all ranks by using `gather_string_offsets`, and then it can be all-to-all
  // communicated using the gathered offsets. For the offset subcolumn, we can first calculate the
  // sizes of all rows by calculating the adjacent differences. Then, the sizes are all-to-all
  // communicated. Once the all-to-all communication finishes, on target rank we can reconstruct
  // the offset subcolumn by using a scan on sizes.

  /* Communicate the number of bytes of string columns */

  // For batch `i`, `string_send_offsets[i, j, k]` represents the start index into char subcolumn
  // to be sent to rank `k` for column `j`. If column `j` is not a string column,
  // `string_send_offsets[i, j]` will be an empty vector. Otherwise, `string_send_offsets[i, j]`
  // will be a vector of length `mpi_size + 1`.
  vector<vector<vector<cudf::size_type>>> string_send_offsets_left;
  vector<vector<vector<cudf::size_type>>> string_send_offsets_right;
  // For batch `i`, `string_recv_offsets[i, j, k]` represents the start index into char subcolumn
  // to receive data from rank `k` for column `j`. If column `j` is not a string column,
  // `string_recv_offsets[i, j]` will be an empty vector. Otherwise, `string_recv_offsets[i, j]`
  // will be a vector of length `mpi_size + 1`.
  vector<vector<vector<int64_t>>> string_recv_offsets_left;
  vector<vector<vector<int64_t>>> string_recv_offsets_right;

  gather_string_offsets(hashed_left->view(),
                        left_offset,
                        over_decom_factor,
                        string_send_offsets_left,
                        string_recv_offsets_left,
                        communicator);

  gather_string_offsets(hashed_right->view(),
                        right_offset,
                        over_decom_factor,
                        string_send_offsets_right,
                        string_recv_offsets_right,
                        communicator);

  /* Calculate the number of bytes from string offsets */

  vector<rmm::device_uvector<cudf::size_type>> string_sizes_send_left;
  vector<rmm::device_uvector<cudf::size_type>> string_sizes_send_right;
  vector<vector<rmm::device_uvector<cudf::size_type>>> string_sizes_recv_left;
  vector<vector<rmm::device_uvector<cudf::size_type>>> string_sizes_recv_right;

  calculate_string_sizes_from_offsets(hashed_left->view(), string_sizes_send_left);
  calculate_string_sizes_from_offsets(hashed_right->view(), string_sizes_send_right);

  allocate_string_sizes_receive_buffer(
    hashed_left->view(), over_decom_factor, recv_offsets_left, string_sizes_recv_left);

  allocate_string_sizes_receive_buffer(
    hashed_right->view(), over_decom_factor, recv_offsets_right, string_sizes_recv_right);

  /* Declare storage for the table after all-to-all communication */

  // left table after all-to-all for each batch
  vector<std::unique_ptr<table>> communicated_left;
  // right table after all-to-all for each batch
  vector<std::unique_ptr<table>> communicated_right;

  allocate_communicated_table(hashed_left->view(),
                              recv_offsets_left,
                              string_recv_offsets_left,
                              over_decom_factor,
                              communicated_left);

  allocate_communicated_table(hashed_right->view(),
                              recv_offsets_right,
                              string_recv_offsets_right,
                              over_decom_factor,
                              communicated_right);

  // Synchronization on the default stream is necessary here because the communicator can use a
  // different stream
  CUDA_RT_CALL(cudaStreamSynchronize(0));

  // *flags* indicates whether each batch has finished communication
  // *flags* uses std::atomic because unsynchronized access to an object which is modified in one
  // thread and read in another is undefined behavior.
  vector<std::atomic<bool>> flags(over_decom_factor);
  vector<std::unique_ptr<table>> batch_join_results(over_decom_factor);

  for (auto &flag : flags) { flag = false; }

  /* Copy from hashed table to communicated table for the current rank */

  // These device-to-device memory copies are performed explicitly here before all-to-all
  // communication and local join, because if they are part of the communication, they could block
  // the host thread (even if they are launched on different streams) while the local join kernel
  // is running, limiting the efficacy of overlapping.

  copy_to_self(hashed_left->view(),
               communicated_left,
               left_offset,
               recv_offsets_left,
               string_send_offsets_left,
               string_recv_offsets_left,
               string_sizes_send_left,
               string_sizes_recv_left,
               over_decom_factor,
               communicator);

  copy_to_self(hashed_right->view(),
               communicated_right,
               right_offset,
               recv_offsets_right,
               string_send_offsets_right,
               string_recv_offsets_right,
               string_sizes_send_right,
               string_sizes_recv_right,
               over_decom_factor,
               communicator);

  /* Launch inner join thread */

  std::thread inner_join_thread(inner_join_func,
                                std::ref(communicated_left),
                                std::ref(communicated_right),
                                std::ref(batch_join_results),
                                left_on,
                                right_on,
                                columns_in_common,
                                std::ref(flags),
                                communicator,
                                report_timing,
                                rmm::mr::get_current_device_resource());

  /* Use the current thread for all-to-all communication */

  for (int ibatch = 0; ibatch < over_decom_factor; ibatch++) {
    if (report_timing) { start_time = high_resolution_clock::now(); }

    vector<AllToAllCommBuffer> all_to_all_comm_buffers;

    // the start and end index for left_offset and right_offset for the ibatch
    size_t start_idx = ibatch * mpi_size;
    size_t end_idx   = (ibatch + 1) * mpi_size + 1;

    append_to_all_to_all_comm_buffers(
      hashed_left->view(),
      communicated_left[ibatch]->mutable_view(),
      vector<cudf::size_type>(&left_offset[start_idx], &left_offset[end_idx]),
      recv_offsets_left[ibatch],
      string_send_offsets_left[ibatch],
      string_recv_offsets_left[ibatch],
      string_sizes_send_left,
      string_sizes_recv_left[ibatch],
      all_to_all_comm_buffers,
      left_compression_options);

    append_to_all_to_all_comm_buffers(
      hashed_right->view(),
      communicated_right[ibatch]->mutable_view(),
      vector<cudf::size_type>(&right_offset[start_idx], &right_offset[end_idx]),
      recv_offsets_right[ibatch],
      string_send_offsets_right[ibatch],
      string_recv_offsets_right[ibatch],
      string_sizes_send_right,
      string_sizes_recv_right[ibatch],
      all_to_all_comm_buffers,
      right_compression_options);

    // all-to-all communication for the ibatch
    if (communicator->group_by_batch()) communicator->start();

    all_to_all_comm(
      all_to_all_comm_buffers, communicator, false, report_timing, preallocated_pinned_buffer);

    if (communicator->group_by_batch()) communicator->stop();

    postprocess_all_to_all_comm(all_to_all_comm_buffers, communicator, false, report_timing);

    calculate_string_offsets_from_sizes(communicated_left[ibatch]->mutable_view(),
                                        string_sizes_recv_left[ibatch]);
    calculate_string_offsets_from_sizes(communicated_right[ibatch]->mutable_view(),
                                        string_sizes_recv_right[ibatch]);

    // mark the communication of ibatch as finished.
    // the join thread is safe to start performing local join on ibatch
    flags[ibatch] = true;

    if (report_timing) {
      stop_time     = high_resolution_clock::now();
      auto duration = duration_cast<milliseconds>(stop_time - start_time);
      std::cout << "Rank " << mpi_rank << ": All-to-all communication on batch " << ibatch
                << " takes " << duration.count() << "ms" << std::endl;
    }
  }

  // hashed left and right tables should not be needed now
  hashed_left.reset();
  hashed_right.reset();

  // wait for all join batches to finish
  inner_join_thread.join();

  /* Merge join results from different batches into a single table */

  vector<cudf::table_view> batch_join_results_view;

  for (auto &table_ptr : batch_join_results) {
    batch_join_results_view.push_back(table_ptr->view());
  }

  return cudf::concatenate(batch_join_results_view);
}