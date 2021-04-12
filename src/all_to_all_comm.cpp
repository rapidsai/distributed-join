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

#include "all_to_all_comm.hpp"

#include "communicator.hpp"
#include "compression.hpp"
#include "error.hpp"
#include "strings_column.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
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

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

using cudf::column;
using cudf::table;
using std::vector;

void communicate_sizes(std::vector<int64_t> const &send_offset,
                       std::vector<int64_t> &recv_offset,
                       Communicator *communicator)
{
  int mpi_size = communicator->mpi_size;
  std::vector<int64_t> send_count(mpi_size, -1);

  for (int irank = 0; irank < mpi_size; irank++) {
    send_count[irank] = send_offset[irank + 1] - send_offset[irank];
  }

  std::vector<int64_t> recv_count(mpi_size, -1);

  // Note: MPI is used for communicating the sizes instead of *Communicator* because
  // *Communicator* is not guaranteed to be able to send/recv host buffers.

  std::vector<MPI_Request> send_req(mpi_size);
  std::vector<MPI_Request> recv_req(mpi_size);

  for (int irank = 0; irank < mpi_size; irank++) {
    MPI_CALL(MPI_Isend(&send_count[irank],
                       1,
                       MPI_INT64_T,
                       irank,
                       exchange_size_tag,
                       MPI_COMM_WORLD,
                       &send_req[irank]));
  }

  for (int irank = 0; irank < mpi_size; irank++) {
    MPI_CALL(MPI_Irecv(&recv_count[irank],
                       1,
                       MPI_INT64_T,
                       irank,
                       exchange_size_tag,
                       MPI_COMM_WORLD,
                       &recv_req[irank]));
  }

  MPI_CALL(MPI_Waitall(mpi_size, send_req.data(), MPI_STATUSES_IGNORE));
  MPI_CALL(MPI_Waitall(mpi_size, recv_req.data(), MPI_STATUSES_IGNORE));

  recv_offset.resize(mpi_size + 1, -1);
  recv_offset[0] = 0;
  std::partial_sum(recv_count.begin(), recv_count.end(), recv_offset.begin() + 1);
}

void communicate_sizes(std::vector<cudf::size_type> const &send_offset,
                       std::vector<int64_t> &recv_offset,
                       Communicator *communicator)
{
  communicate_sizes(
    std::vector<int64_t>(send_offset.begin(), send_offset.end()), recv_offset, communicator);
}

/**
 * Send data from the current rank to other ranks according to offset.
 *
 * Note: This call should be enclosed by communicator->start() and communicator->stop().
 *
 * @param[in] data                The starting address of data to be sent in device buffer.
 * @param[in] offset              Vector of length mpi_size + 1. Items in *data* with indicies from
 * offset[i] to offset[i+1] will be sent to rank i.
 * @param[in] item_size           The size of each item.
 * @param[in] communicator        An instance of 'Communicator' used for communication.
 * @param[in] self_send           Whether sending data to itself. If this argument is false, items
 * in *data* destined for the current rank will not be copied.
 */
static void send_data_by_offset(const void *data,
                                std::vector<int64_t> const &offset,
                                size_t item_size,
                                Communicator *communicator,
                                bool self_send = true)
{
  int mpi_rank = communicator->mpi_rank;
  int mpi_size = communicator->mpi_size;

  for (int itarget_rank = 0; itarget_rank < mpi_size; itarget_rank++) {
    if (!self_send && itarget_rank == mpi_rank) continue;

    // calculate the number of elements to send
    int64_t count = offset[itarget_rank + 1] - offset[itarget_rank];

    // calculate the starting address
    const void *start_addr = (void *)((char *)data + offset[itarget_rank] * item_size);

    // send buffer to the target rank
    communicator->send(start_addr, count, item_size, itarget_rank);
  }
}

/**
 * Receive data sent by 'send_data_by_offset'.
 *
 * Note: This call should be enclosed by communicator->start() and communicator->stop().
 *
 * @param[out] data         Items received from all ranks will be placed contiguously in *data*.
 *     This argument needs to be preallocated.
 * @param[in] offset        The items received from rank i will be stored at the start of
 * `data[offset[i]]`.
 * @param[in] item_size     The size of each item.
 * @param[in] communicator  An instance of 'Communicator' used for communication.
 * @param[in] self_recv     Whether recving data from itself. If this argument is false, items in
 *                          *data* from the current rank will not be received.
 */
static void recv_data_by_offset(void *data,
                                std::vector<int64_t> const &offset,
                                size_t item_size,
                                Communicator *communicator,
                                bool self_recv = true)
{
  int mpi_rank = communicator->mpi_rank;
  int mpi_size = communicator->mpi_size;

  for (int isource_rank = 0; isource_rank < mpi_size; isource_rank++) {
    if (!self_recv && mpi_rank == isource_rank) continue;

    communicator->recv((void *)((char *)data + offset[isource_rank] * item_size),
                       offset[isource_rank + 1] - offset[isource_rank],
                       item_size,
                       isource_rank);
  }
}

void warmup_all_to_all(Communicator *communicator)
{
  int mpi_rank                        = communicator->mpi_rank;
  int mpi_size                        = communicator->mpi_size;
  int64_t size                        = 10'000'000LL;
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource();

  /* Allocate send/recv buffers */

  std::vector<void *> send_buffer(mpi_size, nullptr);
  std::vector<void *> recv_buffer(mpi_size, nullptr);

  for (int irank = 0; irank < mpi_size; irank++) {
    if (irank == mpi_rank) continue;
    send_buffer[irank] = mr->allocate(size / mpi_size, rmm::cuda_stream_default);
    recv_buffer[irank] = mr->allocate(size / mpi_size, rmm::cuda_stream_default);
  }

  CUDA_RT_CALL(cudaStreamSynchronize(0));

  /* Communication */

  communicator->start();

  for (int irank = 0; irank < mpi_size; irank++) {
    if (irank != mpi_rank) communicator->send(send_buffer[irank], size / mpi_size, 1, irank);
  }

  for (int irank = 0; irank < mpi_size; irank++) {
    if (irank != mpi_rank) communicator->recv(recv_buffer[irank], size / mpi_size, 1, irank);
  }

  communicator->stop();

  /* Deallocate send/recv buffers */

  for (int irank = 0; irank < mpi_rank; irank++) {
    mr->deallocate(send_buffer[irank], size / mpi_size, rmm::cuda_stream_default);
    mr->deallocate(recv_buffer[irank], size / mpi_size, rmm::cuda_stream_default);
  }

  CUDA_RT_CALL(cudaStreamSynchronize(0));
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

/**
 * Allocate the table after all-to-all communication.
 *
 * @param[in] input_table Table that needs to be all-to-all communicated.
 * @param[in] recv_offsets Vector of size `mpi_size + 1`, indicating the start row index in
 * *input_table* to be sent to each rank.
 * @param[in] string_recv_offsets Vector with shape `(num_columns, mpi_size + 1)`. The output of
 * `gather_string_offsets`.
 *
 * @return Allocated table after all-to-all communication.
 */
static std::unique_ptr<table> allocate_communicated_table_helper(
  cudf::table_view input_table,
  vector<int64_t> const &recv_offsets,
  vector<vector<int64_t>> const &string_recv_offsets)
{
  vector<std::unique_ptr<column>> communicated_columns;
  for (cudf::size_type icol = 0; icol < input_table.num_columns(); icol++) {
    cudf::column_view input_column = input_table.column(icol);
    cudf::data_type dtype          = input_column.type();

    if (dtype.id() == cudf::type_id::STRING) {
      std::unique_ptr<column> chars_column = cudf::strings::detail::create_chars_child_column(
        recv_offsets.back(), 0, string_recv_offsets[icol].back());
      std::unique_ptr<column> offset_column =
        cudf::make_numeric_column(input_column.child(0).type(), recv_offsets.back() + 1);

      communicated_columns.push_back(cudf::make_strings_column(
        recv_offsets.back(), std::move(offset_column), std::move(chars_column), 0, {}));
    } else {
      communicated_columns.push_back(cudf::make_fixed_width_column(dtype, recv_offsets.back()));
    }
  }
  return std::make_unique<table>(std::move(communicated_columns));
}

/**
 * Explicitly copy data destined to the current rank during all-to-all communication.
 *
 * This function can be used together with `all_to_all_comm` with `include_self = false` for a
 * complete all-to-all communication.
 *
 * @param[in] input_table Table to be all-to-all communicated.
 * @param[in] communicated_tables Table after all-to-all communication.
 * @param[in] send_offset Vector of size `mpi_size + 1` indicating the start row index of
 * `input_table` to be sent to each rank.
 * @param[in] recv_offset Vector of size `mpi_size + 1` indicating the start row index of
 * `communicated_tables` to receive data from each rank.
 * @param[in] string_send_offsets Vector with shape `(num_columns, mpi_size + 1)`, such that
 * `string_send_offsets[j,k]` representing the start index in the char subcolumn of column `j` that
 * needs to be sent to rank `k`.
 * @param[in] string_recv_offsets Vector with shape `(num_columns, mpi_size + 1)`, such that
 * `string_recv_offsets[j,k]` representing the start index in the char subcolumn of column `j` that
 * receives data from rank `k`.
 * @param[in] string_sizes_send String sizes of each row for all string columns.
 * @param[in] string_sizes_recv Receive buffers for string sizes. This argument needs to be
 * preallocated.
 */
static void copy_to_self(cudf::table_view input_table,
                         cudf::mutable_table_view communicated_tables,
                         vector<cudf::size_type> const &send_offsets,
                         vector<int64_t> const &recv_offsets,
                         vector<vector<cudf::size_type>> const &string_send_offsets,
                         vector<vector<int64_t>> const &string_recv_offsets,
                         vector<rmm::device_uvector<cudf::size_type>> const &string_sizes_send,
                         vector<rmm::device_uvector<cudf::size_type>> &string_sizes_recv,
                         Communicator *communicator)
{
  int mpi_rank = communicator->mpi_rank;

  for (cudf::size_type icol = 0; icol < input_table.num_columns(); icol++) {
    cudf::data_type dtype = input_table.column(icol).type();
    if (dtype.id() != cudf::type_id::STRING) {
      // This is a fixed-width column
      cudf::size_type dtype_size = cudf::size_of(dtype);

      CUDA_RT_CALL(cudaMemcpy(
        static_cast<void *>(communicated_tables.column(icol).head<char>() +
                            recv_offsets[mpi_rank] * dtype_size),
        static_cast<const void *>(input_table.column(icol).head<char>() +
                                  static_cast<int64_t>(send_offsets[mpi_rank]) * dtype_size),
        (recv_offsets[mpi_rank + 1] - recv_offsets[mpi_rank]) * dtype_size,
        cudaMemcpyDeviceToDevice));
    } else {
      // This is a string column
      CUDA_RT_CALL(
        cudaMemcpy(string_sizes_recv[icol].data() + recv_offsets[mpi_rank],
                   string_sizes_send[icol].data() + send_offsets[mpi_rank],
                   (recv_offsets[mpi_rank + 1] - recv_offsets[mpi_rank]) * sizeof(cudf::size_type),
                   cudaMemcpyDeviceToDevice));

      CUDA_RT_CALL(cudaMemcpy(
        communicated_tables.column(icol).child(1).head<char>() +
          string_recv_offsets[icol][mpi_rank],
        input_table.column(icol).child(1).head<char>() + string_send_offsets[icol][mpi_rank],
        string_send_offsets[icol][mpi_rank + 1] - string_send_offsets[icol][mpi_rank],
        cudaMemcpyDeviceToDevice));
    }
  }
}

AllToAllCommunicator::AllToAllCommunicator(
  cudf::table_view input_table,
  std::vector<cudf::size_type> offsets,
  Communicator *communicator,
  std::vector<ColumnCompressionOptions> compression_options,
  bool explicit_copy_to_self)
  : input_table(input_table),
    communicator(communicator),
    explicit_copy_to_self(explicit_copy_to_self),
    send_offsets(offsets),
    compression_options(compression_options)
{
  /* Communicate number of rows */

  communicate_sizes(send_offsets, recv_offsets, communicator);

  /* Communicate the number of bytes of string columns */

  gather_string_offsets(
    input_table, send_offsets, string_send_offsets, string_recv_offsets, communicator);

  /* Calculate the number of bytes from string offsets */

  calculate_string_sizes_from_offsets(
    input_table, offsets.front(), offsets.back(), string_sizes_to_send);

  allocate_string_sizes_receive_buffer(input_table, recv_offsets, string_sizes_received);
}

std::unique_ptr<cudf::table> AllToAllCommunicator::allocate_communicated_table()
{
  std::unique_ptr<cudf::table> communicated_tables =
    allocate_communicated_table_helper(input_table, recv_offsets, string_recv_offsets);

  // Synchronization on the default stream is necessary here because subsequently the communicator
  // can use a different stream to receive data into allocated tables
  CUDA_RT_CALL(cudaStreamSynchronize(0));

  if (explicit_copy_to_self) {
    // The device-to-device memory copies are performed explicitly here before all-to-all
    // communication and local join, because if they are part of the communication, they could block
    // the host thread (even if they are launched on different streams) while the local join kernel
    // is running, limiting the efficacy of overlapping.

    copy_to_self(input_table,
                 communicated_tables->mutable_view(),
                 send_offsets,
                 recv_offsets,
                 string_send_offsets,
                 string_recv_offsets,
                 string_sizes_to_send,
                 string_sizes_received,
                 communicator);
  }

  return communicated_tables;
}

void AllToAllCommunicator::launch_communication(cudf::mutable_table_view communicated_table,
                                                bool report_timing,
                                                void *preallocated_pinned_buffer)
{
  vector<AllToAllCommBuffer> all_to_all_comm_buffers;

  append_to_all_to_all_comm_buffers(input_table,
                                    communicated_table,
                                    send_offsets,
                                    recv_offsets,
                                    string_send_offsets,
                                    string_recv_offsets,
                                    string_sizes_to_send,
                                    string_sizes_received,
                                    all_to_all_comm_buffers,
                                    compression_options);

  if (communicator->group_by_batch()) communicator->start();

  all_to_all_comm(all_to_all_comm_buffers,
                  communicator,
                  !explicit_copy_to_self,
                  report_timing,
                  preallocated_pinned_buffer);

  if (communicator->group_by_batch()) communicator->stop();

  postprocess_all_to_all_comm(
    all_to_all_comm_buffers, communicator, !explicit_copy_to_self, report_timing);

  calculate_string_offsets_from_sizes(communicated_table, string_sizes_received);
}
