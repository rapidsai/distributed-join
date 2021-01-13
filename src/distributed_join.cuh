/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>
#include <rmm/thrust_rmm_allocator.h>
#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <cascaded.hpp>
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
#include <nvcomp.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include "comm.cuh"
#include "communicator.h"
#include "error.cuh"
#include "utility.cuh"

using cudf::column;
using cudf::table;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

enum class CompressionMethod { cascaded, lz4 };

struct AllToAllCommBuffer {
  // the buffer to be all-to-all communicated
  const void *send_buffer;
  // the receive buffer for all-to-all communication
  void *recv_buffer;
  // vector of size `mpi_size`, representing the start index of items in `send_buffer` to be sent to
  // each rank
  std::vector<cudf::size_type> send_offsets;
  // vector of size `mpi_size`, representing the start index of items in `recv_offsets` to
  // receive data from each rank
  std::vector<int64_t> recv_offsets;
  // data type of each element
  cudf::data_type dtype;
  // whether to compress buffer before communication
  bool compression;
  // if using compression, the compression method used
  CompressionMethod compression_method;
  // compressed `send_buffer` to be all-to-all communicated
  rmm::device_buffer compressed_send_buffer;
  // the receive buffer for the compressed data
  rmm::device_buffer compressed_recv_buffer;
  // vector of size `mpi_size`, representing the start byte in `compressed_send_buffer` to be sent
  // to each rank
  std::vector<cudf::size_type> compressed_send_offsets;
  // vector of size `mpi_size`, representing the start byte in `compressed_recv_buffer` to
  // receive data from each rank
  std::vector<int64_t> compressed_recv_offsets;
};

/**
 * Communicate number of elements recieved from each rank during all-to-all communication.
 *
 * Note: This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] send_offset Vector of length mpi_size + 1 such that `send_offset[i+1] -
 * send_offset[i]` is the number of elements sent from the current rank to rank i during the
 * all-to-all communication.
 * @param[out] recv_offset Vector of length mpi_size + 1 such that `recv_offset[i+1] -
 * recv_offset[i]` is the number of elements received from rank i during the all-to-all
 * communication. The vector will be resized in this function and does not need to be preallocated.
 */
void communicate_sizes(vector<int64_t> const &send_offset,
                       vector<int64_t> &recv_offset,
                       Communicator *communicator)
{
  int mpi_size = communicator->mpi_size;
  vector<int64_t> send_count(mpi_size, -1);

  for (int irank = 0; irank < mpi_size; irank++) {
    send_count[irank] = send_offset[irank + 1] - send_offset[irank];
  }

  vector<int64_t> recv_count(mpi_size, -1);

  // Note: MPI is used for communicating the sizes instead of *Communicator* because
  // *Communicator* is not guaranteed to be able to send/recv host buffers.

  vector<MPI_Request> send_req(mpi_size);
  vector<MPI_Request> recv_req(mpi_size);

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

void communicate_sizes(vector<cudf::size_type> const &send_offset,
                       vector<int64_t> &recv_offset,
                       Communicator *communicator)
{
  communicate_sizes(
    vector<int64_t>(send_offset.begin(), send_offset.end()), recv_offset, communicator);
}

struct compression_functor {
  /**
   * Compress a buffer using cascaded compression.
   *
   * @param[in] uncompressed_data Input buffer to be compressed.
   * @param[in] uncompressed_count Number of elements to be compressed. Note that in general this
   * is different from the size of the buffer.
   * @param[out] compressed_data Output data after cascaded compression. This argument does not
   * need to be preallocated.
   * @param[out] compressed_size Number of bytes of *compressed_data*.
   */
  template <typename T,
            std::enable_if_t<not cudf::is_timestamp_t<T>::value and
                             not cudf::is_duration_t<T>::value> * = nullptr>
  void operator()(const void *uncompressed_data,
                  size_t uncompressed_count,
                  rmm::device_buffer &compressed_data,
                  size_t &compressed_size)
  {
    if (uncompressed_count == 0) {
      compressed_size = 0;
      return;
    }

    nvcomp::CascadedCompressor<T> compressor(
      static_cast<const T *>(uncompressed_data), uncompressed_count, 1, 1, true);

    const size_t temp_size = compressor.get_temp_size();
    rmm::device_buffer temp_space(temp_size);
    compressed_size = compressor.get_max_output_size(temp_space.data(), temp_size);
    compressed_data = rmm::device_buffer(compressed_size);

    // Set the output buffer to 0 to get away a bug in nvcomp
    CUDA_RT_CALL(cudaMemset(compressed_data.data(), 0, compressed_size));

    compressor.compress_async(
      temp_space.data(), temp_size, compressed_data.data(), &compressed_size, cudaStreamDefault);
    CUDA_RT_CALL(cudaStreamSynchronize(cudaStreamDefault));
  }

  template <
    typename T,
    std::enable_if_t<cudf::is_timestamp_t<T>::value or cudf::is_duration_t<T>::value> * = nullptr>
  void operator()(const void *uncompressed_data,
                  size_t uncompressed_count,
                  rmm::device_buffer &compressed_data,
                  size_t &compressed_size)
  {
    // If the data type is duration or time, use the corresponding arithmetic type
    operator()<typename T::rep>(
      uncompressed_data, uncompressed_count, compressed_data, compressed_size);
  }
};

struct decompressor_functor {
  /**
   * Decompress a buffer previously compressed by `compression_functor{}.operator()`.
   *
   * @param[in] compressed_data Input data to be decompressed.
   * @param[in] compressed_size Size of *compressed_data* in bytes.
   * @param[out] output Decompressed output. This argument needs to be preallocated.
   * @param[in] expected_output_count Expected number of elements in the decompressed buffer.
   */
  template <typename T,
            std::enable_if_t<not cudf::is_timestamp_t<T>::value and
                             not cudf::is_duration_t<T>::value> * = nullptr>
  void operator()(const void *compressed_data,
                  size_t compressed_size,
                  void *output,
                  size_t expected_output_count)
  {
    if (expected_output_count == 0) return;

    nvcomp::Decompressor<T> decompressor(compressed_data, compressed_size, cudaStreamDefault);
    const size_t output_count = decompressor.get_num_elements();
    assert(output_count == expected_output_count);

    const size_t temp_size = decompressor.get_temp_size();
    rmm::device_buffer temp_space(temp_size);

    decompressor.decompress_async(
      temp_space.data(), temp_size, static_cast<T *>(output), output_count, cudaStreamDefault);
    CUDA_RT_CALL(cudaStreamSynchronize(cudaStreamDefault));
  }

  template <
    typename T,
    std::enable_if_t<cudf::is_timestamp_t<T>::value or cudf::is_duration_t<T>::value> * = nullptr>
  void operator()(const void *compressed_data,
                  size_t compressed_size,
                  void *output,
                  size_t expected_output_count)
  {
    // If the data type is duration or time, use the corresponding arithmetic type
    operator()<typename T::rep>(compressed_data, compressed_size, output, expected_output_count);
  }
};

/**
 * Local join thread used for merging incoming partitions and performing local joins.
 *
 * @param[in] communicated_left Left table after all-to-all communication.
 * @param[in] communicated_right Right table after all-to-all communication.
 * @param[out] batch_join_results Inner join result of each batch.
 * @param[in] left_on Column indices from the left table to join on. This argument will be passed
 *     directly *cudf::inner_join*.
 * @param[in] right_on Column indices from the right table to join on. This argument will be
 * passed directly *cudf::inner_join*.
 * @param[in] columns_in_common Vector of pairs of column indices from the left and right table
 * that are in common and only one column will be produced in *batch_join_results*. This argument
 *     will be passed directly *cudf::inner_join*.
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

  for (int ibatch = 0; ibatch < flags.size(); ibatch++) {
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
 * Calculate and communicate the number of bytes sent/received during all-to-all communication for
 * all string columns in all batches.
 *
 * Note: This function needs to be called collectively by all ranks in `MPI_COMM_WORLD`.
 *
 * @param[in] table Table that needs to be all-to-all communicated.
 * @param[in] offsets Vector of size `mpi_size * over_decom_factor` indexed into `table`, such that
 * `offsets[i * mpi_size + k]` is the start index of rows needs to be sent to rank `k` during batch
 * `i`.
 * @param[in] over_decom_factor Number of batches.
 * @param[out] string_send_offsets Vector with shape `(num_batches, num_columns, mpi_size)`, such
 * that `string_send_offsets[i,j,k]` representing the start index in the char subcolumn of column
 * `j` that needs to be sent to rank `k` during batch `i`.
 * @param[out] string_recv_offsets Vector with shape `(num_batches, num_columns, mpi_size)`, such
 * that `string_recv_offsets[i,j,k]` representing the start index in the char subcolumn of column
 * `j` that receives data from rank `k` during batch `i`.
 */
void gather_string_offsets(cudf::table_view table,
                           vector<cudf::size_type> const &offsets,
                           const int over_decom_factor,
                           vector<vector<vector<cudf::size_type>>> &string_send_offsets,
                           vector<vector<vector<int64_t>>> &string_recv_offsets,
                           Communicator *communicator)
{
  int mpi_size = communicator->mpi_size;
  string_send_offsets.resize(over_decom_factor);
  string_recv_offsets.resize(over_decom_factor);

  for (int ibatch = 0; ibatch < over_decom_factor; ibatch++) {
    size_t start_idx = ibatch * mpi_size;
    size_t end_idx   = (ibatch + 1) * mpi_size + 1;
    thrust::device_vector<cudf::size_type> d_offset(&offsets[start_idx], &offsets[end_idx]);

    for (cudf::size_type icol = 0; icol < table.num_columns(); icol++) {
      // 1. If not a string column, push an empty vector
      cudf::data_type dtype = table.column(icol).type();
      if (dtype.id() != cudf::type_id::STRING) {
        string_send_offsets[ibatch].emplace_back();
        string_recv_offsets[ibatch].emplace_back();
        continue;
      } else {
        string_send_offsets[ibatch].emplace_back(mpi_size + 1);
        string_recv_offsets[ibatch].emplace_back(mpi_size + 1);
      }

      // 2. Gather `string_send_offsets` from offset subcolumn and `offsets`
      thrust::device_vector<cudf::size_type> d_string_send_offsets(mpi_size + 1);
      thrust::gather(rmm::exec_policy(rmm::cuda_stream_default)->on(0),
                     d_offset.begin(),
                     d_offset.end(),
                     thrust::device_ptr<const cudf::size_type>(
                       table.column(icol).child(0).head<cudf::size_type>()),
                     d_string_send_offsets.begin());
      CUDA_RT_CALL(cudaMemcpy(string_send_offsets[ibatch][icol].data(),
                              thrust::raw_pointer_cast(d_string_send_offsets.data()),
                              (mpi_size + 1) * sizeof(cudf::size_type),
                              cudaMemcpyDeviceToHost));

      // 3. Communicate string_send_offsets and receive string_recv_offsets
      communicate_sizes(
        string_send_offsets[ibatch][icol], string_recv_offsets[ibatch][icol], communicator);
    }
  }
}

/**
 * Allocate tables after all-to-all communication for all batches.
 *
 * @param[in] input_table Table that needs to be all-to-all communicated.
 * @param[in] recv_offsets Vector of size `(num_batches, mpi_size)`, such that `recv_offsets[i,j]`
 * is the start row index to be sent to rank `j` in batch `i`.
 * @param[in] string_recv_offsets Vector with shape `(num_batches, num_columns, mpi_size)`. The
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
 * Calculate the string size of each row.
 *
 * Note: This function is the reverse of `calculate_string_offsets_from_sizes`.
 *
 * @param[in] input_table Table for which the string sizes are calculated.
 * @param[out] output_sizes Vector of size `num_columns`, where `output_sizes[j]` stores the string
 * size of each row in column `j`.
 */
void calculate_string_sizes_from_offsets(cudf::table_view input_table,
                                         vector<rmm::device_buffer> &output_sizes)
{
  output_sizes.clear();

  for (cudf::size_type icol = 0; icol < input_table.num_columns(); icol++) {
    cudf::column_view input_column = input_table.column(icol);
    if (input_column.type().id() != cudf::type_id::STRING) {
      output_sizes.emplace_back();
      continue;
    }

    output_sizes.emplace_back(input_column.size() * sizeof(cudf::size_type));

    // Assume the first entry of the offset subcolumn is always 0
    thrust::adjacent_difference(
      // rmm::exec_policy(rmm::cuda_stream_default)->on(0),
      thrust::device_ptr<const cudf::size_type>(
        input_column.child(0).begin<const cudf::size_type>() + 1),
      thrust::device_ptr<const cudf::size_type>(input_column.child(0).end<cudf::size_type>()),
      thrust::device_ptr<cudf::size_type>(
        static_cast<cudf::size_type *>(output_sizes[icol].data())));
  }
}

/**
 * Calculate string offsets from sizes.
 *
 * Note: This function is the reverse of `calculate_string_sizes_from_offsets`.
 *
 * @param[out] output_table Calculated offsets will be stored in the string columns of
 * `output_table`.
 * @param[in] input_sizes Vector of size `num_columns`, where `input_sizes[j]` stores the string
 * size of each row in column `j`.
 */
void calculate_string_offsets_from_sizes(cudf::mutable_table_view output_table,
                                         vector<rmm::device_buffer> const &input_sizes)
{
  for (cudf::size_type icol = 0; icol < output_table.num_columns(); icol++) {
    cudf::mutable_column_view output_column = output_table.column(icol);
    if (output_column.type().id() != cudf::type_id::STRING) continue;

    cudf::size_type nrows = output_column.size();
    const cudf::size_type *sizes_start =
      static_cast<const cudf::size_type *>(input_sizes[icol].data());
    const cudf::size_type *sizes_end = sizes_start + nrows;
    thrust::inclusive_scan(
      // rmm::exec_policy(rmm::cuda_stream_default)->on(0),
      thrust::device_ptr<const cudf::size_type>(sizes_start),
      thrust::device_ptr<const cudf::size_type>(sizes_end),
      thrust::device_ptr<cudf::size_type>(
        static_cast<cudf::size_type *>(output_column.child(0).head())) +
        1);
    CUDA_TRY(cudaMemsetAsync(output_column.child(0).head(), 0, sizeof(cudf::size_type), 0));
  }
}

/**
 * Helper function for allocating the receive buffer of string sizes.
 */
void allocate_string_sizes_receive_buffer(cudf::table_view input_table,
                                          int over_decom_factor,
                                          vector<vector<int64_t>> recv_offsets,
                                          vector<vector<rmm::device_buffer>> &string_sizes_recv)
{
  string_sizes_recv.clear();
  string_sizes_recv.resize(over_decom_factor);

  for (int ibatch = 0; ibatch < over_decom_factor; ibatch++) {
    for (cudf::size_type icol = 0; icol < input_table.num_columns(); icol++) {
      if (input_table.column(icol).type().id() != cudf::type_id::STRING) {
        string_sizes_recv[ibatch].emplace_back();
      } else {
        string_sizes_recv[ibatch].emplace_back(recv_offsets[ibatch].back() *
                                               sizeof(cudf::size_type));
      }
    }
  }
}

void copy_to_self(cudf::table_view input_table,
                  vector<std::unique_ptr<table>> &communicated_tables,
                  vector<cudf::size_type> const &send_offsets,
                  vector<vector<int64_t>> const &recv_offsets,
                  vector<vector<vector<cudf::size_type>>> const &string_send_offsets,
                  vector<vector<vector<int64_t>>> const &string_recv_offsets,
                  vector<rmm::device_buffer> const &string_sizes_send,
                  vector<vector<rmm::device_buffer>> &string_sizes_recv,
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
          cudaMemcpy(ADV_PTR<cudf::size_type>(string_sizes_recv[ibatch][icol].data(),
                                              recv_offsets[ibatch][mpi_rank]),
                     ADV_PTR<cudf::size_type>(string_sizes_send[icol].data(),
                                              send_offsets[ibatch * mpi_size + mpi_rank]),
                     (recv_offsets[ibatch][mpi_rank + 1] - recv_offsets[ibatch][mpi_rank]) *
                       sizeof(cudf::size_type),
                     cudaMemcpyDeviceToDevice));

        CUDA_RT_CALL(cudaMemcpy(
          ADV_PTR(communicated_tables[ibatch]->mutable_view().column(icol).child(1).head(),
                  string_recv_offsets[ibatch][icol][mpi_rank]),
          ADV_PTR(input_table.column(icol).child(1).head(),
                  string_send_offsets[ibatch][icol][mpi_rank]),
          string_send_offsets[ibatch][icol][mpi_rank + 1] -
            string_send_offsets[ibatch][icol][mpi_rank],
          cudaMemcpyDeviceToDevice));
      }
    }
  }
}

/**
 * Generate plans for all-to-all communication.
 *
 * Note: This function does not perform the actual communication. It simply puts required
 * information about all-to-all communication (e.g. the send buffer, the receive buffer, offsets,
 * whether to use compression etc.) into `all_to_all_comm_buffers`.
 *
 * @param[in] input Table to be all-to-all communicated.
 * @param[in] output Table after all-to-all communication. This argument needs to be preallocated.
 * The helper function `allocate_communicated_table` can be used to allocate this table.
 * @param[in] send_offset Vector of size `mpi_size + 1` such that `send_offset[i]` represents the
 * start index of `input` to be sent to rank `i`.
 * @param[in] recv_offset Vector of size `mpi_size + 1` such that `recv_offset[i]` represents the
 * start index of `output` to receive data from rank `i`.
 * @param[in] string_send_offsets Vector with shape `(num_columns, mpi_size)`, such
 * that `string_send_offsets[j,k]` representing the start index in the char subcolumn of column
 * `j` that needs to be sent to rank `k`, for the current batch. The helper function
 * `gather_string_offsets` can be used to generate this field.
 * @param[in] string_recv_offsets Vector with shape `(num_columns, mpi_size)`, such
 * that `string_recv_offsets[j,k]` representing the start index in the char subcolumn of column
 * `j` that receives data from rank `k`, for the current batch. The helper function
 * `gather_string_offsets` can be used to generate this field.
 * @param[in] string_sizes_send String sizes of each row for all string columns. The helper function
 * `calculate_string_sizes_from_offsets` can be used to generate this field.
 * @param[in] string_sizes_recv Receive buffers for string sizes. This argument needs to be
 * preallocated. The helper function `allocate_string_sizes_receive_buffer` can be used for
 * allocating the buffers.
 * @param[out] all_to_all_comm_buffers Each element in this vector represents a buffer that needs to
 * be all-to-all communicated.
 */
void preprocess_all_to_all_comm(cudf::table_view input,
                                cudf::mutable_table_view output,
                                vector<cudf::size_type> const &send_offsets,
                                vector<int64_t> const &recv_offsets,
                                vector<vector<cudf::size_type>> const &string_send_offsets,
                                vector<vector<int64_t>> const &string_recv_offsets,
                                vector<rmm::device_buffer> const &string_sizes_send,
                                vector<rmm::device_buffer> &string_sizes_recv,
                                vector<AllToAllCommBuffer> &all_to_all_comm_buffers,
                                bool compression)
{
  for (cudf::size_type icol = 0; icol < input.num_columns(); icol++) {
    cudf::data_type dtype = input.column(icol).type();
    assert(dtype == output.column(icol).type());

    if (dtype.id() != cudf::type_id::STRING) {
      // This is a fixed-width column
      all_to_all_comm_buffers.emplace_back();
      AllToAllCommBuffer &buffer = all_to_all_comm_buffers.back();
      buffer.send_buffer         = input.column(icol).head();
      buffer.recv_buffer         = output.column(icol).head();
      buffer.send_offsets        = send_offsets;
      buffer.recv_offsets        = recv_offsets;
      buffer.dtype               = dtype;
      buffer.compression         = compression;
      buffer.compression_method  = CompressionMethod::cascaded;
    } else {
      // This is a string column
      all_to_all_comm_buffers.emplace_back();
      AllToAllCommBuffer &offset_buffer = all_to_all_comm_buffers.back();
      offset_buffer.send_buffer         = string_sizes_send[icol].data();
      offset_buffer.recv_buffer         = string_sizes_recv[icol].data();
      offset_buffer.send_offsets        = send_offsets;
      offset_buffer.recv_offsets        = recv_offsets;
      offset_buffer.dtype               = input.column(icol).child(0).type();
      offset_buffer.compression         = compression;
      offset_buffer.compression_method  = CompressionMethod::cascaded;

      all_to_all_comm_buffers.emplace_back();
      AllToAllCommBuffer &string_buffer = all_to_all_comm_buffers.back();
      string_buffer.send_buffer         = input.column(icol).child(1).head();
      string_buffer.recv_buffer         = output.column(icol).child(1).head();
      string_buffer.send_offsets        = string_send_offsets[icol];
      string_buffer.recv_offsets        = string_recv_offsets[icol];
      string_buffer.dtype               = input.column(icol).child(1).type();
      string_buffer.compression         = compression;
      string_buffer.compression_method  = CompressionMethod::cascaded;
    }
  }
}

void preprocess_all_to_all_comm(cudf::table_view input,
                                cudf::mutable_table_view output,
                                vector<cudf::size_type> const &send_offsets,
                                vector<int64_t> const &recv_offsets,
                                vector<AllToAllCommBuffer> &all_to_all_comm_buffers,
                                bool compression)
{
  vector<rmm::device_buffer> string_sizes_recv;

  preprocess_all_to_all_comm(input,
                             output,
                             send_offsets,
                             recv_offsets,
                             vector<vector<cudf::size_type>>(),
                             vector<vector<int64_t>>(),
                             vector<rmm::device_buffer>(),
                             string_sizes_recv,
                             all_to_all_comm_buffers,
                             compression);
}

/**
 * Perform all-to-all communication of a single batch according to plans.
 *
 * Note: If the communicator supports grouping by batches, this call is nonblocking and should
 * be enclosed by `communicator->start()` and `communicator->stop()`.
 *
 * This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] all_to_all_comm_buffers Plans for all-to-all communication, generated by
 * `preprocess_all_to_all_comm`.
 * @param[in] communicator An instance of `Communicator` used for communication.
 * @param[in] include_self If true, this function will send the partition destined to the current
 * rank.
 */
void all_to_all_comm(vector<AllToAllCommBuffer> &all_to_all_comm_buffers,
                     Communicator *communicator,
                     bool include_self  = true,
                     bool report_timing = false)
{
  int mpi_rank = communicator->mpi_rank;
  int mpi_size = communicator->mpi_size;

  double start_time;
  double stop_time;
  double total_compression_time  = 0.0;
  double total_uncompressed_size = 0.0;
  double total_compressed_size   = 0.0;

  for (auto &buffer : all_to_all_comm_buffers) {
    if (!buffer.compression) {
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
    // compression is done, we can allocate the compressed send buffer and copy the compressed data
    // into the buffer. The all-to-all communication can reuse the `send_data_by_offset` and
    // `recv_data_by_offset` functions.

    if (report_timing) { start_time = MPI_Wtime(); }

    // Compress each partition in the send buffer separately and store the result in
    // `compressed_buffers`
    vector<rmm::device_buffer> compressed_buffers(mpi_size);
    vector<size_t> compressed_buffer_sizes(mpi_size, 0);

    for (int irank = 0; irank < mpi_size; irank++) {
      if (!include_self && irank == mpi_rank) continue;

      cudf::type_dispatcher(
        buffer.dtype,
        compression_functor{},
        ADV_PTR(buffer.send_buffer, buffer.send_offsets[irank] * cudf::size_of(buffer.dtype)),
        buffer.send_offsets[irank + 1] - buffer.send_offsets[irank],
        compressed_buffers[irank],
        compressed_buffer_sizes[irank]);
    }

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
      total_uncompressed_size += (buffer.send_offsets.back() * cudf::size_of(buffer.dtype));
      total_compressed_size += buffer.compressed_send_offsets.back();
    }

    communicate_sizes(buffer.compressed_send_offsets, buffer.compressed_recv_offsets, communicator);

    // Merge compressed data of all partitions in `compressed_buffers` into a single buffer
    buffer.compressed_send_buffer.resize(buffer.compressed_send_offsets.back());
    for (int irank = 0; irank < mpi_size; irank++) {
      if (!include_self && irank == mpi_rank) continue;

      CUDA_RT_CALL(cudaMemcpy(
        ADV_PTR(buffer.compressed_send_buffer.data(), buffer.compressed_send_offsets[irank]),
        compressed_buffers[irank].data(),
        compressed_buffer_sizes[irank],
        cudaMemcpyDeviceToDevice));
    }
    compressed_buffers.clear();

    // Allocate receive buffer and launch all-to-all communication on the compressed buffer
    buffer.compressed_recv_buffer.resize(buffer.compressed_recv_offsets.back());

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

  if (report_timing) {
    std::cout << "Rank " << mpi_rank << ": compression takes " << total_compression_time * 1e3
              << "ms"
              << " with compression ratio " << total_uncompressed_size / total_compressed_size
              << " and throughput " << total_uncompressed_size / total_compression_time / 1e9
              << "GB/s" << std::endl;
  }
}

/**
 * Actions to be performed after all-to-all communication is finished.
 *
 * Note: The arguments of this function need to match those of `all_to_all_comm`.
 */
void postprocess_all_to_all_comm(vector<AllToAllCommBuffer> &all_to_all_comm_buffers,
                                 Communicator *communicator,
                                 bool include_self  = true,
                                 bool report_timing = false)
{
  int mpi_rank = communicator->mpi_rank;
  int mpi_size = communicator->mpi_size;
  double start_time;
  double stop_time;
  double total_uncompressed_size = 0.0;

  if (report_timing) { start_time = MPI_Wtime(); }

  // Decompress compressed data into destination buffer
  for (auto &buffer : all_to_all_comm_buffers) {
    if (!buffer.compression) continue;

    for (int irank = 0; irank < mpi_size; irank++) {
      if (!include_self && irank == mpi_rank) continue;

      cudf::type_dispatcher(
        buffer.dtype,
        decompressor_functor{},
        ADV_PTR(buffer.compressed_recv_buffer.data(), buffer.compressed_recv_offsets[irank]),
        buffer.compressed_recv_offsets[irank + 1] - buffer.compressed_recv_offsets[irank],
        ADV_PTR(buffer.recv_buffer, buffer.recv_offsets[irank] * cudf::size_of(buffer.dtype)),
        buffer.recv_offsets[irank + 1] - buffer.recv_offsets[irank]);
    }

    if (report_timing)
      total_uncompressed_size += (buffer.recv_offsets.back() * cudf::size_of(buffer.dtype));
  }

  if (report_timing) {
    stop_time       = MPI_Wtime();
    double duration = stop_time - start_time;
    std::cout << "Rank " << mpi_rank << ": decompression takes " << duration * 1e3 << "ms"
              << " with throughput " << total_uncompressed_size / duration / 1e9 << "GB/s"
              << std::endl;
  }
}

/**
 * Top level interface for distributed inner join.
 *
 * This function should be called collectively by all processes in MPI_COMM_WORLD. All arguments
 * are significant for all ranks.
 *
 * The argument `left` and `right` are the left table and the right table distributed on each
 * rank. In other word, the left (right) table to be joined is the concatenation of `left`
 * (`right`) on all ranks. If the whole tables reside on a single rank, you should use
 * *distribute_table* to distribute the table before calling this function.
 *
 * @param[in] left The left table distributed on each rank.
 * @param[in] right The right table distributed on each rank.
 * @param[in] left_on The column indices from `left` to join on.
 * The column from `left` indicated by `left_on[i]` will be compared against the column
 * from `right` indicated by `right_on[i]`.
 * @param[in] right_on The column indices from `right` to join on.
 * The column from `right` indicated by `right_on[i]` will be compared against the column
 * from `left` indicated by `left_on[i]`.
 * @param[in] columns_in_common is a vector of pairs of column indices into
 * `left` and `right`, respectively, that are "in common". For "common"
 * columns, only a single output column will be produced, which is gathered
 * from `left_on` columns. Else, for every column in `left_on` and `right_on`,
 * an output column will be produced.  For each of these pairs (L, R), L
 * should exist in `left_on` and R should exist in `right_on`.
 * @param[in] communicator An instance of `Communicator` used for communication.
 * @param[in] over_decom_factor Over-decomposition factor used for overlapping computation and
 * communication.
 * @param[in] report_timing Whether collect and print timing.
 * @return Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`. The resulting table will be joined columns of
 * `left(including common columns)+right(excluding common columns)`. The join result is the
 * concatenation of the returned tables on all ranks.
 */
std::unique_ptr<table> distributed_inner_join(
  cudf::table_view const &left,
  cudf::table_view const &right,
  vector<cudf::size_type> const &left_on,
  vector<cudf::size_type> const &right_on,
  vector<std::pair<cudf::size_type, cudf::size_type>> const &columns_in_common,
  Communicator *communicator,
  int over_decom_factor = 1,
  bool compression      = false,
  bool report_timing    = false)
{
  if (over_decom_factor == 1) {
    // @TODO: If over_decom_factor is 1, there is no opportunity for overlapping. Therefore,
    // we can get away with using just one thread.
  }

  int mpi_rank = communicator->mpi_rank;
  int mpi_size = communicator->mpi_size;
  std::chrono::time_point<high_resolution_clock> start_time;
  std::chrono::time_point<high_resolution_clock> stop_time;

  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource();

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
  // subcolumn. For the char subcolumn, we need to first gather the offsets in this string subcolumn
  // of all ranks by using `gather_string_offsets`, and then it can be all-to-all communicated using
  // the gathered offsets. For the offset subcolumn, we can first calculate the sizes of all rows
  // by calculating the adjacent differences. Then, the sizes are all-to-all communicated. Once the
  // all-to-all communication finishes, on target rank we can reconstruct the offset subcolumn by
  // using a scan on sizes.

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

  vector<rmm::device_buffer> string_sizes_send_left;
  vector<rmm::device_buffer> string_sizes_send_right;
  vector<vector<rmm::device_buffer>> string_sizes_recv_left;
  vector<vector<rmm::device_buffer>> string_sizes_recv_right;

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

    preprocess_all_to_all_comm(
      hashed_left->view(),
      communicated_left[ibatch]->mutable_view(),
      vector<cudf::size_type>(&left_offset[start_idx], &left_offset[end_idx]),
      recv_offsets_left[ibatch],
      string_send_offsets_left[ibatch],
      string_recv_offsets_left[ibatch],
      string_sizes_send_left,
      string_sizes_recv_left[ibatch],
      all_to_all_comm_buffers,
      compression);

    preprocess_all_to_all_comm(
      hashed_right->view(),
      communicated_right[ibatch]->mutable_view(),
      vector<cudf::size_type>(&right_offset[start_idx], &right_offset[end_idx]),
      recv_offsets_right[ibatch],
      string_send_offsets_right[ibatch],
      string_recv_offsets_right[ibatch],
      string_sizes_send_right,
      string_sizes_recv_right[ibatch],
      all_to_all_comm_buffers,
      compression);

    // all-to-all communication for the ibatch
    if (communicator->group_by_batch()) communicator->start();

    all_to_all_comm(all_to_all_comm_buffers, communicator, false, report_timing);

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

/**
 * This helper function runs compression and decompression on a small buffer to avoid nvcomp's
 * setup time during the actual run.
 */
void warmup_nvcomp()
{
  constexpr size_t warmup_size = 1000;
  rmm::device_buffer input_data(warmup_size * sizeof(int));
  rmm::device_buffer compressed_data;
  rmm::device_buffer decompressed_data(warmup_size * sizeof(int));
  size_t compressed_size;

  compression_functor{}.operator()<int>(
    input_data.data(), warmup_size, compressed_data, compressed_size);

  decompressor_functor{}.operator()<int>(
    compressed_data.data(), compressed_size, decompressed_data.data(), warmup_size);
}
