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

#ifndef __DISTRIBUTED_JOIN
#define __DISTRIBUTED_JOIN

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
#include <cascaded.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/hashing.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <nvcomp.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include "comm.cuh"
#include "communicator.h"
#include "error.cuh"

using cudf::column;
using cudf::table;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

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

/**
 * All-to-all communication of a single batch.
 *
 * Note: This call is nonblocking and should be enclosed by `communicator->start()` and
 * `communicator->stop()`.
 *
 * This function needs to be called collectively by all ranks in MPI_COMM_WORLD. For every ranks,
 * all arguments are significant.
 *
 * @param[in] input Table to be communicated.
 * @param[out] output Table after all-to-all communication. This argument needs to be preallocated.
 * @param[in] send_offset Vector of size `mpi_size + 1` such that `send_offset[i]` represents the
 * start index of `input` to be sent to rank `i`.
 * @param[in] recv_offset Vector of size `mpi_size + 1` such that `recv_offset[i]` represents the
 * start index of `output` to receive data from rank `i`.
 * @param[in] communicator An instance of `Communicator` used for communication.
 * @param[in] include_self If true, this function will send the partition destined to the current
 * rank.
 */
void all_to_all_comm(cudf::table_view input,
                     cudf::mutable_table_view output,
                     vector<cudf::size_type> const &send_offset,
                     vector<int64_t> const &recv_offset,
                     Communicator *communicator,
                     bool include_self = true)
{
  for (cudf::size_type icol = 0; icol < input.num_columns(); icol++) {
    cudf::data_type dtype      = input.column(icol).type();
    cudf::size_type dtype_size = cudf::size_of(dtype);
    if (!communicator->group_by_batch()) communicator->start();

    send_data_by_offset(
      input.column(icol).head(), send_offset, dtype_size, communicator, include_self);

    recv_data_by_offset(
      output.column(icol).head(), recv_offset, dtype_size, communicator, include_self);

    if (!communicator->group_by_batch()) communicator->stop();
  }
}

struct compression_functor {
  template <typename T, std::enable_if_t<std::is_integral<T>::value> * = nullptr>
  void operator()(const void *uncompressed_data,
                  size_t uncompressed_count,
                  rmm::device_buffer &compressed_data,
                  size_t &compressed_size)
  {
    nvcomp::CascadedCompressor<T> compressor(
      static_cast<const T *>(uncompressed_data), uncompressed_count, 1, 1, true);

    const size_t temp_size = compressor.get_temp_size();
    rmm::device_buffer temp_space(temp_size);
    compressed_size = compressor.get_max_output_size(temp_space.data(), temp_size);
    compressed_data = rmm::device_buffer(compressed_size);

    compressor.compress_async(
      temp_space.data(), temp_size, compressed_data.data(), &compressed_size, cudaStreamDefault);
    CUDA_RT_CALL(cudaStreamSynchronize(cudaStreamDefault));
  }

  template <typename T, std::enable_if_t<!std::is_integral<T>::value> * = nullptr>
  void operator()(const void *uncompressed_data,
                  size_t uncompressed_count,
                  rmm::device_buffer &compressed_data,
                  size_t &compressed_size)
  {
    assert(false && "Unsupported data type for cascaded compressor");
  }
};

struct decompressor_functor {
  template <typename T>
  void operator()(const void *compressed_data,
                  size_t compressed_size,
                  void *output,
                  size_t expected_output_count)
  {
    nvcomp::Decompressor<T> decompressor(compressed_data, compressed_size, cudaStreamDefault);
    const size_t output_count = decompressor.get_num_elements();
    assert(output_count == expected_output_count);

    const size_t temp_size = decompressor.get_temp_size();
    rmm::device_buffer temp_space(temp_size);

    decompressor.decompress_async(
      temp_space.data(), temp_size, static_cast<T *>(output), output_count, cudaStreamDefault);
    CUDA_RT_CALL(cudaStreamSynchronize(cudaStreamDefault));
  }
};

void all_to_all_comm_with_compression(cudf::table_view input,
                                      vector<cudf::size_type> const &send_offset,
                                      vector<rmm::device_buffer> &compressed_input,
                                      vector<rmm::device_buffer> &compressed_output,
                                      vector<vector<int64_t>> &compressed_recv_offset,
                                      Communicator *communicator,
                                      bool include_self = true)
{
  cudf::size_type ncols = input.num_columns();
  compressed_input.resize(ncols);
  compressed_output.resize(ncols);
  compressed_recv_offset.resize(ncols);

  int mpi_rank{communicator->mpi_rank};
  int mpi_size{communicator->mpi_size};

  for (cudf::size_type icol = 0; icol < ncols; icol++) {
    cudf::column_view column   = input.column(icol);
    cudf::data_type dtype      = column.type();
    cudf::size_type dtype_size = cudf::size_of(dtype);

    vector<rmm::device_buffer> compressed_column(mpi_size);
    vector<size_t> compressed_column_size(mpi_size, 0);

    // compress each partition in the column separately and store the result in *compressed_column*
    for (int irank = 0; irank < mpi_size; irank++) {
      if (!include_self && irank == mpi_rank) continue;

      cudf::type_dispatcher(
        dtype,
        compression_functor{},
        static_cast<const void *>(column.head<char>() + send_offset[irank] * dtype_size),
        send_offset[irank + 1] - send_offset[irank],
        compressed_column[irank],
        compressed_column_size[irank]);
    }

    // calculate and communicate offsets for compressed column
    vector<int64_t> compressed_send_offset(mpi_size + 1, -1);
    compressed_send_offset[0] = 0;
    for (int irank = 0; irank < mpi_size; irank++) {
      compressed_send_offset[irank + 1] =
        compressed_send_offset[irank] + compressed_column_size[irank];
    }

    communicate_sizes(compressed_send_offset, compressed_recv_offset[icol], communicator);

    // merge compressed data of all partitions in *compressed_column* into a single buffer
    compressed_input[icol] = rmm::device_buffer(compressed_send_offset.back());
    for (int irank = 0; irank < mpi_size; irank++) {
      if (!include_self && irank == mpi_rank) continue;

      CUDA_RT_CALL(
        cudaMemcpy((void *)((char *)compressed_input[icol].data() + compressed_send_offset[irank]),
                   compressed_column[irank].data(),
                   compressed_column_size[irank],
                   cudaMemcpyDeviceToDevice));
    }

    compressed_column.clear();

    // allocate receive buffer and launch all-to-all communication
    compressed_output[icol] = rmm::device_buffer(compressed_recv_offset[icol].back());

    if (!communicator->group_by_batch()) communicator->start();

    send_data_by_offset(
      compressed_input[icol].data(), compressed_send_offset, 1, communicator, include_self);

    recv_data_by_offset(
      compressed_output[icol].data(), compressed_recv_offset[icol], 1, communicator, include_self);

    if (!communicator->group_by_batch()) communicator->stop();
  }
}

void all_to_all_comm_decompression(cudf::mutable_table_view output,
                                   vector<int64_t> const &recv_offset,
                                   vector<rmm::device_buffer> const &compressed_output,
                                   vector<vector<int64_t>> const &compressed_recv_offset,
                                   Communicator *communicator,
                                   bool include_self = true)
{
  int mpi_rank{communicator->mpi_rank};
  int mpi_size{communicator->mpi_size};
  cudf::size_type ncols = output.num_columns();

  for (cudf::size_type icol = 0; icol < ncols; icol++) {
    cudf::column_view column   = output.column(icol);
    cudf::data_type dtype      = column.type();
    cudf::size_type dtype_size = cudf::size_of(dtype);

    for (int irank = 0; irank < mpi_size; irank++) {
      if (!include_self && irank == mpi_rank) continue;

      cudf::type_dispatcher(
        dtype,
        decompressor_functor{},
        (void *)((char *)compressed_output[icol].data() + compressed_recv_offset[icol][irank]),
        compressed_recv_offset[icol][irank + 1] - compressed_recv_offset[icol][irank],
        (void *)(column.head<char>() + recv_offset[irank] * dtype_size),
        recv_offset[irank + 1] - recv_offset[irank]);
    }
  }
}

/**
 * Local join thread used for merging incoming partitions and performing local joins.
 *
 * @param[in] communicated_left Left table after all-to-all communication.
 * @param[in] communicated_right Right table after all-to-all communication.
 * @param[out] batch_join_results Inner join result of each batch.
 * @param[in] left_on Column indices from the left table to join on. This argument will be passed
 *     directly *cudf::inner_join*.
 * @param[in] right_on Column indices from the right table to join on. This argument will be passed
 *     directly *cudf::inner_join*.
 * @param[in] columns_in_common Vector of pairs of column indices from the left and right table that
 *     are in common and only one column will be produced in *batch_join_results*. This argument
 *     will be passed directly *cudf::inner_join*.
 * @param[in] flags *flags[i]* is true if and only if the ith batch has finished the all-to-all
 *     communication.
 * @param[in] report_timing Whether to print the local join time to stderr.
 * @param[in] mr: RMM memory resource.
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
      // If either is empty, the local join will return the other table, which is not desired.
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
      std::cerr << "Rank " << communicator->mpi_rank << ": Local join on batch " << ibatch
                << " takes " << duration.count() << "ms" << std::endl;
    }
  }
}

/**
 * Top level interface for distributed inner join.
 *
 * This function should be called collectively by all processes in MPI_COMM_WORLD. All arguments are
 * significant for all ranks.
 *
 * The argument `left` and `right` are the left table and the right table distributed on each rank.
 * In other word, the left (right) table to be joined is the concatenation of `left` (`right`) on
 * all ranks. If the whole tables reside on a single rank, you should use *distribute_table* to
 * distribute the table before calling this function.
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

  int mpi_rank{communicator->mpi_rank};
  int mpi_size{communicator->mpi_size};
  std::chrono::time_point<high_resolution_clock> start_time;
  std::chrono::time_point<high_resolution_clock> stop_time;

  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource();

  /* Hash partition */

  if (report_timing) { start_time = high_resolution_clock::now(); }

  std::unique_ptr<table> hashed_left;
  vector<cudf::size_type> left_offset;

  std::unique_ptr<table> hashed_right;
  vector<cudf::size_type> right_offset;

  std::tie(hashed_left, left_offset) = cudf::detail::hash_partition(
    left, left_on, mpi_size * over_decom_factor, mr, cudaStreamPerThread);

  std::tie(hashed_right, right_offset) = cudf::detail::hash_partition(
    right, right_on, mpi_size * over_decom_factor, mr, cudaStreamPerThread);

  CUDA_RT_CALL(cudaStreamSynchronize(cudaStreamPerThread));

  left_offset.push_back(left.num_rows());
  right_offset.push_back(right.num_rows());

  if (report_timing) {
    stop_time     = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop_time - start_time);
    std::cerr << "Rank " << mpi_rank << ": Hash partition takes " << duration.count() << "ms"
              << std::endl;
  }

  /* Communicate the offsets of each batch */

  // `recv_offsets[i, j]` represents the number of items received from rank `j` in batch `i`
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

  /* Declare storage for the table after all-to-all communication */

  // left table after all-to-all for each batch
  vector<std::unique_ptr<table>> communicated_left(over_decom_factor);
  // right table after all-to-all for each batch
  vector<std::unique_ptr<table>> communicated_right(over_decom_factor);

  for (int ibatch = 0; ibatch < over_decom_factor; ibatch++) {
    vector<std::unique_ptr<column>> communicated_left_columns;
    for (cudf::size_type icol = 0; icol < hashed_left->num_columns(); icol++) {
      communicated_left_columns.push_back(cudf::make_numeric_column(
        hashed_left->view().column(icol).type(), recv_offsets_left[ibatch].back()));
    }
    communicated_left[ibatch] = std::make_unique<table>(std::move(communicated_left_columns));

    vector<std::unique_ptr<column>> communicated_right_columns;
    for (cudf::size_type icol = 0; icol < hashed_right->num_columns(); icol++) {
      communicated_right_columns.push_back(cudf::make_numeric_column(
        hashed_right->view().column(icol).type(), recv_offsets_right[ibatch].back()));
    }
    communicated_right[ibatch] = std::make_unique<table>(std::move(communicated_right_columns));
  }

  CUDA_RT_CALL(cudaStreamSynchronize(cudaStreamDefault));

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

  for (int ibatch = 0; ibatch < over_decom_factor; ibatch++) {
    for (cudf::size_type icol = 0; icol < hashed_left->num_columns(); icol++) {
      cudf::data_type dtype      = hashed_left->view().column(icol).type();
      cudf::size_type dtype_size = cudf::size_of(dtype);

      CUDA_RT_CALL(cudaMemcpy(
        (void *)((char *)(communicated_left[ibatch]->mutable_view().column(icol).head()) +
                 recv_offsets_left[ibatch][mpi_rank] * dtype_size),
        (void *)((char *)(hashed_left->view().column(icol).head()) +
                 left_offset[ibatch * mpi_size + mpi_rank] * dtype_size),
        (recv_offsets_left[ibatch][mpi_rank + 1] - recv_offsets_left[ibatch][mpi_rank]) *
          dtype_size,
        cudaMemcpyDeviceToDevice));
    }

    for (cudf::size_type icol = 0; icol < hashed_right->num_columns(); icol++) {
      cudf::data_type dtype      = hashed_right->view().column(icol).type();
      cudf::size_type dtype_size = cudf::size_of(dtype);

      CUDA_RT_CALL(cudaMemcpy(
        (void *)((char *)(communicated_right[ibatch]->mutable_view().column(icol).head()) +
                 recv_offsets_right[ibatch][mpi_rank] * dtype_size),
        (void *)((char *)(hashed_right->view().column(icol).head()) +
                 right_offset[ibatch * mpi_size + mpi_rank] * dtype_size),
        (recv_offsets_right[ibatch][mpi_rank + 1] - recv_offsets_right[ibatch][mpi_rank]) *
          dtype_size,
        cudaMemcpyDeviceToDevice));
    }
  }

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

    // the start and end index for left_offset and right_offset for the ibatch
    size_t start_idx = ibatch * mpi_size;
    size_t end_idx   = (ibatch + 1) * mpi_size + 1;

    // temporary buffers for holding compressed data, if needed
    vector<rmm::device_buffer> compressed_input_left;
    vector<rmm::device_buffer> compressed_input_right;
    vector<rmm::device_buffer> compressed_output_left;
    vector<rmm::device_buffer> compressed_output_right;
    vector<vector<int64_t>> compressed_recv_offset_left;
    vector<vector<int64_t>> compressed_recv_offset_right;

    // all-to-all communication for the ibatch
    if (communicator->group_by_batch()) communicator->start();

    if (compression) {
      all_to_all_comm_with_compression(
        hashed_left->view(),
        vector<cudf::size_type>(&left_offset[start_idx], &left_offset[end_idx]),
        compressed_input_left,
        compressed_output_left,
        compressed_recv_offset_left,
        communicator,
        false);

      all_to_all_comm_with_compression(
        hashed_right->view(),
        vector<cudf::size_type>(&right_offset[start_idx], &right_offset[end_idx]),
        compressed_input_right,
        compressed_output_right,
        compressed_recv_offset_right,
        communicator,
        false);
    } else {
      all_to_all_comm(hashed_left->view(),
                      communicated_left[ibatch]->mutable_view(),
                      vector<cudf::size_type>(&left_offset[start_idx], &left_offset[end_idx]),
                      recv_offsets_left[ibatch],
                      communicator,
                      false);

      all_to_all_comm(hashed_right->view(),
                      communicated_right[ibatch]->mutable_view(),
                      vector<cudf::size_type>(&right_offset[start_idx], &right_offset[end_idx]),
                      recv_offsets_right[ibatch],
                      communicator,
                      false);
    }

    if (communicator->group_by_batch()) communicator->stop();

    compressed_input_left.clear();
    compressed_input_right.clear();

    // decompress the received data, if needed
    if (compression) {
      all_to_all_comm_decompression(communicated_left[ibatch]->mutable_view(),
                                    recv_offsets_left[ibatch],
                                    compressed_output_left,
                                    compressed_recv_offset_left,
                                    communicator,
                                    false);

      all_to_all_comm_decompression(communicated_right[ibatch]->mutable_view(),
                                    recv_offsets_right[ibatch],
                                    compressed_output_right,
                                    compressed_recv_offset_right,
                                    communicator,
                                    false);
    }

    // mark the communication of ibatch as finished.
    // the join thread is safe to start performing local join on ibatch
    flags[ibatch] = true;

    if (report_timing) {
      stop_time     = high_resolution_clock::now();
      auto duration = duration_cast<milliseconds>(stop_time - start_time);
      std::cerr << "Rank " << mpi_rank << ": All-to-all communication on batch " << ibatch
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

#endif  // __DISTRIBUTED_JOIN
