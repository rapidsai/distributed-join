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

#include "all_to_all_comm.hpp"
#include "communicator.hpp"
#include "compression.hpp"
#include "error.hpp"
#include "shuffle_on.hpp"

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/join.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cmath>
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
 * Helper function for getting the number of partitions in NVLink communication stage.
 *
 * This function calculates the largest integer that divides *mpi_size* and is not larger than
 * *nvlink_domain_size*.
 */
static int get_nvl_partition_size(int mpi_size, int nvlink_domain_size)
{
  if (nvlink_domain_size >= mpi_size) return mpi_size;

  for (int size = ceil(sqrt(mpi_size)); size > 0; size--) {
    if (mpi_size % size == 0 && size <= nvlink_domain_size) { return size; }
  }

  return 1;
}

static std::unique_ptr<table> local_join_helper(cudf::table_view left,
                                                cudf::table_view right,
                                                vector<cudf::size_type> const &left_on,
                                                vector<cudf::size_type> const &right_on)
{
  if (left.num_rows() && right.num_rows()) {
    // Perform local join only when both left and right tables are not empty.
    // If either is empty, cuDF's inner join will return the other table, which is not desired.
    return cudf::inner_join(left, right, left_on, right_on);
  }

  return std::make_unique<table>();
}

/**
 * Local join thread used for merging incoming partitions and performing local joins.
 *
 * @param[in] communicated_left Left table after all-to-all communication.
 * @param[in] communicated_right Right table after all-to-all communication.
 * @param[out] batch_join_results Inner join result of each batch.
 * @param[in] left_on Column indices from the left table to join on. This argument will be
 * passed directly to *cudf::inner_join*.
 * @param[in] right_on Column indices from the right table to join on. This argument will be
 * passed directly to *cudf::inner_join*.
 * @param[in] flags *flags[i]* is true if and only if the ith batch has finished the all-to-all
 * communication.
 * @param[in] report_timing Whether to print the local join time.
 * @param[in] mr RMM memory resource.
 */
static void inner_join_func(vector<std::unique_ptr<table>> &communicated_left,
                            vector<std::unique_ptr<table>> &communicated_right,
                            vector<std::unique_ptr<table>> &batch_join_results,
                            vector<cudf::size_type> const &left_on,
                            vector<cudf::size_type> const &right_on,
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

    batch_join_results[ibatch] = local_join_helper(
      communicated_left[ibatch]->view(), communicated_right[ibatch]->view(), left_on, right_on);

    if (report_timing) {
      stop_time     = high_resolution_clock::now();
      auto duration = duration_cast<milliseconds>(stop_time - start_time);
      std::cout << "Rank " << communicator->mpi_rank << ": Local join on batch " << ibatch
                << " takes " << duration.count() << "ms" << std::endl;
    }
  }
}

std::unique_ptr<table> distributed_inner_join(
  cudf::table_view left,
  cudf::table_view right,
  vector<cudf::size_type> const &left_on,
  vector<cudf::size_type> const &right_on,
  Communicator *communicator,
  vector<ColumnCompressionOptions> left_compression_options,
  vector<ColumnCompressionOptions> right_compression_options,
  int over_decom_factor,
  bool report_timing,
  void *preallocated_pinned_buffer,
  int nvlink_domain_size)
{
  int mpi_rank = communicator->mpi_rank;
  int mpi_size = communicator->mpi_size;
  std::chrono::time_point<high_resolution_clock> start_time;
  std::chrono::time_point<high_resolution_clock> stop_time;

  int nvlink_partition_size = get_nvl_partition_size(mpi_size, nvlink_domain_size);

  /* Shuffle in Infiniband domain */

  std::unique_ptr<table> shuffled_left_ib;
  std::unique_ptr<table> shuffled_right_ib;

  if (nvlink_partition_size != mpi_size) {
    constexpr uint32_t hash_partition_seed_ib = 87654321;

    shuffled_left_ib = shuffle_on(left,
                                  left_on,
                                  CommunicationGroup(mpi_size, nvlink_partition_size),
                                  communicator,
                                  left_compression_options,
                                  cudf::hash_id::HASH_MURMUR3,
                                  hash_partition_seed_ib,
                                  report_timing,
                                  preallocated_pinned_buffer);

    shuffled_right_ib = shuffle_on(right,
                                   right_on,
                                   CommunicationGroup(mpi_size, nvlink_partition_size),
                                   communicator,
                                   right_compression_options,
                                   cudf::hash_id::HASH_MURMUR3,
                                   hash_partition_seed_ib,
                                   report_timing,
                                   preallocated_pinned_buffer);

    left  = shuffled_left_ib->view();
    right = shuffled_right_ib->view();
  }

  if (nvlink_partition_size == 1) {
    if (report_timing) { start_time = high_resolution_clock::now(); }

    auto join_result = local_join_helper(left, right, left_on, right_on);

    if (report_timing) {
      stop_time     = high_resolution_clock::now();
      auto duration = duration_cast<milliseconds>(stop_time - start_time);
      std::cout << "Rank " << mpi_rank << ": Hash partition takes " << duration.count() << "ms"
                << std::endl;
    }

    return join_result;
  }

  /* Hash partition */

  if (report_timing) { start_time = high_resolution_clock::now(); }

  std::unique_ptr<table> hashed_left;
  vector<cudf::size_type> left_offset;

  std::unique_ptr<table> hashed_right;
  vector<cudf::size_type> right_offset;

  constexpr uint32_t hash_partition_seed = 12345678;

  std::tie(hashed_left, left_offset) =
    cudf::hash_partition(left,
                         left_on,
                         nvlink_partition_size * over_decom_factor,
                         cudf::hash_id::HASH_MURMUR3,
                         hash_partition_seed);

  std::tie(hashed_right, right_offset) =
    cudf::hash_partition(right,
                         right_on,
                         nvlink_partition_size * over_decom_factor,
                         cudf::hash_id::HASH_MURMUR3,
                         hash_partition_seed);

  CUDA_RT_CALL(cudaStreamSynchronize(0));

  shuffled_left_ib.reset();
  shuffled_right_ib.reset();

  left_offset.push_back(left.num_rows());
  right_offset.push_back(right.num_rows());

  if (report_timing) {
    stop_time     = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop_time - start_time);
    std::cout << "Rank " << mpi_rank << ": Hash partition takes " << duration.count() << "ms"
              << std::endl;
  }

  /* Construct AllToAllCommunicator */

  std::vector<AllToAllCommunicator> all_to_all_communicator_left;
  std::vector<AllToAllCommunicator> all_to_all_communicator_right;

  for (int ibatch = 0; ibatch < over_decom_factor; ibatch++) {
    int start_idx = ibatch * nvlink_partition_size;
    int end_idx   = (ibatch + 1) * nvlink_partition_size + 1;

    all_to_all_communicator_left.emplace_back(
      hashed_left->view(),
      vector<cudf::size_type>(&left_offset[start_idx], &left_offset[end_idx]),
      CommunicationGroup(nvlink_partition_size, 1),
      communicator,
      generate_none_compression_options(hashed_left->view()),
      true);

    all_to_all_communicator_right.emplace_back(
      hashed_right->view(),
      vector<cudf::size_type>(&right_offset[start_idx], &right_offset[end_idx]),
      CommunicationGroup(nvlink_partition_size, 1),
      communicator,
      generate_none_compression_options(hashed_right->view()),
      true);
  }

  /* Allocate storage for the table after all-to-all communication */

  vector<std::unique_ptr<table>> communicated_left;
  vector<std::unique_ptr<table>> communicated_right;

  for (int ibatch = 0; ibatch < over_decom_factor; ibatch++) {
    communicated_left.push_back(all_to_all_communicator_left[ibatch].allocate_communicated_table());

    communicated_right.push_back(
      all_to_all_communicator_right[ibatch].allocate_communicated_table());
  }

  // *flags* indicates whether each batch has finished communication
  // *flags* uses std::atomic because unsynchronized access to an object which is modified in one
  // thread and read in another is undefined behavior.
  vector<std::atomic<bool>> flags(over_decom_factor);
  vector<std::unique_ptr<table>> batch_join_results(over_decom_factor);

  for (auto &flag : flags) { flag = false; }

  /* Launch inner join thread */

  std::thread inner_join_thread(inner_join_func,
                                std::ref(communicated_left),
                                std::ref(communicated_right),
                                std::ref(batch_join_results),
                                left_on,
                                right_on,
                                std::ref(flags),
                                communicator,
                                report_timing,
                                rmm::mr::get_current_device_resource());

  /* Use the current thread for all-to-all communication */

  for (int ibatch = 0; ibatch < over_decom_factor; ibatch++) {
    if (report_timing) { start_time = high_resolution_clock::now(); }

    all_to_all_communicator_left[ibatch].launch_communication(
      communicated_left[ibatch]->mutable_view(), report_timing, preallocated_pinned_buffer);

    all_to_all_communicator_right[ibatch].launch_communication(
      communicated_right[ibatch]->mutable_view(), report_timing, preallocated_pinned_buffer);

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
