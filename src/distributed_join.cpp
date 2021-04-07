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
 * @param[in] flags *flags[i]* is true if and only if the ith batch has finished the all-to-all
 *     communication.
 * @param[in] report_timing Whether to print the local join time to stderr.
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

    if (communicated_left[ibatch]->num_rows() && communicated_right[ibatch]->num_rows()) {
      // Perform local join only when both left and right tables are not empty.
      // If either is empty, cuDF's inner join will return the other table, which is not desired.
      batch_join_results[ibatch] = cudf::inner_join(
        communicated_left[ibatch]->view(), communicated_right[ibatch]->view(), left_on, right_on);
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

std::unique_ptr<table> distributed_inner_join(
  cudf::table_view const &left,
  cudf::table_view const &right,
  vector<cudf::size_type> const &left_on,
  vector<cudf::size_type> const &right_on,
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

  constexpr uint32_t hash_partition_seed = 12345678;

  std::tie(hashed_left, left_offset) = cudf::hash_partition(
    left, left_on, mpi_size * over_decom_factor, cudf::hash_id::HASH_MURMUR3, hash_partition_seed);

  std::tie(hashed_right, right_offset) = cudf::hash_partition(right,
                                                              right_on,
                                                              mpi_size * over_decom_factor,
                                                              cudf::hash_id::HASH_MURMUR3,
                                                              hash_partition_seed);

  CUDA_RT_CALL(cudaStreamSynchronize(0));

  left_offset.push_back(left.num_rows());
  right_offset.push_back(right.num_rows());

  if (report_timing) {
    stop_time     = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop_time - start_time);
    std::cout << "Rank " << mpi_rank << ": Hash partition takes " << duration.count() << "ms"
              << std::endl;
  }

  auto all_to_all_communicator_left = AllToAllCommunicator(hashed_left->view(),
                                                           over_decom_factor,
                                                           left_offset,
                                                           communicator,
                                                           left_compression_options,
                                                           true);

  auto all_to_all_communicator_right = AllToAllCommunicator(hashed_right->view(),
                                                            over_decom_factor,
                                                            right_offset,
                                                            communicator,
                                                            right_compression_options,
                                                            true);

  /* Declare storage for the table after all-to-all communication */

  // left table after all-to-all for each batch
  vector<std::unique_ptr<table>> communicated_left =
    all_to_all_communicator_left.allocate_communicated_table();
  // right table after all-to-all for each batch
  vector<std::unique_ptr<table>> communicated_right =
    all_to_all_communicator_right.allocate_communicated_table();

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

    all_to_all_communicator_left.communicate_batch(
      communicated_left[ibatch]->mutable_view(), ibatch, report_timing, preallocated_pinned_buffer);

    all_to_all_communicator_right.communicate_batch(communicated_right[ibatch]->mutable_view(),
                                                    ibatch,
                                                    report_timing,
                                                    preallocated_pinned_buffer);

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
