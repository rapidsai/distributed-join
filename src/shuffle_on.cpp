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

#include "shuffle_on.hpp"

#include "all_to_all_comm.hpp"
#include "communicator.hpp"
#include "compression.hpp"
#include "error.hpp"

#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <vector>

using cudf::table;
using std::vector;

std::unique_ptr<cudf::table> shuffle_on(cudf::table_view const& input,
                                        std::vector<cudf::size_type> const& on_columns,
                                        Communicator* communicator,
                                        std::vector<ColumnCompressionOptions> compression_options,
                                        cudf::hash_id hash_function,
                                        bool report_timing,
                                        void* preallocated_pinned_buffer)
{
  int mpi_size = communicator->mpi_size;

  /* Hash partition */

  std::unique_ptr<table> hashed_input;
  vector<cudf::size_type> offsets;

  std::tie(hashed_input, offsets) =
    cudf::hash_partition(input, on_columns, mpi_size, hash_function);

  CUDA_RT_CALL(cudaStreamSynchronize(0));

  offsets.push_back(hashed_input->num_rows());

  /* All_to_all communication */

  auto all_to_all_communicator =
    AllToAllCommunicator(hashed_input->view(), 1, offsets, communicator, compression_options, true);

  vector<std::unique_ptr<table>> shuffled = all_to_all_communicator.allocate_communicated_table();

  all_to_all_communicator.communicate_batch(
    shuffled[0]->mutable_view(), 0, report_timing, preallocated_pinned_buffer);

  return std::move(shuffled[0]);
}
