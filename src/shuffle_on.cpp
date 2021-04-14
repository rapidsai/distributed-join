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

#include <mpi.h>

#include <memory>
#include <vector>

using cudf::table;
using std::vector;

std::unique_ptr<cudf::table> shuffle_on(cudf::table_view const& input,
                                        std::vector<cudf::size_type> const& on_columns,
                                        CommunicationGroup comm_group,
                                        Communicator* communicator,
                                        std::vector<ColumnCompressionOptions> compression_options,
                                        cudf::hash_id hash_function,
                                        uint32_t hash_seed,
                                        bool report_timing,
                                        void* preallocated_pinned_buffer)
{
  int mpi_rank      = communicator->mpi_rank;
  int ngpus         = comm_group.size();
  double start_time = 0.0;
  double stop_time  = 0.0;

  /* Hash partition */

  std::unique_ptr<table> hashed_input;
  vector<cudf::size_type> offsets;

  if (report_timing) { start_time = MPI_Wtime(); }

  std::tie(hashed_input, offsets) =
    cudf::hash_partition(input, on_columns, ngpus, hash_function, hash_seed);

  CUDA_RT_CALL(cudaStreamSynchronize(0));

  offsets.push_back(hashed_input->num_rows());

  if (report_timing) {
    stop_time = MPI_Wtime();
    std::cout << "Rank " << mpi_rank << ": Hash partition takes " << (stop_time - start_time) * 1e3
              << "ms" << std::endl;
  }

  /* All_to_all communication */

  AllToAllCommunicator all_to_all_communicator(
    hashed_input->view(), offsets, comm_group, communicator, compression_options, true);

  std::unique_ptr<table> shuffled = all_to_all_communicator.allocate_communicated_table();

  all_to_all_communicator.launch_communication(
    shuffled->mutable_view(), report_timing, preallocated_pinned_buffer);

  return shuffled;
}

std::unique_ptr<cudf::table> shuffle_on(cudf::table_view const& input,
                                        std::vector<cudf::size_type> const& on_columns,
                                        Communicator* communicator,
                                        std::vector<ColumnCompressionOptions> compression_options,
                                        cudf::hash_id hash_function,
                                        uint32_t hash_seed,
                                        bool report_timing,
                                        void* preallocated_pinned_buffer)
{
  return shuffle_on(input,
                    on_columns,
                    CommunicationGroup(communicator->mpi_size, 1),
                    communicator,
                    compression_options,
                    hash_function,
                    hash_seed,
                    report_timing,
                    preallocated_pinned_buffer);
}
