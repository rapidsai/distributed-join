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

#ifndef __DISTRIBUTED_CUH
#define __DISTRIBUTED_CUH

#include <iostream>
#include <cstdlib>
#include <vector>
#include <thread>
#include <rmm/rmm.h>
#include <mpi.h>
#include <ucp/api/ucp.h>
#include <cudf/cudf.h>

#include "comm.cuh"
#include "error.cuh"
#include "cudf_helper.cuh"
#include "communicator.h"


template <typename data_type>
void add_constant_to_column(gdf_column *column, data_type constant)
{
    auto buffer_ptr = thrust::device_pointer_cast(reinterpret_cast<data_type *>(column->data));

    thrust::for_each(buffer_ptr, buffer_ptr + column->size,
                     [=] __device__ (data_type &i) {
                        i += constant;
                     });
}


/**
 * This function generates build table and probe table distributed, and it need to be called collectively by all ranks
 * in MPI_COMM_WORLD.
 *
 * @param[out] build_table                 Generated build table distributed on each rank. This argument does not need
 *                                         to be preallocated, but the user of this function is responsible for freeing
 *                                         the table and the device buffer it contains. See `free_table`.
 * @param[in] build_table_size_per_rank    The number of rows of build_table on each rank.
 * @param[out] probe_table                 Generated probe table distributed on each rank. This argument does not need
 *                                         to be preallocated, but the user of this function is responsible for freeing
 *                                         the table and the device buffer it contains. See `free_table`.
 * @param[in] selectivity                  The percentage of keys in the probe table present in the build table.
 * @param[in] rand_max_per_rank            The lottery size on each rank. This argument should be set larger than
 *                                         `build_table_size_per_rank`.
 * @param[in] uniq_build_tbl_keys          Whether the keys in the build table are unique.
 * @param[in] communicator                 An instance of `Communicator` used for communication.
 *
 * Note: assume build_table_size_per_rank % mpi_rank == 0 and probe_table_size_per_rank % mpi_rank == 0.
 */
template<typename KEY_T, typename PAYLOAD_T>
void generate_tables_distributed(
                                 std::vector<gdf_column *> &build_table,
                                 gdf_size_type build_table_size_per_rank,
                                 std::vector<gdf_column *> &probe_table,
                                 gdf_size_type probe_table_size_per_rank,
                                 const double selectivity,
                                 const KEY_T rand_max_per_rank,
                                 const bool uniq_build_tbl_keys,
                                 Communicator *communicator)
{
    // Algorithm used for distributed generation:
    // Rank i generates build and probe table independently with keys randomly selected from range
    // [i*uniq_build_tbl_keys, (i+1)*uniq_build_tbl_keys] (called pre_shuffle_table). Afterwards, pre_shuffle_table
    // will be divided into N chunks with the same number of rows, and then send chunk j to rank j. This all-to-all
    // communication will make each local table have keys uniformly from the whole range.

    // Get MPI information

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Generate local build and probe table on each rank

    std::vector<gdf_column *> pre_shuffle_build_table;
    std::vector<gdf_column *> pre_shuffle_probe_table;

    generate_build_probe_tables<KEY_T, PAYLOAD_T>(
        pre_shuffle_build_table, build_table_size_per_rank,
        pre_shuffle_probe_table, probe_table_size_per_rank,
        selectivity, rand_max_per_rank, uniq_build_tbl_keys
    );

    // Add constant to build and probe table to make sure the range is correct

    add_constant_to_column<KEY_T>(pre_shuffle_build_table[0], rand_max_per_rank * mpi_rank);
    add_constant_to_column<KEY_T>(pre_shuffle_probe_table[0], rand_max_per_rank * mpi_rank);
    add_constant_to_column<PAYLOAD_T>(pre_shuffle_build_table[1], build_table_size_per_rank * mpi_rank);
    add_constant_to_column<PAYLOAD_T>(pre_shuffle_probe_table[1], probe_table_size_per_rank * mpi_rank);

    // Construct buffer offset to indicate the start indices to each rank

    std::vector<int> build_table_offset(mpi_size + 1);
    std::vector<int> probe_table_offset(mpi_size + 1);

    const gdf_size_type build_table_send_size = build_table_size_per_rank / mpi_size;
    const gdf_size_type probe_table_send_size = probe_table_size_per_rank / mpi_size;

    for (int irank = 0; irank < mpi_size; irank ++) {
        build_table_offset[irank] = build_table_send_size * irank;
        probe_table_offset[irank] = probe_table_send_size * irank;
    }

    build_table_offset[mpi_size] = build_table_size_per_rank;
    probe_table_offset[mpi_size] = probe_table_size_per_rank;

    // Send each bucket to the desired target rank

    all_to_all_comm(pre_shuffle_build_table, build_table_offset.data(), build_table, communicator, false);

    all_to_all_comm(pre_shuffle_probe_table, probe_table_offset.data(), probe_table, communicator, false);

    // cleanup

    free_table(pre_shuffle_build_table);
    free_table(pre_shuffle_probe_table);
}

#endif  // __DISTRIBUTED_CUH
