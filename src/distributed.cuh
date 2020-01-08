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


/**
 * Merge local tables from each rank to the root rank.
 *
 * This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[out] received_table    Merged table on root rank. Only significant on root. This argument does not need to be
 *                               preallocated, but the caller of this function is responsible for freeing the table and
 *                               the device memory it contains. See 'free_table'.
 * @param[in] local_table        The local table on each rank to be sent to the master rank. Significant on all ranks.
 * @param[in] communicator       An instance of `Communicator` used for communication.
 */
void collect_tables(
                    std::vector<gdf_column *> &received_table,
                    const std::vector<gdf_column *> &local_table,
                    Communicator *communicator)
{
    int mpi_rank {communicator->mpi_rank};
    int mpi_size {communicator->mpi_size};

    /* Gather table size from each rank */

    int ncols = local_table.size();
    gdf_size_type local_table_size = local_table[0]->size;
    gdf_size_type *table_sizes = nullptr;
    gdf_size_type *table_sizes_scan = nullptr;
    gdf_size_type global_table_size = 0;

    if (mpi_rank == 0) {
        table_sizes = (gdf_size_type *)malloc(mpi_size * sizeof(gdf_size_type));
        table_sizes_scan = (gdf_size_type *)malloc((mpi_size + 1) * sizeof(gdf_size_type));
        table_sizes_scan[0] = 0;
    }

    MPI_CALL(
        MPI_Gather(
            &local_table_size, 1, mpi_dtype_from_c_type<gdf_size_type>(),
            table_sizes, 1, mpi_dtype_from_c_type<gdf_size_type>(),
            0, MPI_COMM_WORLD
        )
    );

    if (mpi_rank == 0) {
        for (int irank = 0; irank < mpi_size; irank++) {
            global_table_size += table_sizes[irank];
        }

        for (int irank = 0; irank < mpi_size; irank++) {
            table_sizes_scan[irank + 1] = table_sizes_scan[irank] + table_sizes[irank];
        }
    }

    /* Construct received table on root */

    if (mpi_rank == 0) {
        received_table.resize(ncols, nullptr);

        for (int icol = 0; icol < ncols; icol++) {
            void *data;
            size_t alloc_size = global_table_size * gdf_dtype_size(local_table[icol]->dtype);

            CHECK_ERROR(RMM_ALLOC(&data, alloc_size, 0), RMM_SUCCESS, "RMM_ALLOC");

            received_table[icol] = new gdf_column;

            CHECK_ERROR(
                gdf_column_view(received_table[icol], data, nullptr, global_table_size, local_table[icol]->dtype),
                GDF_SUCCESS, "gdf_column_view"
            );
        }
    }

    /* Send local table from each rank to root */

    for (int icol = 0; icol < ncols; icol++) {

        gdf_size_type dtype_size = gdf_dtype_size(local_table[icol]->dtype);

        if (mpi_rank == 0) {

            std::vector<comm_handle_t> requests(mpi_size, nullptr);

            for (int irank = 1; irank < mpi_size; irank++) {
                void *start_addr = (void *)((char *)received_table[icol]->data + table_sizes_scan[irank] * dtype_size);

                requests[irank] = communicator->recv(
                    start_addr, table_sizes[irank], dtype_size, irank, collect_table_tag
                );
            }

            CHECK_ERROR(
                cudaMemcpy(received_table[icol]->data, local_table[icol]->data, table_sizes[0] * dtype_size,
                           cudaMemcpyDeviceToDevice),
                cudaSuccess, "cudaMemcpyDeviceToDevice"
            );

            communicator->waitall(requests);

        } else {

            comm_handle_t request = communicator->send(
                local_table[icol]->data, local_table_size, dtype_size, 0, collect_table_tag
            );

            communicator->wait(request);
        }

    }

    free(table_sizes);
    free(table_sizes_scan);
}


#endif  // __DISTRIBUTED_CUH
