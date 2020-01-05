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

/**
 * Send the hashed buckets to each rank, receive the incoming buckets, and merge the incoming buckets.
 *
 * This function needs to be called collectively by all ranks in MPI_COMM_WORLD. For every ranks, all arguments are
 * significant.
 *
 * @param[in] hashed_table   Rearranged table by hash function.
 * @param[in] offset         Vector of length mpi_size + 1 such that offset[i] represents the start index of bucket i
 *                           in 'hashed_table'.
 * @param[out] result_table  The communication result which gathers bucket i of input 'hashed_table' on rank i. This
 *                           argument does not need to be preallocated, but the user of this function is responsible
 *                           for freeing this table as well as the device buffer it contains. See 'free_table'.
 * @param[in] communicator   An instance of `Communicator` used for communication.
 */
void all_to_all_comm(
                     std::vector<gdf_column *> &hashed_table,
                     int *offset,
                     std::vector<gdf_column *> &result_table,
                     Communicator *communicator,
                     bool offset_on_device=true)
{
    int mpi_size {communicator->mpi_size};

    int *offset_host;

    if (offset_on_device) {
        offset_host = (int *)malloc((mpi_size + 1) * sizeof(int));
        CUDA_RT_CALL(cudaMemcpy(offset_host, offset, (mpi_size + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    } else {
        offset_host = offset;
    }

    result_table.resize(hashed_table.size(), nullptr);

    for (int icol = 0; icol < hashed_table.size(); icol++) {

        gdf_size_type dtype_size = gdf_dtype_size(hashed_table[icol]->dtype);

        std::vector<comm_handle_t> send_requests = send_data_by_offset(
            hashed_table[icol]->data, offset_host, dtype_size, communicator
        );

        std::vector<void *> received_data;
        std::vector<int64_t> bucket_count;

        std::vector<comm_handle_t> recv_requests = recv_data_by_offset(
            received_data, bucket_count, dtype_size, communicator
        );

        communicator->waitall(send_requests);
        communicator->waitall(recv_requests);

        int64_t total_count;
        void *merged_data = merge_free_received_offset(received_data, bucket_count, dtype_size, total_count);

        result_table[icol] = new gdf_column;

        GDF_CALL(gdf_column_view(
            result_table[icol], merged_data, nullptr, total_count, hashed_table[icol]->dtype)
        );
    }

    if (offset_on_device)
        free(offset_host);
}


/**
 * Non-overlapped version of distributed inner join on left_table and right_table.
 *
 * This function should be called collectively by all processes in MPI_COMM_WORLD. All arguments are significant for
 * all ranks.
 *
 * @param[in] left_table         The left table to be joined on each rank.
 * @param[in] right_table        The right table to be joined on each rank.
 * @param[out] result_table      The result of left and right table inner join on each rank. This
 *                               argument needs not to be preallocated. The caller is responsible for
 *                               deallocate the memory.
 * @param[in] communicator       An instance of `Communicator` used for communication.
 * @param[in] free_input_tables  If true, this function will free input left_table and right_table when no longer needed.
 */
void distributed_join_single_batch(
                                   std::vector<gdf_column *> &left_table,
                                   std::vector<gdf_column *> &right_table,
                                   std::vector<gdf_column *> &result_table,
                                   Communicator *communicator,
                                   bool free_input_tables=false)
{
    double start = MPI_Wtime();

    std::vector<gdf_column *> hashed_left_table;
    std::vector<gdf_column *> hashed_right_table;

    int *left_offsets;
    int *right_offsets;

    hash_table(left_table, hashed_left_table, left_offsets);
    hash_table(right_table, hashed_right_table, right_offsets);

    if (free_input_tables) {
        free_table(left_table);
        free_table(right_table);
    }

    std::vector<gdf_column *> local_left_table;
    std::vector<gdf_column *> local_right_table;

    all_to_all_comm(hashed_left_table, left_offsets, local_left_table, communicator);
    all_to_all_comm(hashed_right_table, right_offsets, local_right_table, communicator);

    free_table(hashed_left_table);
    free_table(hashed_right_table);

    RMM_CALL(RMM_FREE(left_offsets, 0));
    RMM_CALL(RMM_FREE(right_offsets, 0));

    int result_ncols = local_left_table.size() + local_right_table.size() - 1;

    result_table.resize(result_ncols, nullptr);

    for (int icol = 0; icol < result_ncols; icol++) {
        result_table[icol] = new gdf_column;
        result_table[icol]->data = nullptr;
        result_table[icol]->size = 0;
    }

    int columns_to_join[] = {0};

    gdf_context ctxt = {
        0,                     // input data is not sorted
        gdf_method::GDF_HASH,  // hash based join
        0
    };

    if (local_left_table[0]->size && local_right_table[0] -> size) {
        // Perform local join only when both left and right tables are not empty.
        // If either is empty, the local join will return the other table, which is not desired.

        GDF_CALL(
            gdf_inner_join(
                local_left_table.data(), local_left_table.size(), columns_to_join,
                local_right_table.data(), local_right_table.size(), columns_to_join,
                1, result_ncols, result_table.data(), nullptr, nullptr, &ctxt)
        );
    }

    free_table(local_left_table);
    free_table(local_right_table);
}



/**
 * All-to-all communication without merging.
 *
 * This function can be used in the communication thread of the overlapping design, while the merging can be performed
 * inside the local join thread. This is helpful because merging involves device-to-device copy, which uses SM.
 *
 * This function needs to be called collectively by all ranks in MPI_COMM_WORLD. For every ranks, all arguments are
 * significant.
 *
 * @param[in] hashed_table      Rearranged table by hash function.
 * @param[in] offset            Vector of length mpi_size + 1 such that offset[i] represents the start index of bucket i
 *                              in 'hashed_table'.
 * @param[out] local_buckets    The received table from each rank. local_buckets[i][j] points to the data from rank j of
 *                              column i. This argument does not need to be preallocated, but the caller is responsible
 *                              for freeing this buffer using RMM_FREE.
 * @param[out] bucket_count     The number of items received from each rank. bucket_count[i][j] stores the number of
 *                              items received from rank j of column i.
 * @param[in] communicator      An instance of `Communicator` used for communication.
 * @param[in] offset_on_device  If false, this function will assume argument 'offset' is accessible from host. If true,
 *                              this function will copy 'offset' from device to host.
 */
void all_to_all_comm_exchange_data(
                                   std::vector<gdf_column *> &hashed_table,
                                   int *offset,
                                   std::vector<std::vector<void *> > &local_buckets,  // [icol, ibucket]
                                   std::vector<std::vector<int64_t> > &bucket_count,  // [icol, ibucket]
                                   Communicator *communicator,
                                   bool offset_on_device=true)
{
    int mpi_rank {communicator->mpi_rank};
    int mpi_size {communicator->mpi_size};

    int *offset_host;

    if (offset_on_device) {
        offset_host = (int *)malloc((mpi_size + 1) * sizeof(int));
        CUDA_RT_CALL(cudaMemcpy(offset_host, offset, (mpi_size + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    } else {
        offset_host = offset;
    }

    local_buckets.resize(hashed_table.size());
    bucket_count.resize(hashed_table.size());

    for (int icol = 0; icol < hashed_table.size(); icol++) {
        gdf_size_type dtype_size = gdf_dtype_size(hashed_table[icol]->dtype);

        std::vector<comm_handle_t> send_requests = send_data_by_offset(
            hashed_table[icol]->data, offset_host, dtype_size, communicator, false
        );

        std::vector<comm_handle_t> recv_requests = recv_data_by_offset(
            local_buckets[icol], bucket_count[icol], dtype_size, communicator, false
        );

        local_buckets[icol][mpi_rank] = (void *)((char *)(hashed_table[icol]->data) + offset_host[mpi_rank] * dtype_size);
        bucket_count[icol][mpi_rank] = offset_host[mpi_rank + 1] - offset_host[mpi_rank];

        communicator->waitall(send_requests);
        communicator->waitall(recv_requests);
    }
}

/**
 * Merge and free the received buckets from 'all_to_all_comm_exchange_data'.
 *
 * @param[in] local_buckets     Received table from each rank during 'all_to_all_comm_exchange_data'. The device
 *                              buffer inside will be freed by this function
 * @param[in] bucket_counts     Number of items received from each rank, got from 'all_to_all_comm_exchange_data'
 * @param[out] merged_table     Table formed by merging all buckets in 'local_buckets'. It is the caller of this
 *                              function's responsibility to free this table.
 * @param[in] dtypes            Column types
 */
void all_to_all_merge_data(
                           std::vector<std::vector<void *> > &local_buckets,
                           std::vector<std::vector<int64_t> > &bucket_counts,
                           std::vector<gdf_column *> &merged_table,
                           std::vector<gdf_dtype> &dtypes,
                           Communicator *communicator)
{
    merged_table.resize(local_buckets.size(), nullptr);

    for (int icol = 0; icol < local_buckets.size(); icol++) {
        gdf_size_type dtype_size = gdf_dtype_size(dtypes[icol]);

        int64_t total_count;
        void *merged_data = merge_free_received_offset(
            local_buckets[icol], bucket_counts[icol], dtype_size, total_count, communicator, false
        );

        merged_table[icol] = new gdf_column;

        GDF_CALL(gdf_column_view(merged_table[icol], merged_data, nullptr, total_count, dtypes[icol]));
    }
}


void inner_join_func(
                     std::vector<std::vector<std::vector<void *> > > &local_left_buckets,
                     std::vector<std::vector<std::vector<int64_t> > > &left_bucket_counts,
                     std::vector<gdf_dtype> &left_dtypes,
                     int left_join_cols[],
                     std::vector<std::vector<std::vector<void *> > > &local_right_buckets,
                     std::vector<std::vector<std::vector<int64_t> > > &right_bucket_counts,
                     std::vector<gdf_dtype> &right_dtypes,
                     int right_join_cols[],
                     int num_cols_to_join,
                     std::vector<std::vector<gdf_column *> > &result_table_batches,
                     int result_ncols,
                     gdf_context *join_context,
                     int current_device,
                     std::vector<bool> &flags,
                     Communicator *communicator)
{
    CUDA_RT_CALL(cudaSetDevice(current_device));

    for (int ibatch = 0; ibatch < flags.size(); ibatch ++) {
        while (!flags[ibatch]) {;}

        std::vector<gdf_column *> local_left_table;
        std::vector<gdf_column *> local_right_table;

        all_to_all_merge_data(
            local_left_buckets[ibatch], left_bucket_counts[ibatch], local_left_table, left_dtypes, communicator
        );

        all_to_all_merge_data(
            local_right_buckets[ibatch], right_bucket_counts[ibatch], local_right_table, right_dtypes, communicator
        );

        result_table_batches[ibatch].resize(result_ncols, nullptr);

        for (int icol = 0; icol < result_ncols; icol++) {
            result_table_batches[ibatch][icol] = new gdf_column;
            result_table_batches[ibatch][icol]->data = nullptr;
            result_table_batches[ibatch][icol]->size = 0;
        }

        if (local_left_table[0]->size && local_right_table[0] -> size) {
            // Perform local join only when both left and right tables are not empty.
            // If either is empty, the local join will return the other table, which is not desired.
            GDF_CALL(gdf_inner_join(
                local_left_table.data(), local_left_table.size(), left_join_cols,
                local_right_table.data(), local_right_table.size(), right_join_cols,
                num_cols_to_join, result_table_batches[ibatch].size(), result_table_batches[ibatch].data(),
                nullptr, nullptr, join_context
            ));
        }

        free_table(local_left_table);
        free_table(local_right_table);
    }
}


/**
 * Distributed inner join on left_table and right_table.
 *
 * This function should be called collectively by all processes in MPI_COMM_WORLD. All arguments are significant for
 * all ranks.
 *
 * @param[in] left_table         The left table to be joined on each rank.
 * @param[in] right_table        The right table to be joined on each rank.
 * @param[out] result_table      The result of left and right table inner join on each rank. This
 *                               argument needs not to be preallocated. The caller is responsible for
 *                               deallocate the memory.
 * @param[in] communicator       An instance of `Communicator` used for communication.
 * @param[in] over_decom_factor  Over-decomposition factor.
 */
void distributed_join(
                      std::vector<gdf_column *> &left_table,
                      std::vector<gdf_column *> &right_table,
                      std::vector<gdf_column *> &result_table,
                      Communicator *communicator,
                      int over_decom_factor=1,
                      bool free_input_tables=false)
{
    if (over_decom_factor == 1) {
        distributed_join_single_batch(left_table, right_table, result_table, communicator, free_input_tables);
        return;
    }

    int mpi_size = communicator->mpi_size;

    /* Hash left and right tables into partitions */

    std::vector<gdf_column *> hashed_left_table;
    std::vector<gdf_column *> hashed_right_table;

    int *left_offsets;
    int *right_offsets;
    int *left_offsets_host = (int *)malloc((mpi_size * over_decom_factor + 1) * sizeof(int));
    int *right_offsets_host = (int *)malloc((mpi_size * over_decom_factor + 1) * sizeof(int));

    hash_table(left_table, hashed_left_table, left_offsets, mpi_size * over_decom_factor);
    hash_table(right_table, hashed_right_table, right_offsets, mpi_size * over_decom_factor);

    CUDA_RT_CALL(cudaMemcpy(
        left_offsets_host, left_offsets, (mpi_size * over_decom_factor + 1) * sizeof(int), cudaMemcpyDeviceToHost
    ));

    CUDA_RT_CALL(cudaMemcpy(
        right_offsets_host, right_offsets, (mpi_size * over_decom_factor + 1) * sizeof(int), cudaMemcpyDeviceToHost
    ));

    RMM_CALL(RMM_FREE(left_offsets, 0));
    RMM_CALL(RMM_FREE(right_offsets, 0));

    /* Setup parameters */

    // result_table_batches[i] holds the join result of batch i
    std::vector<std::vector<gdf_column *> > result_table_batches(over_decom_factor);

    int result_ncols = left_table.size() + right_table.size() - 1;

    int columns_to_join[] = {0};

    gdf_context ctxt = {
        0,                     // input data is not sorted
        gdf_method::GDF_HASH,  // hash based join
        0
    };

    if (free_input_tables) {
        free_table(left_table);
        free_table(right_table);
    }

    /* Get column dtypes */

    std::vector<gdf_dtype> left_dtypes(hashed_left_table.size());
    std::vector<gdf_dtype> right_dtypes(hashed_right_table.size());

    for (int icol = 0; icol < hashed_left_table.size(); icol ++) {
        left_dtypes[icol] = hashed_left_table[icol]->dtype;
    }

    for (int icol = 0; icol < hashed_right_table.size(); icol ++) {
        right_dtypes[icol] = hashed_right_table[icol]->dtype;
    }

    /* Launch join thread */

    std::vector<std::vector<std::vector<void *> > > local_left_buckets(over_decom_factor);  // [ibatch, icol, ibucket]
    std::vector<std::vector<std::vector<void *> > > local_right_buckets(over_decom_factor);  // [ibatch, icol, ibucket]
    std::vector<std::vector<std::vector<int64_t> > > left_bucket_counts(over_decom_factor);  // [ibatch, icol, ibucket]
    std::vector<std::vector<std::vector<int64_t> > > right_bucket_counts(over_decom_factor);  // [ibatch, icol, ibucket]
    std::vector<bool> flags(over_decom_factor, false);

    std::thread inner_join_thread(
        inner_join_func,
        std::ref(local_left_buckets), std::ref(left_bucket_counts), std::ref(left_dtypes), columns_to_join,
        std::ref(local_right_buckets), std::ref(right_bucket_counts), std::ref(right_dtypes), columns_to_join,
        1, std::ref(result_table_batches), result_ncols,
        &ctxt, communicator->current_device, std::ref(flags), communicator
    );

    for (int ibatch = 0; ibatch < over_decom_factor; ibatch ++) {

        /* Communicate current batch */

        all_to_all_comm_exchange_data(
            hashed_left_table, left_offsets_host + ibatch * mpi_size,
            local_left_buckets[ibatch], left_bucket_counts[ibatch], communicator, false
        );

        all_to_all_comm_exchange_data(
            hashed_right_table, right_offsets_host + ibatch * mpi_size,
            local_right_buckets[ibatch], right_bucket_counts[ibatch], communicator, false
        );

        flags[ibatch] = true;
    }

    /* Wait for all joins to finish */

    inner_join_thread.join();

    free_table(hashed_left_table);
    free_table(hashed_right_table);

    /* Query how many rows in the join result for each batch */

    std::vector<gdf_size_type> result_idx;
    result_idx.push_back(0);

    for (auto & table : result_table_batches) {
        result_idx.push_back(result_idx.back() + table[0]->size);
    }

    /* Query the size of each column */

    std::vector<gdf_dtype> col_dtypes(result_ncols);

    for (int ibatch = 0; ibatch < over_decom_factor; ibatch ++) {
        if (result_table_batches[ibatch][0] -> size != 0) {
            for (int icol = 0; icol < result_ncols; icol ++) {
                col_dtypes[icol] = result_table_batches[ibatch][icol]->dtype;
            }
        }
    }

    /* Merge batched join result into a single table */

    if (over_decom_factor == 1) {
        result_table = result_table_batches[0];
        return;
    } else {
        result_table.resize(result_ncols, nullptr);

        for (int icol = 0; icol < result_ncols; icol ++) {
            result_table[icol] = new gdf_column;

            gdf_size_type dtype_size = gdf_dtype_size(col_dtypes[icol]);

            void *buffer;
            RMM_CALL(RMM_ALLOC(&buffer, result_idx.back() * dtype_size, 0));

            for (int ibatch = 0; ibatch < over_decom_factor; ibatch ++) {
                void *start_addr = (char *)buffer + result_idx[ibatch] * dtype_size;
                CUDA_RT_CALL(cudaMemcpy(
                    start_addr,
                    result_table_batches[ibatch][icol]->data,
                    result_table_batches[ibatch][icol]->size * dtype_size,
                    cudaMemcpyDeviceToDevice
                ));
            }

            GDF_CALL(gdf_column_view(
                result_table[icol], buffer, nullptr, result_idx.back(), col_dtypes[icol]
            ));
        }

        for (auto & table : result_table_batches)
            free_table(table);
    }

}


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
