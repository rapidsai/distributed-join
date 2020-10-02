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

#include <utility>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <tuple>
#include <type_traits>
#include <chrono>
#include <iostream>

#include <cudf/types.hpp>
#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/detail/hashing.hpp>
#include <cudf/join.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <mpi.h>
#include <rmm/mr/device/per_device_resource.hpp>

#include "error.cuh"
#include "communicator.h"
#include "comm.cuh"

using std::vector;
using cudf::column;
using cudf::table;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;


/**
 * Communicate number of elements recieved from each rank during all-to-all communication.
 *
 * @param[in] send_offset Vector of length mpi_size + 1 such that send_count[i+1] - send_count[i]
 * is the number of elements sent from the current rank to rank i during the all-to-all
 * communication.
 * @param[out] recv_count Vector of length mpi_size such that recv_count[i] is the number of
 * elements received from rank i during the all-to-all communication.
 */
void
communicate_sizes(
    vector<cudf::size_type> const& offset,
    vector<int64_t> &recv_count,
    Communicator *communicator)
{
    int mpi_size = communicator->mpi_size;
    vector<int64_t> send_count(mpi_size, -1);

    for (int irank = 0; irank < mpi_size; irank++) {
        send_count[irank] = offset[irank + 1] - offset[irank];
    }

    recv_count.resize(mpi_size, -1);

    vector<MPI_Request> send_req(mpi_size);
    vector<MPI_Request> recv_req(mpi_size);

    for (int irank = 0; irank < mpi_size; irank++) {
        MPI_CALL( MPI_Isend(
            &send_count[irank], 1, MPI_INT64_T, irank, exchange_size_tag,
            MPI_COMM_WORLD, &send_req[irank]
        ));
    }

    for (int irank = 0; irank < mpi_size; irank++) {
        MPI_CALL( MPI_Irecv(
            &recv_count[irank], 1, MPI_INT64_T, irank, exchange_size_tag,
            MPI_COMM_WORLD, &recv_req[irank]
        ));
    }

    MPI_CALL( MPI_Waitall(mpi_size, send_req.data(), MPI_STATUSES_IGNORE) );
    MPI_CALL( MPI_Waitall(mpi_size, recv_req.data(), MPI_STATUSES_IGNORE) );
}


/**
 * All-to-all communication of a single batch without merging.
 *
 * For distributed join with overlapping communication and computation, this function can be used
 * in the communication thread, while the merging can be performed inside the local join thread.
 * This is necessary because merging involves device-to-device copy, which uses SM.
 *
 * This function needs to be called collectively by all ranks in MPI_COMM_WORLD. For every ranks,
 * all arguments are significant.
 *
 * @param[in] hashed Rearranged table by hash function.
 * @param[in] offset Vector of length mpi_size + 1 such that offset[i] represents the start index
 * of bucket i in `hashed`.
 * @param[out] local_buckets The received table from each rank. local_buckets[i][j] points to the
 * data from rank j of column i. This argument does not need to be preallocated, but the caller is
 * responsible for freeing this buffer using RMM_FREE.
 * @param[out] recv_nrows The number of rows received from each rank. recv_nrows[i] stores the
 * number of rows received from rank i.
 * @param[in] communicator An instance of `Communicator` used for communication.
 *
 * @TODO: Use unique_ptr for managing the lifetime of `local_buckets`.
 */
void
all_to_all_comm(
    cudf::table_view hashed,
    vector<cudf::size_type> const& offset,
    vector<vector<void *> > &local_buckets,  // [icol, ibucket]
    vector<int64_t> &recv_nrows,
    Communicator *communicator)
{
    int mpi_rank {communicator->mpi_rank};

    communicate_sizes(offset, recv_nrows, communicator);

    local_buckets.resize(hashed.num_columns());

    // In this function, offset has type cudf::size_type. In send_data_by_offset, offset has type
    // int. This assert ensures these two types are compatible.
    static_assert(
        std::is_same<int, cudf::size_type>::value,
        "int and size_type are not the same"
    );

    for (cudf::size_type icol = 0; icol < hashed.num_columns(); icol++) {
        cudf::size_type dtype_size = cudf::size_of(hashed.column(icol).type());

        communicator->start();

        // communicate with other ranks
        send_data_by_offset(
            hashed.column(icol).head(), offset, dtype_size, communicator, false
        );

        recv_data_by_offset(
            local_buckets[icol], recv_nrows, dtype_size, communicator, false
        );

        // For bucket to the rank itself, simply compute the pointer to the right location.
        // No communication is necesssary.
        local_buckets[icol][mpi_rank] = (void *)(hashed.column(icol).head<char>()
                                                 + offset[mpi_rank] * dtype_size);

        communicator->stop();
    }
}

/**
 * Merge and free the received buckets from 'all_to_all_comm'.
 *
 * @param[in] buckets Received table from each rank during 'all_to_all_comm'. The device buffer
 * inside will be freed by this function.
 * @param[in] counts Number of items received from each rank, got from 'all_to_all_comm'.
 * @param[in] dtypes Column types
 *
 * @return Table formed by merging all buckets in *buckets*.
 */
std::unique_ptr<table>
all_to_all_merge_data(
    vector<vector<void *> > const& buckets,
    vector<int64_t> const& counts,
    vector<cudf::data_type> const& dtypes,
    Communicator *communicator)
{
    vector<std::unique_ptr<column> > merged_table;

    for (int icol = 0; icol < buckets.size(); icol++) {
        size_t dtype_size = cudf::size_of(dtypes[icol]);

        int64_t total_count;
        rmm::device_buffer merged_data = merge_free_received_offset(
            buckets[icol], counts, dtype_size, total_count, communicator, false
        );

        merged_table.push_back(std::make_unique<column>(dtypes[icol], total_count, std::move(merged_data)));
    }

    return std::make_unique<table>(std::move(merged_table));
}


/**
 * All-to-all communication with merging for a single batch.
 *
 * This function needs to be called collectively by all ranks in MPI_COMM_WORLD. For every ranks,
 * all arguments are significant.
 *
 * @param[in] input Input table to be communicated.
 * @param[in] offset Vector of length mpi_size + 1 such that offset[i] represents the starting row
 * index of bucket i in `input`.
 * @param[in] communicator An instance of `Communicator` used for communication.
 *
 * @return The communication result which gathers bucket i of `input` on rank i.
 */
std::unique_ptr<table>
all_to_all_comm_single_batch(
    cudf::table_view input,
    vector<cudf::size_type> const& offset,
    Communicator *communicator)
{
    vector<int64_t> recv_nrows;
    communicate_sizes(offset, recv_nrows, communicator);

    vector<std::unique_ptr<column> > recv_columns;

    for (cudf::size_type icol = 0; icol < input.num_columns(); icol++) {

        std::size_t dtype_size = cudf::size_of(input.column(icol).type());

        communicator->start();

        send_data_by_offset(input.column(icol).head(), offset, dtype_size, communicator);

        vector<void *> recv_data;

        recv_data_by_offset(recv_data, recv_nrows, dtype_size, communicator);

        communicator->stop();

        int64_t nrows;
        rmm::device_buffer merged_data = merge_free_received_offset(
            recv_data, recv_nrows, dtype_size, nrows, communicator
        );

        recv_columns.push_back(std::make_unique<column>(
            input.column(icol).type(), nrows, std::move(merged_data))
        );
    }

    return std::make_unique<table>(std::move(recv_columns));
}


void
inner_join_func(
    vector<vector<vector<void *> > > const& left_buckets,
    vector<vector<int64_t> > const& left_counts,
    vector<cudf::data_type> const& left_dtypes,
    vector<vector<vector<void *> > > const& right_buckets,
    vector<vector<int64_t> > const& right_counts,
    vector<cudf::data_type> const& right_dtypes,
    vector<std::unique_ptr<table> > &batch_join_results,
    vector<cudf::size_type> const& left_on,
    vector<cudf::size_type> const& right_on,
    vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
    vector<std::atomic<bool> > const& flags,
    Communicator *communicator,
    bool report_timing,
    rmm::mr::device_memory_resource* mr)
{
    CUDA_RT_CALL(cudaSetDevice(communicator->current_device));
    rmm::mr::set_current_device_resource(mr);

    std::chrono::time_point<high_resolution_clock> start_time;
    std::chrono::time_point<high_resolution_clock> stop_time;

    for (int ibatch = 0; ibatch < flags.size(); ibatch++) {
        // busy waiting for all-to-all communication of ibatch to finish
        while (!flags[ibatch]) {;}

        if (report_timing) {
            start_time = high_resolution_clock::now();
        }

        std::unique_ptr<table> local_left = all_to_all_merge_data(
            left_buckets[ibatch], left_counts[ibatch], left_dtypes, communicator
        );

        std::unique_ptr<table> local_right = all_to_all_merge_data(
            right_buckets[ibatch], right_counts[ibatch], right_dtypes, communicator
        );

        if (local_left->num_rows() && local_right->num_rows()) {
            // Perform local join only when both left and right tables are not empty.
            // If either is empty, the local join will return the other table, which is not desired.
            batch_join_results[ibatch] = cudf::inner_join(
                local_left->view(), local_right->view(),
                left_on, right_on, columns_in_common
            );
        } else {
            batch_join_results[ibatch] = std::make_unique<table>();
        }

        if (report_timing) {
            stop_time = high_resolution_clock::now();
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
std::unique_ptr<table>
distributed_inner_join(
    cudf::table_view const& left,
    cudf::table_view const& right,
    vector<cudf::size_type> const& left_on,
    vector<cudf::size_type> const& right_on,
    vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
    Communicator *communicator,
    int over_decom_factor=1,
    bool report_timing=false)
{
    if (over_decom_factor == 1) {
        // @TODO: If over_decom_factor is 1, there is no opportunity for overlapping. Therefore,
        // we can get away with using just one thread.
    }

    int mpi_size = communicator->mpi_size;
    std::chrono::time_point<high_resolution_clock> start_time;
    std::chrono::time_point<high_resolution_clock> stop_time;

    /* Hash partition */

    if (report_timing) {
        start_time = high_resolution_clock::now();
    }

    std::unique_ptr<table> hashed_left;
    vector<cudf::size_type> left_offset;

    std::unique_ptr<table> hashed_right;
    vector<cudf::size_type> right_offset;

    std::tie(hashed_left, left_offset) = cudf::detail::hash_partition(
        left, left_on, mpi_size * over_decom_factor,
        rmm::mr::get_current_device_resource(),
        cudaStreamPerThread
    );

    std::tie(hashed_right, right_offset) = cudf::detail::hash_partition(
        right, right_on, mpi_size * over_decom_factor,
        rmm::mr::get_current_device_resource(),
        cudaStreamPerThread
    );

    CUDA_RT_CALL( cudaStreamSynchronize(cudaStreamPerThread) );

    left_offset.push_back(left.num_rows());
    right_offset.push_back(right.num_rows());

    if (report_timing) {
        stop_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop_time - start_time);
        std::cerr << "Rank " << communicator->mpi_rank << ": Hash partition takes "
                  << duration.count() << "ms" << std::endl;
    }

    /* Get column data types */

    cudf::size_type nleft_columns = hashed_left->num_columns();
    cudf::size_type nright_columns = hashed_right->num_columns();

    vector<cudf::data_type> left_dtypes(nleft_columns);
    vector<cudf::data_type> right_dtypes(nright_columns);

    for (int icol = 0; icol < nleft_columns; icol ++) {
        left_dtypes[icol] = hashed_left->view().column(icol).type();
    }

    for (int icol = 0; icol < nright_columns; icol ++) {
        right_dtypes[icol] = hashed_right->view().column(icol).type();
    }

    /* Declare storage for received buckets */

    vector<vector<vector<void *> > > left_buckets(over_decom_factor);  // [ibatch, icol, ibucket]
    vector<vector<vector<void *> > > right_buckets(over_decom_factor);  // [ibatch, icol, ibucket]
    vector<vector<int64_t> > left_counts(over_decom_factor);  // [ibatch, ibucket]
    vector<vector<int64_t> > right_counts(over_decom_factor);  // [ibatch, ibucket]
    vector<std::atomic<bool> > flags(over_decom_factor);  // whether each batch has finished communication
    vector<std::unique_ptr<table> > batch_join_results(over_decom_factor);

    for (auto &flag : flags) {
        flag = false;
    }

    /* Launch inner join thread */

    std::thread inner_join_thread(
        inner_join_func,
        std::ref(left_buckets), std::ref(left_counts), std::ref(left_dtypes),
        std::ref(right_buckets), std::ref(right_counts), std::ref(right_dtypes),
        std::ref(batch_join_results), left_on, right_on, columns_in_common,
        std::ref(flags), communicator, report_timing, rmm::mr::get_current_device_resource()
    );

    /* Use the current thread for all-to-all communication */

    for (int ibatch = 0; ibatch < over_decom_factor; ibatch ++) {

        if (report_timing) {
            start_time = high_resolution_clock::now();
        }

        // the start and end index for left_offset and right_offset for the ibatch
        size_t start_idx = ibatch * mpi_size;
        size_t end_idx = (ibatch + 1) * mpi_size + 1;

        // all-to-all communication for the ibatch
        all_to_all_comm(
            hashed_left->view(),
            vector<cudf::size_type>(&left_offset[start_idx], &left_offset[end_idx]),
            left_buckets[ibatch], left_counts[ibatch], communicator
        );

        all_to_all_comm(
            hashed_right->view(),
            vector<cudf::size_type>(&right_offset[start_idx], &right_offset[end_idx]),
            right_buckets[ibatch], right_counts[ibatch], communicator
        );

        if (over_decom_factor == 1) {
            hashed_left.reset();
            hashed_right.reset();
        }

        // mark the communication of ibatch as finished.
        // the join thread is safe to start performing local join on ibatch
        flags[ibatch] = true;

        if (report_timing) {
            stop_time = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop_time - start_time);
            std::cerr << "Rank " << communicator->mpi_rank << ": All-to-all communication on batch "
                      << ibatch << " takes " << duration.count() << "ms" << std::endl;
        }
    }

    // wait for all join batches to finish
    inner_join_thread.join();

    // hashed left and right tables should not be needed now
    if (over_decom_factor > 1) {
        hashed_left.reset();
        hashed_right.reset();
    }

    /* Merge join results from different batches into a single table */

    vector<cudf::table_view> batch_join_results_view;

    for (auto &table_ptr : batch_join_results) {
        batch_join_results_view.push_back(table_ptr->view());
    }

    return cudf::concatenate(batch_join_results_view);
}

#endif  // __DISTRIBUTED_JOIN
