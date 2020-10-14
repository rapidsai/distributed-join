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
#include <numeric>

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
 * Note: This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] send_offset Vector of length mpi_size + 1 such that `send_offset[i+1] - send_offset[i]`
 * is the number of elements sent from the current rank to rank i during the all-to-all communication.
 * @param[out] recv_offset Vector of length mpi_size + 1 such that `recv_offset[i+1] - recv_offset[i]`
 * is the number of elements received from rank i during the all-to-all communication. The vector will
 * be resized in this function and does not need to be preallocated.
 */
void
communicate_sizes(
    vector<cudf::size_type> const& send_offset,
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

    recv_offset.resize(mpi_size + 1, -1);
    recv_offset[0] = 0;
    std::partial_sum(recv_count.begin(), recv_count.end(), recv_offset.begin() + 1);
}


/**
 * All-to-all communication of a single batch.
 *
 * This function needs to be called collectively by all ranks in MPI_COMM_WORLD. For every ranks,
 * all arguments are significant.
 *
 * @param[in] input Table to be communicated.
 * @param[in] offset Vector of length `mpi_size + 1` such that `offset[i]` represents the start
 * index of `input` to be sent to rank `i`.
 * @param[in] communicator An instance of `Communicator` used for communication.
 * @param[in] self_copy_stream A CUDA stream used for copying data to the same rank.
 *
 * @return Table after all-to-all communication.
 */
std::unique_ptr<table>
all_to_all_comm(
    cudf::table_view input,
    vector<cudf::size_type> const& offset,
    Communicator *communicator,
    cudaStream_t self_copy_stream = 0)
{
    vector<int64_t> recv_offset;
    communicate_sizes(offset, recv_offset, communicator);

    vector<std::unique_ptr<column> > communicated_columns;

    // In this function, offset has type cudf::size_type. In send_data_by_offset, offset has type
    // int. This assert ensures these two types are compatible.
    static_assert(
        std::is_same<int, cudf::size_type>::value,
        "int and size_type are not the same"
    );

    for (cudf::size_type icol = 0; icol < input.num_columns(); icol++) {
        cudf::data_type dtype = input.column(icol).type();
        cudf::size_type dtype_size = cudf::size_of(dtype);

        communicated_columns.push_back(make_numeric_column(dtype, recv_offset.back()));

        CUDA_RT_CALL( cudaStreamSynchronize(cudaStreamDefault) );

        communicator->start();

        // communicate with other ranks
        send_data_by_offset(
            input.column(icol).head(), offset, dtype_size, communicator, false
        );

        recv_data_by_offset(
            communicated_columns[icol]->mutable_view().head(),
            recv_offset, dtype_size, communicator, false
        );

        communicator->stop();

        // Copy data directly from input buffer to output buffer of the same rank.
        // No communication is necessary.
        int mpi_rank { communicator->mpi_rank };
        CUDA_RT_CALL( cudaMemcpyAsync(
            (void *)((char *)(communicated_columns[icol]->mutable_view().head())
                     + recv_offset[mpi_rank] * dtype_size),
            (void *)((char *)(input.column(icol).head()) + offset[mpi_rank] * dtype_size),
            (recv_offset[mpi_rank + 1] - recv_offset[mpi_rank]) * dtype_size,
            cudaMemcpyDeviceToDevice,
            self_copy_stream)
        );
    }

    CUDA_RT_CALL( cudaStreamSynchronize(self_copy_stream) );

    return std::make_unique<table>(std::move(communicated_columns));
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
void
inner_join_func(
    vector<std::unique_ptr<table> > &communicated_left,
    vector<std::unique_ptr<table> > &communicated_right,
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

        if (communicated_left[ibatch]->num_rows() && communicated_right[ibatch]->num_rows()) {
            // Perform local join only when both left and right tables are not empty.
            // If either is empty, the local join will return the other table, which is not desired.
            batch_join_results[ibatch] = cudf::inner_join(
                communicated_left[ibatch]->view(), communicated_right[ibatch]->view(),
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

    /* Declare storage for the table after all-to-all communication */

    // left table after all-to-all for each batch
    vector<std::unique_ptr<table> > communicated_left(over_decom_factor);
    // right table after all-to-all for each batch
    vector<std::unique_ptr<table> > communicated_right(over_decom_factor);
    // *flags* indicates whether each batch has finished communication
    // *flags* uses std::atomic because unsynchronized access to an object which is modified in one
    // thread and read in another is undefined behavior.
    vector<std::atomic<bool> > flags(over_decom_factor);
    vector<std::unique_ptr<table> > batch_join_results(over_decom_factor);

    for (auto &flag : flags) {
        flag = false;
    }

    /* Launch inner join thread */

    std::thread inner_join_thread(
        inner_join_func,
        std::ref(communicated_left), std::ref(communicated_right),
        std::ref(batch_join_results), left_on, right_on, columns_in_common,
        std::ref(flags), communicator, report_timing, rmm::mr::get_current_device_resource()
    );

    /* Create a priority stream for copying from hash partitioned to communicated table for the same rank */
    cudaStream_t self_copy_stream;
    int least_priority;
    int greatest_priority;
    CUDA_RT_CALL( cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority) );
    CUDA_RT_CALL(
        cudaStreamCreateWithPriority(&self_copy_stream, cudaStreamNonBlocking, greatest_priority) );

    /* Use the current thread for all-to-all communication */

    for (int ibatch = 0; ibatch < over_decom_factor; ibatch ++) {

        if (report_timing) {
            start_time = high_resolution_clock::now();
        }

        // the start and end index for left_offset and right_offset for the ibatch
        size_t start_idx = ibatch * mpi_size;
        size_t end_idx = (ibatch + 1) * mpi_size + 1;

        // all-to-all communication for the ibatch
        communicated_left[ibatch] = all_to_all_comm(
            hashed_left->view(),
            vector<cudf::size_type>(&left_offset[start_idx], &left_offset[end_idx]),
            communicator, self_copy_stream
        );

        communicated_right[ibatch] = all_to_all_comm(
            hashed_right->view(),
            vector<cudf::size_type>(&right_offset[start_idx], &right_offset[end_idx]),
            communicator, self_copy_stream
        );

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

    CUDA_RT_CALL( cudaStreamDestroy(self_copy_stream) );

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
