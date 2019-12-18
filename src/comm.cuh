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

#ifndef __COMM_CUH
#define __COMM_CUH

#include <vector>
#include <ucp/api/ucp.h>
#include <mpi.h>
#include <cassert>

#include "communicator.h"
#include "error.cuh"


enum COMM_TAGS
{
    placeholder_tag,
    offset_tag,
    distribute_col_tag,
    collect_table_tag
};


/**
 * Usage: mpi_dtype_from_c_type<input_type>() returns the MPI datatype corresponding to a native C type "input_type".
 *
 * For example, mpi_dtype_from_c_type<int>() would return MPI_INT32_T.
 */
template <typename c_type>
MPI_Datatype mpi_dtype_from_c_type()
{
    MPI_Datatype mpi_dtype;
    if(std::is_same<c_type,int8_t>::value) mpi_dtype = MPI_INT8_T;
    else if(std::is_same<c_type,uint8_t>::value) mpi_dtype = MPI_UINT8_T;
    else if(std::is_same<c_type,int16_t>::value) mpi_dtype = MPI_INT16_T;
    else if(std::is_same<c_type,uint16_t>::value) mpi_dtype = MPI_UINT16_T;
    else if(std::is_same<c_type,int32_t>::value) mpi_dtype = MPI_INT32_T;
    else if(std::is_same<c_type,uint32_t>::value) mpi_dtype = MPI_UINT32_T;
    else if(std::is_same<c_type,int64_t>::value) mpi_dtype = MPI_INT64_T;
    else if(std::is_same<c_type,uint64_t>::value) mpi_dtype = MPI_UINT64_T;
    else if(std::is_same<c_type,float>::value) mpi_dtype = MPI_FLOAT;
    else if(std::is_same<c_type,double>::value) mpi_dtype = MPI_DOUBLE;

    return mpi_dtype;
}


/**
 * Send data from the current rank to other ranks according to offset.
 *
 * @param[in] data                The starting address of data to be sent in device buffer.
 * @param[in] offset              Array of length mpi_size + 1. Items in *data* with indicies from offset[i] to
 *                                offset[i+1] will be sent to rank i.
 * @param[in] item_size           The size of each item.
 * @param[in] communicator        An instance of 'Communicator' used for communication.
 * @param[in] self_send           Whether sending data to itself. If this argument is false, items in *data* destined for the
 *                                current rank will not be copied, and nullptr is returned as handle.
 * @returns                       A vector holding the handles of all aysnc requests. See 'wait' and 'waitall' in
 *                                'Communicator'.
 */
std::vector<comm_handle_t> send_data_by_offset(
                                               const void *data,
                                               const int *offset,
                                               size_t item_size,
                                               Communicator *communicator,
                                               bool self_send=true)
{
    int mpi_rank {communicator->mpi_rank};
    int mpi_size {communicator->mpi_size};

    std::vector<comm_handle_t> requests(mpi_size, nullptr);

    for (int itarget_rank = 0; itarget_rank < mpi_size; itarget_rank++) {
        if (!self_send && itarget_rank == mpi_rank)
            continue;

        // calculate the number of elements to send
        size_t count = offset[itarget_rank + 1] - offset[itarget_rank];

        // calculate the starting address
        const void *start_addr = (void *)((char *)data + offset[itarget_rank] * item_size);

        // send buffer to the target node
        requests[itarget_rank] = communicator->send(start_addr, count, item_size, itarget_rank, offset_tag);
    }

    return requests;
}


/**
 * Receive the data sent from 'send_data_by_offset'.
 *
 * @param[out] data         The data received from each rank. This argument does not need to be preallocated, but the
 *                          caller is responsible for freeing this buffer using RMM_FREE. See
 *                          'merge_free_received_offset'.
 * @param[out] bucket_count The number of items received from each rank.
 * @param[in] item_size     The size of each item. Used for passing to receive function in UCX.
 * @param[in] communicator  An instance of 'Communicator' used for communication.
 * @param[in] self_recv     Whether recving data from itself. If this argument is false, data[mpi_rank] will be nullptr
 *                          and bucket_count[mpi_rank] will be 0.
 * @returns                 A vector holding the handles of all aysnc requests. See 'wait' and 'waitall' in
 *                          'Communicator'.
 */
std::vector<comm_handle_t> recv_data_by_offset(
                                               std::vector<void *> &data,
                                               std::vector<int64_t> &bucket_count,
                                               size_t item_size,
                                               Communicator *communicator,
                                               bool self_recv=true)
{
    int mpi_rank {communicator->mpi_rank};
    int mpi_size {communicator->mpi_size};

    data.resize(mpi_size, nullptr);
    bucket_count.resize(mpi_size, 0);

    std::vector<comm_handle_t> requests(mpi_size, nullptr);

    for (int isource_rank = 0; isource_rank < mpi_size; isource_rank++) {
        if (!self_recv && mpi_rank == isource_rank)
            continue;

        requests[isource_rank] = communicator->recv(
            &data[isource_rank], &bucket_count[isource_rank], item_size, isource_rank, offset_tag
        );
    }

    return requests;
}


/**
 * Merge received buckets into a single buffer, and free all received buckets.
 *
 * @param[in] received_data        Vector with length of number of ranks, where the ith entry has the data received
 *                                 from rank i. Buffers contained in this argument will be freed in this function. Also
 *                                 see 'recv_data_by_offset'.
 * @param[in] bucket_count         Vector with length of number of ranks, where the ith entry represents the the number
 *                                 of elements of received_data[i]. See 'recv_data_by_offset'.
 * @param[in] item_size            The size of each element.
 * @param[out] total_count         The number of elements in the merged buffer returned from this function. It is the
 *                                 sum of bucket_count.
 * @returns                        Merged device buffer. The user of this function is responsible for freeing this
 *                                 returned buffer using RMM_FREE.
 */
void* merge_free_received_offset(
                                 std::vector<void *> received_data,
                                 const std::vector<int64_t> &bucket_count,
                                 size_t item_size,
                                 int64_t &total_count,
                                 Communicator *communicator=nullptr,
                                 bool self_free=true)
{
    total_count = 0LL;

    for (auto count : bucket_count) {
        total_count += count;
    }

    void* merged_data {nullptr};

    RMM_CALL(RMM_ALLOC(&merged_data, total_count * item_size, 0));

    void* current_data = merged_data;

    for (int irank = 0; irank < bucket_count.size(); irank++) {
        CUDA_RT_CALL(cudaMemcpy(
            current_data, received_data[irank], bucket_count[irank] * item_size, cudaMemcpyDeviceToDevice
        ));

        current_data = (void *)((char *)current_data + bucket_count[irank] * item_size);
    }

    for (int irank = 0; irank < received_data.size(); irank ++) {
        if (!self_free && irank == communicator->mpi_rank)
            continue;

        RMM_CALL(RMM_FREE(received_data[irank], 0));
    }

    return merged_data;
}


#endif // __COMM_CUH
