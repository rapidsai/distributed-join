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

#include <mpi.h>
#include <cudf/types.hpp>
#include <vector>

#include "communicator.h"
#include "error.cuh"

enum COMM_TAGS { placeholder_tag, exchange_size_tag };

/**
 * Usage: mpi_dtype_from_c_type<input_type>() returns the MPI datatype corresponding to a native C
 * type "input_type".
 *
 * For example, mpi_dtype_from_c_type<int>() would return MPI_INT32_T.
 */
template <typename c_type>
MPI_Datatype mpi_dtype_from_c_type()
{
  MPI_Datatype mpi_dtype;
  if (std::is_same<c_type, int8_t>::value)
    mpi_dtype = MPI_INT8_T;
  else if (std::is_same<c_type, uint8_t>::value)
    mpi_dtype = MPI_UINT8_T;
  else if (std::is_same<c_type, int16_t>::value)
    mpi_dtype = MPI_INT16_T;
  else if (std::is_same<c_type, uint16_t>::value)
    mpi_dtype = MPI_UINT16_T;
  else if (std::is_same<c_type, int32_t>::value)
    mpi_dtype = MPI_INT32_T;
  else if (std::is_same<c_type, uint32_t>::value)
    mpi_dtype = MPI_UINT32_T;
  else if (std::is_same<c_type, int64_t>::value)
    mpi_dtype = MPI_INT64_T;
  else if (std::is_same<c_type, uint64_t>::value)
    mpi_dtype = MPI_UINT64_T;
  else if (std::is_same<c_type, float>::value)
    mpi_dtype = MPI_FLOAT;
  else if (std::is_same<c_type, double>::value)
    mpi_dtype = MPI_DOUBLE;

  return mpi_dtype;
}

/**
 * Send data from the current rank to other ranks according to offset.
 *
 * Note: This call should be enclosed by communicator->start() and communicator->stop().
 *
 * @param[in] data                The starting address of data to be sent in device buffer.
 * @param[in] offset              Vector of length mpi_size + 1. Items in *data* with indicies from
 * offset[i] to offset[i+1] will be sent to rank i.
 * @param[in] item_size           The size of each item.
 * @param[in] communicator        An instance of 'Communicator' used for communication.
 * @param[in] self_send           Whether sending data to itself. If this argument is false, items
 * in *data* destined for the current rank will not be copied.
 */
void send_data_by_offset(const void *data,
                         std::vector<int> const &offset,
                         size_t item_size,
                         Communicator *communicator,
                         bool self_send = true)
{
  int mpi_rank{communicator->mpi_rank};
  int mpi_size{communicator->mpi_size};

  for (int itarget_rank = 0; itarget_rank < mpi_size; itarget_rank++) {
    if (!self_send && itarget_rank == mpi_rank) continue;

    // calculate the number of elements to send
    size_t count = offset[itarget_rank + 1] - offset[itarget_rank];

    // calculate the starting address
    const void *start_addr = (void *)((char *)data + offset[itarget_rank] * item_size);

    // send buffer to the target rank
    communicator->send(start_addr, count, item_size, itarget_rank);
  }
}

/**
 * Receive data sent by 'send_data_by_offset'.
 *
 * Note: This call should be enclosed by communicator->start() and communicator->stop().
 *
 * @param[out] data         Items received from all ranks will be placed contiguously in *data*.
 *     This argument needs to be preallocated.
 * @param[in] offset        The items received from rank i will be stored at the start of
 * `data[offset[i]]`.
 * @param[in] item_size     The size of each item.
 * @param[in] communicator  An instance of 'Communicator' used for communication.
 * @param[in] self_recv     Whether recving data from itself. If this argument is false, items in
 *                          *data* from the current rank will not be received.
 */
void recv_data_by_offset(void *data,
                         std::vector<int64_t> const &offset,
                         size_t item_size,
                         Communicator *communicator,
                         bool self_recv = true)
{
  int mpi_rank{communicator->mpi_rank};
  int mpi_size{communicator->mpi_size};

  for (int isource_rank = 0; isource_rank < mpi_size; isource_rank++) {
    if (!self_recv && mpi_rank == isource_rank) continue;

    communicator->recv((void *)((char *)data + offset[isource_rank] * item_size),
                       offset[isource_rank + 1] - offset[isource_rank],
                       item_size,
                       isource_rank);
  }
}

#endif  // __COMM_CUH
