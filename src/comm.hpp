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

#pragma once

#include "communicator.hpp"

#include <cudf/types.hpp>

#include <mpi.h>

#include <cstdint>
#include <type_traits>
#include <vector>

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
 * Communicate number of elements recieved from each rank during all-to-all communication.
 *
 * Note: This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] send_offset Vector of length mpi_size + 1 such that `send_offset[i+1] -
 * send_offset[i]` is the number of elements sent from the current rank to rank i during the
 * all-to-all communication.
 * @param[out] recv_offset Vector of length mpi_size + 1 such that `recv_offset[i+1] -
 * recv_offset[i]` is the number of elements received from rank i during the all-to-all
 * communication. The vector will be resized in this function and does not need to be preallocated.
 */
void communicate_sizes(std::vector<int64_t> const &send_offset,
                       std::vector<int64_t> &recv_offset,
                       Communicator *communicator);

void communicate_sizes(std::vector<cudf::size_type> const &send_offset,
                       std::vector<int64_t> &recv_offset,
                       Communicator *communicator);
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
                         std::vector<int64_t> const &offset,
                         size_t item_size,
                         Communicator *communicator,
                         bool self_send = true);

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
                         bool self_recv = true);

void warmup_all_to_all(Communicator *communicator);
