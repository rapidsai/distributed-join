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

#include "comm.hpp"

#include "communicator.hpp"
#include "error.hpp"

#include <cudf/types.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <mpi.h>

#include <cuda_runtime.h>

#include <numeric>
#include <vector>

void communicate_sizes(std::vector<int64_t> const &send_offset,
                       std::vector<int64_t> &recv_offset,
                       Communicator *communicator)
{
  int mpi_size = communicator->mpi_size;
  std::vector<int64_t> send_count(mpi_size, -1);

  for (int irank = 0; irank < mpi_size; irank++) {
    send_count[irank] = send_offset[irank + 1] - send_offset[irank];
  }

  std::vector<int64_t> recv_count(mpi_size, -1);

  // Note: MPI is used for communicating the sizes instead of *Communicator* because
  // *Communicator* is not guaranteed to be able to send/recv host buffers.

  std::vector<MPI_Request> send_req(mpi_size);
  std::vector<MPI_Request> recv_req(mpi_size);

  for (int irank = 0; irank < mpi_size; irank++) {
    MPI_CALL(MPI_Isend(&send_count[irank],
                       1,
                       MPI_INT64_T,
                       irank,
                       exchange_size_tag,
                       MPI_COMM_WORLD,
                       &send_req[irank]));
  }

  for (int irank = 0; irank < mpi_size; irank++) {
    MPI_CALL(MPI_Irecv(&recv_count[irank],
                       1,
                       MPI_INT64_T,
                       irank,
                       exchange_size_tag,
                       MPI_COMM_WORLD,
                       &recv_req[irank]));
  }

  MPI_CALL(MPI_Waitall(mpi_size, send_req.data(), MPI_STATUSES_IGNORE));
  MPI_CALL(MPI_Waitall(mpi_size, recv_req.data(), MPI_STATUSES_IGNORE));

  recv_offset.resize(mpi_size + 1, -1);
  recv_offset[0] = 0;
  std::partial_sum(recv_count.begin(), recv_count.end(), recv_offset.begin() + 1);
}

void communicate_sizes(std::vector<cudf::size_type> const &send_offset,
                       std::vector<int64_t> &recv_offset,
                       Communicator *communicator)
{
  communicate_sizes(
    std::vector<int64_t>(send_offset.begin(), send_offset.end()), recv_offset, communicator);
}

void send_data_by_offset(const void *data,
                         std::vector<int64_t> const &offset,
                         size_t item_size,
                         Communicator *communicator,
                         bool self_send)
{
  int mpi_rank = communicator->mpi_rank;
  int mpi_size = communicator->mpi_size;

  for (int itarget_rank = 0; itarget_rank < mpi_size; itarget_rank++) {
    if (!self_send && itarget_rank == mpi_rank) continue;

    // calculate the number of elements to send
    int64_t count = offset[itarget_rank + 1] - offset[itarget_rank];

    // calculate the starting address
    const void *start_addr = (void *)((char *)data + offset[itarget_rank] * item_size);

    // send buffer to the target rank
    communicator->send(start_addr, count, item_size, itarget_rank);
  }
}

void recv_data_by_offset(void *data,
                         std::vector<int64_t> const &offset,
                         size_t item_size,
                         Communicator *communicator,
                         bool self_recv)
{
  int mpi_rank = communicator->mpi_rank;
  int mpi_size = communicator->mpi_size;

  for (int isource_rank = 0; isource_rank < mpi_size; isource_rank++) {
    if (!self_recv && mpi_rank == isource_rank) continue;

    communicator->recv((void *)((char *)data + offset[isource_rank] * item_size),
                       offset[isource_rank + 1] - offset[isource_rank],
                       item_size,
                       isource_rank);
  }
}

void warmup_all_to_all(Communicator *communicator)
{
  int mpi_rank                        = communicator->mpi_rank;
  int mpi_size                        = communicator->mpi_size;
  int64_t size                        = 10'000'000LL;
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource();

  /* Allocate send/recv buffers */

  std::vector<void *> send_buffer(mpi_size, nullptr);
  std::vector<void *> recv_buffer(mpi_size, nullptr);

  for (int irank = 0; irank < mpi_size; irank++) {
    if (irank == mpi_rank) continue;
    send_buffer[irank] = mr->allocate(size / mpi_size, rmm::cuda_stream_default);
    recv_buffer[irank] = mr->allocate(size / mpi_size, rmm::cuda_stream_default);
  }

  CUDA_RT_CALL(cudaStreamSynchronize(0));

  /* Communication */

  communicator->start();

  for (int irank = 0; irank < mpi_size; irank++) {
    if (irank != mpi_rank) communicator->send(send_buffer[irank], size / mpi_size, 1, irank);
  }

  for (int irank = 0; irank < mpi_size; irank++) {
    if (irank != mpi_rank) communicator->recv(recv_buffer[irank], size / mpi_size, 1, irank);
  }

  communicator->stop();

  /* Deallocate send/recv buffers */

  for (int irank = 0; irank < mpi_rank; irank++) {
    mr->deallocate(send_buffer[irank], size / mpi_size, rmm::cuda_stream_default);
    mr->deallocate(recv_buffer[irank], size / mpi_size, rmm::cuda_stream_default);
  }

  CUDA_RT_CALL(cudaStreamSynchronize(0));
}
