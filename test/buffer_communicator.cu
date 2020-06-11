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

#include <cstdint>
#include <vector>
#include <iostream>
#include <cassert>

#include "../src/communicator.h"
#include "../src/error.cuh"

static constexpr int64_t COUNT = 50'000'000LL;


__global__ void set_data(uint64_t *start_addr, uint64_t size, uint64_t start_val)
{
    const int ithread = threadIdx.x + blockDim.x * blockIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (uint64_t ielement = ithread; ielement < size; ielement += stride) {
        start_addr[ielement] = (start_val + ielement);
    }
}


__global__ void test_correctness(uint64_t *start_addr, uint64_t size, uint64_t start_val)
{
    const int ithread = threadIdx.x + blockDim.x * blockIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (uint64_t ielement = ithread; ielement < size; ielement += stride) {
        assert(start_addr[ielement] == (start_val + ielement));
    }
}


int main(int argc, char *argv[])
{
    UCXBufferCommunicator communicator;
    communicator.initialize(argc, argv);

    int mpi_rank = communicator.mpi_rank;
    int mpi_size = communicator.mpi_size;

    communicator.setup_cache(2 * mpi_size, 20'000'000LL);
    communicator.warmup_cache();

    /* Send and recv data */

    uint64_t *send_buf {nullptr};
    std::vector<uint64_t *> recv_buf(mpi_size, nullptr);

    std::vector<comm_handle_t> send_reqs(mpi_size, nullptr);
    std::vector<comm_handle_t> recv_reqs(mpi_size, nullptr);

    RMM_CALL(RMM_ALLOC(&send_buf, COUNT * sizeof(uint64_t), 0));

    int grid_size {-1};
    int block_size {-1};

    CUDA_RT_CALL(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, set_data));
    set_data<<<grid_size, block_size>>>(send_buf, COUNT, COUNT * mpi_rank);

    for (int irank = 0; irank < mpi_size; irank ++) {
        if (irank != mpi_rank) {
            send_reqs[irank] = communicator.send((void *)send_buf, COUNT, sizeof(uint64_t), irank, 32);
        }
    }

    int64_t count_received;

    for (int irank = mpi_size - 1; irank >= 0; irank --) {
        if (irank != mpi_rank) {
            recv_reqs[irank] = communicator.recv(
                (void **)&recv_buf[irank], &count_received, sizeof(uint64_t), irank, 32
            );
        }
    }

    communicator.waitall(send_reqs);
    communicator.waitall(recv_reqs);

    assert(count_received == COUNT);

    /* Test the correctness */

    for (int irank = 0; irank < mpi_size; irank ++) {
        if (irank != mpi_rank) {
            test_correctness<<<grid_size, block_size>>>(recv_buf[irank], COUNT, COUNT * irank);
        }
    }

    /* Cleanup */

    RMM_CALL(RMM_FREE(send_buf, 0));
    for (int irank = 0; irank < mpi_size; irank ++) {
        if (irank != mpi_rank) {
            RMM_CALL(RMM_FREE(recv_buf[irank], 0));
        }
    }

    communicator.finalize();

    if (mpi_rank == 0) {
        std::cerr << "Test case \"buffer_communicator\" passes successfully.\n";
    }

    return 0;
}
