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
#include <mpi.h>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "../src/topology.cuh"
#include "../src/communicator.h"
#include "../src/error.cuh"

static int64_t COUNT = 50'000'000LL;


void parse_command_line_arguments(int argc, char *argv[])
{
    for (int iarg = 0; iarg < argc; iarg++) {
        if (!strcmp(argv[iarg], "--count")) {
            COUNT = atol(argv[iarg + 1]);
        }
    }
}


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
    /* Initialize topology */

    setup_topology(argc, argv);

    /* Parse command line arguments */

    parse_command_line_arguments(argc, argv);

    /* Initialize memory pool */

    size_t free_memory, total_memory;
    CUDA_RT_CALL(cudaMemGetInfo(&free_memory, &total_memory));
    const size_t pool_size = free_memory - 5LL * (1LL << 29);  // free memory - 500MB

    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr {mr, pool_size, pool_size};
    rmm::mr::set_current_device_resource(&pool_mr);

    /* Initialize communicator */

    int mpi_rank;
    int mpi_size;
    MPI_CALL( MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) );
    MPI_CALL( MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) );

    UCXCommunicator* communicator = initialize_ucx_communicator(
        true, 2 * mpi_size, 20'000'000LL
    );

    /* Send and recv data */

    rmm::device_buffer send_buf {COUNT * sizeof(uint64_t), 0};
    std::vector<uint64_t *> recv_buf(mpi_size, nullptr);

    std::vector<comm_handle_t> send_reqs(mpi_size, nullptr);
    std::vector<comm_handle_t> recv_reqs(mpi_size, nullptr);

    int grid_size {-1};
    int block_size {-1};

    CUDA_RT_CALL(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, set_data));
    set_data<<<grid_size, block_size>>>(
        static_cast<uint64_t *>(send_buf.data()), COUNT, COUNT * mpi_rank
    );

    for (int irank = 0; irank < mpi_size; irank ++) {
        if (irank != mpi_rank) {
            send_reqs[irank] = communicator->send(send_buf.data(), COUNT, sizeof(uint64_t), irank, 32);
        }
    }

    int64_t count_received;

    for (int irank = mpi_size - 1; irank >= 0; irank --) {
        if (irank != mpi_rank) {
            recv_reqs[irank] = communicator->recv(
                (void **)&recv_buf[irank], &count_received, sizeof(uint64_t), irank, 32
            );
        }
    }

    communicator->waitall(send_reqs);
    communicator->waitall(recv_reqs);

    assert(count_received == COUNT);

    /* Test the correctness */

    for (int irank = 0; irank < mpi_size; irank ++) {
        if (irank != mpi_rank) {
            test_correctness<<<grid_size, block_size>>>(recv_buf[irank], COUNT, COUNT * irank);
        }
    }

    /* Cleanup */

    for (int irank = 0; irank < mpi_size; irank ++) {
        if (irank != mpi_rank) {
            rmm::mr::get_current_device_resource()->deallocate(
                recv_buf[irank], COUNT, cudaStreamDefault
            );
        }
    }

    CUDA_RT_CALL( cudaStreamSynchronize(cudaStreamDefault) );
    communicator->finalize();
    delete communicator;

    if (mpi_rank == 0) {
        std::cerr << "Test case \"buffer_communicator\" passes successfully.\n";
    }

    return 0;
}
