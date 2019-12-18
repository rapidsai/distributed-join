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

#include <vector>
#include <rmm/rmm.h>
#include <mpi.h>
#include <iostream>
#include <cuda_profiler_api.h>

#include "../src/communicator.h"
#include "../src/error.cuh"

#define SIZE 800'000'000LL
#define BUFFER_SIZE 25'000'000LL
#define REPEAT 4


int main(int argc, char *argv[])
{
    UCXBufferCommunicator communicator;
    communicator.initialize(argc, argv);

    int mpi_rank = communicator.mpi_rank;
    int mpi_size = communicator.mpi_size;

    communicator.setup_cache(2 * mpi_size, BUFFER_SIZE);
    communicator.warmup_cache();

    /* Allocate data buffers */

    std::vector<void *> send_buffer(mpi_size, nullptr);
    std::vector<void *> recv_buffer(mpi_size, nullptr);

    for (int irank = 0; irank < mpi_size; irank ++) {
        RMM_CALL(RMM_ALLOC(&send_buffer[irank], SIZE / mpi_size, 0));
    }

    std::vector<comm_handle_t> send_reqs(mpi_size, nullptr);
    std::vector<comm_handle_t> recv_reqs(mpi_size, nullptr);

    /* Communication */

    UCX_CALL(ucp_worker_flush(communicator.ucp_worker));
    MPI_Barrier(MPI_COMM_WORLD);
    cudaProfilerStart();
    double start = MPI_Wtime();

    for (int icol = 0; icol < REPEAT; icol ++)
    {
        for (int irank = 0; irank < mpi_size; irank++) {
            if (irank != mpi_rank)
                send_reqs[irank] = communicator.send(send_buffer[irank], SIZE / mpi_size, 1, irank, 20);
            else
                send_reqs[irank] = nullptr;
        }

        for (int irank = 0; irank < mpi_size; irank++) {
            if (irank != mpi_rank)
                recv_reqs[irank] = communicator.recv(&recv_buffer[irank], nullptr, 1, irank, 20);
            else
                recv_reqs[irank] = nullptr;
        }

        communicator.waitall(send_reqs);
        communicator.waitall(recv_reqs);

        for (int irank = 0; irank < mpi_rank; irank ++)
            RMM_CALL(RMM_FREE(recv_buffer[irank], 0));
    }

    double stop = MPI_Wtime();
    cudaProfilerStop();

    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        std::cerr << "Elasped time (s) " << stop - start << std::endl;
        std::cerr << "Bandwidth (GB/s) " << (double)SIZE * (mpi_size - 5) * REPEAT / (stop - start) / 1e9 << std::endl;
    }

    /* Cleanup */

    for(int irank = 0; irank < mpi_rank; irank++) {
        RMM_CALL(RMM_FREE(send_buffer[irank], 0));
    }

    communicator.finalize();
    return 0;
}
