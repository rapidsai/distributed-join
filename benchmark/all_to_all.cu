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

#include <cuda_profiler_api.h>
#include <mpi.h>
#include <cstdint>
#include <iostream>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/communicator.h"
#include "../src/error.cuh"
#include "../src/topology.cuh"

static int64_t SIZE                     = 800'000'000LL;
static int64_t COMMUNICATOR_BUFFER_SIZE = 25'000'000LL;
static int REPEAT                       = 4;
static bool WARM_UP                     = false;
static std::string COMMUNICATOR_NAME    = "UCX";
static std::string REGISTRATION_METHOD  = "preregistered";

void parse_command_line_arguments(int argc, char *argv[])
{
  for (int iarg = 0; iarg < argc; iarg++) {
    if (!strcmp(argv[iarg], "--size")) { SIZE = atol(argv[iarg + 1]); }

    if (!strcmp(argv[iarg], "--buffer-size")) { COMMUNICATOR_BUFFER_SIZE = atol(argv[iarg + 1]); }

    if (!strcmp(argv[iarg], "--repeat")) { REPEAT = atoi(argv[iarg + 1]); }

    if (!strcmp(argv[iarg], "--warm-up")) { WARM_UP = true; }

    if (!strcmp(argv[iarg], "--communicator")) { COMMUNICATOR_NAME = argv[iarg + 1]; }

    if (!strcmp(argv[iarg], "--registration-method")) { REGISTRATION_METHOD = argv[iarg + 1]; }
  }
}

void report_configuration()
{
  MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));

  int mpi_rank;
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
  if (mpi_rank != 0) return;

  std::cout << "========== Parameters ==========" << std::endl;
  std::cout << std::boolalpha;
  std::cout << "Size: " << SIZE << std::endl;
  std::cout << "Communicator: " << COMMUNICATOR_NAME << std::endl;
  if (COMMUNICATOR_NAME == "UCX") {
    std::cout << "Registration method: " << REGISTRATION_METHOD << std::endl;
    if (REGISTRATION_METHOD == "buffer")
      std::cout << "Communicator buffer size: " << COMMUNICATOR_BUFFER_SIZE << std::endl;
  }
  std::cout << "Repeat: " << REPEAT << std::endl;
  std::cout << "Warmup: " << WARM_UP << std::endl;
  std::cout << "================================" << std::endl;
}

int main(int argc, char *argv[])
{
  /* Initialize topology */

  setup_topology(argc, argv);

  int mpi_rank;
  int mpi_size;
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

  /* Parse command line arguments */

  parse_command_line_arguments(argc, argv);
  report_configuration();

  /* Initialize communicator and memory pool */

  Communicator *communicator{nullptr};
  registered_memory_resource *registered_mr{nullptr};
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> *pool_mr{nullptr};

  setup_memory_pool_and_communicator(communicator,
                                     registered_mr,
                                     pool_mr,
                                     COMMUNICATOR_NAME,
                                     REGISTRATION_METHOD,
                                     COMMUNICATOR_BUFFER_SIZE);

  /* Warmup if necessary */

  if (WARM_UP) {
    const int64_t WARMUP_BUFFER_SIZE = 4'000'000LL;
    std::vector<void *> warmup_send_buffer(mpi_size, nullptr);
    std::vector<void *> warmup_recv_buffer(mpi_size, nullptr);

    for (int irank = 0; irank < mpi_size; irank++) {
      warmup_send_buffer[irank] = pool_mr->allocate(WARMUP_BUFFER_SIZE, cudaStreamDefault);
      warmup_recv_buffer[irank] = pool_mr->allocate(WARMUP_BUFFER_SIZE, cudaStreamDefault);
    }

    CUDA_RT_CALL(cudaStreamSynchronize(cudaStreamDefault));

    communicator->start();

    for (int irank = 0; irank < mpi_size; irank++) {
      if (irank != mpi_rank) {
        communicator->send(warmup_send_buffer[irank], WARMUP_BUFFER_SIZE, 1, irank);
      }
    }

    for (int irank = 0; irank < mpi_size; irank++) {
      if (irank != mpi_rank) {
        communicator->recv(warmup_recv_buffer[irank], WARMUP_BUFFER_SIZE, 1, irank);
      }
    }

    communicator->stop();

    for (int irank = 0; irank < mpi_rank; irank++) {
      pool_mr->deallocate(warmup_send_buffer[irank], WARMUP_BUFFER_SIZE, cudaStreamDefault);
      pool_mr->deallocate(warmup_recv_buffer[irank], WARMUP_BUFFER_SIZE, cudaStreamDefault);
    }

    CUDA_RT_CALL(cudaStreamSynchronize(cudaStreamDefault));
  }

  /* Allocate data buffers */

  std::vector<void *> send_buffer(mpi_size, nullptr);
  std::vector<void *> recv_buffer(mpi_size, nullptr);

  for (int irank = 0; irank < mpi_size; irank++) {
    send_buffer[irank] = pool_mr->allocate(SIZE / mpi_size, cudaStreamDefault);
    recv_buffer[irank] = pool_mr->allocate(SIZE / mpi_size, cudaStreamDefault);
  }

  CUDA_RT_CALL(cudaStreamSynchronize(cudaStreamDefault));

  /* Communication */

  MPI_Barrier(MPI_COMM_WORLD);
  cudaProfilerStart();
  double start = MPI_Wtime();

  for (int icol = 0; icol < REPEAT; icol++) {
    communicator->start();

    for (int irank = 0; irank < mpi_size; irank++) {
      if (irank != mpi_rank) communicator->send(send_buffer[irank], SIZE / mpi_size, 1, irank);
    }

    for (int irank = 0; irank < mpi_size; irank++) {
      if (irank != mpi_rank) communicator->recv(recv_buffer[irank], SIZE / mpi_size, 1, irank);
    }

    communicator->stop();
  }

  double stop = MPI_Wtime();
  cudaProfilerStop();

  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 0) {
    std::cerr << "Elasped time (s) " << stop - start << std::endl;
    std::cerr << "Bandwidth (GB/s) "
              << (double)SIZE / mpi_size * (mpi_size - 1) * REPEAT / (stop - start) / 1e9
              << std::endl;
  }

  /* Cleanup */

  for (int irank = 0; irank < mpi_rank; irank++) {
    pool_mr->deallocate(send_buffer[irank], SIZE / mpi_size, cudaStreamDefault);
    pool_mr->deallocate(recv_buffer[irank], SIZE / mpi_size, cudaStreamDefault);
  }

  CUDA_RT_CALL(cudaStreamSynchronize(cudaStreamDefault));

  destroy_memory_pool_and_communicator(
    communicator, registered_mr, pool_mr, COMMUNICATOR_NAME, REGISTRATION_METHOD);

  MPI_CALL(MPI_Finalize());

  return 0;
}
