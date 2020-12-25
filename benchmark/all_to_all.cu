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

static int REPEAT                           = 4;
static std::string COMMUNICATOR_NAME        = "UCX";
static std::string REGISTRATION_METHOD      = "preregistered";
static int64_t COMMUNICATOR_BUFFER_SIZE     = 25'000'000LL;
static constexpr int64_t WARMUP_BUFFER_SIZE = 4'000'000LL;
static const std::vector<int64_t> SIZES{1'000'000LL,
                                        2'000'000LL,
                                        4'000'000LL,
                                        8'000'000LL,
                                        16'000'000LL,
                                        32'000'000LL,
                                        64'000'000LL,
                                        128'000'000LL,
                                        256'000'000LL,
                                        512'000'000LL,
                                        1024'000'000LL,
                                        2048'000'000LL,
                                        4096'000'000LL};

void parse_command_line_arguments(int argc, char *argv[])
{
  for (int iarg = 0; iarg < argc; iarg++) {
    if (!strcmp(argv[iarg], "--repeat")) { REPEAT = atoi(argv[iarg + 1]); }

    if (!strcmp(argv[iarg], "--communicator")) { COMMUNICATOR_NAME = argv[iarg + 1]; }

    if (!strcmp(argv[iarg], "--registration-method")) { REGISTRATION_METHOD = argv[iarg + 1]; }

    if (!strcmp(argv[iarg], "--buffer-size")) { COMMUNICATOR_BUFFER_SIZE = atol(argv[iarg + 1]); }
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
  std::cout << "Communicator: " << COMMUNICATOR_NAME << std::endl;
  if (COMMUNICATOR_NAME == "UCX") {
    std::cout << "Registration method: " << REGISTRATION_METHOD << std::endl;
    if (REGISTRATION_METHOD == "buffer")
      std::cout << "Communicator buffer size: " << COMMUNICATOR_BUFFER_SIZE << std::endl;
  }
  std::cout << "Repeat: " << REPEAT << std::endl;
  std::cout << "================================" << std::endl;
}

void run_all_to_all(int64_t size,
                    Communicator *communicator,
                    rmm::mr::device_memory_resource *mr,
                    bool print_result = true)
{
  int mpi_rank;
  int mpi_size;
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

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

  MPI_Barrier(MPI_COMM_WORLD);
  cudaProfilerStart();
  double start = MPI_Wtime();

  for (int run = 0; run < REPEAT; run++) {
    communicator->start();

    for (int irank = 0; irank < mpi_size; irank++) {
      if (irank != mpi_rank) communicator->send(send_buffer[irank], size / mpi_size, 1, irank);
    }

    for (int irank = 0; irank < mpi_size; irank++) {
      if (irank != mpi_rank) communicator->recv(recv_buffer[irank], size / mpi_size, 1, irank);
    }

    communicator->stop();
  }

  double stop = MPI_Wtime();
  cudaProfilerStop();

  MPI_Barrier(MPI_COMM_WORLD);

  if (mpi_rank == 0 && print_result) {
    std::cout << "Size (MB): " << size / 1e6 << ", "
              << "Elasped time (s): " << stop - start << ", "
              << "Bandwidth per GPU (GB/s): "
              << (double)size / mpi_size * (mpi_size - 1) * REPEAT / (stop - start) / 1e9
              << std::endl;
  }

  /* Deallocate send/recv buffers */

  for (int irank = 0; irank < mpi_rank; irank++) {
    mr->deallocate(send_buffer[irank], size / mpi_size, rmm::cuda_stream_default);
    mr->deallocate(recv_buffer[irank], size / mpi_size, rmm::cuda_stream_default);
  }

  CUDA_RT_CALL(cudaStreamSynchronize(0));
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

  /* Warmup */

  run_all_to_all(WARMUP_BUFFER_SIZE, communicator, pool_mr, false);

  /* Benchmark */

  for (const int64_t &size : SIZES) run_all_to_all(size, communicator, pool_mr, true);

  /* Cleanup */

  destroy_memory_pool_and_communicator(
    communicator, registered_mr, pool_mr, COMMUNICATOR_NAME, REGISTRATION_METHOD);

  MPI_CALL(MPI_Finalize());

  return 0;
}
