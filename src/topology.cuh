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

#include <mpi.h>
#include <iostream>

#include "error.cuh"

void setup_topology(int argc, char *argv[])
{
  int device_count;
  int mpi_rank;

  MPI_CALL(MPI_Init(&argc, &argv));
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

  CUDA_RT_CALL(cudaGetDeviceCount(&device_count));
  std::cout << "Device count: " << device_count << std::endl;

  int current_device = mpi_rank % device_count;

  CUDA_RT_CALL(cudaSetDevice(current_device));
  std::cout << "Rank " << mpi_rank << " select " << current_device << "/" << device_count << " GPU"
            << std::endl;
}
