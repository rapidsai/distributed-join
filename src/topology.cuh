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

#include "error.cuh"
#include "registered_memory_resource.hpp"

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <mpi.h>

#include <iostream>

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

/**
 * Setup RMM memory pool and communicator.
 *
 * This function will set the current device's memory pool. The memory pool and communicator
 * initialized in this function can be destroyed by *destroy_memory_pool_and_communicator*.
 *
 * @param[out]: communicator Communicator to be constructed.
 * @param[out]: registered_mr If the memory pool needs to be preregistered, this argument holds
 * pointer to the registered memory resource. If not preregistered, this argument will be *nullptr*.
 * @param[out]: pool_mr RMM memory resource for memory pool.
 * @param[in]: communicator_name Can be either "NCCL" or "UCX".
 * @param[in]: registration_method If using UCX communicator, this argument can be either "none",
 * "buffer" or "preregistered".
 * @param[in]: communicator_buffer_size If the registration_method is set to "buffer", this argument
 * controls the size of the communication buffer used by the communicator.
 */
void setup_memory_pool_and_communicator(
  Communicator *&communicator,
  registered_memory_resource *&registered_mr,
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> *&pool_mr,
  std::string communicator_name,
  std::string registration_method,
  int64_t communicator_buffer_size)
{
  int mpi_size;
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

  registered_mr = nullptr;

  // Calculate the memory pool size
  size_t free_memory, total_memory;
  CUDA_RT_CALL(cudaMemGetInfo(&free_memory, &total_memory));
  const size_t pool_size = free_memory - 5LL * (1LL << 29);  // free memory - 500MB

  if (communicator_name == "NCCL") {
    communicator = new NCCLCommunicator;
    communicator->initialize();
    pool_mr = new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(
      rmm::mr::get_current_device_resource(), pool_size, pool_size);
    rmm::mr::set_current_device_resource(pool_mr);
  } else if (communicator_name == "UCX") {
    if (registration_method == "buffer") {
      // For UCX with buffer communicator, a memory pool is first constructed so that the
      // communication buffers will be allocated in memory pool.
      pool_mr = new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(
        rmm::mr::get_current_device_resource(), pool_size, pool_size);
      rmm::mr::set_current_device_resource(pool_mr);
      // *2 because buffers are needed for both sends and receives
      const int num_comm_buffers = 2 * mpi_size;
      communicator               = initialize_ucx_communicator(
        true, num_comm_buffers, communicator_buffer_size / num_comm_buffers - 100'000LL);
    } else if (registration_method == "preregistered") {
      // For UCX with preregistered memory pool, a communicator is first constructed so that
      // `registered_memory_resource` can use the communicator for buffer registrations.
      UCXCommunicator *ucx_communicator = initialize_ucx_communicator(false, 0, 0);
      communicator                      = ucx_communicator;
      registered_mr                     = new registered_memory_resource(ucx_communicator);
      pool_mr = new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(
        registered_mr, pool_size, pool_size);
      rmm::mr::set_current_device_resource(pool_mr);
    } else if (registration_method == "none") {
      communicator = initialize_ucx_communicator(false, 0, 0);
      pool_mr      = new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(
        rmm::mr::get_current_device_resource(), pool_size, pool_size);
      rmm::mr::set_current_device_resource(pool_mr);
    } else {
      throw std::runtime_error("Unknown registration method");
    }
  } else {
    throw std::runtime_error("Unknown communicator name");
  }
}

void destroy_memory_pool_and_communicator(
  Communicator *communicator,
  registered_memory_resource *registered_mr,
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> *pool_mr,
  std::string communicator_name,
  std::string registration_method)
{
  if (communicator_name == "UCX" && registration_method == "buffer") {
    // When finalizing buffer communicator, communication buffers need be deallocated, so
    // `finalize` needs to be called before the memory pool is deleted.
    communicator->finalize();
    delete pool_mr;
    delete registered_mr;
  } else {
    // For registered memory resouce, the memory pool needs to be deleted before finalizing
    // the communicator, so that all buffers can be deregistered through UCX.
    // For every other scenario, the order of deleting memory pool and finalizing the communicator
    // does not matter, and we just choose this path.
    delete pool_mr;
    delete registered_mr;
    communicator->finalize();
  }

  delete communicator;
}
