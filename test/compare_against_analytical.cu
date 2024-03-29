/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "../src/communicator.hpp"
#include "../src/distribute_table.hpp"
#include "../src/distributed_join.hpp"
#include "../src/error.hpp"
#include "../src/setup.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <mpi.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

using cudf::table;

__global__ void verify_correctness(
  const int *col0, const int *col1, const int *col2, const int *col3, int size)
{
  for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += blockDim.x * gridDim.x) {
    assert(col0[i] % 15 == 0);
    assert(col1[i] == col0[i] / 3);
    assert(col2[i] % 15 == 0);
    assert(col3[i] == col2[i] / 5);
    assert(col0[i] == col2[i]);
  }
}

/**
 * This helper function generates the left/right table used for testing join.
 *
 * There are two columns in each table. The first column is filled with consecutive multiple of
 * argument *multiple*, and is used as key column. For example, if *multiple* is 3, the column
 * contains 0,3,6,9...etc. The second column is filled with consecutive integers and is used as
 * payload column.
 */
std::unique_ptr<table> generate_table(cudf::size_type size, int multiple)
{
  std::vector<std::unique_ptr<cudf::column>> new_table;

  // construct the key column
  auto key_column = cudf::make_numeric_column(cudf::data_type(cudf::type_id::INT32), size);
  auto key_buffer = key_column->mutable_view().head<int>();
  thrust::sequence(thrust::device, key_buffer, key_buffer + size, 0, multiple);
  new_table.push_back(std::move(key_column));

  // construct the payload column
  auto payload_column = cudf::make_numeric_column(cudf::data_type(cudf::type_id::INT32), size);
  auto payload_buffer = payload_column->mutable_view().head<int>();
  thrust::sequence(thrust::device, payload_buffer, payload_buffer + size);
  new_table.push_back(std::move(payload_column));

  return std::make_unique<table>(std::move(new_table));
}

void run_test(cudf::size_type size,  // must be a multiple of 5
              int over_decomposition_factor,
              bool compression,
              int nvlink_domain_size,
              Communicator *communicator)
{
  assert(size % 5 == 0);

  int mpi_rank;
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

  /* Generate input tables */

  std::unique_ptr<table> left_table;
  std::unique_ptr<table> right_table;
  cudf::table_view left_view;
  cudf::table_view right_view;

  if (mpi_rank == 0) {
    left_table  = generate_table(size, 3);
    right_table = generate_table(size, 5);

    left_view  = left_table->view();
    right_view = right_table->view();

    CUDA_RT_CALL(cudaStreamSynchronize(cudaStreamDefault));
  }

  /* Distribute input tables among ranks */

  auto local_left_table  = distribute_table(left_view, communicator);
  auto local_right_table = distribute_table(right_view, communicator);

  /* Generate compression options */

  std::vector<ColumnCompressionOptions> left_compression_options =
    generate_compression_options_distributed(local_left_table->view(), compression);
  std::vector<ColumnCompressionOptions> right_compression_options =
    generate_compression_options_distributed(local_right_table->view(), compression);

  /* Distributed join */

  auto join_result = distributed_inner_join(local_left_table->view(),
                                            local_right_table->view(),
                                            {0},
                                            {0},
                                            communicator,
                                            left_compression_options,
                                            right_compression_options,
                                            over_decomposition_factor,
                                            false,
                                            nullptr,
                                            nvlink_domain_size);

  /* Merge table from worker ranks to the root rank */

  std::unique_ptr<table> merged_table = collect_tables(join_result->view(), communicator);

  /* Verify Correctness */

  if (mpi_rank == 0) {
    const int block_size{128};
    int nblocks{-1};

    CUDA_RT_CALL(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, verify_correctness, block_size, 0));

    // Since the key has to be a multiple of 5, the join result size is the size of left table
    // divided by 5.
    assert(merged_table->num_rows() == size / 5);

    verify_correctness<<<nblocks, block_size>>>(merged_table->get_column(0).view().head<int>(),
                                                merged_table->get_column(1).view().head<int>(),
                                                merged_table->get_column(2).view().head<int>(),
                                                merged_table->get_column(3).view().head<int>(),
                                                merged_table->num_rows());

    CUDA_RT_CALL(cudaDeviceSynchronize());
    std::cerr << "Test case (" << size << "," << over_decomposition_factor << "," << compression
              << "," << nvlink_domain_size << ") passes successfully.\n";
  }
}

int main(int argc, char *argv[])
{
  MPI_CALL(MPI_Init(&argc, &argv));
  set_cuda_device();

  /* Initialize memory pool */

  const size_t pool_size = 960'000'000;  // 960MB

  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource();
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr{mr, pool_size, pool_size};
  rmm::mr::set_current_device_resource(&pool_mr);

  /* Initialize communicator */

  int mpi_size;
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

  UCXCommunicator *communicator = initialize_ucx_communicator(
    true,
    2 * mpi_size,  // *2 because buffers are needed for both sends and receives
    100'000LL);

  /* Run test */

  // Note: temporarily disable some test cases because nvcomp's cascaded selector can raise
  // "Floating point exception" if the input buffer is smaller than sample_size * num_samples.

  run_test(30'000, 1, false, 1, communicator);
  run_test(300'000, 1, false, 1, communicator);
  // run_test(300'000, 1, true, 1, communicator);
  run_test(300'000, 4, false, 1, communicator);
  // run_test(300'000, 4, true, 1, communicator);
  run_test(3'000'000, 1, true, 1, communicator);
  run_test(3'000'000, 4, true, 1, communicator);
  run_test(3'000'000, 4, true, 2, communicator);

  /* Cleanup */

  communicator->finalize();
  delete communicator;

  MPI_CALL(MPI_Finalize());

  return 0;
}
