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
#include <iostream>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <cudf/join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "../src/communicator.h"
#include "../src/distribute_table.cuh"
#include "../src/distributed_join.cuh"
#include "../src/error.cuh"
#include "../src/generate_table.cuh"
#include "../src/registered_memory_resource.hpp"
#include "../src/topology.cuh"

using cudf::table;

template <typename data_type>
__global__ void verify_correctness(const data_type *data1,
                                   const data_type *data2,
                                   cudf::size_type size)
{
  const cudf::size_type start_idx = threadIdx.x + blockDim.x * blockIdx.x;
  const cudf::size_type stride    = blockDim.x * gridDim.x;

  for (cudf::size_type idx = start_idx; idx < size; idx += stride) {
    assert(data1[idx] == data2[idx]);
  }
}

template <typename KEY_T, typename PAYLOAD_T>
void run_test(cudf::size_type build_table_size,
              cudf::size_type probe_table_size,
              double selectivity,
              bool is_build_table_key_unique,
              int over_decomposition_factor,
              bool compression,
              Communicator *communicator)
{
  int mpi_rank;
  int mpi_size;
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

  /* Generate build table and probe table and compute reference solution */

  std::unique_ptr<table> build;
  std::unique_ptr<table> probe;
  std::unique_ptr<table> reference;

  cudf::table_view build_view;
  cudf::table_view probe_view;

  if (mpi_rank == 0) {
    KEY_T rand_max_val = build_table_size * 2;

    std::tie(build, probe) = generate_build_probe_tables<KEY_T, PAYLOAD_T>(
      build_table_size, probe_table_size, selectivity, rand_max_val, is_build_table_key_unique);

    build_view = build->view();
    probe_view = probe->view();

    reference = cudf::inner_join(
      build->view(), probe->view(), {0}, {0}, {std::pair<cudf::size_type, cudf::size_type>(0, 0)});
  }

  std::unique_ptr<table> local_build = distribute_table(build_view, communicator);
  std::unique_ptr<table> local_probe = distribute_table(probe_view, communicator);

  /* Distributed join */

  std::unique_ptr<table> join_result_all_ranks =
    distributed_inner_join(local_build->view(),
                           local_probe->view(),
                           {0},
                           {0},
                           {std::pair<cudf::size_type, cudf::size_type>(0, 0)},
                           communicator,
                           over_decomposition_factor,
                           compression);

  /* Send join result from all ranks to the root rank */

  std::unique_ptr<table> join_result = collect_tables(join_result_all_ranks->view(), communicator);

  /* Verify correctness */

  if (mpi_rank == 0) {
    // Although join_result and reference should contain the same table, rows may be reordered.
    // Therefore, we first sort both tables and then compare

    cudf::size_type nrows = reference->num_rows();
    assert(join_result->num_rows() == nrows);

    std::unique_ptr<table> join_sorted      = cudf::sort(join_result->view());
    std::unique_ptr<table> reference_sorted = cudf::sort(reference->view());

    // Get the number of thread blocks based on thread block size

    const int block_size = 128;
    int nblocks{-1};

    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &nblocks, verify_correctness<KEY_T>, block_size, 0));

    // There should be three columns in the result table. The first column is the joined key
    // column. The second and third column comes from the payload column from the left and
    // the right input table, respectively.

    // Verify the first column (key column) is correct.

    verify_correctness<KEY_T>
      <<<nblocks, block_size>>>(join_sorted->view().column(0).head<KEY_T>(),
                                reference_sorted->view().column(0).head<KEY_T>(),
                                nrows);

    // Verify the remaining two payload columns are correct.

    for (cudf::size_type icol = 1; icol <= 2; icol++) {
      verify_correctness<PAYLOAD_T>
        <<<nblocks, block_size>>>(join_sorted->view().column(icol).head<PAYLOAD_T>(),
                                  reference_sorted->view().column(icol).head<PAYLOAD_T>(),
                                  nrows);
    }

    CUDA_RT_CALL(cudaDeviceSynchronize());

    std::cerr << "Test case (" << build_table_size << "," << probe_table_size << "," << selectivity
              << "," << is_build_table_key_unique << "," << over_decomposition_factor << ","
              << compression << ") passes successfully.\n";
  }
}

int main(int argc, char *argv[])
{
  /* Initialize topology */

  setup_topology(argc, argv);

  /* Initialize communicator */

  UCXCommunicator *communicator = initialize_ucx_communicator(false, 0, 0);

  /* Initialize memory pool */

  size_t free_memory, total_memory;
  CUDA_RT_CALL(cudaMemGetInfo(&free_memory, &total_memory));
  const size_t pool_size = free_memory - 5LL * (1LL << 29);  // free memory - 500MB

  registered_memory_resource mr(communicator);
  auto *pool_mr =
    new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(&mr, pool_size, pool_size);
  rmm::mr::set_current_device_resource(pool_mr);

  /* run test */

  run_test<int32_t, int32_t>(1'000'000, 5'000'000, 0.3, true, 10, false, communicator);
  run_test<int32_t, int32_t>(1'000'000, 5'000'000, 0.3, true, 10, true, communicator);
  run_test<int64_t, int64_t>(1'000'000, 5'000'000, 0.3, true, 10, false, communicator);
  run_test<int64_t, int64_t>(1'000'000, 5'000'000, 0.3, true, 10, true, communicator);
  run_test<int32_t, int32_t>(1'000'000, 5'000'000, 1.0, true, 10, false, communicator);
  run_test<int32_t, int32_t>(1'000'000, 5'000'000, 1.0, true, 10, true, communicator);
  run_test<int64_t, int64_t>(1'000'000, 5'000'000, 1.0, true, 10, false, communicator);
  run_test<int64_t, int64_t>(1'000'000, 5'000'000, 1.0, true, 10, true, communicator);
  run_test<int32_t, int32_t>(1'000'000, 1'000'000, 0.3, true, 10, false, communicator);
  run_test<int32_t, int32_t>(1'000'000, 1'000'000, 0.3, true, 10, true, communicator);
  run_test<int64_t, int64_t>(1'000'000, 1'000'000, 0.3, true, 10, false, communicator);
  run_test<int64_t, int64_t>(1'000'000, 1'000'000, 0.3, true, 10, true, communicator);
  run_test<int32_t, int32_t>(1'000'000, 5'000'000, 0.3, true, 1, false, communicator);
  run_test<int32_t, int32_t>(1'000'000, 5'000'000, 0.3, true, 1, true, communicator);
  run_test<int64_t, int64_t>(1'000'000, 5'000'000, 0.3, true, 1, false, communicator);
  run_test<int64_t, int64_t>(1'000'000, 5'000'000, 0.3, true, 1, true, communicator);

  /* Cleanup */

  delete pool_mr;
  communicator->finalize();
  delete communicator;

  return 0;
}
