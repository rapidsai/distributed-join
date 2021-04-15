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

/*
This test case compares the result of distributed join on multiple GPUs to the result of
cudf::inner_join on a single GPU.

Specifically, it follows the following steps:
1. The root rank constructs a random build table and a random probe table.
2. The root rank runs cudf::inner_join on the newly constructed tables.
3. The root rank distributes the build and probe table across all ranks.
4. All ranks run distibuted join collectively.
5. Each rank sends the distributed join result to the root rank.
6. The root rank assembles the received results into a single table and compares it to the result of
step 2.
*/

#include "../src/communicator.hpp"
#include "../src/distribute_table.hpp"
#include "../src/distributed_join.hpp"
#include "../src/error.hpp"
#include "../src/generate_table.cuh"
#include "../src/registered_memory_resource.hpp"
#include "../src/setup.hpp"

#include <cudf/join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using cudf::table;

template <typename dtype>
std::string dtype_to_string()
{
  if (std::is_same<dtype, int32_t>::value) {
    return "int32_t";
  } else if (std::is_same<dtype, int64_t>::value) {
    return "int64_t";
  } else if (std::is_same<dtype, cudf::timestamp_D>::value) {
    return "timestamp_D";
  } else if (std::is_same<dtype, cudf::timestamp_ms>::value) {
    return "timestamp_ms";
  } else if (std::is_same<dtype, cudf::timestamp_ns>::value) {
    return "timestamp_ns";
  } else if (std::is_same<dtype, cudf::duration_D>::value) {
    return "duration_D";
  } else if (std::is_same<dtype, cudf::duration_s>::value) {
    return "duration_s";
  } else if (std::is_same<dtype, cudf::duration_us>::value) {
    return "duration_us";
  } else {
    return "unknown_t";
  }
}

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
              int nvlink_domain_size,
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

    reference = cudf::inner_join(build->view(), probe->view(), {0}, {0});
  }

  std::unique_ptr<table> local_build = distribute_table(build_view, communicator);
  std::unique_ptr<table> local_probe = distribute_table(probe_view, communicator);

  /* Generate compression options */

  std::vector<ColumnCompressionOptions> build_compression_options =
    generate_compression_options_distributed(local_build->view(), compression);
  std::vector<ColumnCompressionOptions> probe_compression_options =
    generate_compression_options_distributed(local_probe->view(), compression);

  /* Distributed join */

  std::unique_ptr<table> join_result_all_ranks = distributed_inner_join(local_build->view(),
                                                                        local_probe->view(),
                                                                        {0},
                                                                        {0},
                                                                        communicator,
                                                                        build_compression_options,
                                                                        probe_compression_options,
                                                                        over_decomposition_factor,
                                                                        false,
                                                                        nullptr,
                                                                        nvlink_domain_size);

  /* Send join result from all ranks to the root rank */

  std::unique_ptr<table> join_result = collect_tables(join_result_all_ranks->view(), communicator);

  /* Verify correctness */

  if (mpi_rank == 0) {
    // Compare the number of columns
    cudf::size_type ncols = reference->num_columns();
    assert(join_result->num_columns() == ncols);
    assert(ncols == 4);

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

    // There should be four columns in the result table. The first two columns are from the left
    // table, and the last two columns are from the right table.

    verify_correctness<KEY_T>
      <<<nblocks, block_size>>>(join_sorted->view().column(0).head<KEY_T>(),
                                reference_sorted->view().column(0).head<KEY_T>(),
                                nrows);

    verify_correctness<PAYLOAD_T>
      <<<nblocks, block_size>>>(join_sorted->view().column(1).head<PAYLOAD_T>(),
                                reference_sorted->view().column(1).head<PAYLOAD_T>(),
                                nrows);

    verify_correctness<KEY_T>
      <<<nblocks, block_size>>>(join_sorted->view().column(2).head<KEY_T>(),
                                reference_sorted->view().column(2).head<KEY_T>(),
                                nrows);

    verify_correctness<PAYLOAD_T>
      <<<nblocks, block_size>>>(join_sorted->view().column(3).head<PAYLOAD_T>(),
                                reference_sorted->view().column(3).head<PAYLOAD_T>(),
                                nrows);

    CUDA_RT_CALL(cudaDeviceSynchronize());

    std::cerr << std::boolalpha;
    std::cerr << "Test case (" << dtype_to_string<KEY_T>() << "," << dtype_to_string<PAYLOAD_T>()
              << "," << build_table_size << "," << probe_table_size << "," << selectivity << ","
              << is_build_table_key_unique << "," << over_decomposition_factor << "," << compression
              << "," << nvlink_domain_size << ") passes successfully.\n";
  }
}

int main(int argc, char *argv[])
{
  MPI_CALL(MPI_Init(&argc, &argv));
  set_cuda_device();

  /* Initialize communicator */

  UCXCommunicator *communicator = initialize_ucx_communicator(false, 0, 0);

  /* Initialize memory pool */

  const size_t pool_size = 1'500'000'000;  // 1.5GB

  registered_memory_resource mr(communicator);
  auto *pool_mr =
    new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(&mr, pool_size, pool_size);
  rmm::mr::set_current_device_resource(pool_mr);

  /* run test */

  run_test<int32_t, int32_t>(1'000'000, 5'000'000, 0.3, true, 10, false, 1, communicator);
  run_test<int32_t, int32_t>(1'000'000, 5'000'000, 0.3, true, 10, true, 1, communicator);
  run_test<int64_t, int64_t>(1'000'000, 5'000'000, 0.3, true, 10, false, 1, communicator);
  run_test<int64_t, int64_t>(1'000'000, 5'000'000, 0.3, true, 10, true, 1, communicator);
  run_test<int32_t, int32_t>(1'000'000, 5'000'000, 1.0, true, 10, false, 1, communicator);
  run_test<int32_t, int32_t>(1'000'000, 5'000'000, 1.0, true, 10, true, 1, communicator);
  run_test<int64_t, int64_t>(1'000'000, 5'000'000, 1.0, true, 10, false, 1, communicator);
  run_test<int64_t, int64_t>(1'000'000, 5'000'000, 1.0, true, 10, true, 1, communicator);
  run_test<int32_t, int32_t>(1'000'000, 1'000'000, 0.3, true, 10, false, 1, communicator);
  run_test<int32_t, int32_t>(1'000'000, 1'000'000, 0.3, true, 10, true, 1, communicator);
  run_test<int64_t, int64_t>(1'000'000, 1'000'000, 0.3, true, 10, false, 1, communicator);
  run_test<int64_t, int64_t>(1'000'000, 1'000'000, 0.3, true, 10, true, 1, communicator);
  run_test<int32_t, int32_t>(1'000'000, 5'000'000, 0.3, true, 1, false, 1, communicator);
  run_test<int32_t, int32_t>(1'000'000, 5'000'000, 0.3, true, 1, true, 1, communicator);
  run_test<int64_t, int64_t>(1'000'000, 5'000'000, 0.3, true, 1, false, 1, communicator);
  run_test<int64_t, int64_t>(1'000'000, 5'000'000, 0.3, true, 1, true, 1, communicator);
  run_test<int64_t, cudf::timestamp_D>(1'000'000, 1'000'000, 0.3, true, 1, false, 1, communicator);
  run_test<int64_t, cudf::timestamp_D>(1'000'000, 1'000'000, 0.3, true, 1, true, 1, communicator);
  run_test<int64_t, cudf::timestamp_ms>(1'000'000, 1'000'000, 0.3, true, 1, false, 1, communicator);
  run_test<int64_t, cudf::timestamp_ms>(1'000'000, 1'000'000, 0.3, true, 1, true, 1, communicator);
  run_test<int64_t, cudf::timestamp_ns>(1'000'000, 1'000'000, 0.3, true, 1, false, 1, communicator);
  run_test<int64_t, cudf::timestamp_ns>(1'000'000, 1'000'000, 0.3, true, 1, true, 1, communicator);
  run_test<int64_t, cudf::duration_D>(1'000'000, 1'000'000, 0.3, true, 1, false, 1, communicator);
  run_test<int64_t, cudf::duration_D>(1'000'000, 1'000'000, 0.3, true, 1, true, 1, communicator);
  run_test<int64_t, cudf::duration_s>(1'000'000, 1'000'000, 0.3, true, 1, false, 1, communicator);
  run_test<int64_t, cudf::duration_s>(1'000'000, 1'000'000, 0.3, true, 1, true, 1, communicator);
  run_test<int64_t, cudf::duration_us>(1'000'000, 1'000'000, 0.3, true, 1, false, 1, communicator);
  run_test<int64_t, cudf::duration_us>(1'000'000, 1'000'000, 0.3, true, 1, true, 1, communicator);
  run_test<int32_t, int32_t>(1'000'000, 1'000'000, 0.3, true, 1, false, 2, communicator);
  run_test<int32_t, int32_t>(1'000'000, 1'000'000, 0.3, true, 1, true, 2, communicator);
  run_test<int32_t, int32_t>(1'000'000, 1'000'000, 0.3, true, 10, false, 2, communicator);
  run_test<int32_t, int32_t>(1'000'000, 1'000'000, 0.3, true, 10, true, 2, communicator);

  /* Cleanup */

  delete pool_mr;
  communicator->finalize();
  delete communicator;

  MPI_CALL(MPI_Finalize());

  return 0;
}
