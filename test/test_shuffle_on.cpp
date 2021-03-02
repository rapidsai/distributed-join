/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
 * Test the correctness of shuffle_on implementation. This test has the following steps:
 *
 * 1. Each GPU independently generate a table with a single key column, filled with random integers.
 * 2. The generated table is shuffled across GPUs, using identity hash function.
 * 3. Each GPU verifies the shuffled keys has the same remainders modulo the number of MPI ranks.
 */

#include "../src/communicator.hpp"
#include "../src/error.hpp"
#include "../src/setup.hpp"
#include "../src/shuffle_on.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <mpi.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

using cudf::column;
using cudf::table;

std::unique_ptr<table> generate_table(cudf::size_type size)
{
  std::vector<std::unique_ptr<column>> columns;

  auto key_column = cudf::make_numeric_column(cudf::data_type(cudf::type_id::INT32), size);
  auto key_buffer = key_column->mutable_view().head<int>();
  for (int ielement = 0; ielement < size; ielement++) {
    key_buffer[ielement] = rand() % (size * 10);
  }

  columns.push_back(std::move(key_column));

  return std::make_unique<table>(std::move(columns));
}

void run_test(int nrows_per_gpu, bool compression, Communicator *communicator)
{
  auto input_table = generate_table(nrows_per_gpu);

  auto compression_options =
    generate_compression_options_distributed(input_table->view(), compression);

  std::unique_ptr<cudf::table> output_table = shuffle_on(
    input_table->view(), {0}, communicator, compression_options, cudf::hash_id::HASH_IDENTITY);

  assert(output_table->view().num_columns() == 1);
  cudf::size_type num_rows_shuffled = output_table->view().column(0).size();
  auto key_buffer                   = output_table->view().column(0).head<int>();

  if (num_rows_shuffled != 0) {
    int mod_result = key_buffer[0] % communicator->mpi_size;
    for (cudf::size_type ielement = 0; ielement < num_rows_shuffled; ielement++) {
      assert(key_buffer[ielement] % communicator->mpi_size == mod_result);
    }
  }

  MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
  if (communicator->mpi_rank == 0) {
    std::cerr << std::boolalpha;
    std::cerr << "Test case (" << nrows_per_gpu << "," << compression << ") passes successfully.\n";
  }
}

int main(int argc, char *argv[])
{
  MPI_CALL(MPI_Init(&argc, &argv));
  set_cuda_device();

  rmm::mr::managed_memory_resource mr;
  rmm::mr::set_current_device_resource(&mr);

  UCXCommunicator *communicator = initialize_ucx_communicator(false, 0, 0);

  run_test(1'000'000, false, communicator);
  run_test(1'000'000, true, communicator);

  communicator->finalize();
  delete communicator;
  MPI_CALL(MPI_Finalize());

  return 0;
}
