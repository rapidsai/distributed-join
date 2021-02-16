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

#include "../src/communicator.hpp"
#include "../src/distributed_join.hpp"
#include "../src/error.hpp"
#include "../src/setup.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/sequence.h>

#include <mpi.h>

#include <cassert>
#include <cstring>
#include <memory>

/**
 * Generate an input table used for this test case.
 *
 * The generated table has `nelements_per_gpu` rows and two columns. The first column is the key
 * column, with sequence `start_value`, `start_value+multiple`, `start_value+multiple*2`, etc. For
 * example, if `start_value` is `0`, and `multiple` is `3`, the key column will contain the sequence
 * 0,3,6,9,12,etc. The payload column is a string column. The string contained in a row with key `k`
 * has length `k % 7 + 1`, and all filled the `k % 26`th letter in lowercase. For example, if `k=2`,
 * the string is `ccc`, and if `k=5`, the string is `ffffff`.
 */
std::unique_ptr<cudf::table> generate_table(cudf::size_type nelements_per_gpu,
                                            int start_value,
                                            int multiple)
{
  // First pass: calculate string subcolumn size
  cudf::size_type string_column_size = 0;
  for (cudf::size_type ielement = 0; ielement < nelements_per_gpu; ielement++) {
    int current_value = start_value + ielement * multiple;
    string_column_size += (current_value % 7 + 1);
  }

  // Allocate buffers for the string column
  std::vector<char> strings(string_column_size);
  std::vector<cudf::size_type> offsets(nelements_per_gpu + 1);

  // Second pass, fill the string subcolumn
  cudf::size_type current_offset = 0;
  for (cudf::size_type ielement = 0; ielement < nelements_per_gpu; ielement++) {
    int current_value = start_value + ielement * multiple;
    int current_size  = current_value % 7 + 1;
    char current_char = 'a' + current_value % 26;
    offsets[ielement] = current_offset;
    memset(strings.data() + current_offset, current_char, current_size);
    current_offset += current_size;
  }

  offsets[nelements_per_gpu] = current_offset;

  // Construct the payload column
  std::unique_ptr<cudf::column> payload_column = cudf::make_strings_column(strings, offsets);

  // Construct the key column
  std::unique_ptr<cudf::column> key_column =
    cudf::make_numeric_column(cudf::data_type(cudf::type_id::INT32), nelements_per_gpu);
  int *key_buffer = key_column->mutable_view().head<int>();
  thrust::sequence(
    rmm::exec_policy(0)->on(0), key_buffer, key_buffer + nelements_per_gpu, start_value, multiple);

  std::vector<std::unique_ptr<cudf::column>> new_table;
  new_table.push_back(std::move(key_column));
  new_table.push_back(std::move(payload_column));

  return std::make_unique<cudf::table>(std::move(new_table));
}

void run_test(cudf::size_type nelements_per_gpu, bool compression, Communicator *communicator)
{
  int mpi_rank = communicator->mpi_rank;
  int mpi_size = communicator->mpi_size;

  std::unique_ptr<cudf::table> left_table =
    generate_table(nelements_per_gpu, nelements_per_gpu * mpi_rank * 3, 3);

  std::unique_ptr<cudf::table> right_table =
    generate_table(nelements_per_gpu, nelements_per_gpu * mpi_rank * 5, 5);

  /* Generate compression options */

  std::vector<ColumnCompressionOptions> left_compression_options =
    generate_compression_options_distributed(left_table->view(), compression);
  std::vector<ColumnCompressionOptions> right_compression_options =
    generate_compression_options_distributed(right_table->view(), compression);

  auto join_result = distributed_inner_join(left_table->view(),
                                            right_table->view(),
                                            {0},
                                            {0},
                                            {std::pair<cudf::size_type, cudf::size_type>(0, 0)},
                                            communicator,
                                            left_compression_options,
                                            right_compression_options,
                                            1);

  int num_rows = join_result->num_rows();
  int total_nrows;
  MPI_CALL(MPI_Allreduce(&num_rows, &total_nrows, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

  assert(total_nrows == nelements_per_gpu * mpi_size / 5);

  if (num_rows == 0) return;

  cudf::column_view key_column     = join_result->view().column(0);
  cudf::column_view payload_column = join_result->view().column(1);

  for (cudf::size_type irow = 0; irow < num_rows; irow++) {
    int key = *(key_column.begin<int>() + irow);
    assert(key % 15 == 0);
    cudf::size_type start_idx = *(payload_column.child(0).begin<cudf::size_type>() + irow);
    cudf::size_type end_idx   = *(payload_column.child(0).begin<cudf::size_type>() + irow + 1);
    assert(end_idx - start_idx == key % 7 + 1);

    for (; start_idx < end_idx; start_idx++)
      assert(*(payload_column.child(1).begin<char>() + start_idx) == 'a' + key % 26);
  }
}

int main(int argc, char *argv[])
{
  MPI_CALL(MPI_Init(&argc, &argv));
  set_cuda_device();

  /* Setup memory pool */

  rmm::mr::managed_memory_resource mr;
  rmm::mr::set_current_device_resource(&mr);

  /* Initialize communicator */

  UCXCommunicator *communicator = initialize_ucx_communicator(false, 0, 0);

  /* Run tests */

  // Note: temporarily disable some test cases because nvcomp's cascaded selector can raise
  // "Floating point exception" if the input buffer is smaller than sample_size * num_samples.

  // run_test(12'000, true, communicator);
  run_test(12'000, false, communicator);
  run_test(120'000, true, communicator);
  run_test(120'000, false, communicator);

  /* Cleanup */

  communicator->finalize();
  delete communicator;

  MPI_CALL(MPI_Finalize());

  return 0;
}
