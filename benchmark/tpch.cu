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
This benchmark needs split TPC-H lineitem and orders table in csv format with delimiter '|'. The
lineitem table must be named "lineitem00", "lineitem01", etc. The orders table must be named
"orders00", "orders01", etc. Each rank will read its corresponding split files. For example, rank 0
will read "lineitem00" and "orders00"; rank 2 will read "lineitem02" and "orders02".

Parameters:
--data-folder    The forder containing the split files.
--orders         Comma-seperated list of column indices for orders table. Must contain 0.
--lineitem       Comma-seperated list of column indices for lineitem table. Must contain 0.
--compression    If specified, compressed data before all-to-all communication.

Example:
UCX_MEMTYPE_CACHE=n UCX_TLS=sm,cuda_copy,cuda_ipc mpirun -n 4 --cpus-per-rank 2 benchmark/tpch
--data-folder <path-to-data-folder> --orders 0,1,2 --lineitem 0,1,2,3 --compression
*/

#include "../src/distributed_join.cuh"
#include "../src/topology.cuh"

#include <cudf/column/column_view.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

static std::string data_folderpath;
static std::vector<int> orders_columns;
static std::vector<int> lineitem_columns;
static bool compression = false;

std::vector<int> split(char *str)
{
  std::vector<int> split_result;
  char *ptr = strtok(str, ",");

  while (ptr != NULL) {
    split_result.push_back(atoi(ptr));
    ptr = strtok(NULL, ",");
  }

  return split_result;
}

void parse_command_line_arguments(int argc, char *argv[])
{
  for (int iarg = 0; iarg < argc; iarg++) {
    if (!strcmp(argv[iarg], "--data-folder")) { data_folderpath = argv[iarg + 1]; }

    if (!strcmp(argv[iarg], "--orders")) { orders_columns = split(argv[iarg + 1]); }

    if (!strcmp(argv[iarg], "--lineitem")) { lineitem_columns = split(argv[iarg + 1]); }

    if (!strcmp(argv[iarg], "--compression")) { compression = true; }
  }
}

void report_configuration()
{
  MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));

  int mpi_rank;
  int mpi_size;
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
  if (mpi_rank != 0) return;

  std::cout << "========== Parameters ==========" << std::endl;
  std::cout << std::boolalpha;
  std::cout << "Data folder: " << data_folderpath << std::endl;
  std::cout << "Lineitem columns: ";
  std::copy(
    lineitem_columns.begin(), lineitem_columns.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "Orders columns: ";
  std::copy(
    orders_columns.begin(), orders_columns.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "Compression: " << compression << std::endl;
  std::cout << "================================" << std::endl;
}

int64_t calculate_table_size(cudf::table_view input_table)
{
  int64_t table_size = 0;

  for (cudf::size_type icol = 0; icol < input_table.num_columns(); icol++) {
    cudf::column_view current_column = input_table.column(icol);
    cudf::data_type dtype            = current_column.type();
    if (cudf::is_fixed_width(dtype)) {
      table_size += (cudf::size_of(dtype) * current_column.size());
    } else {
      assert(dtype.id() == cudf::type_id::STRING);
      table_size += current_column.child(1).size();
    }
  }

  return table_size;
}

int main(int argc, char *argv[])
{
  setup_topology(argc, argv);
  parse_command_line_arguments(argc, argv);
  report_configuration();

  /* Initialize communicator and memory pool */

  Communicator *communicator{nullptr};
  registered_memory_resource *registered_mr{nullptr};
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> *pool_mr{nullptr};

  setup_memory_pool_and_communicator(
    communicator, registered_mr, pool_mr, "UCX", "preregistered", 0);

  // Read order table

  std::stringstream index_string_stream;
  index_string_stream << std::setw(2) << std::setfill('0') << communicator->mpi_rank;
  std::string index_string = index_string_stream.str();

  std::string orders_filepath   = data_folderpath + "/orders" + index_string;
  std::string lineitem_filepath = data_folderpath + "/lineitem" + index_string;

  cudf::io::csv_reader_options orders_options =
    cudf::io::csv_reader_options::builder(cudf::io::source_info(orders_filepath));
  orders_options.set_delimiter('|');
  orders_options.set_use_cols_indexes(orders_columns);
  orders_options.set_infer_date_indexes({4});
  auto orders_table = cudf::io::read_csv(orders_options);

  // Read lineitem table

  cudf::io::csv_reader_options lineitem_options =
    cudf::io::csv_reader_options::builder(cudf::io::source_info(lineitem_filepath));
  lineitem_options.set_delimiter('|');
  lineitem_options.set_use_cols_indexes(lineitem_columns);
  lineitem_options.set_infer_date_indexes({10, 11, 12});
  auto lineitem_table = cudf::io::read_csv(lineitem_options);

  // Calculate input sizes

  int64_t input_size_irank = 0;
  int64_t input_size_total;

  input_size_irank += calculate_table_size(orders_table.tbl->view());
  input_size_irank += calculate_table_size(lineitem_table.tbl->view());

  MPI_CALL(
    MPI_Allreduce(&input_size_irank, &input_size_total, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD));

  // Perform distributed join

  CUDA_RT_CALL(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  auto join_result = distributed_inner_join(orders_table.tbl->view(),
                                            lineitem_table.tbl->view(),
                                            {0},
                                            {0},
                                            {std::pair<cudf::size_type, cudf::size_type>(0, 0)},
                                            communicator,
                                            1,
                                            compression);

  MPI_Barrier(MPI_COMM_WORLD);
  double stop = MPI_Wtime();

  if (communicator->mpi_rank == 0) {
    double elapsed_time = stop - start;
    std::cout << "Average size per rank (GB): " << input_size_total / communicator->mpi_size / 1e9
              << std::endl;
    std::cout << "Elasped time (s): " << elapsed_time << std::endl;
    std::cout << "Throughput (GB/s): " << input_size_total / 1e9 / elapsed_time << std::endl;
  }

  // Cleanup

  join_result.reset();
  lineitem_table.tbl.reset();
  orders_table.tbl.reset();

  destroy_memory_pool_and_communicator(
    communicator, registered_mr, pool_mr, "UCX", "preregistered");

  MPI_CALL(MPI_Finalize());

  return 0;
}
