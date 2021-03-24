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
This benchmark expects split TPC-H lineitem and orders tables in parquet format. The
lineitem tables must be named "lineitem00.parquet", "lineitem01.parquet", etc. The orders table must
be named "orders00.parquet", "orders01.parquet", etc. Each rank will read its corresponding split
files. For example, rank 0 will read "lineitem00.parquet" and "orders00.parquet"; rank 2 will read
"lineitem02.parquet" and "orders02.parquet".

To get the split parquet files, we can
1. Use `tpch-dbgen` to generate TPC-H tables with desired scale factor. See
   https://github.com/electrum/tpch-dbgen
2. Split the generated tables, e.g.
    split -C <size-of-each-split-file> --numeric-suffixes lineitem.tbl lineitem
3. Convert the split tables to parquet format, e.g.
    python scripts/tpch_to_parquet.py <path-to-folder-with-split-files>

Parameters:
--data-folder    The forder containing the split parquet files.
--orders         Comma-seperated list of column indices for orders table. Must contain 0.
--lineitem       Comma-seperated list of column indices for lineitem table. Must contain 0.
--compression    If specified, compressed data before all-to-all communication.

Example:
UCX_MEMTYPE_CACHE=n UCX_TLS=sm,cuda_copy,cuda_ipc mpirun -n 4 --cpus-per-rank 2 benchmark/tpch
--data-folder <path-to-data-folder> --orders O_ORDERKEY --lineitem L_ORDERKEY,L_SHIPDATE,L_SUPPKEY
--compression
*/

#include "../src/all_to_all_comm.hpp"
#include "../src/compression.hpp"
#include "../src/distributed_join.hpp"
#include "../src/setup.hpp"
#include "utility.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <mpi.h>

#include <cuda_runtime.h>

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
static std::vector<std::string> orders_columns;
static std::vector<std::string> lineitem_columns;
static bool compression = false;

std::vector<std::string> split(char *str)
{
  std::vector<std::string> split_result;
  char *ptr = strtok(str, ",");

  while (ptr != NULL) {
    split_result.emplace_back(ptr);
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
  std::copy(lineitem_columns.begin(),
            lineitem_columns.end(),
            std::ostream_iterator<std::string>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "Orders columns: ";
  std::copy(orders_columns.begin(),
            orders_columns.end(),
            std::ostream_iterator<std::string>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "Compression: " << compression << std::endl;
  std::cout << "================================" << std::endl;
}

int main(int argc, char *argv[])
{
  MPI_CALL(MPI_Init(&argc, &argv));
  set_cuda_device();

  parse_command_line_arguments(argc, argv);
  report_configuration();

  // Initialize communicator and memory pool

  Communicator *communicator{nullptr};
  registered_memory_resource *registered_mr{nullptr};
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> *pool_mr{nullptr};

  setup_memory_pool_and_communicator(
    communicator, registered_mr, pool_mr, "UCX", "preregistered", 0);

  void *preallocated_pinned_buffer;
  CUDA_RT_CALL(
    cudaMallocHost(&preallocated_pinned_buffer, communicator->mpi_size * sizeof(size_t)));

  // Read input tables

  std::stringstream index_string_stream;
  index_string_stream << std::setw(2) << std::setfill('0') << communicator->mpi_rank;
  std::string index_string = index_string_stream.str();

  std::string orders_filepath   = data_folderpath + "/orders" + index_string + ".parquet";
  std::string lineitem_filepath = data_folderpath + "/lineitem" + index_string + ".parquet";

  cudf::io::parquet_reader_options orders_options =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(orders_filepath));
  orders_options.set_columns(orders_columns);
  auto orders_table = cudf::io::read_parquet(orders_options);

  cudf::io::parquet_reader_options lineitem_options =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(lineitem_filepath));
  lineitem_options.set_columns(lineitem_columns);
  auto lineitem_table = cudf::io::read_parquet(lineitem_options);

  /*
  // Print the data types used for the lineitem table for debugging
  for (cudf::size_type icol = 0; icol < lineitem_table.tbl->view().num_columns(); icol++) {
    std::cout << (int32_t)lineitem_table.tbl->view().column(icol).type().id() << " ";
  }
  std::cout << std::endl;
  */

  // Calculate input sizes

  int64_t input_size_irank = 0;
  int64_t input_size_total;

  input_size_irank += calculate_table_size(orders_table.tbl->view());
  input_size_irank += calculate_table_size(lineitem_table.tbl->view());

  MPI_CALL(
    MPI_Allreduce(&input_size_irank, &input_size_total, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD));

  // Warmup all-to-all

  warmup_all_to_all(communicator);

  // Warmup nvcomp

  if (compression) { warmup_nvcomp(); }

  // Generate compression options

  std::vector<ColumnCompressionOptions> orders_compression_options =
    generate_compression_options_distributed(orders_table.tbl->view(), compression);
  std::vector<ColumnCompressionOptions> lineitem_compression_options =
    generate_compression_options_distributed(lineitem_table.tbl->view(), compression);

  if (communicator->mpi_rank == 0) {
    std::cout << "Orders table compression options: " << std::endl;
    print_compression_options(orders_compression_options);
    std::cout << "Lineitem table compression options: " << std::endl;
    print_compression_options(lineitem_compression_options);
  }

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
                                            orders_compression_options,
                                            lineitem_compression_options,
                                            1,
                                            true,
                                            preallocated_pinned_buffer);

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

  CUDA_RT_CALL(cudaFreeHost(preallocated_pinned_buffer));

  destroy_memory_pool_and_communicator(
    communicator, registered_mr, pool_mr, "UCX", "preregistered");

  MPI_CALL(MPI_Finalize());

  return 0;
}
