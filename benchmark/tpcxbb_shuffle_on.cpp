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
This benchmark tests shuffle performance on TPCx-BB "webstream" table.

Parameters:
--data-folder      Forder containing the parquet files of TPCx-BB "webstream" table.
--nfiles-per-rank  Number of Parquet files read for input on each rank.
--compression      If specified, compressed data before all-to-all communication.
*/

#include "../src/all_to_all_comm.hpp"
#include "../src/communicator.hpp"
#include "../src/error.hpp"
#include "../src/registered_memory_resource.hpp"
#include "../src/setup.hpp"
#include "../src/shuffle_on.hpp"
#include "utility.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>

#include <mpi.h>

#include <cuda_runtime.h>

#include <dirent.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>

using cudf::table;
using cudf::table_view;

static std::string data_folderpath = "";
static int nfiles_per_rank         = 1;
static bool compression            = false;

void parse_command_line_arguments(int argc, char *argv[])
{
  for (int iarg = 0; iarg < argc; iarg++) {
    if (!strcmp(argv[iarg], "--data-folder")) { data_folderpath = argv[iarg + 1]; }

    if (!strcmp(argv[iarg], "--nfiles-per-rank")) { nfiles_per_rank = atoi(argv[iarg + 1]); }

    if (!strcmp(argv[iarg], "--compression")) { compression = true; }
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
  std::cout << "Data folder: " << data_folderpath << std::endl;
  std::cout << "Files per rank: " << nfiles_per_rank << std::endl;
  std::cout << "Compression: " << compression << std::endl;
  std::cout << "================================" << std::endl;
}

int main(int argc, char *argv[])
{
  MPI_CALL(MPI_Init(&argc, &argv));
  set_cuda_device();

  // Parse command line arguments

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

  // Get a vector of parquet file names in data_folder

  int num_input_files = 0;
  std::vector<std::string> file_names;

  if (communicator->mpi_rank == 0) {
    DIR *data_folder = opendir(data_folderpath.c_str());
    if (!data_folder) {
      std::cerr << "Cannot open directory\n";
      return EXIT_FAILURE;
    }

    struct dirent *next_entry;
    while ((next_entry = readdir(data_folder)) != NULL) {
      std::string file_name = next_entry->d_name;
      if (file_name.find(".parquet") != std::string::npos) {
        // if the current file name contains ".parquet"
        file_names.push_back(file_name);
        num_input_files++;
      }
    }
    closedir(data_folder);
  }

  MPI_CALL(MPI_Bcast(&num_input_files, 1, MPI_INT, 0, MPI_COMM_WORLD));

  constexpr int max_file_name_length = 100;
  char file_name_bcast[max_file_name_length];
  for (int ifile = 0; ifile < num_input_files; ifile++) {
    if (communicator->mpi_rank == 0) {
      strncpy(file_name_bcast, file_names[ifile].c_str(), max_file_name_length);
    }

    MPI_CALL(MPI_Bcast(file_name_bcast, max_file_name_length, MPI_CHAR, 0, MPI_COMM_WORLD));
    if (communicator->mpi_rank != 0) { file_names.emplace_back(file_name_bcast); }
  }

  std::sort(file_names.begin(), file_names.end());

  // Read parquet files

  std::vector<std::unique_ptr<table>> input_tables;

  for (int ifile = 0; ifile < nfiles_per_rank; ifile++) {
    int file_index = ifile * communicator->mpi_size + communicator->mpi_rank;
    if (file_index >= num_input_files) break;
    std::string filepath = data_folderpath + "/" + file_names[file_index];
    cudf::io::parquet_reader_options cuio_options =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath));
    cuio_options.set_columns(
      {"wcs_user_sk", "wcs_item_sk", "wcs_click_date_sk", "wcs_click_time_sk"});
    input_tables.push_back(cudf::io::read_parquet(cuio_options).tbl);
  }

  // Combine input tables into a single table

  std::vector<table_view> input_table_views;
  input_table_views.reserve(input_tables.size());
  for (auto const &input_table : input_tables) { input_table_views.push_back(input_table->view()); }
  std::unique_ptr<table> combined_input = cudf::concatenate(input_table_views);
  input_tables.clear();

  // Remove rows with NULL value in "wcs_user_sk" and "wcs_item_sk"

  std::unique_ptr<table> combined_input_filtered =
    cudf::drop_nulls(combined_input->view(), {0, 1}, 2);
  combined_input.reset();
  std::cout << "Rank " << communicator->mpi_rank << " input table has "
            << combined_input_filtered->view().num_rows() << " rows." << std::endl;

  // Calculate input sizes

  int64_t input_size_irank = calculate_table_size(combined_input_filtered->view());
  int64_t input_size_total;

  MPI_CALL(
    MPI_Allreduce(&input_size_irank, &input_size_total, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD));

  // Warmup

  warmup_all_to_all(communicator);
  if (compression) { warmup_nvcomp(); }

  std::vector<ColumnCompressionOptions> compression_options =
    generate_compression_options_distributed(combined_input_filtered->view(), compression);

  if (communicator->mpi_rank == 0) {
    for (size_t icol = 0; icol < compression_options.size(); icol++) {
      nvcompCascadedFormatOpts format = compression_options[icol].cascaded_format;
      std::cout << "Column " << icol << " RLE=" << format.num_RLEs
                << ", Delta=" << format.num_deltas << ", Bitpack=" << format.use_bp << std::endl;
    }
  }

  // Benchmark shuffle_on

  CUDA_RT_CALL(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  std::unique_ptr<table> shuffle_result = shuffle_on(combined_input_filtered->view(),
                                                     {0},
                                                     communicator,
                                                     compression_options,
                                                     cudf::hash_id::HASH_MURMUR3,
                                                     true,
                                                     preallocated_pinned_buffer);

  MPI_Barrier(MPI_COMM_WORLD);
  double stop = MPI_Wtime();

  if (communicator->mpi_rank == 0) {
    double elapsed_time = stop - start;
    std::cout << "Elasped time (s): " << elapsed_time << std::endl;
    std::cout << "Throughput (GB/s): " << input_size_total / 1e9 / elapsed_time << std::endl;
  }

  // Cleanup

  combined_input_filtered.reset();
  shuffle_result.reset();

  CUDA_RT_CALL(cudaFreeHost(preallocated_pinned_buffer));

  destroy_memory_pool_and_communicator(
    communicator, registered_mr, pool_mr, "UCX", "preregistered");

  MPI_CALL(MPI_Finalize());

  return 0;
}
