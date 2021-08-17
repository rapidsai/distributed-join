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

/*
This benchmark runs distributed join on random keys. Both the left and the right tables contain two
columns. The key column consists of random integers and the payload column consists of row ids.

Parameters:

**--key-type {int32_t,int64_t}**

Data type for the key columns. Default: `int64_t`.

**--payload-type {int32_t,int64_t}**

Data type for the payload columns. Default: `int64_t`.

**--build-table-nrows [INTEGER]**

Number of rows in the build table per GPU. Default: `100'000'000`.

**--probe-table-nrows [INTEGER]**

Number of rows in the probe table per GPU. Default: `100'000'000`.

**--selectivity [FLOAT]**

The probability (in range 0.0 - 1.0) of each probe table row has matches in the build table.
Default: `0.3`.

**--duplicate-build-keys**

If specified, key columns of the build table are allowed to have duplicates.

**--over-decomposition-factor [INTEGER]**

Partition the input tables into (over decomposition factor) * (number of GPUs) buckets, which is
used for computation-communication overlap. This argument has to be an integer >= 1. Higher number
means smaller batch size. `1` means no overlap. Default: `1`.

**--communicator [STR]**

This option can be either "UCX" or "NCCL", which controls what communicator to use. Default: `UCX`.

**--registration-method [STR]**

If the UCX communicator is selected, this option can be either "none", "preregistered" or "buffer",
to control how registration is performed for GPUDirect RDMA.
- "none": No preregistration.
- "preregistered": The whole RMM memory pool will be preregistered.
- "buffer": Preregister a set of communication buffers. The communication in distributed join will
go through these buffers.
*/

#include "../src/communicator.hpp"
#include "../src/compression.hpp"
#include "../src/distributed_join.hpp"
#include "../src/error.hpp"
#include "../src/generate_table.cuh"
#include "../src/registered_memory_resource.hpp"
#include "../src/setup.hpp"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda_profiler_api.h>

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

static std::string key_type     = "int64_t";
static std::string payload_type = "int64_t";

static cudf::size_type BUILD_TABLE_NROWS_EACH_RANK = 100'000'000;
static cudf::size_type PROBE_TABLE_NROWS_EACH_RANK = 100'000'000;
static double SELECTIVITY                          = 0.3;
static bool IS_BUILD_TABLE_KEY_UNIQUE              = true;
static int OVER_DECOMPOSITION_FACTOR               = 1;
static std::string COMMUNICATOR_NAME               = "UCX";
static std::string REGISTRATION_METHOD             = "preregistered";
static int64_t COMMUNICATOR_BUFFER_SIZE            = 1'600'000'000LL;
static bool COMPRESSION                            = false;
static int NVLINK_DOMAIN_SIZE                      = 1;
static bool REPORT_TIMING                          = false;

void parse_command_line_arguments(int argc, char *argv[])
{
  for (int iarg = 0; iarg < argc; iarg++) {
    if (!strcmp(argv[iarg], "--key-type")) { key_type = argv[iarg + 1]; }

    if (!strcmp(argv[iarg], "--payload-type")) { payload_type = argv[iarg + 1]; }

    if (!strcmp(argv[iarg], "--build-table-nrows")) {
      BUILD_TABLE_NROWS_EACH_RANK = atoi(argv[iarg + 1]);
    }

    if (!strcmp(argv[iarg], "--probe-table-nrows")) {
      PROBE_TABLE_NROWS_EACH_RANK = atoi(argv[iarg + 1]);
    }

    if (!strcmp(argv[iarg], "--selectivity")) { SELECTIVITY = atof(argv[iarg + 1]); }

    if (!strcmp(argv[iarg], "--duplicate-build-keys")) { IS_BUILD_TABLE_KEY_UNIQUE = false; }

    if (!strcmp(argv[iarg], "--over-decomposition-factor")) {
      OVER_DECOMPOSITION_FACTOR = atoi(argv[iarg + 1]);
    }

    if (!strcmp(argv[iarg], "--communicator")) { COMMUNICATOR_NAME = argv[iarg + 1]; }

    if (!strcmp(argv[iarg], "--compression")) { COMPRESSION = true; }

    if (!strcmp(argv[iarg], "--registration-method")) { REGISTRATION_METHOD = argv[iarg + 1]; }

    if (!strcmp(argv[iarg], "--nvlink-domain-size")) { NVLINK_DOMAIN_SIZE = atoi(argv[iarg + 1]); }

    if (!strcmp(argv[iarg], "--report-timing")) { REPORT_TIMING = true; }
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
  std::cout << "Key type: " << key_type << std::endl;
  std::cout << "Payload type: " << payload_type << std::endl;
  std::cout << "Number of rows in the build table: "
            << static_cast<uint64_t>(BUILD_TABLE_NROWS_EACH_RANK) * mpi_size / 1e6 << " million"
            << std::endl;
  std::cout << "Number of rows in the probe table: "
            << static_cast<uint64_t>(PROBE_TABLE_NROWS_EACH_RANK) * mpi_size / 1e6 << " million"
            << std::endl;
  std::cout << "Selectivity: " << SELECTIVITY << std::endl;
  std::cout << "Keys in build table are unique: " << IS_BUILD_TABLE_KEY_UNIQUE << std::endl;
  std::cout << "Over-decomposition factor: " << OVER_DECOMPOSITION_FACTOR << std::endl;
  std::cout << "Communicator: " << COMMUNICATOR_NAME << std::endl;
  if (COMMUNICATOR_NAME == "UCX")
    std::cout << "Registration method: " << REGISTRATION_METHOD << std::endl;
  std::cout << "Compression: " << COMPRESSION << std::endl;
  std::cout << "NVLink domain size: " << NVLINK_DOMAIN_SIZE << std::endl;
  std::cout << "================================" << std::endl;
}

int main(int argc, char *argv[])
{
  MPI_CALL(MPI_Init(&argc, &argv));
  set_cuda_device();

  /* Parse command line arguments */

  parse_command_line_arguments(argc, argv);
  report_configuration();

  cudf::size_type RAND_MAX_VAL =
    std::max(BUILD_TABLE_NROWS_EACH_RANK, PROBE_TABLE_NROWS_EACH_RANK) * 2;

  /* Initialize communicator and memory pool */

  int mpi_rank;
  int mpi_size;
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

  Communicator *communicator{nullptr};
  // `registered_mr` holds reference to the registered memory resource, and *nullptr* if registered
  // memory resource is not used.
  registered_memory_resource *registered_mr{nullptr};
  // pool_mr need to live on heap because for registered memory resources, the memory pool needs
  // to deallocated before UCX cleanup, which can be achieved by calling the destructor of
  // `poll_mr`.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> *pool_mr{nullptr};

  setup_memory_pool_and_communicator(communicator,
                                     registered_mr,
                                     pool_mr,
                                     COMMUNICATOR_NAME,
                                     REGISTRATION_METHOD,
                                     COMMUNICATOR_BUFFER_SIZE);

  void *preallocated_pinned_buffer;
  CUDA_RT_CALL(cudaMallocHost(&preallocated_pinned_buffer, mpi_size * sizeof(size_t)));

  /* Warmup nvcomp */

  if (COMPRESSION) { warmup_nvcomp(); }

  /* Generate build table and probe table on each rank */

  std::unique_ptr<cudf::table> left;
  std::unique_ptr<cudf::table> right;

#define generate_tables(KEY_T, PAYLOAD_T)                                        \
  {                                                                              \
    std::tie(left, right) =                                                      \
      generate_tables_distributed<KEY_T, PAYLOAD_T>(BUILD_TABLE_NROWS_EACH_RANK, \
                                                    PROBE_TABLE_NROWS_EACH_RANK, \
                                                    SELECTIVITY,                 \
                                                    RAND_MAX_VAL,                \
                                                    IS_BUILD_TABLE_KEY_UNIQUE,   \
                                                    communicator);               \
  }

#define generate_tables_key_type(KEY_T)                 \
  {                                                     \
    if (payload_type == "int64_t") {                    \
      generate_tables(KEY_T, int64_t)                   \
    } else if (payload_type == "int32_t") {             \
      generate_tables(KEY_T, int32_t)                   \
    } else {                                            \
      throw std::runtime_error("Unknown payload type"); \
    }                                                   \
  }

  if (key_type == "int64_t") {
    generate_tables_key_type(int64_t)
  } else if (key_type == "int32_t") {
    generate_tables_key_type(int32_t)
  } else {
    throw std::runtime_error("Unknown key type");
  }

  /* Generate compression options */

  std::vector<ColumnCompressionOptions> left_compression_options =
    generate_compression_options_distributed(left->view(), COMPRESSION);
  std::vector<ColumnCompressionOptions> right_compression_options =
    generate_compression_options_distributed(right->view(), COMPRESSION);

  /* Distributed join */

  CUDA_RT_CALL(cudaDeviceSynchronize());

  MPI_Barrier(MPI_COMM_WORLD);
  cudaProfilerStart();
  double start = MPI_Wtime();

  std::unique_ptr<cudf::table> join_result = distributed_inner_join(left->view(),
                                                                    right->view(),
                                                                    {0},
                                                                    {0},
                                                                    communicator,
                                                                    left_compression_options,
                                                                    right_compression_options,
                                                                    OVER_DECOMPOSITION_FACTOR,
                                                                    REPORT_TIMING,
                                                                    preallocated_pinned_buffer,
                                                                    NVLINK_DOMAIN_SIZE);

  MPI_Barrier(MPI_COMM_WORLD);
  double stop = MPI_Wtime();
  cudaProfilerStop();

  if (mpi_rank == 0) { std::cout << "Elasped time (s) " << stop - start << std::endl; }

  /* Cleanup */
  left.reset();
  right.reset();
  join_result.reset();
  CUDA_RT_CALL(cudaFreeHost(preallocated_pinned_buffer));
  CUDA_RT_CALL(cudaDeviceSynchronize());

  destroy_memory_pool_and_communicator(
    communicator, registered_mr, pool_mr, COMMUNICATOR_NAME, REGISTRATION_METHOD);

  MPI_CALL(MPI_Finalize());

  return 0;
}
