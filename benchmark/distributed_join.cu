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

#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <utility>
#include <tuple>
#include <cstdint>
#include <cstring>
#include <cstdlib>

#include <mpi.h>
#include <cuda_profiler_api.h>

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/cnmem_memory_resource.hpp>

#include "../src/topology.cuh"
#include "../src/communicator.h"
#include "../src/error.cuh"
#include "../src/generate_table.cuh"
#include "../src/distributed_join.cuh"

#define KEY_T int64_t
#define PAYLOAD_T int64_t

static cudf::size_type BUILD_TABLE_NROWS_EACH_RANK = 100'000'000;
static cudf::size_type PROBE_TABLE_NROWS_EACH_RANK = 100'000'000;
static double SELECTIVITY = 0.3;
static bool IS_BUILD_TABLE_KEY_UNIQUE = true;
static int OVER_DECOMPOSITION_FACTOR = 1;
static bool USE_BUFFER_COMMUNICATOR = false;


void parse_command_line_arguments(int argc, char *argv[])
{
    for (int iarg = 0; iarg < argc; iarg++) {
        if (!strcmp(argv[iarg], "--build-table-nrows")) {
            BUILD_TABLE_NROWS_EACH_RANK = atoi(argv[iarg + 1]);
        }

        if (!strcmp(argv[iarg], "--probe-table-nrows")) {
            PROBE_TABLE_NROWS_EACH_RANK = atoi(argv[iarg + 1]);
        }

        if (!strcmp(argv[iarg], "--selectivity")) {
            SELECTIVITY = atof(argv[iarg + 1]);
        }

        if (!strcmp(argv[iarg], "--duplicate-build-keys")) {
            IS_BUILD_TABLE_KEY_UNIQUE = false;
        }

        if (!strcmp(argv[iarg], "--over-decomposition-factor")) {
            OVER_DECOMPOSITION_FACTOR = atoi(argv[iarg + 1]);
        }

        if (!strcmp(argv[iarg], "--use-buffer-communicator")) {
            USE_BUFFER_COMMUNICATOR = true;
        }
    }
}


int main(int argc, char *argv[])
{
    /* Initialize topology */

    setup_topology(argc, argv);

    /* Parse command line arguments */

    parse_command_line_arguments(argc, argv);

    KEY_T RAND_MAX_VAL = std::max(BUILD_TABLE_NROWS_EACH_RANK, PROBE_TABLE_NROWS_EACH_RANK) * 2;

    /* Initialize memory pool */

    size_t free_memory, total_memory;
    CUDA_RT_CALL(cudaMemGetInfo(&free_memory, &total_memory));
    const size_t pool_size = free_memory - 5LL * (1LL << 29);  // free memory - 500MB

    rmm::mr::cnmem_memory_resource cnmem_mr {pool_size};
    rmm::mr::set_default_resource(&cnmem_mr);

    /* Initialize communicator */

    int mpi_rank;
    int mpi_size;
    MPI_CALL( MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) );
    MPI_CALL( MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) );

    UCXCommunicator* communicator = initialize_ucx_communicator(
        USE_BUFFER_COMMUNICATOR, 2 * mpi_size, 800'000'000LL / mpi_size - 100'000LL
    );

    /* Generate build table and probe table on each node */

    std::unique_ptr<cudf::table> left;
    std::unique_ptr<cudf::table> right;

    std::tie(left, right) = generate_tables_distributed<KEY_T, PAYLOAD_T>(
        BUILD_TABLE_NROWS_EACH_RANK, PROBE_TABLE_NROWS_EACH_RANK,
        SELECTIVITY, RAND_MAX_VAL, IS_BUILD_TABLE_KEY_UNIQUE,
        communicator
    );

    /* Distributed join */

    CUDA_RT_CALL(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    cudaProfilerStart();
    double start = MPI_Wtime();

    std::unique_ptr<cudf::table> join_result = distributed_inner_join(
        left->view(), right->view(),
        {0}, {0}, {std::pair<cudf::size_type, cudf::size_type>(0, 0)},
        communicator, OVER_DECOMPOSITION_FACTOR
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double stop = MPI_Wtime();
    cudaProfilerStop();

    if (mpi_rank == 0) {
        std::cout << "Elasped time (s) " << stop - start << std::endl;
    }

    /* Cleanup */

    communicator->finalize();
    delete communicator;

    return 0;
}
