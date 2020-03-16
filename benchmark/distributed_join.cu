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

#include <mpi.h>
#include <cuda_profiler_api.h>

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include "../src/communicator.h"
#include "../src/error.cuh"
#include "../src/generate_table.cuh"
#include "../src/distributed_join.cuh"

#define KEY_T int64_t
#define PAYLOAD_T int64_t

static constexpr cudf::size_type BUILD_TABLE_NROWS_EACH_RANK = 100'000'000;
static constexpr cudf::size_type PROBE_TABLE_NROWS_EACH_RANK = 100'000'000;
static constexpr double SELECTIVITY = 0.3;
static constexpr KEY_T RAND_MAX_VAL = 200'000'000;
static constexpr bool IS_BUILD_TABLE_KEY_UNIQUE = true;
static constexpr int OVER_DECOMPOSITION_FACTOR = 1;

using cudf::experimental::table;


int main(int argc, char *argv[])
{
    /* Initialize communication */

    UCXBufferCommunicator communicator;
    communicator.initialize(argc, argv);

    int mpi_rank {communicator.mpi_rank};
    int mpi_size {communicator.mpi_size};

    communicator.setup_cache(2 * mpi_size, 800'000'000LL / mpi_size - 100'000LL);
    communicator.warmup_cache();

    /* Generate build table and probe table on each node */

    std::unique_ptr<table> left;
    std::unique_ptr<table> right;

    std::tie(left, right) = generate_tables_distributed<KEY_T, PAYLOAD_T>(
        BUILD_TABLE_NROWS_EACH_RANK, PROBE_TABLE_NROWS_EACH_RANK,
        SELECTIVITY, RAND_MAX_VAL, IS_BUILD_TABLE_KEY_UNIQUE,
        &communicator
    );

    /* Distributed join */

    CUDA_RT_CALL(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    cudaProfilerStart();
    double start = MPI_Wtime();

    std::unique_ptr<table> join_result = distributed_inner_join(
        left->view(), right->view(),
        {0}, {0}, {std::pair<cudf::size_type, cudf::size_type>(0, 0)},
        &communicator, OVER_DECOMPOSITION_FACTOR
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double stop = MPI_Wtime();
    cudaProfilerStop();

    if (mpi_rank == 0) {
        std::cout << "Elasped time (s) " << stop - start << std::endl;
    }

    /* Cleanup */

    communicator.finalize();

    return 0;
}
