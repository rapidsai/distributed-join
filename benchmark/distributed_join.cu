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

#include <cstdlib>
#include <cstring>
#include <vector>
#include <rmm/rmm.h>
#include <mpi.h>
#include <ucp/api/ucp.h>
#include <cuda_profiler_api.h>
#include <algorithm>

#include "../src/cudf_helper.cuh"
#include "../src/distributed.cuh"
#include "../src/error.cuh"
#include "../src/comm.cuh"

#define BUILD_TABLE_SIZE_EACH_RANK 100'000'000
#define PROBE_TABLE_SIZE_EACH_RANK 100'000'000
#define SELECTIVITY 0.3
#define RAND_MAX_VAL 200'000'000
#define IS_BUILD_TABLE_KEY_UNIQUE true
#define OVER_DECOMPOSITION_FACTOR 1

#define KEY_T int64_t
#define PAYLOAD_T int64_t


int main(int argc, char *argv[])
{
    /* Initialize communication */

    UCXBufferCommunicator communicator;
    communicator.initialize(argc, argv);

    int mpi_rank {communicator.mpi_rank};
    int mpi_size {communicator.mpi_size};

    communicator.setup_cache(2 * mpi_size, std::max(250'000LL, 800'000'000LL / mpi_size / 50));
    communicator.warmup_cache();

    /* Generate build table and probe table on each node */

    std::vector<gdf_column *> local_build_table;
    std::vector<gdf_column *> local_probe_table;

    generate_tables_distributed<KEY_T, PAYLOAD_T>(
        local_build_table, BUILD_TABLE_SIZE_EACH_RANK,
        local_probe_table, PROBE_TABLE_SIZE_EACH_RANK,
        SELECTIVITY, RAND_MAX_VAL, IS_BUILD_TABLE_KEY_UNIQUE,
        &communicator
    );

    /* Distributed join */

    std::vector<gdf_column *> distributed_result;

    CUDA_RT_CALL(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    cudaProfilerStart();
    double start = MPI_Wtime();

    distributed_join(
        local_build_table, local_probe_table, distributed_result,
        &communicator, OVER_DECOMPOSITION_FACTOR
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double stop = MPI_Wtime();
    cudaProfilerStop();

    if (mpi_rank == 0) {
        std::cout << "Elasped time (s) " << stop - start << std::endl;
    }

    /* Cleanup */

    free_table(local_build_table);
    free_table(local_probe_table);
    free_table(distributed_result);

    communicator.finalize();

    return 0;
}
