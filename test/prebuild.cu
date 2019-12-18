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

#include <rmm/rmm.h>
#include <mpi.h>
#include <ucp/api/ucp.h>
#include <vector>

#include "../src/comm.cuh"
#include "../src/error.cuh"
#include "../src/distributed.cuh"
#include "../src/error.cuh"


#define SIZE 30000
#define OVER_DECOMPOSITION_FACTOR 1


__global__ void fill_buffer(int *buffer, int multiple)
{
    for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE; i += blockDim.x * gridDim.x)
        buffer[i] = i * multiple;
}


__global__ void verify_prebuild_correctness(int *key, int *col1, int *col2, int size)
{
    for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += blockDim.x * gridDim.x) {
        assert(key[i] % 15 == 0);
        assert(col1[i] == key[i] / 3);
        assert(col2[i] == key[i] / 5);
    }
}


gdf_column* generate_column(int *buffer)
{
    gdf_column* new_column = new gdf_column;

    CHECK_ERROR(
        gdf_column_view(new_column, buffer, nullptr, SIZE, GDF_INT32),
        GDF_SUCCESS, "gdf_column_view"
    );

    return new_column;
}


void generate_table(std::vector<gdf_column *> &table, int multiple)
{
    table.resize(2, nullptr);

    int *col1_buffer;
    CHECK_ERROR(RMM_ALLOC(&col1_buffer, SIZE * sizeof(int), 0), RMM_SUCCESS, "RMM_ALLOC");

    const int block_size = 128;
    int nblocks {-1};

    CHECK_ERROR(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, fill_buffer, block_size, 0),
        cudaSuccess, "cudaOccupancyMaxActiveBlocksPerMultiprocessor"
    );

    fill_buffer<<<nblocks, block_size>>>(col1_buffer, multiple);

    table[0] = generate_column(col1_buffer);

    int *col2_buffer;
    CHECK_ERROR(RMM_ALLOC(&col2_buffer, SIZE * sizeof(int), 0), RMM_SUCCESS, "RMM_ALLOC");

    CHECK_ERROR(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, linear_sequence<int, size_t>, block_size, 0),
        cudaSuccess, "cudaOccupancyMaxActiveBlocksPerMultiprocessor"
    );

    linear_sequence<<<nblocks, block_size>>>(col2_buffer, SIZE);

    table[1] = generate_column(col2_buffer);
}


int main(int argc, char *argv[])
{
    /* Initialize communication */

    UCXBufferCommunicator communicator;
    communicator.initialize(argc, argv);

    int mpi_rank {communicator.mpi_rank};
    int mpi_size {communicator.mpi_size};

    communicator.setup_cache(2 * mpi_size, 100'000LL);
    communicator.warmup_cache();

    /* Generate tables */

    std::vector<gdf_column *> left_table;
    std::vector<gdf_column *> right_table;

    if (mpi_rank == 0) {
        generate_table(left_table, 3);
        generate_table(right_table, 5);
    }

    std::vector<gdf_column *> local_left_table = distribute_table(left_table, &communicator);
    std::vector<gdf_column *> local_right_table = distribute_table(right_table, &communicator);

    /* Distributed join */

    std::vector<gdf_column *> join_result;

    distributed_join(local_left_table, local_right_table, join_result, &communicator, OVER_DECOMPOSITION_FACTOR);

    std::vector<gdf_column *> merged_join_result;

    collect_tables(merged_join_result, join_result, &communicator);

    /* Verify Correctness */

    const int block_size = 128;
    int nblocks {-1};

    CHECK_ERROR(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, verify_prebuild_correctness, block_size, 0),
        cudaSuccess, "cudaOccupancyMaxActiveBlocksPerMultiprocessor"
    );

    if (mpi_rank == 0) {
        assert(merged_join_result[0]->size == 6000);

        verify_prebuild_correctness<<<nblocks, block_size>>>(
            (int *)merged_join_result[1]->data,
            (int *)merged_join_result[0]->data,
            (int *)merged_join_result[2]->data,
            merged_join_result[0]->size
        );
    }

    /* Cleanup */

    free_table(local_left_table);
    free_table(local_right_table);

    if (mpi_rank == 0) {
        free_table(merged_join_result);
        free_table(left_table);
        free_table(right_table);
    }

    free_table(join_result);

    communicator.finalize();

    return 0;
}
