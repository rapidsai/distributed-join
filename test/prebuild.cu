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

#include <vector>
#include <iostream>
#include <cassert>
#include <memory>
#include <mpi.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/join.hpp>

#include "../src/communicator.h"
#include "../src/error.cuh"
#include "../src/distribute_table.cuh"
#include "../src/distributed_join.cuh"

using cudf::experimental::table;

#define SIZE 30000
#define OVER_DECOMPOSITION_FACTOR 1


__global__ void fill_buffer(int *buffer, int multiple)
{
    for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE; i += blockDim.x * gridDim.x)
        buffer[i] = i * multiple;
}


__global__ void verify_correctness(const int *key, const int *col1, const int *col2, int size)
{
    for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += blockDim.x * gridDim.x) {
        assert(key[i] % 15 == 0);
        assert(col1[i] == key[i] / 3);
        assert(col2[i] == key[i] / 5);
    }
}


/**
 * This helper function generates the left/right table used for testing join.
 *
 * There are two columns in each table. The first column is filled with consecutive multiple of
 * argument *multiple*, and is used as key column. For example, if *multiple* is 3, the column
 * contains 0,3,6,9...etc. The second column is filled with consecutive integers and is used as
 * payload column.
 */
std::unique_ptr<table>
generate_table(int multiple)
{
    std::vector<std::unique_ptr<cudf::column> > new_table;

    // compute the number of thread blocks and thread block size for fill_buffer kernel
    const int block_size {128};
    int nblocks {-1};

    CUDA_RT_CALL(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, fill_buffer, block_size, 0)
    );

    // construct the key column
    auto key_column = cudf::make_numeric_column(cudf::data_type(cudf::INT32), SIZE);
    fill_buffer<<<nblocks, block_size>>>(key_column->mutable_view().head<int>(), multiple);
    new_table.push_back(std::move(key_column));

    // construct the payload column
    auto payload_column = cudf::make_numeric_column(cudf::data_type(cudf::INT32), SIZE);
    fill_buffer<<<nblocks, block_size>>>(payload_column->mutable_view().head<int>(), 1);
    new_table.push_back(std::move(payload_column));

    return std::make_unique<table>(std::move(new_table));
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

    /* Generate input tables */

    std::unique_ptr<table> left_table;
    std::unique_ptr<table> right_table;

    if (mpi_rank == 0) {
        left_table = generate_table(3);
        right_table = generate_table(5);
    }

    /* Distribute input tables among ranks */

    auto local_left_table = distribute_table(left_table.get(), &communicator);
    auto local_right_table = distribute_table(right_table.get(), &communicator);

    /* Distributed join */

    auto join_result = distributed_inner_join(
        local_left_table->view(), local_right_table->view(),
        {0}, {0}, {std::pair<cudf::size_type, cudf::size_type>(0, 0)},
        &communicator
    );

    /* Merge table from worker ranks to the root rank */

    std::unique_ptr<table> merged_table = collect_tables(join_result->view(), &communicator);

    /* Verify Correctness */

    if (mpi_rank == 0) {
        const int block_size {128};
        int nblocks {-1};

        CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &nblocks, verify_correctness, block_size, 0
        ));

        assert(merged_table->num_rows() == 6000);

        verify_correctness<<<nblocks, block_size>>>(
            merged_table->get_column(0).view().head<int>(),
            merged_table->get_column(1).view().head<int>(),
            merged_table->get_column(2).view().head<int>(),
            merged_table->num_rows()
        );
    }

    /* Cleanup */

    communicator.finalize();

    if (mpi_rank == 0) {
        std::cerr << "Test case \"prebuild\" passes successfully.\n";
    }

    return 0;
}
