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
void generate_table(std::vector<std::unique_ptr<cudf::column> > &table, int multiple)
{
    assert(table.size() == 0);

    // compute the number of thread blocks and thread block size for fill_buffer kernel
    const int block_size {128};
    int nblocks {-1};

    CUDA_RT_CALL(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, fill_buffer, block_size, 0)
    );

    // construct the key column
    auto key_column = cudf::make_numeric_column(cudf::data_type(cudf::INT32), SIZE);
    fill_buffer<<<nblocks, block_size>>>(key_column->mutable_view().head<int>(), multiple);
    table.push_back(std::move(key_column));

    // construct the payload column
    auto payload_column = cudf::make_numeric_column(cudf::data_type(cudf::INT32), SIZE);
    fill_buffer<<<nblocks, block_size>>>(payload_column->mutable_view().head<int>(), 1);
    table.push_back(std::move(payload_column));
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

    std::vector<std::unique_ptr<cudf::column> > left_table;
    std::vector<std::unique_ptr<cudf::column> > right_table;

    if (mpi_rank == 0) {
        generate_table(left_table, 3);
        generate_table(right_table, 5);
    }

    /* Distribute input tables among ranks */

    // @TODO: uncomment this after porting distribute_table to new column API
    // auto local_left_table = distribute_table(left_table, &communicator);
    // auto local_right_table = distribute_table(right_table, &communicator);

    /* Distributed join */

    // @TODO: currently this shared-memory join is just a placeholder
    std::unique_ptr<cudf::experimental::table> join_result;

    if (mpi_rank == 0) {
        cudf::table_view left_table_view { {left_table[0]->view(), left_table[1]->view()} };
        cudf::table_view right_table_view { {right_table[0]->view(), right_table[1]->view()} };

        join_result = cudf::experimental::inner_join(
            left_table_view, right_table_view,
            {0}, {0}, {std::pair<cudf::size_type, cudf::size_type>(0, 0)}
        );
    }

    /* Verify Correctness */

    const int block_size {128};
    int nblocks {-1};

    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &nblocks, verify_correctness, block_size, 0
    ));

    if (mpi_rank == 0) {
        assert(join_result->num_rows() == 6000);

        verify_correctness<<<nblocks, block_size>>>(
            join_result->get_column(0).view().head<int>(),
            join_result->get_column(1).view().head<int>(),
            join_result->get_column(2).view().head<int>(),
            join_result->num_rows()
        );
    }

    /* Cleanup */

    communicator.finalize();

    if (mpi_rank == 0) {
        std::cerr << "Test case \"prebuild\" passes successfully.\n";
    }

    return 0;
}
