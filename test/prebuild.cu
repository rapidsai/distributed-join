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
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include "../src/topology.cuh"
#include "../src/communicator.h"
#include "../src/error.cuh"
#include "../src/distribute_table.cuh"
#include "../src/distributed_join.cuh"

using cudf::table;

static cudf::size_type SIZE = 30000;  // must be a multiple of 5
static int OVER_DECOMPOSITION_FACTOR = 1;


void parse_command_line_arguments(int argc, char *argv[])
{
    for (int iarg = 0; iarg < argc; iarg++) {
        if (!strcmp(argv[iarg], "--size")) {
            SIZE = atoi(argv[iarg + 1]);
        }

        if (!strcmp(argv[iarg], "--over-decomposition-factor")) {
            OVER_DECOMPOSITION_FACTOR = atoi(argv[iarg + 1]);
        }
    }
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

    // construct the key column
    auto key_column = cudf::make_numeric_column(cudf::data_type(cudf::type_id::INT32), SIZE);
    auto key_buffer = key_column->mutable_view().head<int>();
    thrust::sequence(thrust::device, key_buffer, key_buffer + SIZE, 0, multiple);
    new_table.push_back(std::move(key_column));

    // construct the payload column
    auto payload_column = cudf::make_numeric_column(cudf::data_type(cudf::type_id::INT32), SIZE);
    auto payload_buffer = payload_column->mutable_view().head<int>();
    thrust::sequence(thrust::device, payload_buffer, payload_buffer + SIZE);
    new_table.push_back(std::move(payload_column));

    return std::make_unique<table>(std::move(new_table));
}


int main(int argc, char *argv[])
{
    /* Initialize topology */

    setup_topology(argc, argv);

    /* Parse command line arguments */

    parse_command_line_arguments(argc, argv);

    /* Initialize memory pool */

    size_t free_memory, total_memory;
    CUDA_RT_CALL(cudaMemGetInfo(&free_memory, &total_memory));
    const size_t pool_size = free_memory - 5LL * (1LL << 29);  // free memory - 500MB

    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr {mr, pool_size, pool_size};
    rmm::mr::set_current_device_resource(&pool_mr);

    /* Initialize communicator */

    int mpi_rank;
    int mpi_size;
    MPI_CALL( MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) );
    MPI_CALL( MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) );

    UCXCommunicator* communicator = initialize_ucx_communicator(
        true, 2 * mpi_size, 100'000LL
    );

    /* Generate input tables */

    std::unique_ptr<table> left_table;
    std::unique_ptr<table> right_table;
    cudf::table_view left_view;
    cudf::table_view right_view;

    if (mpi_rank == 0) {
        left_table = generate_table(3);
        right_table = generate_table(5);

        left_view = left_table->view();
        right_view = right_table->view();
    }

    /* Distribute input tables among ranks */

    auto local_left_table = distribute_table(left_view, communicator);
    auto local_right_table = distribute_table(right_view, communicator);

    /* Distributed join */

    auto join_result = distributed_inner_join(
        local_left_table->view(), local_right_table->view(),
        {0}, {0}, {std::pair<cudf::size_type, cudf::size_type>(0, 0)},
        communicator, OVER_DECOMPOSITION_FACTOR
    );

    /* Merge table from worker ranks to the root rank */

    std::unique_ptr<table> merged_table = collect_tables(join_result->view(), communicator);

    /* Verify Correctness */

    if (mpi_rank == 0) {
        const int block_size {128};
        int nblocks {-1};

        CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &nblocks, verify_correctness, block_size, 0
        ));

        // Since the key has to be a multiple of 5, the join result size is the size of left table
        // divided by 5.
        assert(merged_table->num_rows() == SIZE / 5);

        verify_correctness<<<nblocks, block_size>>>(
            merged_table->get_column(0).view().head<int>(),
            merged_table->get_column(1).view().head<int>(),
            merged_table->get_column(2).view().head<int>(),
            merged_table->num_rows()
        );
    }

    /* Cleanup */

    communicator->finalize();
    delete communicator;

    if (mpi_rank == 0) {
        std::cerr << "Test case \"prebuild\" passes successfully.\n";
    }

    return 0;
}
