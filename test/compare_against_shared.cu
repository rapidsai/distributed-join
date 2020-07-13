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
#include <memory>
#include <utility>
#include <tuple>

#include <cudf/table/table.hpp>
#include <cudf/join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/cnmem_memory_resource.hpp>

#include "../src/topology.cuh"
#include "../src/communicator.h"
#include "../src/error.cuh"
#include "../src/generate_table.cuh"
#include "../src/distribute_table.cuh"
#include "../src/distributed_join.cuh"

using cudf::table;

#define KEY_T int
#define PAYLOAD_T int

static cudf::size_type BUILD_TABLE_SIZE = 1'000'000;
static cudf::size_type PROBE_TABLE_SIZE = 5'000'000;
static double SELECTIVITY = 0.3;
static bool IS_BUILD_TABLE_KEY_UNIQUE = true;
static int OVER_DECOMPOSITION_FACTOR = 10;


void parse_command_line_arguments(int argc, char *argv[])
{
    for (int iarg = 0; iarg < argc; iarg++) {
        if (!strcmp(argv[iarg], "--build-table-nrows")) {
            BUILD_TABLE_SIZE = atoi(argv[iarg + 1]);
        }

        if (!strcmp(argv[iarg], "--probe-table-nrows")) {
            PROBE_TABLE_SIZE = atoi(argv[iarg + 1]);
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
    }
}


template<typename data_type>
__global__ void
verify_correctness(const data_type *data1, const data_type *data2, cudf::size_type size)
{
    const cudf::size_type start_idx = threadIdx.x + blockDim.x * blockIdx.x;
    const cudf::size_type stride = blockDim.x * gridDim.x;

    for (cudf::size_type idx = start_idx; idx < size; idx += stride) {
        assert(data1[idx] == data2[idx]);
    }
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

    rmm::mr::cnmem_memory_resource mr {pool_size};
    rmm::mr::set_default_resource(&mr);

    /* Initialize communicator */

    int mpi_rank;
    int mpi_size;
    MPI_CALL( MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) );
    MPI_CALL( MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) );

    UCXCommunicator* communicator = initialize_ucx_communicator(
        true, 2 * 3 * 2 * 2 * mpi_size, 1'000'000LL
    );

    /* Generate build table and probe table and compute reference solution */

    std::unique_ptr<table> build;
    std::unique_ptr<table> probe;
    std::unique_ptr<table> reference;

    cudf::table_view build_view;
    cudf::table_view probe_view;

    if (mpi_rank == 0) {
        KEY_T RAND_MAX_VAL = BUILD_TABLE_SIZE * 2;

        std::tie(build, probe) = generate_build_probe_tables<KEY_T, PAYLOAD_T>(
            BUILD_TABLE_SIZE, PROBE_TABLE_SIZE, SELECTIVITY, RAND_MAX_VAL, IS_BUILD_TABLE_KEY_UNIQUE
        );

        build_view = build->view();
        probe_view = probe->view();

        reference = cudf::inner_join(
            build->view(), probe->view(),
            {0}, {0}, {std::pair<cudf::size_type, cudf::size_type>(0, 0)}
        );
    }

    std::unique_ptr<table> local_build = distribute_table(build_view, communicator);
    std::unique_ptr<table> local_probe = distribute_table(probe_view, communicator);

    /* Distributed join */

    std::unique_ptr<table> join_result_all_ranks = distributed_inner_join(
        local_build->view(), local_probe->view(),
        {0}, {0}, {std::pair<cudf::size_type, cudf::size_type>(0, 0)},
        communicator, OVER_DECOMPOSITION_FACTOR
    );

    /* Send join result from all ranks to the root rank */

    std::unique_ptr<table> join_result = collect_tables(
        join_result_all_ranks->view(), communicator
    );

    /* Verify correctness */

    if (mpi_rank == 0) {
        // Although join_result and reference should contain the same table, rows may be reordered.
        // Therefore, we first sort both tables and then compare

        cudf::size_type nrows = reference->num_rows();
        assert(join_result->num_rows() == nrows);

        std::unique_ptr<table> join_sorted = cudf::sort(join_result->view());
        std::unique_ptr<table> reference_sorted = cudf::sort(reference->view());

        // Get the number of thread blocks based on thread block size

        const int block_size = 128;
        int nblocks {-1};

        CUDA_RT_CALL(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &nblocks, verify_correctness<KEY_T>, block_size, 0
            )
        );

        // There should be three columns in the result table. The first column is the joined key
        // column. The second and third column comes from the payload column from the left and
        // the right input table, respectively.

        // Verify the first column (key column) is correct.

        verify_correctness<KEY_T><<<nblocks, block_size>>>(
            join_sorted->view().column(0).head<KEY_T>(),
            reference_sorted->view().column(0).head<KEY_T>(),
            nrows
        );

        // Verify the remaining two payload columns are correct.

        for (cudf::size_type icol = 1; icol <= 2; icol++) {
            verify_correctness<PAYLOAD_T><<<nblocks, block_size>>>(
                join_sorted->view().column(icol).head<PAYLOAD_T>(),
                reference_sorted->view().column(icol).head<PAYLOAD_T>(),
                nrows
            );
        }
    }

    /* Cleanup */

    communicator->finalize();
    delete communicator;

    if (mpi_rank == 0) {
        std::cerr << "Test case \"compare_against_shared\" passes successfully.\n";
    }

    return 0;
}
