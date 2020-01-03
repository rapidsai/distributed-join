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
#include <iostream>

#include "../src/cudf_helper.cuh"
#include "../src/distributed.cuh"
#include "../src/error.cuh"
#include "../src/comm.cuh"

#define BUILD_TABLE_SIZE 1'000'000
#define PROBE_TABLE_SIZE 5'000'000
#define SELECTIVITY 0.3
#define RAND_MAX_VAL 2'000'000
#define IS_BUILD_TABLE_KEY_UNIQUE true
#define OVER_DECOMPOSITION_FACTOR 10

#define KEY_T int
#define PAYLOAD_T int


int main(int argc, char *argv[])
{
    /* Initialize communication */

    UCXBufferCommunicator communicator;
    communicator.initialize(argc, argv);

    int mpi_rank {communicator.mpi_rank};
    int mpi_size {communicator.mpi_size};

    communicator.setup_cache(2 * 3 * 2 * 2 * mpi_size, 1'000'000LL);
    communicator.warmup_cache();

    /* Generate build table and probe table and compute reference solution */

    std::vector<gdf_column *> build_table;
    std::vector<gdf_column *> probe_table;
    std::vector<gdf_column *> reference_result;

    gdf_context ctxt = {
        0,                     // input data is not sorted
        gdf_method::GDF_HASH,  // hash based join
        0
    };

    int columns_to_join[] = {0};

    if (mpi_rank == 0) {
        generate_build_probe_tables<KEY_T, PAYLOAD_T>(
            build_table, BUILD_TABLE_SIZE, probe_table, PROBE_TABLE_SIZE,
            SELECTIVITY, RAND_MAX_VAL, IS_BUILD_TABLE_KEY_UNIQUE
        );

        reference_result.resize(build_table.size() + probe_table.size() - 1, nullptr);

        for (auto & col_ptr : reference_result) {
            col_ptr = new gdf_column;
        }

        CHECK_ERROR(
            gdf_inner_join(build_table.data(), build_table.size(), columns_to_join,
                           probe_table.data(), probe_table.size(), columns_to_join,
                           1, build_table.size() + probe_table.size() - 1, reference_result.data(),
                           nullptr, nullptr, &ctxt),
            GDF_SUCCESS, "gdf_inner_join"
        );

    }

    std::vector<gdf_column *> local_build_table = distribute_table(build_table, &communicator);
    std::vector<gdf_column *> local_probe_table = distribute_table(probe_table, &communicator);

    /* Distributed join */

    std::vector<gdf_column *> distributed_result;

    distributed_join(
        local_build_table, local_probe_table, distributed_result,
        &communicator, OVER_DECOMPOSITION_FACTOR
    );

    if (mpi_rank == 0) {
        free_table(build_table);
        free_table(probe_table);
    }

    free_table(local_build_table);
    free_table(local_probe_table);

    std::vector<gdf_column *> received_table;

    collect_tables(received_table, distributed_result, &communicator);
    free_table(distributed_result);

    if (mpi_rank == 0) {
        // hold the indices of sort result
        gdf_column refernece_idx;
        gdf_column received_idx;

        gdf_size_type size = reference_result[0]->size;
        int ncols = reference_result.size();

        assert(size == received_table[0]->size);

        /* Allocate device memory for sort indices */

        void *data;

        CHECK_ERROR(RMM_ALLOC(&data, size * sizeof(int), 0), RMM_SUCCESS, "RMM_ALLOC");

        CHECK_ERROR(
            gdf_column_view(&refernece_idx, data, nullptr, size, GDF_INT32),
            GDF_SUCCESS, "gdf_column_view"
        );

        CHECK_ERROR(RMM_ALLOC(&data, size * sizeof(int), 0), RMM_SUCCESS, "RMM_ALLOC");

        CHECK_ERROR(
            gdf_column_view(&received_idx, data, nullptr, size, GDF_INT32),
            GDF_SUCCESS, "gdf_column_view"
        );

        /* Sort the reference table and reference table */

        std::vector<int8_t> asc_desc(ncols, 0);
        int8_t *asc_desc_dev;
        CHECK_ERROR(RMM_ALLOC(&asc_desc_dev, ncols * sizeof(int8_t), 0), RMM_SUCCESS, "RMM_ALLOC");
        CHECK_ERROR(
            cudaMemcpy(asc_desc_dev, asc_desc.data(), ncols * sizeof(int8_t), cudaMemcpyHostToDevice),
            cudaSuccess, "cudaMemcpy"
        );

        CHECK_ERROR(
            gdf_order_by(reference_result.data(), asc_desc_dev, ncols, &refernece_idx, &ctxt),
            GDF_SUCCESS, "gdf_order_by"
        );

        CHECK_ERROR(
            gdf_order_by(received_table.data(), asc_desc_dev, ncols, &received_idx, &ctxt),
            GDF_SUCCESS, "gdf_order_by"
        );

        /* Verify correctness */

        const int block_size = 128;
        int nblocks {-1};

        CHECK_ERROR(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, verify_correctness<int>, block_size, 0),
            cudaSuccess, "cudaOccupancyMaxActiveBlocksPerMultiprocessor"
        );

        for (int icol = 0; icol < ncols; icol++) {
            verify_correctness<int><<<nblocks, block_size>>>(
                (int *)reference_result[icol]->data,
                (int *)refernece_idx.data,
                (int *)received_table[icol]->data,
                (int *)received_idx.data,
                size
            );
        }

        CHECK_ERROR(RMM_FREE(asc_desc_dev, 0), RMM_SUCCESS, "RMM_FREE");
        CHECK_ERROR(gdf_column_free(&refernece_idx), GDF_SUCCESS, "gdf_column_free");
        CHECK_ERROR(gdf_column_free(&received_idx), GDF_SUCCESS, "gdf_column_free");

    }

    /* Cleanup */

    if (mpi_rank == 0) {
        free_table(reference_result);
        free_table(received_table);
    }

    communicator.finalize();

    if (mpi_rank == 0) {
        std::cerr << "Test case \"compare_against_shared\" passes successfully.\n";
    }

    return 0;
}
