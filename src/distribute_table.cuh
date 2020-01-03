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

#ifndef __DISTRIBUTE_TABLE
#define __DISTRIBUTE_TABLE

#include <vector>

#include <cudf/types.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include "comm.cuh"
#include "error.cuh"
#include "communicator.h"

/**
 * Helper function for calculating the number of rows of a local table.
 *
 * This function is useful, for example, for calculating the size on root and allocating receive
 * buffer on workers when distributing tables from root node to worker nodes. This function mainly
 * solves the problem when the number of ranks does not divide the number of rows in the global
 * table.
 *
 * @param[in] global_table_size     Number of rows in the global table.
 * @param[in] mpi_rank              Target rank for which this function will calculate the local
 *                                  table size.
 * @param[in] mpi_size              Total number of MPI ranks.
 *
 * @returns                         Number of rows in the local table of rank *mpi_rank*.
 */
inline cudf::size_type
get_local_table_size(cudf::size_type global_table_size, int mpi_rank, int mpi_size)
{
    cudf::size_type local_table_size = global_table_size / mpi_size;

    if (mpi_rank < global_table_size % mpi_size) {
        local_table_size++;
    }

    return local_table_size;
}


/**
 * Distribute a column from root rank to worker ranks.
 *
 * This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in]  global_col     The global table column. Significant only on root rank.
 * @param[out] local_col      The local table column to be filled with data in *global_col*.
 *                            Significant on all ranks.
 * @param[in]  communicator   An instance of `Communicator` used for communication.
 */
void
distribute_cols(
    cudf::column_view global_col,
    cudf::mutable_column_view local_col,
    Communicator *communicator)
{
    /* Get MPI information */

    int mpi_rank {communicator->mpi_rank};
    int mpi_size {communicator->mpi_size};

    std::vector<comm_handle_t> requests;
    std::size_t dtype_size = cudf::size_of(local_col.type());

    if (mpi_rank == 0) {
        // master node

        cudf::size_type global_size = global_col.size();
        requests.resize(mpi_size);

        // Send global_col to each slave node

        for (cudf::size_type irank = 1; irank < mpi_size; irank++) {
            cudf::size_type start_idx = std::min<cudf::size_type>(irank, global_size % mpi_size)
                                        + (global_size / mpi_size) * irank;
            cudf::size_type irank_size = get_local_table_size(global_size, irank, mpi_size);
            void *start_addr = (void *)(global_col.head<char>() + start_idx * dtype_size);

            requests[irank] = communicator->send(
                start_addr, irank_size, dtype_size, irank, distribute_col_tag
            );
        }

        // Fill master node's local_col

        cudf::size_type rank0_size = get_local_table_size(global_size, 0, mpi_size);

        CUDA_RT_CALL(cudaMemcpy(
            local_col.head(), global_col.head(), rank0_size * dtype_size,
            cudaMemcpyDeviceToDevice
        ));

        communicator->waitall(requests);

    } else {
        // slave node

        comm_handle_t request = communicator->recv(
            local_col.head(), local_col.size(), dtype_size, 0, distribute_col_tag
        );

        communicator->wait(request);
    }
}


/**
 * Distribute a table from root rank to worker ranks.
 *
 * This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] global_table     The global table. Only significant on root rank.
 * @param[in] communicator     An instance of `Communicator` used for communication.
 *
 * @returns                    The local table on each rank.
 */
std::unique_ptr<cudf::experimental::table>
distribute_table(
    cudf::experimental::table *global_table,
    Communicator *communicator)
{
    /* Get MPI information */

    int mpi_rank {communicator->mpi_rank};
    int mpi_size {communicator->mpi_size};
    MPI_Datatype mpi_size_type = mpi_dtype_from_c_type<cudf::size_type>();

    /* Broadcast global table size */

    cudf::size_type global_table_size {-1};

    if (mpi_rank == 0) {
        global_table_size = global_table->num_rows();
    }

    MPI_Bcast(&global_table_size, 1, mpi_size_type, 0, MPI_COMM_WORLD);

    /* Broadcast number of columns */

    cudf::size_type ncols {-1};

    if (mpi_rank == 0) {
        ncols = global_table->num_columns();
    }

    MPI_Bcast(&ncols, 1, mpi_size_type, 0, MPI_COMM_WORLD);

    /* Broadcast column datatype */

    std::vector<cudf::data_type> columns_dtype(ncols);

    for (cudf::size_type icol = 0; icol < ncols; icol++) {

        if (mpi_rank == 0)
            columns_dtype[icol] = global_table->view().column(icol).type();

        MPI_Bcast(&columns_dtype[icol], sizeof(cudf::size_type), MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    /* Allocate local tables across nodes */

    cudf::size_type local_table_size = get_local_table_size(global_table_size, mpi_rank, mpi_size);

    std::vector<std::unique_ptr<cudf::column> > local_table;

    for (int icol = 0; icol < ncols; icol++) {
        auto new_column = cudf::make_numeric_column(columns_dtype[icol], local_table_size);
        local_table.push_back(std::move(new_column));
    }

    /* Send table from root to all nodes */

    for (int icol = 0; icol < ncols; icol++) {
        cudf::column_view global_col;

        if (mpi_rank == 0)
            global_col = global_table->view().column(icol);

        distribute_cols(global_col, local_table[icol]->mutable_view(), communicator);
    }

    return std::make_unique<cudf::experimental::table>(std::move(local_table));
}

#endif
