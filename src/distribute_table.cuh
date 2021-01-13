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

#include <memory>
#include <vector>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include "comm.cuh"
#include "communicator.h"
#include "error.cuh"

/**
 * Helper function for calculating the number of rows of a local table.
 *
 * This function is useful, for example, for calculating the size on root and allocating receive
 * buffer on workers when distributing tables from root rank to worker ranks. This function mainly
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
inline cudf::size_type get_local_table_size(cudf::size_type global_table_size,
                                            int mpi_rank,
                                            int mpi_size)
{
  cudf::size_type local_table_size = global_table_size / mpi_size;

  if (mpi_rank < global_table_size % mpi_size) { local_table_size++; }

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
void distribute_cols(cudf::column_view global_col,
                     cudf::mutable_column_view local_col,
                     Communicator *communicator)
{
  /* Get MPI information */

  int mpi_rank = communicator->mpi_rank;
  int mpi_size = communicator->mpi_size;

  std::size_t dtype_size = cudf::size_of(local_col.type());

  communicator->start();

  if (mpi_rank == 0) {
    // Send global_col from the root rank to each worker rank

    cudf::size_type global_size = global_col.size();

    for (cudf::size_type irank = 1; irank < mpi_size; irank++) {
      cudf::size_type start_idx =
        std::min<cudf::size_type>(irank, global_size % mpi_size) + (global_size / mpi_size) * irank;
      cudf::size_type irank_size = get_local_table_size(global_size, irank, mpi_size);
      void *start_addr           = (void *)(global_col.head<char>() + start_idx * dtype_size);

      communicator->send(start_addr, irank_size, dtype_size, irank);
    }

    // Fill the root rank's local_col

    cudf::size_type rank0_size = get_local_table_size(global_size, 0, mpi_size);

    CUDA_RT_CALL(cudaMemcpy(
      local_col.head(), global_col.head(), rank0_size * dtype_size, cudaMemcpyDeviceToDevice));

  } else {
    // worker rank
    communicator->recv(local_col.head(), local_col.size(), dtype_size, 0);
  }

  communicator->stop();
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
std::unique_ptr<cudf::table> distribute_table(cudf::table_view global_table,
                                              Communicator *communicator)
{
  /* Get MPI information */

  int mpi_rank               = communicator->mpi_rank;
  int mpi_size               = communicator->mpi_size;
  MPI_Datatype mpi_size_type = mpi_dtype_from_c_type<cudf::size_type>();

  /* Broadcast global table size */

  cudf::size_type global_table_size{-1};

  if (mpi_rank == 0) { global_table_size = global_table.num_rows(); }

  MPI_Bcast(&global_table_size, 1, mpi_size_type, 0, MPI_COMM_WORLD);

  /* Broadcast number of columns */

  cudf::size_type ncols{-1};

  if (mpi_rank == 0) { ncols = global_table.num_columns(); }

  MPI_Bcast(&ncols, 1, mpi_size_type, 0, MPI_COMM_WORLD);

  /* Broadcast column datatype */

  std::vector<cudf::data_type> columns_dtype(ncols);

  for (cudf::size_type icol = 0; icol < ncols; icol++) {
    if (mpi_rank == 0) columns_dtype[icol] = global_table.column(icol).type();

    MPI_Bcast(&columns_dtype[icol], sizeof(cudf::size_type), MPI_CHAR, 0, MPI_COMM_WORLD);
  }

  /* Allocate local tables across ranks */

  cudf::size_type local_table_size = get_local_table_size(global_table_size, mpi_rank, mpi_size);

  std::vector<std::unique_ptr<cudf::column>> local_table;

  for (int icol = 0; icol < ncols; icol++) {
    auto new_column = cudf::make_numeric_column(columns_dtype[icol], local_table_size);
    local_table.push_back(std::move(new_column));
  }

  CUDA_RT_CALL(cudaStreamSynchronize(cudaStreamDefault));

  /* Send table from the root rank to all ranks */

  for (int icol = 0; icol < ncols; icol++) {
    cudf::column_view global_col;

    if (mpi_rank == 0) global_col = global_table.column(icol);

    distribute_cols(global_col, local_table[icol]->mutable_view(), communicator);
  }

  return std::make_unique<cudf::table>(std::move(local_table));
}

/**
 * Merge tables from all worker ranks to the root rank.
 *
 * This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] table The table on each rank to be sent to the master rank. Significant on all ranks.
 * @param[in] communicator An instance of `Communicator` used for communication.
 *
 * @return Merged table on the root rank. `nullptr` on all other ranks.
 */
std::unique_ptr<cudf::table> collect_tables(cudf::table_view table, Communicator *communicator)
{
  int mpi_rank = communicator->mpi_rank;
  int mpi_size = communicator->mpi_size;

  int ncols = table.num_columns();
  int nrows = table.num_rows();

  /* Send the table size to the root */

  std::vector<cudf::size_type> table_nrows(mpi_size, -1);
  MPI_CALL(MPI_Gather(&nrows,
                      1,
                      mpi_dtype_from_c_type<cudf::size_type>(),
                      table_nrows.data(),
                      1,
                      mpi_dtype_from_c_type<cudf::size_type>(),
                      0,
                      MPI_COMM_WORLD));

  /* Compute the scan of table_size on root */

  std::vector<cudf::size_type> table_nrows_scan(mpi_size + 1, -1);
  if (mpi_rank == 0) {
    table_nrows_scan[0] = 0;
    for (int irank = 0; irank < mpi_size; irank++) {
      table_nrows_scan[irank + 1] = table_nrows_scan[irank] + table_nrows[irank];
    }
  }

  /* Construct receive buffer on root */

  std::vector<std::unique_ptr<cudf::column>> merged_columns;
  if (mpi_rank == 0) {
    for (int icol = 0; icol < ncols; icol++) {
      merged_columns.push_back(
        std::move(cudf::make_numeric_column(table.column(icol).type(), table_nrows_scan.back())));
    }

    CUDA_RT_CALL(cudaStreamSynchronize(cudaStreamDefault));
  }

  /* Send table from each rank to root */

  for (int icol = 0; icol < ncols; icol++) {
    std::size_t dtype_size = cudf::size_of(table.column(icol).type());

    communicator->start();

    if (mpi_rank == 0) {
      for (int irank = 1; irank < mpi_size; irank++) {
        void *start_addr = (void *)(merged_columns[icol]->mutable_view().head<char>() +
                                    table_nrows_scan[irank] * dtype_size);

        communicator->recv(start_addr, table_nrows[irank], dtype_size, irank);
      }

      CUDA_RT_CALL(cudaMemcpy(merged_columns[icol]->mutable_view().head(),
                              table.column(icol).head(),
                              table_nrows[0] * dtype_size,
                              cudaMemcpyDeviceToDevice));
    } else {
      communicator->send(table.column(icol).head(), nrows, dtype_size, 0);
    }

    communicator->stop();
  }

  if (mpi_rank == 0) {
    return std::make_unique<cudf::table>(std::move(merged_columns));
  } else {
    return std::unique_ptr<cudf::table>(nullptr);
  }
}

#endif
