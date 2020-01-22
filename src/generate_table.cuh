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

#ifndef __GENERATE_TABLE_CUH
#define __GENERATE_TABLE_CUH

#include <vector>
#include <utility>
#include <memory>
#include <stdexcept>
#include <tuple>

#include <cudf/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>

#include "error.cuh"
#include "../generate_dataset/generate_dataset.cuh"
#include "distributed_join.cuh"

using cudf::experimental::table;


template <typename col_type>
cudf::data_type gdf_dtype_from_col_type()
{
    if(std::is_same<col_type,int8_t>::value) return cudf::data_type(cudf::INT8);
    else if(std::is_same<col_type,uint8_t>::value) return cudf::data_type(cudf::INT8);
    else if(std::is_same<col_type,int16_t>::value) return cudf::data_type(cudf::INT16);
    else if(std::is_same<col_type,uint16_t>::value) return cudf::data_type(cudf::INT16);
    else if(std::is_same<col_type,int32_t>::value) return cudf::data_type(cudf::INT32);
    else if(std::is_same<col_type,uint32_t>::value) return cudf::data_type(cudf::INT32);
    else if(std::is_same<col_type,int64_t>::value) return cudf::data_type(cudf::INT64);
    else if(std::is_same<col_type,uint64_t>::value) return cudf::data_type(cudf::INT64);
    else if(std::is_same<col_type,float>::value) return cudf::data_type(cudf::FLOAT32);
    else if(std::is_same<col_type,double>::value) return cudf::data_type(cudf::FLOAT64);
    else throw std::runtime_error("Unknown data type for cuDF");
}


/**
* Generate a build table and a probe table for testing distributed join.
*
* Both the build table and the probe table have two columns. The first column is the key column,
* with datatype KEY_T. The second column is the payload column, with datatype PAYLOAD_T.
*
* @param[in] build_table_nrows The number of rows in the build table.
* @param[in] probe_table_nrows The number of rows in the probe table.
* @param[in] selectivity Propability with which an element of the probe table is present in the
* build table.
* @param[in] rand_max Maximum random number to generate, i.e., random numbers are integers from
* [0, rand_max].
* @param[in] uniq_build_tbl_keys If each key in the build table should appear exactly once.
*
* @return A pair of generated build table and probe table.
*/
template<typename KEY_T, typename PAYLOAD_T>
std::pair<std::unique_ptr<table>, std::unique_ptr<table> >
generate_build_probe_tables(cudf::size_type build_table_nrows,
                            cudf::size_type probe_table_nrows,
                            const double selectivity,
                            const KEY_T rand_max,
                            const bool uniq_build_tbl_keys)
{
    // Allocate device memory for the generated columns

    std::vector<std::unique_ptr<cudf::column> > build;
    std::vector<std::unique_ptr<cudf::column> > probe;

    build.push_back(std::move(
        cudf::make_numeric_column(gdf_dtype_from_col_type<KEY_T>(), build_table_nrows)
    ));

    build.push_back(std::move(
        cudf::make_numeric_column(gdf_dtype_from_col_type<PAYLOAD_T>(), build_table_nrows)
    ));

    probe.push_back(std::move(
        cudf::make_numeric_column(gdf_dtype_from_col_type<KEY_T>(), probe_table_nrows)
    ));

    probe.push_back(std::move(
        cudf::make_numeric_column(gdf_dtype_from_col_type<PAYLOAD_T>(), probe_table_nrows)
    ));

    // Generate build and probe table data

    generate_input_tables<KEY_T, cudf::size_type>(
        build[0]->mutable_view().head<KEY_T>(), build_table_nrows,
        probe[0]->mutable_view().head<KEY_T>(), probe_table_nrows,
        selectivity, rand_max, uniq_build_tbl_keys
    );

    linear_sequence<PAYLOAD_T, cudf::size_type><<<(build_table_nrows+127)/128,128>>>(
        build[1]->mutable_view().head<PAYLOAD_T>(), build_table_nrows
    );

    linear_sequence<PAYLOAD_T, cudf::size_type><<<(probe_table_nrows+127)/128,128>>>(
        probe[1]->mutable_view().head<PAYLOAD_T>(), probe_table_nrows
    );

    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    // return the generated tables

    auto build_table = std::make_unique<table>(std::move(build));
    auto probe_table = std::make_unique<table>(std::move(probe));

    return std::make_pair(std::move(build_table), std::move(probe_table));
}


template <typename data_type>
void add_constant_to_column(cudf::mutable_column_view column, data_type constant)
{
    auto buffer_ptr = thrust::device_pointer_cast(column.head<data_type>());

    thrust::for_each(buffer_ptr, buffer_ptr + column.size(),
                     [=] __device__ (data_type &i) {
                        i += constant;
                     });
}


/**
 * This function generates build table and probe table distributed, and it need to be called
 * collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] build_table_nrows_per_rank   The number of rows of build table on each rank.
 * @param[in] probe_table_nrows_per_rank   The number of rows of probe table on each rank.
 * @param[in] selectivity                  The percentage of keys in the probe table present in the build table.
 * @param[in] rand_max_per_rank            The lottery size on each rank. This argument should be set larger than
 *                                         `build_table_size_per_rank`.
 * @param[in] uniq_build_tbl_keys          Whether the keys in the build table are unique.
 * @param[in] communicator                 An instance of `Communicator` used for communication.
 *
 * Note: require build_table_size_per_rank % mpi_rank == 0 and probe_table_size_per_rank % mpi_rank == 0.
 *
 * @return A pair of generated build and probe table distributed on each rank.
 */
template<typename KEY_T, typename PAYLOAD_T>
std::pair<std::unique_ptr<table>, std::unique_ptr<table> >
generate_tables_distributed(
    cudf::size_type build_table_nrows_per_rank,
    cudf::size_type probe_table_nrows_per_rank,
    const double selectivity,
    const KEY_T rand_max_per_rank,
    const bool uniq_build_tbl_keys,
    Communicator *communicator)
{
    // Algorithm used for distributed generation:
    // Rank i generates build and probe table independently with keys randomly selected from range
    // [i*uniq_build_tbl_keys, (i+1)*uniq_build_tbl_keys] (called pre_shuffle_table). Afterwards,
    // pre_shuffle_table will be divided into N chunks with the same number of rows, and then send
    // chunk j to rank j. This all-to-all communication will make each local table have keys
    // uniformly from the whole range.

    // Get MPI information

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Generate local build and probe table on each rank

    std::unique_ptr<table> pre_shuffle_build_table;
    std::unique_ptr<table> pre_shuffle_probe_table;

    std::tie(pre_shuffle_build_table, pre_shuffle_probe_table) = \
        generate_build_probe_tables<KEY_T, PAYLOAD_T>(
            build_table_nrows_per_rank, probe_table_nrows_per_rank,
            selectivity, rand_max_per_rank, uniq_build_tbl_keys
        );

    // Add constant to build and probe table to make sure the range is correct

    add_constant_to_column<KEY_T>(
        pre_shuffle_build_table->mutable_view().column(0), rand_max_per_rank * mpi_rank
    );

    add_constant_to_column<KEY_T>(
        pre_shuffle_probe_table->mutable_view().column(0), rand_max_per_rank * mpi_rank
    );

    add_constant_to_column<PAYLOAD_T>(
        pre_shuffle_build_table->mutable_view().column(1), build_table_nrows_per_rank * mpi_rank
    );

    add_constant_to_column<PAYLOAD_T>(
        pre_shuffle_probe_table->mutable_view().column(1), probe_table_nrows_per_rank * mpi_rank
    );

    // Construct buffer offset to indicate the start indices to each rank

    std::vector<cudf::size_type> build_table_offset(mpi_size + 1);
    std::vector<cudf::size_type> probe_table_offset(mpi_size + 1);

    for (cudf::size_type irank = 0; irank <= mpi_size; irank ++) {
        build_table_offset[irank] = build_table_nrows_per_rank / mpi_size * irank;
        probe_table_offset[irank] = probe_table_nrows_per_rank / mpi_size * irank;
    }

    // Send each bucket to the desired target rank

    std::unique_ptr<table> build_table = all_to_all_comm_single_batch(
        pre_shuffle_build_table->view(), build_table_offset, communicator
    );

    std::unique_ptr<table> probe_table = all_to_all_comm_single_batch(
        pre_shuffle_probe_table->view(), probe_table_offset, communicator
    );

    return std::make_pair(std::move(build_table), std::move(probe_table));
}

#endif // __GENERATE_TABLE_CUH
