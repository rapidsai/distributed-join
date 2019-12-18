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

#ifndef __CUDF_HELPER_CUH
#define __CUDF_HELPER_CUH

#include <cudf/cudf.h>
#include <string>

#include "error.cuh"
#include "../generate_dataset/generate_dataset.cuh"


template <typename col_type>
gdf_dtype gdf_dtype_from_col_type()
{
    if(std::is_same<col_type,int8_t>::value) return GDF_INT8;
    else if(std::is_same<col_type,uint8_t>::value) return GDF_INT8;
    else if(std::is_same<col_type,int16_t>::value) return GDF_INT16;
    else if(std::is_same<col_type,uint16_t>::value) return GDF_INT16;
    else if(std::is_same<col_type,int32_t>::value) return GDF_INT32;
    else if(std::is_same<col_type,uint32_t>::value) return GDF_INT32;
    else if(std::is_same<col_type,int64_t>::value) return GDF_INT64;
    else if(std::is_same<col_type,uint64_t>::value) return GDF_INT64;
    else if(std::is_same<col_type,float>::value) return GDF_FLOAT32;
    else if(std::is_same<col_type,double>::value) return GDF_FLOAT64;
    else return GDF_invalid;
}


/**
 * Generate a build table and a probe table for testing join performance.
 *
 * Both the build table and the probe table have two columns. The first column is the key column,
 * with datatype KEY_T. The second column is the payload column, with datatype PAYLOAD_T. Both the
 * arguments build_table and probe_table do not need to be preallocated. It is the caller's
 * responsibility to free memory of build_table and probe_table allocated by this function.
 *
 * @param[out] build_table         The build table to generate.
 * @param[in] build_table_size     The number of rows in the build table.
 * @param[out] probe_table         The probe table to generate.
 * @param[in] probe_table_size     The number of rows in the probe table.
 * @param[in] selectivity          Propability with which an element of the probe table is present in
 *                                 the build table.
 * @param[in] rand_max             Maximum random number to generate, i.e., random numbers are
 *                                 integers from [0, rand_max].
 * @param[in] uniq_build_tbl_keys  If each key in the build table should appear exactly once.
 */
template<typename KEY_T, typename PAYLOAD_T>
void generate_build_probe_tables(std::vector<gdf_column *> &build_table,
                                 gdf_size_type build_table_size,
                                 std::vector<gdf_column *> &probe_table,
                                 gdf_size_type probe_table_size,
                                 const double selectivity,
                                 const KEY_T rand_max,
                                 const bool uniq_build_tbl_keys)
{
    // Allocate device memory for generating data

    KEY_T *build_key_data {nullptr};
    PAYLOAD_T *build_payload_data {nullptr};
    KEY_T *probe_key_data {nullptr};
    PAYLOAD_T *probe_payload_data {nullptr};

    RMM_CALL(RMM_ALLOC(&build_key_data, build_table_size * sizeof(KEY_T), 0));

    RMM_CALL(RMM_ALLOC(&build_payload_data, build_table_size * sizeof(PAYLOAD_T), 0));

    RMM_CALL(RMM_ALLOC(&probe_key_data, probe_table_size * sizeof(KEY_T), 0));

    RMM_CALL(RMM_ALLOC(&probe_payload_data, probe_table_size * sizeof(PAYLOAD_T), 0));

    // Generate build and probe table data

    generate_input_tables<KEY_T, gdf_size_type>(
        build_key_data, build_table_size, probe_key_data, probe_table_size,
        selectivity, rand_max, uniq_build_tbl_keys
    );

    linear_sequence<PAYLOAD_T, gdf_size_type><<<(build_table_size+127)/128,128>>>(
        build_payload_data, build_table_size
    );

    linear_sequence<PAYLOAD_T, gdf_size_type><<<(probe_table_size+127)/128,128>>>(
        probe_payload_data, probe_table_size
    );

    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    // Generate build and probe table from data

    gdf_dtype gdf_key_t = gdf_dtype_from_col_type<KEY_T>();
    gdf_dtype gdf_payload_t = gdf_dtype_from_col_type<PAYLOAD_T>();

    build_table.resize(2, nullptr);

    for (auto & column_ptr : build_table) {
        column_ptr = new gdf_column;
    }

    GDF_CALL(gdf_column_view(build_table[0], build_key_data, nullptr, build_table_size, gdf_key_t));

    GDF_CALL(gdf_column_view(build_table[1], build_payload_data, nullptr, build_table_size, gdf_payload_t));

    probe_table.resize(2, nullptr);

    for (auto & column_ptr : probe_table) {
        column_ptr = new gdf_column;
    }

    GDF_CALL(gdf_column_view(probe_table[0], probe_key_data, nullptr, probe_table_size, gdf_key_t));

    GDF_CALL(gdf_column_view(probe_table[1], probe_payload_data, nullptr, probe_table_size, gdf_payload_t));
}


/**
 * Free the table as well as the device buffer it contains.
 *
 * @param[in] table    The table to be freed.
 */
void free_table(std::vector<gdf_column *> & table)
{
    for (auto & column_ptr : table) {
        if (column_ptr->size > 0)
            GDF_CALL(gdf_column_free(column_ptr));

        delete column_ptr;
    }
}


/**
 * Verify two buffers are the same after reordering.
 *
 * This function verify that data1[idx1[i]] == data2[idx2[i]] for all 0 <= i < size.
 */
template<typename key_type>
__global__ void verify_correctness(key_type *data1, int *idx1, key_type *data2, int *idx2, int size)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += blockDim.x * gridDim.x) {
        assert(data1[idx1[i]] == data2[idx2[i]]);
    }
}


#endif // __CUDF_HELPER_CUH
