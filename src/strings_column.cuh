/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "comm.cuh"
#include "communicator.h"
#include "error.cuh"

#include <rmm/thrust_rmm_allocator.h>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/adjacent_difference.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/scan.h>

#include <vector>

using std::vector;

/**
 * Calculate and communicate the number of bytes sent/received during all-to-all communication for
 * all string columns in all batches.
 *
 * Note: This function needs to be called collectively by all ranks in `MPI_COMM_WORLD`.
 *
 * @param[in] table Table that needs to be all-to-all communicated.
 * @param[in] offsets Vector of size `mpi_size * over_decom_factor + 1` indexed into `table`, such
 * that `offsets[i * mpi_size + k]` is the start index of rows needs to be sent to rank `k` during
 * batch `i`.
 * @param[in] over_decom_factor Number of batches.
 * @param[out] string_send_offsets Vector with shape `(num_batches, num_columns, mpi_size + 1)`,
 * such that `string_send_offsets[i,j,k]` representing the start index in the char subcolumn of
 * column `j` that needs to be sent to rank `k` during batch `i`.
 * @param[out] string_recv_offsets Vector with shape `(num_batches, num_columns, mpi_size + 1)`,
 * such that `string_recv_offsets[i,j,k]` representing the start index in the char subcolumn of
 * column `j` that receives data from rank `k` during batch `i`.
 */
void gather_string_offsets(cudf::table_view table,
                           vector<cudf::size_type> const &offsets,
                           const int over_decom_factor,
                           vector<vector<vector<cudf::size_type>>> &string_send_offsets,
                           vector<vector<vector<int64_t>>> &string_recv_offsets,
                           Communicator *communicator)
{
  int mpi_size = communicator->mpi_size;
  string_send_offsets.resize(over_decom_factor);
  string_recv_offsets.resize(over_decom_factor);

  for (int ibatch = 0; ibatch < over_decom_factor; ibatch++) {
    size_t start_idx = ibatch * mpi_size;
    size_t end_idx   = (ibatch + 1) * mpi_size + 1;
    thrust::device_vector<cudf::size_type> d_offset(&offsets[start_idx], &offsets[end_idx]);

    for (cudf::size_type icol = 0; icol < table.num_columns(); icol++) {
      // 1. If not a string column, push an empty vector
      cudf::data_type dtype = table.column(icol).type();
      if (dtype.id() != cudf::type_id::STRING) {
        string_send_offsets[ibatch].emplace_back();
        string_recv_offsets[ibatch].emplace_back();
        continue;
      } else {
        string_send_offsets[ibatch].emplace_back(mpi_size + 1);
        string_recv_offsets[ibatch].emplace_back(mpi_size + 1);
      }

      // 2. Gather `string_send_offsets` from offset subcolumn and `offsets`
      thrust::device_vector<cudf::size_type> d_string_send_offsets(mpi_size + 1);
      thrust::gather(rmm::exec_policy(rmm::cuda_stream_default)->on(0),
                     d_offset.begin(),
                     d_offset.end(),
                     thrust::device_ptr<const cudf::size_type>(
                       table.column(icol).child(0).head<cudf::size_type>()),
                     d_string_send_offsets.begin());
      CUDA_RT_CALL(cudaMemcpy(string_send_offsets[ibatch][icol].data(),
                              thrust::raw_pointer_cast(d_string_send_offsets.data()),
                              (mpi_size + 1) * sizeof(cudf::size_type),
                              cudaMemcpyDeviceToHost));

      // 3. Communicate string_send_offsets and receive string_recv_offsets
      communicate_sizes(
        string_send_offsets[ibatch][icol], string_recv_offsets[ibatch][icol], communicator);
    }
  }
}

/**
 * Calculate the string size of each row.
 *
 * Note: This function is the reverse of `calculate_string_offsets_from_sizes`.
 *
 * @param[in] input_table Table for which the string sizes are calculated.
 * @param[out] output_sizes Vector of size `num_columns`, where `output_sizes[j]` stores the string
 * size of each row in column `j`.
 */
void calculate_string_sizes_from_offsets(cudf::table_view input_table,
                                         vector<rmm::device_uvector<cudf::size_type>> &output_sizes)
{
  output_sizes.clear();

  for (cudf::size_type icol = 0; icol < input_table.num_columns(); icol++) {
    cudf::column_view input_column = input_table.column(icol);
    if (input_column.type().id() != cudf::type_id::STRING) {
      output_sizes.emplace_back(0, rmm::cuda_stream_default);
      continue;
    }

    output_sizes.emplace_back(input_column.size(), rmm::cuda_stream_default);

    // Assume the first entry of the offset subcolumn is always 0
    thrust::adjacent_difference(
      // rmm::exec_policy(rmm::cuda_stream_default)->on(0),
      thrust::device_ptr<const cudf::size_type>(
        input_column.child(0).begin<const cudf::size_type>() + 1),
      thrust::device_ptr<const cudf::size_type>(input_column.child(0).end<cudf::size_type>()),
      thrust::device_ptr<cudf::size_type>(output_sizes[icol].data()));
  }
}

/**
 * Calculate string offsets from sizes.
 *
 * Note: This function is the reverse of `calculate_string_sizes_from_offsets`.
 *
 * @param[out] output_table Calculated offsets will be stored in the string columns of
 * `output_table`.
 * @param[in] input_sizes Vector of size `num_columns`, where `input_sizes[j]` stores the string
 * size of each row in column `j`.
 */
void calculate_string_offsets_from_sizes(
  cudf::mutable_table_view output_table,
  vector<rmm::device_uvector<cudf::size_type>> const &input_sizes)
{
  for (cudf::size_type icol = 0; icol < output_table.num_columns(); icol++) {
    cudf::mutable_column_view output_column = output_table.column(icol);
    if (output_column.type().id() != cudf::type_id::STRING) continue;

    cudf::size_type nrows              = output_column.size();
    const cudf::size_type *sizes_start = input_sizes[icol].data();
    const cudf::size_type *sizes_end   = sizes_start + nrows;
    thrust::inclusive_scan(
      // rmm::exec_policy(rmm::cuda_stream_default)->on(0),
      thrust::device_ptr<const cudf::size_type>(sizes_start),
      thrust::device_ptr<const cudf::size_type>(sizes_end),
      thrust::device_ptr<cudf::size_type>(
        static_cast<cudf::size_type *>(output_column.child(0).head())) +
        1);
    CUDA_TRY(cudaMemsetAsync(output_column.child(0).head(), 0, sizeof(cudf::size_type), 0));
  }
}

/**
 * Helper function for allocating the receive buffer of string sizes.
 */
void allocate_string_sizes_receive_buffer(
  cudf::table_view input_table,
  int over_decom_factor,
  vector<vector<int64_t>> recv_offsets,
  vector<vector<rmm::device_uvector<cudf::size_type>>> &string_sizes_recv)
{
  string_sizes_recv.clear();
  string_sizes_recv.resize(over_decom_factor);

  for (int ibatch = 0; ibatch < over_decom_factor; ibatch++) {
    for (cudf::size_type icol = 0; icol < input_table.num_columns(); icol++) {
      if (input_table.column(icol).type().id() != cudf::type_id::STRING) {
        string_sizes_recv[ibatch].emplace_back(0, rmm::cuda_stream_default);
      } else {
        string_sizes_recv[ibatch].emplace_back(recv_offsets[ibatch].back(),
                                               rmm::cuda_stream_default);
      }
    }
  }
}
