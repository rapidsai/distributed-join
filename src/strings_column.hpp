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

#include "communicator.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <vector>

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
void gather_string_offsets(
  cudf::table_view table,
  std::vector<cudf::size_type> const &offsets,
  const int over_decom_factor,
  std::vector<std::vector<std::vector<cudf::size_type>>> &string_send_offsets,
  std::vector<std::vector<std::vector<int64_t>>> &string_recv_offsets,
  Communicator *communicator);

/**
 * Calculate the string size of each row.
 *
 * Note: This function is the reverse of `calculate_string_offsets_from_sizes`.
 *
 * @param[in] input_table Table for which the string sizes are calculated.
 * @param[out] output_sizes Vector of size `num_columns`, where `output_sizes[j]` stores the string
 * size of each row in column `j`.
 */
void calculate_string_sizes_from_offsets(
  cudf::table_view input_table, std::vector<rmm::device_uvector<cudf::size_type>> &output_sizes);

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
  std::vector<rmm::device_uvector<cudf::size_type>> const &input_sizes);

/**
 * Helper function for allocating the receive buffer of string sizes.
 */
void allocate_string_sizes_receive_buffer(
  cudf::table_view input_table,
  int over_decom_factor,
  std::vector<std::vector<int64_t>> recv_offsets,
  std::vector<std::vector<rmm::device_uvector<cudf::size_type>>> &string_sizes_recv);
