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
#include "compression.hpp"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <vector>

/**
 * Top level interface for distributed inner join.
 *
 * This function should be called collectively by all processes in MPI_COMM_WORLD. All arguments
 * are significant for all ranks.
 *
 * The argument `left` and `right` are the left table and the right table distributed on each
 * rank. In other word, the left (right) table to be joined is the concatenation of `left`
 * (`right`) on all ranks. If the whole tables reside on a single rank, you should use
 * *distribute_table* to distribute the table before calling this function.
 *
 * @param[in] left The left table distributed on each rank.
 * @param[in] right The right table distributed on each rank.
 * @param[in] left_on The column indices from `left` to join on.
 * The column from `left` indicated by `left_on[i]` will be compared against the column
 * from `right` indicated by `right_on[i]`.
 * @param[in] right_on The column indices from `right` to join on.
 * The column from `right` indicated by `right_on[i]` will be compared against the column
 * from `left` indicated by `left_on[i]`.
 * @param[in] communicator An instance of `Communicator` used for communication.
 * @param[in] left_compression_options Vector of length equal to the number of columns in *left*,
 * indicating whether/how each column of the left table needs to be compressed before communication.
 * @param[in] right_compression_options Vector of length equal to the number of columns in *right*,
 * indicating whether/how each column of the right table needs to be compressed before
 * communication.
 * @param[in] over_decom_factor Over-decomposition factor used for overlapping computation and
 * communication.
 * @param[in] report_timing Whether collect and print timing.
 * @param[in] preallocated_pinned_buffer Preallocated page-locked host buffer with size at least
 * `mpi_size * sizeof(size_t)`, used for holding the compressed sizes.
 * @return Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`. The resulting table will be joined columns of
 * `left(including common columns)+right(excluding common columns)`. The join result is the
 * concatenation of the returned tables on all ranks.
 */
std::unique_ptr<cudf::table> distributed_inner_join(
  cudf::table_view const &left,
  cudf::table_view const &right,
  std::vector<cudf::size_type> const &left_on,
  std::vector<cudf::size_type> const &right_on,
  Communicator *communicator,
  std::vector<ColumnCompressionOptions> left_compression_options,
  std::vector<ColumnCompressionOptions> right_compression_options,
  int over_decom_factor            = 1,
  bool report_timing               = false,
  void *preallocated_pinned_buffer = nullptr);
