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

#ifndef __DISTRIBUTED_JOIN
#define __DISTRIBUTED_JOIN

#include <utility>
#include <vector>
#include <memory>

#include <cudf/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/hashing.hpp>

#include "error.cuh"
#include "communicator.h"

/**
 * Top level interface for distributed inner join.
 *
 * This function should be called collectively by all processes in MPI_COMM_WORLD. All arguments are
 * significant for all ranks.
 *
 * The argument `left` and `right` are the left table and the right table distributed on each rank.
 * In other word, the left (right) table to be joined is the concatenation of `left` (`right`) on
 * all ranks. If the whole tables reside on a single rank, you should use *distribute_table* to
 * distribute the table before calling this function.
 *
 * @param[in] left The left table distributed on each rank.
 * @param[in] right The right table distributed on each rank.
 * @param[in] left_on The column indices from `left` to join on.
 * The column from `left` indicated by `left_on[i]` will be compared against the column
 * from `right` indicated by `right_on[i]`.
 * @param[in] right_on The column indices from `right` to join on.
 * The column from `right` indicated by `right_on[i]` will be compared against the column
 * from `left` indicated by `left_on[i]`.
 * @param[in] columns_in_common is a vector of pairs of column indices into
 * `left` and `right`, respectively, that are "in common". For "common"
 * columns, only a single output column will be produced, which is gathered
 * from `left_on` columns. Else, for every column in `left_on` and `right_on`,
 * an output column will be produced.  For each of these pairs (L, R), L
 * should exist in `left_on` and R should exist in `right_on`.
 * @param[in] communicator An instance of `Communicator` used for communication.
 * @param[in] over_decom_factor Over-decomposition factor.
 * @return Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`. The resulting table will be joined columns of
 * `left(including common columns)+right(excluding common columns)`. The join result is the
 * concatenation of the returned tables on all ranks.
 */
std::unique_ptr<cudf::experimental::table>
distributed_inner_join(
    cudf::table_view const& left,
    cudf::table_view const& right,
    std::vector<cudf::size_type> const& left_on,
    std::vector<cudf::size_type> const& right_on,
    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
    Communicator *communicator,
    int over_decom_factor=1)
{
    if (over_decom_factor == 1) {
        // @TODO: If over_decom_factor is 1, there is no opportunity for overlapping. Therefore,
        // we can get away with using just one thread.
    }

    int mpi_size = communicator->mpi_size;

    // @TODO: The returned partitions need to be handled differently once
    // https://github.com/rapidsai/cudf/pull/3636 is merged.
    auto tmp_left = cudf::hash_partition(left, left_on, mpi_size * over_decom_factor);
    auto tmp_right = cudf::hash_partition(right, right_on, mpi_size * over_decom_factor);

    // @TODO: The follow section is not need once https://github.com/rapidsai/cudf/pull/3636 is
    // merged.
    // {{{
    std::vector<cudf::size_type> left_offset {0};
    std::vector<cudf::table_view> tables_to_concat;
    for (auto &table_ptr : tmp_left) {
        left_offset.push_back(left_offset.back() + table_ptr->num_rows());
        tables_to_concat.push_back(table_ptr->view());
    }
    auto hashed_left = cudf::experimental::concatenate(tables_to_concat);

    std::vector<cudf::size_type> right_offset {0};
    tables_to_concat.clear();
    for (auto &table_ptr : tmp_right) {
        right_offset.push_back(right_offset.back() + table_ptr->num_rows());
        tables_to_concat.push_back(table_ptr->view());
    }
    auto hashed_right = cudf::experimental::concatenate(tables_to_concat);
    // }}}

    return std::unique_ptr<cudf::experimental::table>(nullptr);
}

#endif  // __DISTRIBUTED_JOIN
