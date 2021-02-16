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

#include <cascaded.hpp>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <vector>

struct AllToAllCommBuffer {
  // the buffer to be all-to-all communicated
  const void *send_buffer;
  // the receive buffer for all-to-all communication
  void *recv_buffer;
  // vector of size `mpi_size + 1`, the start index of items in `send_buffer` to be sent to
  // each rank
  std::vector<int64_t> send_offsets;
  // vector of size `mpi_size + 1`, the start index of items in `recv_buffer` to receive data from
  // each rank
  std::vector<int64_t> recv_offsets;
  // data type of each element
  cudf::data_type dtype;
  // the compression method used
  CompressionMethod compression_method;
  // cascaded compression format
  nvcompCascadedFormatOpts cascaded_format;
  // compressed `send_buffer` to be all-to-all communicated
  rmm::device_buffer compressed_send_buffer;
  // the receive buffer for the compressed data
  rmm::device_buffer compressed_recv_buffer;
  // vector of size `mpi_size + 1`, the start byte in `compressed_send_buffer` to be sent to each
  // rank
  std::vector<int64_t> compressed_send_offsets;
  // vector of size `mpi_size + 1`, the start byte in `compressed_recv_buffer` to receive data from
  // each rank
  std::vector<int64_t> compressed_recv_offsets;

  AllToAllCommBuffer(const void *send_buffer,
                     void *recv_buffer,
                     std::vector<int64_t> send_offsets,
                     std::vector<int64_t> recv_offsets,
                     cudf::data_type dtype,
                     CompressionMethod compression_method,
                     nvcompCascadedFormatOpts cascaded_format)
    : send_buffer(send_buffer),
      recv_buffer(recv_buffer),
      send_offsets(send_offsets),
      recv_offsets(recv_offsets),
      dtype(dtype),
      compression_method(compression_method),
      cascaded_format(cascaded_format)
  {
  }
};

/**
 * Generate plans for all-to-all communication.
 *
 * Note: This function does not perform the actual communication. It simply puts required
 * information about all-to-all communication (e.g. the send buffer, the receive buffer, offsets,
 * whether to use compression etc.) into `all_to_all_comm_buffers`.
 *
 * @param[in] input Table to be all-to-all communicated.
 * @param[in] output Table after all-to-all communication. This argument needs to be preallocated.
 * The helper function `allocate_communicated_table` can be used to allocate this table.
 * @param[in] send_offset Vector of size `mpi_size + 1` such that `send_offset[i]` represents the
 * start index of `input` to be sent to rank `i`.
 * @param[in] recv_offset Vector of size `mpi_size + 1` such that `recv_offset[i]` represents the
 * start index of `output` to receive data from rank `i`.
 * @param[in] string_send_offsets Vector with shape `(num_columns, mpi_size + 1)`, such
 * that `string_send_offsets[j,k]` representing the start index in the char subcolumn of column
 * `j` that needs to be sent to rank `k`, for the current batch. The helper function
 * `gather_string_offsets` can be used to generate this field.
 * @param[in] string_recv_offsets Vector with shape `(num_columns, mpi_size + 1)`, such
 * that `string_recv_offsets[j,k]` representing the start index in the char subcolumn of column
 * `j` that receives data from rank `k`, for the current batch. The helper function
 * `gather_string_offsets` can be used to generate this field.
 * @param[in] string_sizes_send String sizes of each row for all string columns. The helper function
 * `calculate_string_sizes_from_offsets` can be used to generate this field.
 * @param[in] string_sizes_recv Receive buffers for string sizes. This argument needs to be
 * preallocated. The helper function `allocate_string_sizes_receive_buffer` can be used for
 * allocating the buffers.
 * @param[out] all_to_all_comm_buffers Each element in this vector represents a buffer that needs to
 * be all-to-all communicated.
 * @param[in] compression_options Vector of length equal to the number of columns in *input*,
 * indicating whether/how each column needs to be compressed before communication.
 */
void append_to_all_to_all_comm_buffers(
  cudf::table_view input,
  cudf::mutable_table_view output,
  std::vector<cudf::size_type> const &send_offsets,
  std::vector<int64_t> const &recv_offsets,
  std::vector<std::vector<cudf::size_type>> const &string_send_offsets,
  std::vector<std::vector<int64_t>> const &string_recv_offsets,
  std::vector<rmm::device_uvector<cudf::size_type>> const &string_sizes_send,
  std::vector<rmm::device_uvector<cudf::size_type>> &string_sizes_recv,
  std::vector<AllToAllCommBuffer> &all_to_all_comm_buffers,
  std::vector<ColumnCompressionOptions> const &compression_options);

void append_to_all_to_all_comm_buffers(cudf::table_view input,
                                       cudf::mutable_table_view output,
                                       std::vector<cudf::size_type> const &send_offsets,
                                       std::vector<int64_t> const &recv_offsets,
                                       std::vector<AllToAllCommBuffer> &all_to_all_comm_buffers,
                                       std::vector<ColumnCompressionOptions> compression_options);

/**
 * Perform all-to-all communication of a single batch according to plans.
 *
 * Note: If the communicator supports grouping by batches, this call is nonblocking and should
 * be enclosed by `communicator->start()` and `communicator->stop()`.
 *
 * This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] all_to_all_comm_buffers Plans for all-to-all communication, generated by
 * `append_to_all_to_all_comm_buffers`.
 * @param[in] communicator An instance of `Communicator` used for communication.
 * @param[in] include_self If true, this function will send the partition destined to the current
 * rank.
 * @param[in] preallocated_pinned_buffer Preallocated page-locked host buffer with size at least
 * `mpi_size * sizeof(size_t)`, used for holding the compressed sizes.
 */
void all_to_all_comm(std::vector<AllToAllCommBuffer> &all_to_all_comm_buffers,
                     Communicator *communicator,
                     bool include_self                = true,
                     bool report_timing               = false,
                     void *preallocated_pinned_buffer = nullptr);

/**
 * Actions to be performed after all-to-all communication is finished.
 *
 * Note: The arguments of this function need to match those of `all_to_all_comm`.
 */
void postprocess_all_to_all_comm(std::vector<AllToAllCommBuffer> &all_to_all_comm_buffers,
                                 Communicator *communicator,
                                 bool include_self  = true,
                                 bool report_timing = false);

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
 * @param[in] columns_in_common is a vector of pairs of column indices into
 * `left` and `right`, respectively, that are "in common". For "common"
 * columns, only a single output column will be produced, which is gathered
 * from `left_on` columns. Else, for every column in `left_on` and `right_on`,
 * an output column will be produced.  For each of these pairs (L, R), L
 * should exist in `left_on` and R should exist in `right_on`.
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
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const &columns_in_common,
  Communicator *communicator,
  std::vector<ColumnCompressionOptions> left_compression_options,
  std::vector<ColumnCompressionOptions> right_compression_options,
  int over_decom_factor            = 1,
  bool report_timing               = false,
  void *preallocated_pinned_buffer = nullptr);
