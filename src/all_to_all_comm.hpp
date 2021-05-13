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

#pragma once

#include "communicator.hpp"
#include "compression.hpp"

#include <cascaded.hpp>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <mpi.h>

#include <cstdint>
#include <type_traits>
#include <vector>

enum COMM_TAGS { placeholder_tag, exchange_size_tag };

/**
 * Usage: mpi_dtype_from_c_type<input_type>() returns the MPI datatype corresponding to a native C
 * type "input_type".
 *
 * For example, mpi_dtype_from_c_type<int>() would return MPI_INT32_T.
 */
template <typename c_type>
MPI_Datatype mpi_dtype_from_c_type()
{
  MPI_Datatype mpi_dtype;
  if (std::is_same<c_type, int8_t>::value)
    mpi_dtype = MPI_INT8_T;
  else if (std::is_same<c_type, uint8_t>::value)
    mpi_dtype = MPI_UINT8_T;
  else if (std::is_same<c_type, int16_t>::value)
    mpi_dtype = MPI_INT16_T;
  else if (std::is_same<c_type, uint16_t>::value)
    mpi_dtype = MPI_UINT16_T;
  else if (std::is_same<c_type, int32_t>::value)
    mpi_dtype = MPI_INT32_T;
  else if (std::is_same<c_type, uint32_t>::value)
    mpi_dtype = MPI_UINT32_T;
  else if (std::is_same<c_type, int64_t>::value)
    mpi_dtype = MPI_INT64_T;
  else if (std::is_same<c_type, uint64_t>::value)
    mpi_dtype = MPI_UINT64_T;
  else if (std::is_same<c_type, float>::value)
    mpi_dtype = MPI_FLOAT;
  else if (std::is_same<c_type, double>::value)
    mpi_dtype = MPI_DOUBLE;

  return mpi_dtype;
}

class CommunicationGroup {
 public:
  /**
   * CommunicationGroup represents a group of ranks for all-to-all communication.
   *
   * The group of ranks is determined by the following two filters on MPI_COMM_WORLD:
   *
   * First, all ranks are partitioned into "grid" with size *grid_size*.
   *   {{0,1,2,...,grid_size-1},
   *    {grid_size,grid_size+1,grid_size+2,...,grid_size*2-1},
   *    ...
   *   }
   * *grid_size* must divide `mpi_size`.
   *
   * Second, for each grid, ranks are sampled with spacing *stride*. For example, if *stride* is 2,
   * {0,2,4,6,8} is a group, while {1,3,5,7,9} is another group. *stride* must divide *grid_size*.
   *
   * For example, if there are 16 ranks, with grid_size 8, and stride 2, we have the following
   * groups: {0,2,4,6}, {1,3,5,7}, {8,10,12,14}, {9,11,13,15}.
   */
  CommunicationGroup(int grid_size, int stride = 1) : grid_size(grid_size), stride(stride)
  {
    assert(grid_size % stride == 0 && "Group size should be a multiple of stride");
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    group_start = mpi_rank / grid_size * grid_size + mpi_rank % stride;
  }

  // Get the size of the communication group
  int size() { return grid_size / stride; }

  // Get the global MPI rank in MPI_COMM_WORLD corresponding to local index in the group
  int get_global_rank(int local_idx) { return group_start + local_idx * stride; }

  // Get the local index in the group corresponding to the current rank
  int get_local_idx() { return (mpi_rank - group_start) / stride; }

 private:
  int mpi_rank;
  int group_start;
  int grid_size;
  int stride;
};

/**
 * Communicate number of elements recieved from each rank during all-to-all communication.
 *
 * Note: This function needs to be called collectively by all ranks in *comm_group*.
 *
 * @param[in] send_offset Vector of length `comm_group_size + 1` such that `send_offset[i+1] -
 * send_offset[i]` is the number of elements sent from the current rank to local rank `i` during the
 * all-to-all communication.
 * @param[out] recv_offset Vector of length `comm_group_size + 1` such that `recv_offset[i+1] -
 * recv_offset[i]` is the number of elements received from local rank `i` during the all-to-all
 * communication. The vector will be resized in this function and does not need to be preallocated.
 */
void communicate_sizes(std::vector<int64_t> const &send_offset,
                       std::vector<int64_t> &recv_offset,
                       CommunicationGroup comm_group,
                       Communicator *communicator);

void communicate_sizes(std::vector<cudf::size_type> const &send_offset,
                       std::vector<int64_t> &recv_offset,
                       CommunicationGroup comm_group,
                       Communicator *communicator);

void warmup_all_to_all(Communicator *communicator);

struct AllToAllCommBuffer {
  // the buffer to be all-to-all communicated
  const void *send_buffer;
  // the receive buffer for all-to-all communication
  void *recv_buffer;
  // vector of size `comm_group_size + 1`, the start index of items in `send_buffer` to be sent to
  // each rank
  std::vector<int64_t> send_offsets;
  // vector of size `comm_group_size + 1`, the start index of items in `recv_buffer` to receive data
  // from each rank
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
  // vector of size `comm_group_size + 1`, the start byte in `compressed_send_buffer` to be sent to
  // each rank
  std::vector<int64_t> compressed_send_offsets;
  // vector of size `comm_group_size + 1`, the start byte in `compressed_recv_buffer` to receive
  // data from each rank
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
 * @param[in] send_offset Vector of size `comm_group_size + 1` such that `send_offset[i]` represents
 * the start index of `input` to be sent to local rank `i`.
 * @param[in] recv_offset Vector of size `comm_group_size + 1` such that `recv_offset[i]` represents
 * the start index of `output` to receive data from local rank `i`.
 * @param[in] string_send_offsets Vector with shape `(num_columns, comm_group_size + 1)`, such
 * that `string_send_offsets[j,k]` representing the start index in the char subcolumn of column
 * `j` that needs to be sent to local rank `k`. The helper function `gather_string_offsets` can be
 * used to generate this field.
 * @param[in] string_recv_offsets Vector with shape `(num_columns, comm_group_size + 1)`, such
 * that `string_recv_offsets[j,k]` representing the start index in the char subcolumn of column
 * `j` that receives data from local rank `k`. The helper function `gather_string_offsets` can be
 * used to generate this field.
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
 * Perform all-to-all communication according to plans.
 *
 * Note: If the communicator supports grouping by batches, this call is nonblocking and should
 * be enclosed by `communicator->start()` and `communicator->stop()`.
 *
 * This function needs to be called collectively by all ranks in *comm_group*.
 *
 * @param[in] all_to_all_comm_buffers Plans for all-to-all communication, generated by
 * `append_to_all_to_all_comm_buffers`. Note that the send/recv offsets specified must be compatible
 * with *comm_group*.
 * @param[in] communicator An instance of `Communicator` used for communication.
 * @param[in] include_current_rank If true, this function will send the partition destined to the
 * current rank.
 * @param[in] preallocated_pinned_buffer Preallocated page-locked host buffer with size at least
 * `comm_group_size * sizeof(size_t)`, used for holding the compressed sizes.
 */
void all_to_all_comm(std::vector<AllToAllCommBuffer> &all_to_all_comm_buffers,
                     CommunicationGroup comm_group,
                     Communicator *communicator,
                     bool include_current_rank        = true,
                     bool report_timing               = false,
                     void *preallocated_pinned_buffer = nullptr);

/**
 * Actions to be performed after all-to-all communication is finished.
 *
 * Note: The arguments of this function need to match those of `all_to_all_comm`.
 */
void postprocess_all_to_all_comm(std::vector<AllToAllCommBuffer> &all_to_all_comm_buffers,
                                 CommunicationGroup comm_group,
                                 Communicator *communicator,
                                 bool include_current_rank = true,
                                 bool report_timing        = false);

/**
 * High-level interface for all-to-all communicating a cuDF table.
 */
class AllToAllCommunicator {
  // Note: General stategy for the string columns during all-to-all communication
  // Each string column in cuDF consists of two subcolumns: a char subcolumn and an offset
  // subcolumn. For the char subcolumn, we need to first gather the offsets in this string
  // subcolumn of all ranks by using `gather_string_offsets`, and then it can be all-to-all
  // communicated using the gathered offsets. For the offset subcolumn, we can first calculate the
  // sizes of all rows by calculating the adjacent differences. Then, the sizes are all-to-all
  // communicated. Once the all-to-all communication finishes, on target rank we can reconstruct
  // the offset subcolumn by using a scan on sizes.

 public:
  /**
   * Note: This custructor needs to be called collectively for all ranks in *comm_group*.
   *
   * @param[in] input_table Table to be all-to-all communicated.
   * @param[in] offsets Vector of length `comm_group_size + 1`, indexed into *input_table*,
   * representing the start/end row index to send to each rank.
   * @param[in] compression_options Vector of length equal to the number of columns, indicating
   * whether/how each column needs to be compressed before communication.
   * @param[in] explicit_copy_to_current_rank If true, rows destined to the current rank are copied
   * using explicit device-to-device memory copy instead of going through communicator.
   */
  AllToAllCommunicator(cudf::table_view input_table,
                       std::vector<cudf::size_type> offsets,
                       CommunicationGroup comm_group,
                       Communicator *communicator,
                       std::vector<ColumnCompressionOptions> compression_options,
                       bool explicit_copy_to_current_rank = false);

  /**
   * This variant of *AllToAllCommunicator* uses a communication group with all ranks and
   * stride 1.
   */
  AllToAllCommunicator(cudf::table_view input_table,
                       std::vector<cudf::size_type> offsets,
                       Communicator *communicator,
                       std::vector<ColumnCompressionOptions> compression_options,
                       bool explicit_copy_to_current_rank = false);

  AllToAllCommunicator(const AllToAllCommunicator &) = delete;
  AllToAllCommunicator &operator=(const AllToAllCommunicator &) = delete;
  AllToAllCommunicator(AllToAllCommunicator &&)                 = default;

  /**
   * Allocate the tables after all-to-all communication.
   *
   * Note: This function uses the default stream for allocation and is synchronous to the host
   * thread.
   *
   * @return Allocated table.
   */
  std::unique_ptr<cudf::table> allocate_communicated_table();

  /**
   * Launch the all-to-all communication.
   *
   * Note: This function needs to be called collectively by all ranks in *comm_group*.
   * Note: This function will block the host thread until the communication is completed.
   *
   * @param[in] communicated_table Preallocated table for receiving incoming data.
   * @param[in] preallocated_pinned_buffer Preallocated page-locked host buffer with size at least
   * `comm_group_size * sizeof(size_t)`, used for holding the compressed sizes.
   */
  void launch_communication(cudf::mutable_table_view communicated_table,
                            bool report_timing               = false,
                            void *preallocated_pinned_buffer = nullptr);

 private:
  cudf::table_view input_table;
  CommunicationGroup comm_group;
  Communicator *communicator;
  bool explicit_copy_to_current_rank;
  std::vector<cudf::size_type> send_offsets;
  // Start row index in the communicated table to receive data from each rank.
  std::vector<int64_t> recv_offsets;
  // `string_send_offsets[j, k]` represents the start index into char subcolumn to be sent to local
  // rank `k` for column `j`. If column `j` is not a string column, `string_send_offsets[j]` will be
  // an empty vector. Otherwise, `string_send_offsets[j]` will be a vector of length
  // `comm_group_size + 1`.
  std::vector<std::vector<cudf::size_type>> string_send_offsets;
  // `string_recv_offsets[j, k]` represents the start index into char subcolumn
  // to receive data from local rank `k` for column `j`. If column `j` is not a string column,
  // `string_recv_offsets[j]` will be an empty vector. Otherwise, `string_recv_offsets[j]`
  // will be a vector of length `comm_group_size + 1`.
  std::vector<std::vector<int64_t>> string_recv_offsets;
  std::vector<rmm::device_uvector<cudf::size_type>> string_sizes_to_send;
  std::vector<rmm::device_uvector<cudf::size_type>> string_sizes_received;
  std::vector<ColumnCompressionOptions> compression_options;
};
