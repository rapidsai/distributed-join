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

#include "error.hpp"

#include <cascaded.hpp>
#include <nvcomp.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>
#include <simt/type_traits>

#include <mpi.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

using std::vector;

enum class CompressionMethod { none, cascaded, lz4 };

/* A structure outlining how to compress a column */
struct ColumnCompressionOptions {
  CompressionMethod compression_method;
  nvcompCascadedFormatOpts cascaded_format;
  vector<ColumnCompressionOptions> children_compression_options;

  ColumnCompressionOptions() = default;

  ColumnCompressionOptions(CompressionMethod compression_method)
    : compression_method(compression_method)
  {
  }

  ColumnCompressionOptions(CompressionMethod compression_method,
                           nvcompCascadedFormatOpts cascaded_format)
    : compression_method(compression_method), cascaded_format(cascaded_format)
  {
  }

  ColumnCompressionOptions(CompressionMethod compression_method,
                           nvcompCascadedFormatOpts cascaded_format,
                           vector<ColumnCompressionOptions> children_compression_options)
    : compression_method(compression_method),
      cascaded_format(cascaded_format),
      children_compression_options(children_compression_options)
  {
  }
};

struct compression_functor {
  /**
   * Compress a vector of buffers using cascaded compression.
   *
   * @param[in] uncompressed_data Input buffers to be compressed.
   * @param[in] uncompressed_counts Number of elements to be compressed for each buffer in
   * *uncompressed_data*. Note that in general this is different from the size of the buffer.
   * @param[out] compressed_data Compressed buffers after cascaded compression. This argument does
   * not need to be preallocated.
   * @param[out] compressed_sizes Number of bytes for each buffer in *compressed_data*.
   * @param[in] streams CUDA streams used for the compression kernels.
   */
  template <
    typename T,
    std::enable_if_t<!cudf::is_timestamp_t<T>::value && !cudf::is_duration_t<T>::value> * = nullptr>
  void operator()(vector<const void *> const &uncompressed_data,
                  vector<cudf::size_type> const &uncompressed_counts,
                  vector<rmm::device_buffer> &compressed_data,
                  size_t *compressed_sizes,
                  vector<rmm::cuda_stream_view> const &streams,
                  nvcompCascadedFormatOpts cascaded_format)
  {
    size_t npartitions = uncompressed_counts.size();
    compressed_data.resize(npartitions);

    vector<rmm::device_buffer> temp_spaces(npartitions);
    vector<size_t> temp_sizes(npartitions);

    for (size_t ipartition = 0; ipartition < npartitions; ipartition++) {
      if (uncompressed_counts[ipartition] == 0) {
        compressed_sizes[ipartition] = 0;
        continue;
      }

      nvcomp::CascadedCompressor<T> compressor(
        static_cast<const T *>(uncompressed_data[ipartition]),
        uncompressed_counts[ipartition],
        cascaded_format.num_RLEs,
        cascaded_format.num_deltas,
        cascaded_format.use_bp);

      temp_sizes[ipartition]  = compressor.get_temp_size();
      temp_spaces[ipartition] = rmm::device_buffer(temp_sizes[ipartition], streams[ipartition]);
      compressed_sizes[ipartition] =
        compressor.get_max_output_size(temp_spaces[ipartition].data(), temp_sizes[ipartition]);
      compressed_data[ipartition] =
        rmm::device_buffer(compressed_sizes[ipartition], streams[ipartition]);
    }

    for (size_t ipartition = 0; ipartition < npartitions; ipartition++) {
      if (uncompressed_counts[ipartition] == 0) continue;

      nvcomp::CascadedCompressor<T> compressor(
        static_cast<const T *>(uncompressed_data[ipartition]),
        uncompressed_counts[ipartition],
        cascaded_format.num_RLEs,
        cascaded_format.num_deltas,
        cascaded_format.use_bp);

      // Set the output buffer to 0 to get away a bug in nvcomp
      CUDA_RT_CALL(cudaMemsetAsync(compressed_data[ipartition].data(),
                                   0,
                                   compressed_sizes[ipartition],
                                   streams[ipartition].value()));

      compressor.compress_async(temp_spaces[ipartition].data(),
                                temp_sizes[ipartition],
                                compressed_data[ipartition].data(),
                                &compressed_sizes[ipartition],
                                streams[ipartition].value());
    }
  }

  template <
    typename T,
    std::enable_if_t<cudf::is_timestamp_t<T>::value || cudf::is_duration_t<T>::value> * = nullptr>
  void operator()(vector<const void *> const &uncompressed_data,
                  vector<cudf::size_type> const &uncompressed_counts,
                  vector<rmm::device_buffer> &compressed_data,
                  size_t *compressed_sizes,
                  vector<rmm::cuda_stream_view> const &streams,
                  nvcompCascadedFormatOpts cascaded_format)
  {
    // If the data type is duration or time, use the corresponding arithmetic type
    operator()<typename T::rep>(uncompressed_data,
                                uncompressed_counts,
                                compressed_data,
                                compressed_sizes,
                                streams,
                                cascaded_format);
  }
};

struct decompression_functor {
  /**
   * Decompress a vector of buffers previously compressed by `compression_functor{}.operator()`.
   *
   * @param[in] compressed_data Vector of input data to be decompressed.
   * @param[in] compressed_sizes Sizes of *compressed_data* in bytes.
   * @param[out] outputs Decompressed outputs. This argument needs to be preallocated.
   * @param[in] expected_output_counts Expected number of elements in the decompressed buffers.
   */
  template <
    typename T,
    std::enable_if_t<!cudf::is_timestamp_t<T>::value && !cudf::is_duration_t<T>::value> * = nullptr>
  void operator()(vector<const void *> const &compressed_data,
                  vector<int64_t> const &compressed_sizes,
                  vector<void *> const &outputs,
                  vector<int64_t> const &expected_output_counts,
                  vector<rmm::cuda_stream_view> const &streams)
  {
    size_t npartitions = compressed_sizes.size();

    vector<rmm::device_buffer> temp_spaces(npartitions);
    vector<size_t> temp_sizes(npartitions);

    // nvcomp::Decompressor objects are reused in the two passes below since nvcomp::Decompressor
    // constructor can be synchrnous to the host thread. new operator is used instead of
    // std::vector because the copy constructor in nvcomp::Decompressor is deleted.

    nvcomp::Decompressor<T> **decompressors = new nvcomp::Decompressor<T> *[npartitions];
    memset(decompressors, 0, sizeof(nvcomp::Decompressor<T> *) * npartitions);

    for (size_t ipartition = 0; ipartition < npartitions; ipartition++) {
      if (expected_output_counts[ipartition] == 0) continue;

      decompressors[ipartition] = new nvcomp::Decompressor<T>(
        compressed_data[ipartition], compressed_sizes[ipartition], streams[ipartition].value());

      const size_t output_count = decompressors[ipartition]->get_num_elements();
      assert(output_count == expected_output_counts[ipartition]);

      temp_sizes[ipartition]  = decompressors[ipartition]->get_temp_size();
      temp_spaces[ipartition] = rmm::device_buffer(temp_sizes[ipartition], streams[ipartition]);
    }

    for (int ipartition = 0; ipartition < npartitions; ipartition++) {
      if (expected_output_counts[ipartition] == 0) continue;

      decompressors[ipartition]->decompress_async(temp_spaces[ipartition].data(),
                                                  temp_sizes[ipartition],
                                                  static_cast<T *>(outputs[ipartition]),
                                                  expected_output_counts[ipartition],
                                                  streams[ipartition].value());
    }

    for (int ipartition = 0; ipartition < npartitions; ipartition++)
      delete decompressors[ipartition];

    delete[] decompressors;
  }

  template <
    typename T,
    std::enable_if_t<cudf::is_timestamp_t<T>::value || cudf::is_duration_t<T>::value> * = nullptr>
  void operator()(vector<const void *> const &compressed_data,
                  vector<int64_t> const &compressed_sizes,
                  vector<void *> const &outputs,
                  vector<int64_t> const &expected_output_counts,
                  vector<rmm::cuda_stream_view> const &streams)
  {
    // If the data type is duration or time, use the corresponding arithmetic type
    operator()<typename T::rep>(
      compressed_data, compressed_sizes, outputs, expected_output_counts, streams);
  }
};

template <typename T>
using is_cascaded_supported = simt::std::disjunction<std::is_same<int8_t, T>,
                                                     std::is_same<uint8_t, T>,
                                                     std::is_same<int16_t, T>,
                                                     std::is_same<uint16_t, T>,
                                                     std::is_same<int32_t, T>,
                                                     std::is_same<uint32_t, T>,
                                                     std::is_same<int64_t, T>,
                                                     std::is_same<uint64_t, T>>;

struct cascaded_selector_functor {
  /**
   * Generate cascaded compression configuration options using auto-selector.
   *
   * @param[in] uncompressed_data Data used by auto-selector.
   * @param[in] byte_len Number of bytes in *uncompressed_data*.
   *
   * @returns Cascaded compression configuration options for *uncompressed_data*.
   */
  template <typename T, std::enable_if_t<is_cascaded_supported<T>::value> * = nullptr>
  nvcompCascadedFormatOpts operator()(const void *uncompressed_data, size_t byte_len)
  {
    nvcompCascadedSelectorOpts selector_opts;
    selector_opts.sample_size = 1024;
    selector_opts.num_samples = 100;

    nvcomp::CascadedSelector<T> selector(uncompressed_data, byte_len, selector_opts);

    size_t temp_bytes = selector.get_temp_size();
    rmm::device_buffer temp_space(temp_bytes);

    double estimate_ratio;
    return selector.select_config(temp_space.data(), temp_bytes, &estimate_ratio, 0);
  }

  template <
    typename T,
    std::enable_if_t<cudf::is_timestamp_t<T>::value || cudf::is_duration_t<T>::value> * = nullptr>
  nvcompCascadedFormatOpts operator()(const void *uncompressed_data, size_t byte_len)
  {
    // If the data type is duration or time, use the corresponding arithmetic type
    return operator()<typename T::rep>(uncompressed_data, byte_len);
  }

  template <typename T,
            std::enable_if_t<!is_cascaded_supported<T>::value && !cudf::is_timestamp_t<T>::value &&
                             !cudf::is_duration_t<T>::value> * = nullptr>
  nvcompCascadedFormatOpts operator()(const void *uncompressed_data, size_t byte_len)
  {
    throw std::runtime_error("Unsupported type for CascadedSelector");
    return nvcompCascadedFormatOpts();
  }
};

/**
 * Generate compression options using auto selector.
 *
 * @param[in] input_table Table for which to generate compression options.
 *
 * @returns Vector of length equal to number of columns in *input_table*, where each element
 * representing the compression options for each column.
 */
vector<ColumnCompressionOptions> generate_auto_select_compression_options(
  cudf::table_view input_table)
{
  vector<ColumnCompressionOptions> compression_options;

  for (cudf::size_type icol = 0; icol < input_table.num_columns(); icol++) {
    cudf::column_view input_column = input_table.column(icol);
    cudf::data_type dtype          = input_column.type();
    if (dtype.id() == cudf::type_id::STRING) {
      vector<ColumnCompressionOptions> children_options;

      // offset subcolumn
      cudf::data_type offset_dtype = input_column.child(0).type();
      nvcompCascadedFormatOpts offset_cascaded_opts =
        cudf::type_dispatcher(offset_dtype,
                              cascaded_selector_functor{},
                              input_column.child(0).head(),
                              input_column.child(0).size() * cudf::size_of(offset_dtype));
      children_options.emplace_back(CompressionMethod::cascaded, offset_cascaded_opts);

      // do not compress char subcolumn
      children_options.emplace_back(CompressionMethod::none);

      compression_options.emplace_back(
        CompressionMethod::none, nvcompCascadedFormatOpts(), children_options);
    } else {
      nvcompCascadedFormatOpts column_cascaded_opts =
        cudf::type_dispatcher(dtype,
                              cascaded_selector_functor{},
                              input_column.head(),
                              input_column.size() * cudf::size_of(dtype));

      compression_options.emplace_back(CompressionMethod::cascaded, column_cascaded_opts);
    }
  }

  return compression_options;
}

/**
 * Generate compression options that no compression should be performed.
 *
 * @param[in] input_table Table for which to generate compression options.
 *
 * @returns Vector of length equal to number of columns in *input_table*, where each element
 * representing the compression options for each column.
 */
vector<ColumnCompressionOptions> generate_none_compression_options(cudf::table_view input_table)
{
  vector<ColumnCompressionOptions> compression_options;

  for (cudf::size_type icol = 0; icol < input_table.num_columns(); icol++) {
    if (input_table.column(icol).type().id() == cudf::type_id::STRING) {
      vector<ColumnCompressionOptions> children_options;
      // offset subcolumn
      children_options.emplace_back(CompressionMethod::none);
      // char subcolumn
      children_options.emplace_back(CompressionMethod::none);
      compression_options.emplace_back(
        CompressionMethod::none, nvcompCascadedFormatOpts(), children_options);
    } else {
      compression_options.emplace_back(CompressionMethod::none);
    }
  }

  return compression_options;
}

/**
 * Broadcast the compression options of a column from the root rank to all ranks.
 *
 * Note: This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] input_column Column which *input_options* is associated with. This argument is
 * significant on all ranks.
 * @param[in] input_options Compression options associated with *input_column* that needs to be
 * broadcasted. This argument is only significant on the root rank.
 *
 * @returns Broadcasted compression options on all ranks.
 */
ColumnCompressionOptions broadcast_compression_options(cudf::column_view input_column,
                                                       ColumnCompressionOptions input_options)
{
  int mpi_rank;
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

  cudf::data_type dtype = input_column.type();

  CompressionMethod compression_method     = input_options.compression_method;
  nvcompCascadedFormatOpts cascaded_format = input_options.cascaded_format;
  vector<ColumnCompressionOptions> children_compression_options;

  MPI_CALL(MPI_Bcast(&compression_method, sizeof(CompressionMethod), MPI_CHAR, 0, MPI_COMM_WORLD));
  MPI_CALL(
    MPI_Bcast(&cascaded_format, sizeof(nvcompCascadedFormatOpts), MPI_CHAR, 0, MPI_COMM_WORLD));

  if (dtype.id() == cudf::type_id::STRING) {
    ColumnCompressionOptions compression_options;

    // offset subcolumn
    if (mpi_rank == 0) { compression_options = input_options.children_compression_options[0]; }
    children_compression_options.push_back(
      broadcast_compression_options(input_column.child(0), compression_options));

    // char subcolumn
    if (mpi_rank == 0) { compression_options = input_options.children_compression_options[1]; }
    children_compression_options.push_back(
      broadcast_compression_options(input_column.child(1), compression_options));
  }

  return ColumnCompressionOptions(
    compression_method, cascaded_format, children_compression_options);
}

/**
 * Broadcast the compression options of a table from the root rank to all ranks.
 *
 * Note: This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] input_table Table which *input_options* is associated with. This argument is
 * significant on all ranks.
 * @param[in] input_options Vector of lenght equal to the number of columns in *input_table*,
 * representing compression options associated with *input_table* that needs to be
 * broadcasted. Each element represents the compression option of one column in *input_table*. This
 * argument is only significant on the root rank.
 *
 * @returns Broadcasted compression options on all ranks.
 */
vector<ColumnCompressionOptions> broadcast_compression_options(
  cudf::table_view input_table, vector<ColumnCompressionOptions> input_options)
{
  int mpi_rank;
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

  vector<ColumnCompressionOptions> output_options;

  for (cudf::size_type icol = 0; icol < input_table.num_columns(); icol++) {
    ColumnCompressionOptions input_options_icol;
    if (mpi_rank == 0) { input_options_icol = input_options[icol]; }

    output_options.push_back(
      broadcast_compression_options(input_table.column(icol), input_options_icol));
  }

  return output_options;
}

/**
 * Generate the same compression option on all ranks.
 *
 * Note: This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] input_table Table to generate compression options on. This argument is significant on
 * all ranks.
 * @param[in] compression Whether to use compression. If *true*, the compression options will be
 * generated by auto selector on the root rank. If *false*, compression options indicating no
 * compression will be generated.
 *
 * @returns Compression options for *input_table* on all ranks.
 */
vector<ColumnCompressionOptions> generate_compression_options_distributed(
  cudf::table_view input_table, bool compression)
{
  if (!compression) { return generate_none_compression_options(input_table); }

  int mpi_rank;
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

  vector<ColumnCompressionOptions> compression_options;
  if (mpi_rank == 0) {
    compression_options = generate_auto_select_compression_options(input_table);
  }

  compression_options = broadcast_compression_options(input_table, compression_options);

  return compression_options;
}
