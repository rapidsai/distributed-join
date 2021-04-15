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

#include "all_to_all_comm.hpp"
#include "communicator.hpp"
#include "compression.hpp"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <vector>

/**
 * Shuffle the table according to the hash values.
 *
 * Note: This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] input Input table to be shuffled.
 * @param[in] on_columns Columns used when computing the hash value of each row.
 * @param[in] compression_options Vector of length equal to the number of columns in *input*,
 * indicating whether/how each column needs to be compressed before communication.
 * @param[in] hash_function Hash function used for computing the hash value of each row.
 * @param[in] hash_seed Hash seed used with *hash_function*.
 * @param[in] preallocated_pinned_buffer Preallocated page-locked host buffer with size at least
 * `mpi_size * sizeof(size_t)`, used for holding the compressed sizes.
 */
std::unique_ptr<cudf::table> shuffle_on(cudf::table_view const& input,
                                        std::vector<cudf::size_type> const& on_columns,
                                        CommunicationGroup comm_group,
                                        Communicator* communicator,
                                        std::vector<ColumnCompressionOptions> compression_options,
                                        cudf::hash_id hash_function = cudf::hash_id::HASH_MURMUR3,
                                        uint32_t hash_seed          = cudf::DEFAULT_HASH_SEED,
                                        bool report_timing          = false,
                                        void* preallocated_pinned_buffer = nullptr);

/**
 * This variant of *shuffle_on* uses a communication group with all ranks and stride 1.
 */
std::unique_ptr<cudf::table> shuffle_on(cudf::table_view const& input,
                                        std::vector<cudf::size_type> const& on_columns,
                                        Communicator* communicator,
                                        std::vector<ColumnCompressionOptions> compression_options,
                                        cudf::hash_id hash_function = cudf::hash_id::HASH_MURMUR3,
                                        uint32_t hash_seed          = cudf::DEFAULT_HASH_SEED,
                                        bool report_timing          = false,
                                        void* preallocated_pinned_buffer = nullptr);
