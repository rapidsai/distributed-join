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

#include <memory>
#include <vector>

std::unique_ptr<cudf::table> shuffle_on(cudf::table_view const& input,
                                        std::vector<cudf::size_type> const& on_columns,
                                        Communicator* communicator,
                                        std::vector<ColumnCompressionOptions> compression_options,
                                        cudf::hash_id hash_function = cudf::hash_id::HASH_MURMUR3,
                                        bool report_timing          = false,
                                        void* preallocated_pinned_buffer = nullptr);
