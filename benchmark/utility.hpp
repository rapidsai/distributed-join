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

#include "../src/compression.hpp"

#include <nvcomp/cascaded.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <vector>

/**
 * Calculate the table sizes in bytes.
 *
 * Note: This function only support tables with fixed-width and string columns.
 */
inline int64_t calculate_table_size(cudf::table_view input_table)
{
  int64_t table_size = 0;

  for (auto &current_column : input_table) {
    cudf::data_type dtype = current_column.type();
    if (cudf::is_fixed_width(dtype)) {
      table_size += (cudf::size_of(dtype) * current_column.size());
    } else {
      assert(dtype.id() == cudf::type_id::STRING);
      table_size += current_column.child(1).size();
    }
  }

  return table_size;
}

inline void print_compression_options(std::vector<ColumnCompressionOptions> &compression_options)
{
  for (size_t icol = 0; icol < compression_options.size(); icol++) {
    nvcompCascadedFormatOpts format = compression_options[icol].cascaded_format;
    std::cout << "Column " << icol << " RLE=" << format.num_RLEs << ", Delta=" << format.num_deltas
              << ", Bitpack=" << format.use_bp << std::endl;
  }
}
