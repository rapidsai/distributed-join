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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <memory>

/**
 * Distribute a table from root rank to worker ranks.
 *
 * This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] global_table     The global table. Only significant on root rank.
 * @param[in] communicator     An instance of `Communicator` used for communication.
 *
 * @returns                    The local table on each rank.
 */
std::unique_ptr<cudf::table> distribute_table(cudf::table_view global_table,
                                              Communicator *communicator);

/**
 * Merge tables from all worker ranks to the root rank.
 *
 * This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] table The table on each rank to be sent to the master rank. Significant on all ranks.
 * @param[in] communicator An instance of `Communicator` used for communication.
 *
 * @return Merged table on the root rank. `nullptr` on all other ranks.
 */
std::unique_ptr<cudf::table> collect_tables(cudf::table_view table, Communicator *communicator);
