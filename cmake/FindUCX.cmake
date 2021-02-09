# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

find_path(UCX_INCLUDE_DIR NAMES ucp/api/ucp.h)

find_library(UCS_LIBRARY NAMES ucs)
find_library(UCT_LIBRARY NAMES uct)
find_library(UCP_LIBRARY NAMES ucp)

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(UCX DEFAULT_MSG UCX_INCLUDE_DIR UCS_LIBRARY UCT_LIBRARY UCP_LIBRARY)

set(UCX_LIBRARIES ${UCS_LIBRARY} ${UCT_LIBRARY} ${UCP_LIBRARY})

if (UCX_FOUND)
  mark_as_advanced(UCX_INCLUDE_DIR UCX_LIBRARIES)
endif ()
