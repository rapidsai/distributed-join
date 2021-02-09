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

find_path(RMM_INCLUDE_DIR NAMES rmm/device_buffer.hpp)

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(RMM DEFAULT_MSG RMM_INCLUDE_DIR)

if (RMM_FOUND)
  mark_as_advanced(RMM_INCLUDE_DIR)
  set(RMM_INCLUDE_DIRS ${RMM_INCLUDE_DIR})
endif ()
