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

find_path(NVCOMP_INCLUDE_DIR NAMES nvcomp.hpp)
find_library(NVCOMP_LIBRARIES NAMES nvcomp)

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(NVCOMP DEFAULT_MSG NVCOMP_LIBRARIES NVCOMP_INCLUDE_DIR)

if (NVCOMP_FOUND)
  mark_as_advanced(NVCOMP_INCLUDE_DIR NVCOMP_LIBRARIES)
endif ()
