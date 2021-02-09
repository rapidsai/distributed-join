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

find_path(CUDF_INCLUDE_DIR NAMES cudf/join.hpp)

set(CUDF_LIBRARY_NAMES cudf cudf_base cudf_join cudf_hash cudf_partitioning cudf_io)
set(CUDF_LIBRARIES "")
foreach(CUDF_LIBRARY_NAME ${CUDF_LIBRARY_NAMES})
  find_library(${CUDF_LIBRARY_NAME}_LIBRARY NAMES ${CUDF_LIBRARY_NAME} REQUIRED)
  list(APPEND CUDF_LIBRARIES ${${CUDF_LIBRARY_NAME}_LIBRARY})
endforeach()

get_filename_component(CUDF_LIBPATH ${cudf_LIBRARY} DIRECTORY)

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(CUDF DEFAULT_MSG CUDF_LIBRARIES CUDF_INCLUDE_DIR)

if(CUDF_FOUND)
  mark_as_advanced(CUDF_INCLUDE_DIR CUDF_LIBRARIES)
  set(CUDF_INCLUDE_DIRS ${CUDF_INCLUDE_DIR} ${CUDF_INCLUDE_DIR}/libcudf/libcudacxx)
endif()
