# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

function(include_and_link_dependencies target_name)
  set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})

  target_include_directories(${target_name} PUBLIC "${CMAKE_SOURCE_DIR}/src")
  target_include_directories(${target_name} PUBLIC "${CUDAToolkit_INCLUDE_DIRS}")
  target_include_directories(${target_name} PUBLIC "${NCCL_INCLUDE_DIRS}")
  target_include_directories(${target_name} PUBLIC "${UCX_INCLUDE_DIRS}")
  target_include_directories(${target_name} PUBLIC "${NVCOMP_INCLUDE_DIR}")
  target_include_directories(${target_name} PUBLIC "${MPI_CXX_INCLUDE_DIRS}")
  target_include_directories(${target_name} PUBLIC "${CUDF_INCLUDE_DIRS}")
  target_include_directories(${target_name} PUBLIC "${RMM_INCLUDE_DIRS}")

  target_link_libraries(${target_name} PUBLIC ${NCCL_LIBRARIES})
  target_link_libraries(${target_name} PUBLIC ${UCX_LIBRARIES})
  target_link_libraries(${target_name} PUBLIC ${NVCOMP_LIBRARIES})
  target_link_libraries(${target_name} PUBLIC MPI::MPI_CXX)
  target_link_libraries(${target_name} PUBLIC ${CUDF_LIBRARIES})
  target_link_libraries(${target_name} PUBLIC CUDA::cudart)

  target_compile_options(${target_name} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
  target_compile_options(${target_name} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)
endfunction()

function(build_executables sources)
  foreach(source IN LISTS ${sources})
    get_filename_component(target_name ${source} NAME_WLE)
    add_executable(${target_name} ${source})
    include_and_link_dependencies(${target_name})
    target_link_libraries(${target_name} PUBLIC distributed)
  endforeach()
endfunction()
