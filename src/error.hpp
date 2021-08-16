/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstdio>
#include <cstdlib>

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                               \
  {                                                                                      \
    cudaError_t cudaStatus = call;                                                       \
    if (cudaSuccess != cudaStatus) {                                                     \
      fprintf(stderr,                                                                    \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
              #call,                                                                     \
              __LINE__,                                                                  \
              __FILE__,                                                                  \
              cudaGetErrorString(cudaStatus),                                            \
              cudaStatus);                                                               \
      exit(1);                                                                           \
    }                                                                                    \
  }
#endif

#ifndef UCX_CALL
#define UCX_CALL(call)                                               \
  {                                                                  \
    ucs_status_t status = call;                                      \
    if (UCS_OK != status) {                                          \
      fprintf(stderr,                                                \
              "\"%s\" in line %d of file %s failed with %s (%d).\n", \
              #call,                                                 \
              __LINE__,                                              \
              __FILE__,                                              \
              ucs_status_string(status),                             \
              status);                                               \
      exit(1);                                                       \
    }                                                                \
  }
#endif

#ifndef MPI_CALL
#define MPI_CALL(call)                                               \
  {                                                                  \
    int status = call;                                               \
    if (MPI_SUCCESS != status) {                                     \
      int len;                                                       \
      char estring[MPI_MAX_ERROR_STRING];                            \
      MPI_Error_string(status, estring, &len);                       \
      fprintf(stderr,                                                \
              "\"%s\" in line %d of file %s failed with %s (%d).\n", \
              #call,                                                 \
              __LINE__,                                              \
              __FILE__,                                              \
              estring,                                               \
              status);                                               \
      exit(1);                                                       \
    }                                                                \
  }
#endif

#ifndef NCCL_CALL
#define NCCL_CALL(call)                                                          \
  {                                                                              \
    ncclResult_t status = call;                                                  \
    if (ncclSuccess != status) {                                                 \
      fprintf(stderr,                                                            \
              "ERROR: nccl call \"%s\" in line %d of file %s failed with %s.\n", \
              #call,                                                             \
              __LINE__,                                                          \
              __FILE__,                                                          \
              ncclGetErrorString(status));                                       \
      exit(1);                                                                   \
    }                                                                            \
  }
#endif

#define CHECK_ERROR(rtv, expected_value, msg)                                        \
  {                                                                                  \
    if (rtv != expected_value) {                                                     \
      fprintf(stderr, "ERROR on line %d of file %s: %s\n", __LINE__, __FILE__, msg); \
      std::cerr << rtv << std::endl;                                                 \
      exit(1);                                                                       \
    }                                                                                \
  }
