/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

template <typename T = char>
inline void *ADV_PTR(void *ptr, size_t offset)
{
  return static_cast<void *>(static_cast<T *>(ptr) + offset);
}

template <typename T = char>
inline const void *ADV_PTR(const void *ptr, size_t offset)
{
  return static_cast<const void *>(static_cast<const T *>(ptr) + offset);
}
