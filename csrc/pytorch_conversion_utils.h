/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <ATen/ATen.h>

inline at::Tensor vec_to_tensor(const std::vector<int64_t>& vec) {
  return at::tensor(vec, at::dtype(at::kLong).device(at::kCPU));
}

inline std::vector<int64_t> tensor_to_vec(const at::Tensor& tensor) {
  const size_t size = tensor.numel();
  // const int64_t* first = tensor.const_data_ptr<int64_t>();
  // Failed to find a matched PyTorch version to address the error below,
  // so we use `data_ptr` (that is always defined) instead of `const_data_ptr`
  // that is conditionally defined with `std::enable_if`.
  // Error:
  // flashinfer/flashinfer_kernels.abi3.so: error: symbol lookup error: undefined symbol: _ZNK2at10TensorBase14const_data_ptrIlTnNSt9enable_ifIXntsr3stdE10is_const_vIT_EEiE4typeELi0EEEPKS3_v (fatal)
  const int64_t* first = tensor.data_ptr<int64_t>();
  const int64_t* last = first + size;
  return std::vector(first, last);
}
