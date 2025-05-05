/*
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHINFER_ATTENTION_VARIANTS_CUH_
#define FLASHINFER_ATTENTION_VARIANTS_CUH_

#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__) || defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__)) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif

#include <cstdint>
#include <type_traits>

#include "../math.cuh"

namespace flashinfer {

  template <typename T>
  __device__ __forceinline__ float T2float(T q) {
    if constexpr(std::is_same<T, __half>::value)
	    return __half2float(q);
    else
	    return float(q);
  }

  template <typename T>
  __device__ __forceinline__ T float2T(float q) {
    if constexpr(std::is_same<T, __half>::value)
	    return __float2half(q);
    else
	    return float(q);
  }

  template <typename T>
  __device__ __forceinline__ T float2T_unsafe(float q) {
    static_assert(std::is_same_v<T, __hip_bfloat16>);

	  union f2bf { float f; __hip_bfloat16 bf[2]; } _f2bf;
    _f2bf.f = q;
    return _f2bf.bf[1];
  }

  template <typename T>
  __device__ __forceinline__ float T2float_unsafe(T q) {
    static_assert(std::is_same_v<T, __hip_bfloat16>);

	  union bf2f { float f; __hip_bfloat16 bf[2]; } _bf2f = { .f = 0 } ;
    _bf2f.bf[1] = q;
    return _bf2f.f;
  }

// Query Transform function that multiplies the query matrix by sm_scale
template <typename ParamsT_>
struct StandardAttention {
  using ParamsT = ParamsT_;
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  using IdType = typename ParamsT::IdType;
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;

  // Create closure
  __device__ __host__ StandardAttention(const ParamsT& params, uint32_t batch_idx,
                                        uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
  }

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    return T2float(q) * params.sm_scale * math::log2e;
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return logits;
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return true;
  }
};

template <typename ParamsT_>
struct CustomMaskAttention {
  using ParamsT = ParamsT_;
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  static constexpr bool use_softmax = true;

  uint8_t* custom_mask_ptr;
  uint32_t window_left, qo_len, kv_len;

  // Create closure
  __device__ __host__ CustomMaskAttention(const ParamsT& params, uint32_t batch_idx,
                                          uint8_t* smem_ptr) {
    custom_mask_ptr = params.get_batch_local_mask_ptr(batch_idx);
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
  }

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return logits;
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    const uint32_t offset = qo_idx * kv_len + kv_idx;
    return ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
  }
};

template <typename ParamsT_>
struct SlidingWindowAttention {
  using ParamsT = ParamsT_;
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  using IdType = typename ParamsT::IdType;
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;

  // Create closure
  __device__ __host__ __forceinline__ SlidingWindowAttention(const ParamsT& params,
                                                             uint32_t batch_idx,
                                                             uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = (params.window_left >= 0) ? params.window_left : kv_len;
  }

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return logits;
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return (kv_idx + qo_len + window_left >= kv_len + qo_idx);
  }
};

template <typename ParamsT>
struct LogitsSoftCap {
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;

  __device__ __host__ LogitsSoftCap(const ParamsT& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
  }

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    return float(q) * params.sm_scale * math::ptx_rcp(params.logits_soft_cap);
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return params.logits_soft_cap * math::log2e * float(math::tanh(logits));
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return true;
  }
};

template <typename ParamsT>
struct ALIBIAttention {
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  using IdType = typename ParamsT::IdType;
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;

  __device__ __host__ ALIBIAttention(const ParamsT& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
  }

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return logits + params.alibi_slopes[qo_head_idx] * float(int(kv_idx) - int(qo_idx));
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return true;
  }
};

constexpr uint32_t CUSTOM_MASK = 1U;
constexpr uint32_t SLIDING_WINDOW = 2U;
constexpr uint32_t LOGITS_SOFT_CAP = 4U;
constexpr uint32_t ALIBI = 8U;

constexpr uint32_t get_variant_code(bool use_custom_mask, bool use_sliding_window,
                                    bool use_logits_soft_cap, bool use_alibi) {
  return (use_custom_mask ? CUSTOM_MASK : 0U) | (use_sliding_window ? SLIDING_WINDOW : 0U) |
         (use_logits_soft_cap ? LOGITS_SOFT_CAP : 0U) | (use_alibi ? ALIBI : 0U);
}

template <typename ParamsT_, uint32_t VARIANT_CODE>
struct ComposedAttention {
  using ParamsT = ParamsT_;
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  using IdType = typename ParamsT::IdType;
  static constexpr bool use_softmax = true;
  static constexpr bool use_custom_mask = (VARIANT_CODE & CUSTOM_MASK) != 0;
  static constexpr bool use_sliding_window = (VARIANT_CODE & SLIDING_WINDOW) != 0;
  static constexpr bool use_logits_soft_cap = (VARIANT_CODE & LOGITS_SOFT_CAP) != 0;
  static constexpr bool use_alibi = (VARIANT_CODE & ALIBI) != 0;

  uint32_t qo_len, kv_len;
  uint8_t* custom_mask_ptr;
  uint32_t window_left;

  // Create closure
  __device__ __host__ ComposedAttention(const ParamsT& params, uint32_t batch_idx,
                                        uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    if constexpr (use_custom_mask) {
      custom_mask_ptr = params.get_batch_local_mask_ptr(batch_idx);
    }
    if constexpr (use_sliding_window) {
      window_left = (params.window_left >= 0) ? params.window_left : kv_len;
    }
  }

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    if constexpr (use_logits_soft_cap) {
      if constexpr(std::is_same<T, __hip_bfloat16>::value) {
        //return float2T_unsafe<T>(T2float_unsafe(q) * params.sm_scale * math::ptx_rcp(params.logits_soft_cap));
        return float2T_unsafe<T>(T2float_unsafe(q) * params.sm_scale * __builtin_amdgcn_rcpf(params.logits_soft_cap));
      } else {
        return float2T<T>(T2float(q) * params.sm_scale * math::ptx_rcp(params.logits_soft_cap));
      }
    } else {
      return float2T<T>(T2float(q) * params.sm_scale * math::log2e);
    }
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    if constexpr (use_alibi) {
      logits = logits + params.alibi_slopes[qo_head_idx] * float(int(kv_idx) - int(qo_idx));
    }
    if constexpr (use_logits_soft_cap) {
      //float ex2 = __builtin_amdgcn_exp2f((logits*2.0)*math::log2e);
      //float tnh_ = ((ex2-1.0)/(ex2+1.0));
      //logits = params.logits_soft_cap * math::log2e * ((ex2-1.0)/(ex2+1.0));
      //APPROX...
      float e, r, s, t, d;
      float a = (float)logits;
      s = fabsf (a);
      t = -math::log2e * 2.0f * s;
      e = __builtin_amdgcn_exp2f(t);
      d = e + 1.0f;
      r = __builtin_amdgcn_rcpf(d);
      r = e*(-r)+r;//fmaf (e, -r, r);
      if (s < 4.997253418e-3f) r = a;
      union fipnr {float f; unsigned int i;};
      fipnr r_; r_.f = r;
      fipnr a_; a_.f = a;
      //if (!isnan(a)) //UNSAFE...
      { r_.i = (r_.i|(a_.i&0x80000000)); r = r_.f; } // r = copysignf_pos (r, a);
      logits = params.logits_soft_cap * math::log2e * r;
    }
    return logits;
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    bool mask = true;
    if constexpr (use_custom_mask) {
      const uint32_t offset = qo_idx * kv_len + kv_idx;
      mask &= ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
    }
    if constexpr (use_sliding_window) {
      mask &= (kv_idx + qo_len + window_left >= kv_len + qo_idx);
    }
    return mask;
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_VARIANTS_CUH_
