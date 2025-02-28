/*
 * Copyright (c) 2023 by FlashInfer team.
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
#ifndef FLASHINFER_PREFILL_CUH_
#define FLASHINFER_PREFILL_CUH_

#include "../gpu_defines_cuda_hip.h"

#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__) || defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__)) || defined(__CUDACC_RTC__)
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#endif

#include "../cp_async.cuh"
#include "../fastdiv.cuh"
#include "../frag_layout_swizzle.cuh"
#include "../layout.cuh"
#include "../math.cuh"
#include "../mma.cuh"
#include "../page.cuh"
#include "../permuted_smem.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "cascade.cuh"
#include "mask.cuh"
#include "variants.cuh"


#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
// TODO: Better place to put these custom functions.
__device__ __hip_bfloat162 convert_half2_to_bfloat162(const __half2 h2) {
  // Extract the two FP16 components.
  __half h0 = __low2half(h2);  // Lower half of __half2.
  __half h1 = __high2half(h2);  // Higher half of __half2.

  // Manually convert to BF16 by truncating the mantissa.
  __hip_bfloat16 bf0 = __float2bfloat16(__half2float(h0));
  __hip_bfloat16 bf1 = __float2bfloat16(__half2float(h1));

  // Combine the two BF16 components into a __hip_bfloat162.
  return __hip_bfloat162(bf0, bf1);
}

__device__ __half2 convert_bfloat162_to_half2(const __hip_bfloat162 bf2) {
  // Extract the two BF16 components.
  __hip_bfloat16 bf0 = __low2bfloat16(bf2);
  __hip_bfloat16 bf1 = __high2bfloat16(bf2);

  // Convert BF16 to FP16.
  __half h0 = __float2half(__bfloat162float(bf0));
  __half h1 = __float2half(__bfloat162float(bf1));

  // Combine into a __half2.
  return __halves2half2(h0, h1);
}
#endif

namespace flashinfer {

namespace cg = cooperative_groups;
using cp_async::SharedMemFillMode;
using mma::MMAMode;

constexpr uint32_t WARP_SIZE = 64;
constexpr uint32_t CUDA_WARP_SIZE = 32;

using B16_t = uint16_t;
using B16x4_t = B16_t __attribute__((ext_vector_type(4)));
using B32_t = uint32_t;
using B32x4_t = B32_t __attribute__((ext_vector_type(4)));

template <typename DType>
struct mfma_m16n16k16_f32
{
  static_assert(sizeof(DType) == 1);

  using ab_fragment_type = B16x4_t;
  using c_fragment_type = B32x4_t;

  static __device__ __forceinline__ c_fragment_type run(
    ab_fragment_type,
    ab_fragment_type,
    c_fragment_type
  ) {
    return c_fragment_type{};
  }
};

template <>
struct mfma_m16n16k16_f32<__half>
{
  using ab_fragment_type = B16x4_t;
  using c_fragment_type = B32x4_t;

  static __device__ __forceinline__ c_fragment_type run(
    ab_fragment_type a,
    ab_fragment_type b,
    c_fragment_type c
  ) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
  }
};

template <>
struct mfma_m16n16k16_f32<__hip_bfloat16>
{
  using ab_fragment_type = B16x4_t;
  using c_fragment_type = B32x4_t;

  static __device__ __forceinline__ c_fragment_type run(
    ab_fragment_type a,
    ab_fragment_type b,
    c_fragment_type c
  ) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
  }
};

constexpr uint32_t get_num_warps_q(const uint32_t cta_tile_q) {
  if (cta_tile_q > 16) {
    return 4;
  } else {
    return 1;
  }
}

constexpr uint32_t get_num_warps_kv(const uint32_t cta_tile_kv) {
  return 4 / get_num_warps_q(cta_tile_kv);
}

constexpr uint32_t get_num_mma_q(const uint32_t cta_tile_q) {
  if (cta_tile_q > 64) {
    return 2;
  } else {
    return 1;
  }
}

namespace {

template <PosEncodingMode POS_ENCODING_MODE, typename DTypeKV, typename DTypeQKAccum>
constexpr bool is_invalid_configuration(uint32_t NUM_MMA_Q, uint32_t NUM_MMA_D, uint32_t NUM_MMA_KV,
                                        uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV) {
  return ((NUM_MMA_D < 4) || (NUM_MMA_D == 4 && NUM_MMA_KV % 2 == 1) ||
          (NUM_MMA_D > 4 && NUM_MMA_D % (2 * NUM_WARPS_Q) != 0) ||
          (NUM_MMA_Q * (8 * NUM_MMA_D + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256) ||
          (sizeof(DTypeKV) == 1 && NUM_MMA_KV * 2 % NUM_WARPS_Q != 0) ||
          (sizeof(DTypeKV) == 1 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama));
}

template <uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV>
__device__ __forceinline__ uint32_t get_warp_idx_q() {
  if constexpr (NUM_WARPS_Q == 1) {
    return 0;
  } else {
    return threadIdx.y;
  }
}

template <uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV>
__device__ __forceinline__ uint32_t get_warp_idx_kv() {
  if constexpr (NUM_WARPS_KV == 1) {
    return 0;
  } else {
    return threadIdx.z;
  }
}

template <uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV>
__device__ __forceinline__ uint32_t get_warp_idx() {
  return get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>() * NUM_WARPS_Q +
         get_warp_idx_q<NUM_WARPS_Q, NUM_WARPS_KV>();
}

/*!
 * \brief Apply Llama style rotary embedding to two 16x16 fragments.
 * \tparam T The data type of the input fragments.
 * \param x_first_half First fragment x[offset:offset+16, j*16:(j+1)*16]
 * \param x_second_half Second fragment x[offset:offset*16, j*16+d/2:(j+1)*16+d/2]
 * \param rope_freq Rope frequency
 * \param offset The offset of the first row in both fragments.
 * \note The sin/cos computation is slow, especially for A100 GPUs which has low
 *   non tensor-ops flops, will optimize in the future.
 */
template <typename T>
__device__ __forceinline__ void k_frag_apply_llama_rope(T* x_first_half, T* x_second_half,
                                                        const float* rope_freq,
                                                        const uint32_t kv_offset) {
  static_assert(sizeof(T) == 2);
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 2 3
    // ---------
    // 4 5 | 6 7
    uint32_t i = reg_id / 4, j = (reg_id % 4) / 2;
    __sincosf(float(kv_offset + 8 * i) * rope_freq[2 * j + reg_id % 2], &sin, &cos);
    //tmp = x_first_half[reg_id];
    //x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    //x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
//hipFIXED
    tmp = T2float<T>(x_first_half[reg_id]);
    x_first_half[reg_id] = float2T<T>(tmp * cos - T2float<T>(x_second_half[reg_id]) * sin);
    x_second_half[reg_id] = float2T<T>(T2float<T>(x_second_half[reg_id]) * cos + tmp * sin);
  }
}

template <typename T>
__device__ __forceinline__ void q_frag_apply_llama_rope(T* x_first_half, T* x_second_half,
                                                        const float* rope_freq,
                                                        const uint32_t qo_packed_offset,
                                                        const uint_fastdiv group_size) {
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 4 5
    // ---------
    // 2 3 | 6 7
    uint32_t i = ((reg_id % 4) / 2), j = (reg_id / 4);
    __sincosf(float((qo_packed_offset + 8 * i) / group_size) * rope_freq[2 * j + reg_id % 2], &sin,
              &cos);
    //tmp = x_first_half[reg_id];
    //x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    //x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
//hipFIXED
    tmp = T2float<T>(x_first_half[reg_id]);
    x_first_half[reg_id] = float2T<T>(tmp * cos - T2float<T>(x_second_half[reg_id]) * sin);
    x_second_half[reg_id] = float2T<T>(T2float<T>(x_second_half[reg_id]) * cos + tmp * sin);
  }
}

template <typename T, typename IdType>
__device__ __forceinline__ void q_frag_apply_llama_rope_with_pos(T* x_first_half, T* x_second_half,
                                                                 const float* rope_freq,
                                                                 const uint32_t qo_packed_offset,
                                                                 const uint_fastdiv group_size,
                                                                 const IdType* q_offset) {
  float pos[2] = {static_cast<float>(q_offset[qo_packed_offset / group_size]),
                  static_cast<float>(q_offset[(qo_packed_offset + 8) / group_size])};
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 4 5
    // ---------
    // 2 3 | 6 7
    uint32_t i = ((reg_id % 4) / 2), j = (reg_id / 4);
    __sincosf(pos[i] * rope_freq[2 * j + reg_id % 2], &sin, &cos);
    //tmp = x_first_half[reg_id];
    //x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    //x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
//hipFIXED
    tmp = T2float<T>(x_first_half[reg_id]);
    x_first_half[reg_id] = float2T<T>(tmp * cos - T2float<T>(x_second_half[reg_id]) * sin);
    x_second_half[reg_id] = float2T<T>(T2float<T>(x_second_half[reg_id]) * cos + tmp * sin);
  }
}

/*!
 * \brief Produce k/v fragments from global memory to shared memory.
 * \tparam fill_mode The fill mode of the shared memory.
 * \tparam NUM_MMA_D The number of fragments in y dimension.
 * \tparam NUM_MMA_KV The number of fragments in z dimension.
 * \tparam num_warps The number of warps in the threadblock.
 * \tparam T The data type of the input tensor.
 * \param smem The shared memory to store kv fragments.
 * \param gptr The global memory pointer.
 * \param kv_idx_base The base kv index.
 * \param kv_len The length of kv tensor.
 */
template <SharedMemFillMode fill_mode, uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV,
          uint32_t NUM_MMA_D, uint32_t NUM_MMA_KV, SwizzleMode swizzle_mode, typename T>
__device__ __forceinline__ void produce_kv(smem_t<swizzle_mode> smem, uint32_t* smem_offset,
                                           T** gptr, const uint32_t kv_stride_n,
                                           const uint32_t kv_idx_base, const uint32_t kv_len) {
  // NOTE(Zihao): for fp8, this function doesn't work for head_dim = 64 at the moment
  constexpr uint32_t head_dim = NUM_MMA_D * 16;
  constexpr uint32_t num_warps = NUM_WARPS_Q * NUM_WARPS_KV;
  constexpr uint32_t channel_size_128b_kv = head_dim / num_elems_per_128b<T>();
  const uint32_t warp_idx = get_warp_idx<NUM_WARPS_Q, NUM_WARPS_KV>(), lane_idx = threadIdx.x;

  if constexpr (swizzle_mode == SwizzleMode::k128B) {
    uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
    // NOTE(Zihao): NUM_MMA_KV * 4 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 4 / num_warps
    static_assert(NUM_MMA_KV * 4 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(T)); ++j) {
        //smem.load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
//hipFIXED
        smem.template load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
        *smem_offset = smem.template advance_offset_by_column<8>(*smem_offset, j);
        *gptr += 8 * num_elems_per_128b<T>();
      }
      kv_idx += num_warps * 4;
      *smem_offset =
          smem.template advance_offset_by_row<num_warps * 4, channel_size_128b_kv>(*smem_offset) -
          sizeof(T) * NUM_MMA_D;
      *gptr += num_warps * 4 * kv_stride_n - sizeof(T) * NUM_MMA_D * num_elems_per_128b<T>();
    }
    *smem_offset -= NUM_WARPS_KV * NUM_MMA_KV * 16 * channel_size_128b_kv;
  } else {
    uint32_t kv_idx = kv_idx_base + warp_idx * 8 + lane_idx / 4;
    // NOTE(Zihao): NUM_MMA_KV * 2 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 2 / num_warps
    static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
      //smem.load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
//hipFIXED
      smem.template load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
      *smem_offset =
          smem.template advance_offset_by_row<num_warps * 8, channel_size_128b_kv>(*smem_offset);
      kv_idx += num_warps * 8;
      *gptr += num_warps * 8 * kv_stride_n;
    }
    *smem_offset -= NUM_WARPS_KV * NUM_MMA_KV * 16 * channel_size_128b_kv;
  }
}

template <bool produce_v, uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV, uint32_t NUM_MMA_D,
          uint32_t NUM_MMA_KV, SwizzleMode swizzle_mode, typename DType, typename IdType>
__device__ __forceinline__ void page_produce_kv(smem_t<swizzle_mode> smem, uint32_t* smem_offset,
                                                const paged_kv_t<DType, IdType>& paged_kv,
                                                const uint32_t kv_idx_base, const size_t* kv_offset,
                                                const uint32_t kv_len) {
  if (CUDA_WARP_SIZE <= threadIdx.x) {
    return;
  }

  // NOTE(Zihao): for fp8, this function doesn't work for head_dim = 64 at the moment
  constexpr SharedMemFillMode fill_mode =
      produce_v ? SharedMemFillMode::kFillZero : SharedMemFillMode::kNoFill;
  constexpr uint32_t head_dim = NUM_MMA_D * 16;
  constexpr uint32_t num_warps = NUM_WARPS_Q * NUM_WARPS_KV;
  constexpr uint32_t channel_size_128b_kv = head_dim / num_elems_per_128b<DType>();
  const uint32_t warp_idx = get_warp_idx<NUM_WARPS_Q, NUM_WARPS_KV>(), lane_idx = threadIdx.x;
  if constexpr (swizzle_mode == SwizzleMode::k128B) {
    uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
    // NOTE(Zihao): NUM_MMA_KV * 4 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 4 / num_warps
    static_assert(NUM_MMA_KV * 4 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
      DType* gptr = produce_v ? paged_kv.v_data + kv_offset[i] : paged_kv.k_data + kv_offset[i];
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DType)); ++j) {
        //smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
//hipFIXED
        smem.template load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
        *smem_offset = smem.template advance_offset_by_column<8>(*smem_offset, j);
        gptr += 8 * num_elems_per_128b<DType>();
      }
      kv_idx += num_warps * 4;
      *smem_offset =
          smem.template advance_offset_by_row<num_warps * 4, channel_size_128b_kv>(*smem_offset) -
          sizeof(DType) * NUM_MMA_D;
    }
    *smem_offset -= NUM_WARPS_KV * NUM_MMA_KV * 16 * channel_size_128b_kv;
  } else {
    uint32_t kv_idx = kv_idx_base + warp_idx * 8 + lane_idx / 4;
    // NOTE(Zihao): NUM_MMA_KV * 2 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 2 / num_warps
    static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
      DType* gptr = produce_v ? paged_kv.v_data + kv_offset[i] : paged_kv.k_data + kv_offset[i];
      //smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
//hipFIXED
      smem.template load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
      kv_idx += num_warps * 8;
      *smem_offset =
          smem.template advance_offset_by_row<num_warps * 8, channel_size_128b_kv>(*smem_offset);
    }
    *smem_offset -= NUM_WARPS_KV * NUM_MMA_KV * 16 * channel_size_128b_kv;
  }
}

template <uint32_t NUM_MMA_D>
__device__ __forceinline__ void init_rope_freq(float (*rope_freq)[4],
                                               const float log2_rope_rcp_scale,
                                               const float log2_rope_rcp_theta) {
  constexpr uint32_t head_dim = NUM_MMA_D * 16;
  const uint32_t lane_idx = threadIdx.x;
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D / 2; ++mma_d) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      rope_freq[mma_d][j] =
          math::ptx_exp2(log2_rope_rcp_scale +
                         log2_rope_rcp_theta *
                             float(2 * ((mma_d * 16 + (j / 2) * 8 + (lane_idx % 4) * 2 + (j % 2)) %
                                        (head_dim / 2))) /
                             float(head_dim));
    }
  }
}

template <uint32_t NUM_MMA_Q, uint32_t NUM_MMA_D, typename DTypeQKAccum, typename AttentionVariant>
__device__ __forceinline__ void init_states(AttentionVariant variant, float (*o_frag)[NUM_MMA_D][4],
                                            DTypeQKAccum (*m)[1], float (*d)[1]) {
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < NUM_MMA_D; ++mma_d) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
        o_frag[mma_q][mma_d][reg_id] = 0.f;
      }
    }
  }

  if constexpr (variant.use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 1; ++j) {
        m[mma_q][j] = DTypeQKAccum(-math::inf);
        d[mma_q][j] = 1.f;
      }
    }
  }
}

template <uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV, uint32_t NUM_MMA_Q, uint32_t NUM_MMA_D,
          SwizzleMode swizzle_mode, typename DTypeQ>
__device__ __forceinline__ void load_q_global_smem(uint32_t packed_offset,
                                                   const uint32_t qo_upper_bound,
                                                   DTypeQ* q_ptr_base, const uint32_t q_stride_n,
                                                   const uint32_t q_stride_h,
                                                   const uint_fastdiv group_size,
                                                   smem_t<swizzle_mode>* q_smem) {
  if (CUDA_WARP_SIZE <= threadIdx.x) {
    return;
  }

  constexpr uint32_t head_dim = NUM_MMA_D * 16;
  constexpr uint32_t channel_size_128b_q = head_dim / num_elems_per_128b<DTypeQ>();
  const uint32_t lane_idx = threadIdx.x, warp_idx_x = get_warp_idx_q<NUM_WARPS_Q, NUM_WARPS_KV>();

  if (get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>() == 0) {
    uint32_t q_smem_offset_w = q_smem->template get_permuted_offset<channel_size_128b_q>(
        warp_idx_x * NUM_MMA_Q * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        uint32_t q, r;
        group_size.divmod(packed_offset + lane_idx / 8 + mma_q * 16 + j * 4, q, r);
        const uint32_t q_idx = q;
        DTypeQ* q_ptr = q_ptr_base + q * q_stride_n + r * q_stride_h;
#pragma unroll
        for (uint32_t mma_do = 0; mma_do < NUM_MMA_D / 4; ++mma_do) {
          // load q fragment from gmem to smem
          //q_smem->load_128b_async<SharedMemFillMode::kNoFill>(q_smem_offset_w, q_ptr,
//hipFIXED
          q_smem->template load_128b_async<SharedMemFillMode::kNoFill>(q_smem_offset_w, q_ptr,
                                                              q_idx < qo_upper_bound);
          q_smem_offset_w = q_smem->template advance_offset_by_column<8>(q_smem_offset_w, mma_do);
          q_ptr += 8 * num_elems_per_128b<DTypeQ>();
        }
        q_smem_offset_w =
            q_smem->template advance_offset_by_row<4, channel_size_128b_q>(q_smem_offset_w) -
            (NUM_MMA_D / 4) * 8;
      }
    }
  }
}

template <uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV, uint32_t NUM_MMA_Q, uint32_t NUM_MMA_D,
          SwizzleMode swizzle_mode, typename DTypeQ>
__device__ __forceinline__ void q_smem_inplace_apply_rotary(
    const uint32_t q_packed_idx, const uint32_t qo_len, const uint32_t kv_len,
    const uint_fastdiv group_size, smem_t<swizzle_mode>* q_smem, uint32_t* q_smem_offset_r,
    float (*rope_freq)[4]) {
  if (get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>() == 0) {
    constexpr uint32_t head_dim = NUM_MMA_D * 16;
    constexpr uint32_t channel_size_128b_q = head_dim / num_elems_per_128b<DTypeQ>();
    const uint32_t lane_idx = threadIdx.x;
    uint32_t q_frag_local[2][4];
    static_assert(NUM_MMA_D % 4 == 0, "NUM_MMA_D must be a multiple of 4");
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
      uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
#pragma unroll
      for (uint32_t mma_di = 0; mma_di < NUM_MMA_D / 2; ++mma_di) {
//FIXME
#if 0  // disable MMA on ROCm platform
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
#endif // disable MMA on ROCm platform
        uint32_t q_smem_offset_r_last_half =
            q_smem->template advance_offset_by_column<NUM_MMA_D>(q_smem_offset_r_first_half, 0);
//FIXME
#if 0  // disable MMA on ROCm platform
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
#endif // disable MMA on ROCm platform
//FIXME
#if 0  // disable LLaMA-style RoPE for now
        q_frag_apply_llama_rope<DTypeQ>(
            (DTypeQ*)q_frag_local[0], (DTypeQ*)q_frag_local[1], rope_freq[mma_di],
            q_packed_idx + kv_len * group_size - qo_len * group_size + mma_q * 16 + lane_idx / 4,
            group_size);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half =
            q_smem->template advance_offset_by_column<2>(q_smem_offset_r_first_half, mma_di);
#endif // disable LLaMA-style RoPE for now
      }
      *q_smem_offset_r += 16 * channel_size_128b_q;
    }
    *q_smem_offset_r -= NUM_MMA_Q * 16 * channel_size_128b_q;
  }
}

template <uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV, uint32_t NUM_MMA_Q, uint32_t NUM_MMA_D,
          SwizzleMode swizzle_mode, typename DTypeQ, typename IdType>
__device__ __forceinline__ void q_smem_inplace_apply_rotary_with_pos(
    const uint32_t q_packed_idx_base, const IdType* q_offset, smem_t<swizzle_mode>* q_smem,
    const uint_fastdiv group_size, uint32_t* q_smem_offset_r, float (*rope_freq)[4]) {
  if (get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>() == 0) {
    constexpr uint32_t head_dim = NUM_MMA_D * 16;
    constexpr uint32_t channel_size_128b_q = head_dim / num_elems_per_128b<DTypeQ>();
    const uint32_t lane_idx = threadIdx.x;
    uint32_t q_frag_local[2][4];
    static_assert(NUM_MMA_D % 4 == 0, "NUM_MMA_D must be a multiple of 4");
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
      uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
#pragma unroll
      for (uint32_t mma_di = 0; mma_di < NUM_MMA_D / 2; ++mma_di) {
//FIXME
#if 0  // disable MMA on ROCm platform
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
#endif // disable MMA on ROCm platform
        uint32_t q_smem_offset_r_last_half =
            q_smem->template advance_offset_by_column<NUM_MMA_D>(q_smem_offset_r_first_half, 0);
//FIXME
#if 0  // disable MMA on ROCm platform
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
#endif // disable MMA on ROCm platform
        q_frag_apply_llama_rope_with_pos<DTypeQ>(
            (DTypeQ*)q_frag_local[0], (DTypeQ*)q_frag_local[1], rope_freq[mma_di],
            q_packed_idx_base + mma_q * 16 + lane_idx / 4, group_size, q_offset);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half =
            q_smem->template advance_offset_by_column<2>(q_smem_offset_r_first_half, mma_di);
      }
      *q_smem_offset_r += 16 * channel_size_128b_q;
    }
    *q_smem_offset_r -= NUM_MMA_Q * 16 * channel_size_128b_q;
  }
}

template <uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV, uint32_t NUM_MMA_Q, uint32_t NUM_MMA_D,
          SwizzleMode swizzle_mode, typename AttentionVariant>
__device__ __forceinline__ void q_smem_inplace_transform(
    const typename AttentionVariant::ParamsT& params, AttentionVariant variant,
    smem_t<swizzle_mode>* q_smem) {
  if (CUDA_WARP_SIZE <= threadIdx.x) {
    return;
  }

  using DTypeQ = typename AttentionVariant::DTypeQ;
  const uint32_t warp_idx = get_warp_idx<NUM_WARPS_Q, NUM_WARPS_KV>(), lane_idx = threadIdx.x;
  constexpr uint32_t head_dim = NUM_MMA_D * 16;
  constexpr uint32_t channel_size_128b_q = head_dim / num_elems_per_128b<DTypeQ>();
  constexpr uint32_t num_warps = NUM_WARPS_Q * NUM_WARPS_KV;
#pragma unroll
  for (uint32_t i = 0; i < NUM_MMA_Q * head_dim / (NUM_WARPS_KV * 16); ++i) {
    vec_t<DTypeQ, 8> tmp;
    tmp.load((DTypeQ*)(q_smem->base) + (i * num_warps + warp_idx) * 256 + lane_idx * 8);
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      tmp[reg_id] = variant.QueryTransform(params, tmp[reg_id]);
    }
    tmp.store((DTypeQ*)(q_smem->base) + (i * num_warps + warp_idx) * 256 + lane_idx * 8);
  }
}

template <uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV, uint32_t NUM_MMA_D, uint32_t NUM_MMA_KV,
          SwizzleMode swizzle_mode, typename DTypeKV>
__device__ __forceinline__ void k_smem_inplace_apply_rotary(const uint32_t kv_idx_base,
                                                            smem_t<swizzle_mode>* k_smem,
                                                            uint32_t* k_smem_offset_r,
                                                            float (*rope_freq)[4]) {
  static_assert(sizeof(DTypeKV) == 2);
  constexpr uint32_t head_dim = NUM_MMA_D * 16;
  constexpr uint32_t channel_size_128b_kv = head_dim / num_elems_per_128b<DTypeKV>();
  uint32_t k_frag_local[2][4];
  const uint32_t lane_idx = threadIdx.x;
  if constexpr (NUM_MMA_D == 4 && NUM_WARPS_Q == 4) {
    static_assert(NUM_WARPS_KV == 1);
    const uint32_t warp_idx = get_warp_idx_q<NUM_WARPS_Q, NUM_WARPS_KV>();
    // horizontal-axis: y
    // horizontal-axis: y
    // vertical-axis: z
    //         | 1-16       | 16-32      | 32-48      | 48-64      |
    // | 1-16  | warp_idx=0 | warp_idx=1 | warp_idx=0 | warp_idx=1 |
    // | 16-32 | warp_idx=2 | warp_idx=3 | warp_idx=2 | warp_idx=3 |
    static_assert(NUM_MMA_KV % 2 == 0, "when NUM_MMA_D == 4, NUM_MMA_KV must be a multiple of 2");
    uint32_t kv_idx = kv_idx_base + (warp_idx / 2) * 16 + lane_idx / 4;
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) + (warp_idx / 2) * 16 * channel_size_128b_kv;
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV / 2; ++i) {
      // uint32_t mma_kv = warp_idx / 2 + i * 2;
      uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
      uint32_t mma_di = (warp_idx % 2);
      k_smem->ldmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
      uint32_t k_smem_offset_r_last_half =
          k_smem->template advance_offset_by_column<4>(k_smem_offset_r_first_half, 0);
      k_smem->ldmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
      k_frag_apply_llama_rope<DTypeKV>((DTypeKV*)k_frag_local[0], (DTypeKV*)k_frag_local[1],
                                       rope_freq[mma_di], kv_idx);
      k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
      k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
      *k_smem_offset_r += 32 * channel_size_128b_kv;
      kv_idx += 32;
    }
    *k_smem_offset_r = (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) -
                       ((warp_idx / 2) + NUM_MMA_KV) * 16 * channel_size_128b_kv;
  } else {
    const uint32_t warp_idx_x = get_warp_idx_q<NUM_WARPS_Q, NUM_WARPS_KV>(),
                   warp_idx_z = get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>();
    static_assert(NUM_MMA_D % (2 * NUM_WARPS_Q) == 0);
    // horizontal axis: y
    // vertical axis: z
    // | (warp_idx_z, warp_idx_x)       | 1-16   | 16-32  | 32-48  | 48-64  | ...
    // | 1-16*NUM_MMA_KV               | (0, 0) | (0, 1) | (0, 2) | (0, 3) | ...
    // | 16*NUM_MMA_KV-32*NUM_MMA_KV  | (1, 0) | (1, 1) | (1, 2) | (1, 3) | ...
    // ...
    uint32_t kv_idx = kv_idx_base + (warp_idx_z * NUM_MMA_KV * 16) + lane_idx / 4;
    *k_smem_offset_r = *k_smem_offset_r ^ (0x2 * warp_idx_x);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV; ++i) {
      uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / (2 * NUM_WARPS_Q); ++j) {
        uint32_t mma_di = warp_idx_x + j * NUM_WARPS_Q;
        k_smem->ldmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
        uint32_t k_smem_offset_r_last_half =
            k_smem->template advance_offset_by_column<NUM_MMA_D>(k_smem_offset_r_first_half, 0);
        k_smem->ldmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
        k_frag_apply_llama_rope<DTypeKV>((DTypeKV*)k_frag_local[0], (DTypeKV*)k_frag_local[1],
                                         rope_freq[mma_di], kv_idx);
        k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
        k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
        k_smem_offset_r_first_half = k_smem->template advance_offset_by_column<2 * NUM_WARPS_Q>(
            k_smem_offset_r_first_half, mma_di);
      }
      *k_smem_offset_r += 16 * channel_size_128b_kv;
      kv_idx += 16;
    }
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * warp_idx_x)) - NUM_MMA_KV * 16 * channel_size_128b_kv;
  }
}

//FIXME
template <uint32_t NUM_MMA_Q, uint32_t NUM_MMA_D, uint32_t NUM_MMA_KV, SwizzleMode swizzle_mode_q,
          SwizzleMode swizzle_mode_kv, typename DTypeQ, typename DTypeKV, typename DTypeQKAccum>
__device__ __forceinline__ void compute_qk(smem_t<swizzle_mode_q>* q_smem,
                                           uint32_t* q_smem_offset_r,
                                           smem_t<swizzle_mode_kv>* k_smem,
                                           uint32_t* k_smem_offset_r,
                                           DTypeQKAccum (*s_frag)[NUM_MMA_KV][4]) {
  constexpr uint32_t head_dim = NUM_MMA_D * 16;
  constexpr uint32_t channel_size_128b_q = head_dim / num_elems_per_128b<DTypeQ>();
  constexpr uint32_t channel_size_128b_kv = head_dim / num_elems_per_128b<DTypeKV>();

  const uint32_t real_lane_idx = threadIdx.x;

  using ab_frag_type = typename mfma_m16n16k16_f32<DTypeQ>::ab_fragment_type;
  using c_frag_type = typename mfma_m16n16k16_f32<DTypeQ>::c_fragment_type;

  ab_frag_type a_frag[NUM_MMA_Q];
  ab_frag_type b_frag;

  // compute q*k^T
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D; ++mma_d) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
//FIXME
#if 0  // disable MMA on ROCm platform
      q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[mma_q]);
#else
      // thread 0 read first 4 elements, thread 16 read last 4 elements, etc...
      auto* values = reinterpret_cast<unsigned char*>(q_smem->base + *q_smem_offset_r) \
         + sizeof(ab_frag_type) * (real_lane_idx / 16 % 2);
      memcpy(a_frag + mma_q, values, sizeof(ab_frag_type));
#endif // disable MMA on ROCm platform
      *q_smem_offset_r =
          q_smem->template advance_offset_by_row<16, channel_size_128b_q>(*q_smem_offset_r);
    }

    *q_smem_offset_r = q_smem->template advance_offset_by_column<2>(*q_smem_offset_r, mma_d) -
                       NUM_MMA_Q * 16 * channel_size_128b_q;

#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
      if constexpr (sizeof(DTypeKV) == 1) {
        uint32_t b_frag_f8[2];
        if (mma_d % 2 == 0) {
//FIXME
#if 0  // disable MMA on ROCm platform
          k_smem->ldmatrix_m8n8x4_left_half(*k_smem_offset_r, b_frag_f8);
#endif // disable MMA on ROCm platform
        } else {
//FIXME
#if 0  // disable MMA on ROCm platform
          k_smem->ldmatrix_m8n8x4_right_half(*k_smem_offset_r, b_frag_f8);
#endif // disable MMA on ROCm platform
        }
        b_frag_f8[0] = frag_layout_swizzle_16b_to_8b(b_frag_f8[0]);
        b_frag_f8[1] = frag_layout_swizzle_16b_to_8b(b_frag_f8[1]);
        //vec_cast<DTypeQ, DTypeKV>::cast<8>((DTypeQ*)b_frag, (DTypeKV*)b_frag_f8);
//hipFIXED
        if constexpr(std::is_same<DTypeQ, __half>::value)
        if constexpr(std::is_same<DTypeKV, __half>::value)
        vec_cast<__half, __half>::cast<8>((DTypeQ*)b_frag, (DTypeKV*)b_frag_f8);
      } else {
//FIXME
#if 0  // disable MMA on ROCm platform
        k_smem->ldmatrix_m8n8x4(*k_smem_offset_r, b_frag);
#else
        // thread 0 read first 4 elements, thread 16 read last 4 elements, etc...
        auto* values = reinterpret_cast<unsigned char*>(k_smem->base + *k_smem_offset_r) \
          + sizeof(ab_frag_type) * (real_lane_idx / 16 % 2);
        memcpy(&b_frag, values, sizeof(ab_frag_type));
#endif // disable MMA on ROCm platform
      }
      *k_smem_offset_r =
          k_smem->template advance_offset_by_row<16, channel_size_128b_kv>(*k_smem_offset_r);

#pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
        if constexpr (std::is_same_v<DTypeQKAccum, float>) {
          if (mma_d == 0) {
//FIXME
#if 0  // disable MMA on ROCm platform
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeQ, MMAMode::kInit>(
                s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
#else
            *(c_frag_type*)s_frag[mma_q][mma_kv] = mfma_m16n16k16_f32<DTypeQ>::run(
              b_frag, a_frag[mma_q], c_frag_type{});
#endif // disable MMA on ROCm platform
          } else {
//FIXME
#if 0  // disable MMA on ROCm platform
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeQ>(s_frag[mma_q][mma_kv], a_frag[mma_q],
                                                              b_frag);
#else
            *(c_frag_type*)s_frag[mma_q][mma_kv] = mfma_m16n16k16_f32<DTypeQ>::run(
              b_frag, a_frag[mma_q], *(c_frag_type*)s_frag[mma_q][mma_kv]);
#endif // disable MMA on ROCm platform
          }
        } else if (std::is_same_v<DTypeQKAccum, half>) {
          if (mma_d == 0) {
//FIXME
#if 0  // disable MMA on ROCm platform
            mma::mma_sync_m16n16k16_row_col_f16f16f16<MMAMode::kInit>(
                (uint32_t*)s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
#endif // disable MMA on ROCm platform
          } else {
//FIXME
#if 0  // disable MMA on ROCm platform
            mma::mma_sync_m16n16k16_row_col_f16f16f16((uint32_t*)s_frag[mma_q][mma_kv],
                                                      a_frag[mma_q], b_frag);
#endif // disable MMA on ROCm platform
          }
        }
      }
    }
    if constexpr (sizeof(DTypeKV) == 1) {
      if (mma_d % 2 == 1) {
        *k_smem_offset_r =
            k_smem->template advance_offset_by_column<2>(*k_smem_offset_r, mma_d / 2);
      }
      *k_smem_offset_r -= NUM_MMA_KV * 16 * channel_size_128b_kv;
    } else {
      *k_smem_offset_r = k_smem->template advance_offset_by_column<2>(*k_smem_offset_r, mma_d) -
                         NUM_MMA_KV * 16 * channel_size_128b_kv;
    }
  }
  *q_smem_offset_r -= NUM_MMA_D * 2;
  *k_smem_offset_r -= NUM_MMA_D * sizeof(DTypeKV);
}

template <uint32_t NUM_MMA_Q, uint32_t NUM_MMA_D, uint32_t NUM_MMA_KV, typename AttentionVariant,
          typename DTypeQKAccum>
__device__ __forceinline__ void logits_transform(const typename AttentionVariant::ParamsT& params,
                                                 AttentionVariant variant, const uint32_t batch_idx,
                                                 const uint32_t qo_packed_idx_base,
                                                 const uint32_t kv_idx_base, const uint32_t qo_len,
                                                 const uint32_t kv_len,
                                                 const uint_fastdiv group_size,
                                                 DTypeQKAccum (*s_frag)[NUM_MMA_KV][4]) {
  const uint32_t real_lane_idx = threadIdx.x, kv_head_idx = blockIdx.z;
  uint32_t q[NUM_MMA_Q], r[NUM_MMA_Q];
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
    group_size.divmod(qo_packed_idx_base + mma_q * 16 + real_lane_idx % 16, q[mma_q],
                      r[mma_q]);
  }

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
        const uint32_t q_idx = q[mma_q], kv_idx = kv_idx_base + mma_kv * 16 +
                                                  (real_lane_idx / 16) * 4 + reg_id;
        const uint32_t qo_head_idx = kv_head_idx * group_size + r[mma_q];
        s_frag[mma_q][mma_kv][reg_id] =
            variant.LogitsTransform(params, s_frag[mma_q][mma_kv][reg_id], batch_idx, q_idx, kv_idx,
                                    qo_head_idx, kv_head_idx);
      }
    }
  }
}

template <MaskMode MASK_MODE, uint32_t NUM_MMA_Q, uint32_t NUM_MMA_D, uint32_t NUM_MMA_KV,
          typename AttentionVariant, typename DTypeQKAccum>
__device__ __forceinline__ void logits_mask(const typename AttentionVariant::ParamsT& params,
                                            AttentionVariant variant, const uint32_t batch_idx,
                                            const uint32_t qo_packed_idx_base,
                                            const uint32_t kv_idx_base, const uint32_t qo_len,
                                            const uint32_t kv_len, const uint32_t chunk_end,
                                            const uint_fastdiv group_size,
                                            DTypeQKAccum (*s_frag)[NUM_MMA_KV][4]) {
  const uint32_t real_lane_idx = threadIdx.x, kv_head_idx = blockIdx.z;
  uint32_t q[NUM_MMA_Q], r[NUM_MMA_Q];
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
    group_size.divmod(qo_packed_idx_base + mma_q * 16 + real_lane_idx % 16, q[mma_q],
                      r[mma_q]);
  }

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
        const uint32_t q_idx = q[mma_q], kv_idx = kv_idx_base + mma_kv * 16 +
                                                  (real_lane_idx / 16) * 4 + reg_id;
        const uint32_t qo_head_idx = kv_head_idx * group_size + r[mma_q];
        const bool mask =
            (!(MASK_MODE == MaskMode::kCausal
                   ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= chunk_end))
                   : kv_idx >= chunk_end)) &&
            variant.LogitsMask(params, batch_idx, q_idx, kv_idx, qo_head_idx, kv_head_idx);
        s_frag[mma_q][mma_kv][reg_id] =
            (mask) ? s_frag[mma_q][mma_kv][reg_id]
                   : (variant.use_softmax ? DTypeQKAccum(-math::inf) : DTypeQKAccum(0.f));
      }
    }
  }
}

template <uint32_t NUM_MMA_Q, uint32_t NUM_MMA_D, uint32_t NUM_MMA_KV, typename DTypeQKAccum,
          typename AttentionVariant>
__device__ __forceinline__ void update_mdo_states(AttentionVariant variant,
                                                  DTypeQKAccum (*s_frag)[NUM_MMA_KV][4],
                                                  float (*o_frag)[NUM_MMA_D][4],
                                                  DTypeQKAccum (*m)[1], float (*d)[1]) {
  if constexpr (variant.use_softmax) {
    if constexpr (std::is_same_v<DTypeQKAccum, float>) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t j = 0; j < 1; ++j) {
          float m_prev = m[mma_q][j];
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
            float m_local =
                max(max(s_frag[mma_q][mma_kv][j * 2 + 0], s_frag[mma_q][mma_kv][j * 2 + 1]),
                    max(s_frag[mma_q][mma_kv][j * 2 + 2], s_frag[mma_q][mma_kv][j * 2 + 3]));
            m[mma_q][j] = max(m[mma_q][j], m_local);
          }
          m[mma_q][j] = max(m[mma_q][j], math::shfl_xor_sync(m[mma_q][j], 0x10));
          m[mma_q][j] = max(m[mma_q][j], math::shfl_xor_sync(m[mma_q][j], 0x20));

          float o_scale = math::ptx_exp2(m_prev - m[mma_q][j]);
          d[mma_q][j] *= o_scale;
#pragma unroll
          for (uint32_t mma_d = 0; mma_d < NUM_MMA_D; ++mma_d) {
            o_frag[mma_q][mma_d][j * 2 + 0] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 1] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 2] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 3] *= o_scale;
          }
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
            s_frag[mma_q][mma_kv][j * 2 + 0] =
                math::ptx_exp2(s_frag[mma_q][mma_kv][j * 2 + 0] - m[mma_q][j]);
            s_frag[mma_q][mma_kv][j * 2 + 1] =
                math::ptx_exp2(s_frag[mma_q][mma_kv][j * 2 + 1] - m[mma_q][j]);
            s_frag[mma_q][mma_kv][j * 2 + 2] =
                math::ptx_exp2(s_frag[mma_q][mma_kv][j * 2 + 2] - m[mma_q][j]);
            s_frag[mma_q][mma_kv][j * 2 + 3] =
                math::ptx_exp2(s_frag[mma_q][mma_kv][j * 2 + 3] - m[mma_q][j]);
          }
        }
      }
    } else if constexpr (std::is_same_v<DTypeQKAccum, half>) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
        half m_prev[2];
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          m_prev[j] = m[mma_q][j];
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
            __hip_bfloat162 temp_bf162 = __hmax2(convert_half2_to_bfloat162(*(half2*)&s_frag[mma_q][mma_kv][j * 2]),
                                                 convert_half2_to_bfloat162(*(half2*)&s_frag[mma_q][mma_kv][j * 2 + 4]));
            half2 m_local = convert_bfloat162_to_half2(temp_bf162);
#else
            half2 m_local = __hmax2(*(half2*)&s_frag[mma_q][mma_kv][j * 2],
                                    *(half2*)&s_frag[mma_q][mma_kv][j * 2 + 4]);
#endif
            m[mma_q][j] = __hmax(m[mma_q][j], __hmax(m_local.x, m_local.y));
          }
        }
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
        __hip_bfloat162 temp_bf162 = __hmax2(convert_half2_to_bfloat162(*(half2*)&m[mma_q]),
                                             convert_half2_to_bfloat162(math::shfl_xor_sync(*(half2*)&m[mma_q], 0x2)));
        *(half2*)&m[mma_q] = convert_bfloat162_to_half2(temp_bf162);

        temp_bf162 = __hmax2(convert_half2_to_bfloat162(*(half2*)&m[mma_q]),
                             convert_half2_to_bfloat162(math::shfl_xor_sync(*(half2*)&m[mma_q], 0x1)));
        *(half2*)&m[mma_q] = convert_bfloat162_to_half2(temp_bf162);
#else
        *(half2*)&m[mma_q] =
            __hmax2(*(half2*)&m[mma_q], math::shfl_xor_sync(*(half2*)&m[mma_q], 0x2));
        *(half2*)&m[mma_q] =
            __hmax2(*(half2*)&m[mma_q], math::shfl_xor_sync(*(half2*)&m[mma_q], 0x1));
#endif
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          float o_scale = math::ptx_exp2(float(m_prev[j] - m[mma_q][j]));
          d[mma_q][j] *= o_scale;
#pragma unroll
          for (uint32_t mma_d = 0; mma_d < NUM_MMA_D; ++mma_d) {
            o_frag[mma_q][mma_d][j * 2 + 0] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 1] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 4] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 5] *= o_scale;
          }
          half2 m2 = make_half2(m[mma_q][j], m[mma_q][j]);
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
//FIXME
#if 0  // disable PTX exp2() on ROCm platform
            *(half2*)&s_frag[mma_q][mma_kv][j * 2] =
                math::ptx_exp2(*(half2*)&s_frag[mma_q][mma_kv][j * 2] - m2);
            *(half2*)&s_frag[mma_q][mma_kv][j * 2 + 4] =
                math::ptx_exp2(*(half2*)&s_frag[mma_q][mma_kv][j * 2 + 4] - m2);
#endif // disable PTX exp2() on ROCm platform
          }
        }
      }
    }
  }
}

template <uint32_t NUM_MMA_Q, uint32_t NUM_MMA_D, uint32_t NUM_MMA_KV, SwizzleMode swizzle_mode,
          typename DTypeQ, typename DTypeKV, typename DTypeQKAccum, typename AttentionVariant>
__device__ __forceinline__ void compute_sfm_v(AttentionVariant variant,
                                              smem_t<swizzle_mode>* v_smem,
                                              uint32_t i, uint32_t j,
                                              DTypeQKAccum (*s_frag)[NUM_MMA_KV][4],
                                              float (*o_frag)[NUM_MMA_D][4], float (*d)[1]) {
  constexpr uint32_t head_dim = NUM_MMA_D * 16;
  constexpr uint32_t channel_size_128b_kv = head_dim / num_elems_per_128b<DTypeKV>();
  const uint32_t real_lane_idx = threadIdx.x;

  DTypeQ s_frag_f16[NUM_MMA_Q][NUM_MMA_KV][4];
  if constexpr (std::is_same_v<DTypeQKAccum, float>) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
        //vec_cast<DTypeQ, float>::cast<8>(s_frag_f16[mma_q][mma_kv], s_frag[mma_q][mma_kv]);
//hipFIXED
        vec_cast<DTypeQ, float>::template cast<4>(s_frag_f16[mma_q][mma_kv], s_frag[mma_q][mma_kv]);
      }
    }
  }

  if constexpr (variant.use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
        if constexpr (std::is_same_v<DTypeQKAccum, float>) {
//FIXME
#if 0  // disable MMA on ROCm platform
          mma::rowsum_f16f16f32(d[mma_q], s_frag_f16[mma_q][mma_kv]);
#else
          DTypeQKAccum local_rowsum = (
            T2float<DTypeQ>(s_frag_f16[mma_q][mma_kv][0]) + T2float<DTypeQ>(s_frag_f16[mma_q][mma_kv][1]) + 
            T2float<DTypeQ>(s_frag_f16[mma_q][mma_kv][2]) + T2float<DTypeQ>(s_frag_f16[mma_q][mma_kv][3])
          );
          local_rowsum = local_rowsum + math::shfl_xor_sync(local_rowsum, 0x10);
          local_rowsum = local_rowsum + math::shfl_xor_sync(local_rowsum, 0x20);

          d[mma_q][0] = d[mma_q][0] + local_rowsum;
#endif // disable MMA on ROCm platform
        } else {
//FIXME
#if 0  // disable MMA on ROCm platform
          mma::rowsum_f16f16f32(d[mma_q], s_frag[mma_q][mma_kv]);
#endif // disable MMA on ROCm platform
        }
      }
    }
  }

  using ab_frag_type = typename mfma_m16n16k16_f32<DTypeKV>::ab_fragment_type;
  using c_frag_type = typename mfma_m16n16k16_f32<DTypeKV>::c_fragment_type;

#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < NUM_MMA_D; ++mma_d) {
      ab_frag_type b_frag;
      if constexpr (sizeof(DTypeKV) == 1) {
        uint32_t b_frag_f8[2];
        if (mma_d % 2 == 0) {
//FIXME
#if 0  // disable MMA on ROCm platform
          v_smem->ldmatrix_m8n8x4_trans_left_half(*v_smem_offset_r, b_frag_f8);
#endif // disable MMA on ROCm platform
        } else {
//FIXME
#if 0  // disable MMA on ROCm platform
          v_smem->ldmatrix_m8n8x4_trans_right_half(*v_smem_offset_r, b_frag_f8);
#endif // disable MMA on ROCm platform
        }
        b_frag_f8[0] = frag_layout_swizzle_16b_to_8b_trans(b_frag_f8[0]);
        b_frag_f8[1] = frag_layout_swizzle_16b_to_8b_trans(b_frag_f8[1]);
        //vec_cast<DTypeQ, DTypeKV>::cast<8>((DTypeQ*)b_frag, (DTypeKV*)b_frag_f8);
//hipFIXED
        if constexpr(std::is_same<DTypeQ, __half>::value)
        if constexpr(std::is_same<DTypeKV, __half>::value)
        vec_cast<__half, __half>::cast<8>((DTypeQ*)b_frag, (DTypeKV*)b_frag_f8);
        swap(reinterpret_cast<uint32_t*>(&b_frag)[1], reinterpret_cast<uint32_t*>(&b_frag)[2]);
      } else {
//FIXME
#if 0  // disable MMA on ROCm platform
        v_smem->ldmatrix_m8n8x4_trans(*v_smem_offset_r, b_frag);
#else
        DTypeKV data[4];
        static_assert(sizeof(DTypeKV) == 2);
        for (uint32_t elem = 0; elem < 4; ++elem) {
          uint32_t offset = v_smem->template get_permuted_offset<channel_size_128b_kv>(i + elem, j);
          data[elem] = reinterpret_cast<DTypeKV*>(v_smem->base + offset)[real_lane_idx % 8];
        }
        memcpy(&b_frag, data, sizeof(ab_frag_type));
#endif // disable MMA on ROCm platform
      }
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
        if constexpr (std::is_same_v<DTypeQKAccum, float>) {
//FIXME
#if 0  // disable MMA on ROCm platform
          mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeQ>(
              o_frag[mma_q][mma_d], (uint32_t*)(s_frag_f16[mma_q][mma_kv]), b_frag);
#else
          *(c_frag_type*)(o_frag[mma_q][mma_d]) = mfma_m16n16k16_f32<DTypeKV>::run(
            b_frag, *(ab_frag_type*)(s_frag_f16[mma_q][mma_kv]), *(c_frag_type*)(o_frag[mma_q][mma_d]));
#endif // disable MMA on ROCm platform
        } else {
//FIXME
#if 0  // disable MMA on ROCm platform
          mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeQ>(
              o_frag[mma_q][mma_d], (uint32_t*)s_frag[mma_q][mma_kv], b_frag);
#endif // disable MMA on ROCm platform
        }
      }
      if constexpr (sizeof(DTypeKV) == 1) {
        if (mma_d % 2 == 1) {
#if 0
          *v_smem_offset_r =
              v_smem->template advance_offset_by_column<2>(*v_smem_offset_r, mma_d / 2);
#endif
        }
      } else {
#if 0
        *v_smem_offset_r = v_smem->template advance_offset_by_column<2>(*v_smem_offset_r, mma_d);
#else
        j += 2;
#endif
      }
    }
#if 0
    *v_smem_offset_r =
        v_smem->template advance_offset_by_row<16, channel_size_128b_kv>(*v_smem_offset_r) -
        sizeof(DTypeKV) * NUM_MMA_D;
#else
    i += 16;
    j -= sizeof(DTypeKV) * NUM_MMA_D;
#endif
  }
#if 0
  *v_smem_offset_r -= 16 * NUM_MMA_KV * channel_size_128b_kv;
#endif
}

template <uint32_t NUM_MMA_Q, uint32_t NUM_MMA_D, typename DTypeQKAccum, typename AttentionVariant>
__device__ __forceinline__ void normalize_d(AttentionVariant variant, float (*o_frag)[NUM_MMA_D][4],
                                            DTypeQKAccum (*m)[1], float (*d)[1]) {
  if constexpr (variant.use_softmax) {
    float d_rcp[NUM_MMA_Q][1];
    // compute reciprocal of d
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 1; ++j) {
        d_rcp[mma_q][j] =
            (m[mma_q][j] != DTypeQKAccum(-math::inf)) ? math::ptx_rcp(d[mma_q][j]) : 0.f;
      }
    }

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D; ++mma_d) {
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
          o_frag[mma_q][mma_d][reg_id] =
              o_frag[mma_q][mma_d][reg_id] * d_rcp[mma_q][0];
        }
      }
    }
  }
}

/*!
 * \brief Synchronize the states of the MDO kernel across the threadblock along threadIdx.z.
 */
template <uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV, uint32_t NUM_MMA_Q, uint32_t NUM_MMA_D,
          typename DTypeQKAccum, typename AttentionVariant>
__device__ __forceinline__ void threadblock_sync_mdo_states(
    AttentionVariant variant, float (*o_frag)[NUM_MMA_D][4], float* smem_workspace,
    DTypeQKAccum (*m)[1], float (*d)[1], const uint32_t warp_idx, const uint32_t lane_idx) {
  // only necessary when blockDim.z > 1
  if constexpr (NUM_WARPS_KV > 1) {
    float2* smem_md = (float2*)(smem_workspace +
                                NUM_MMA_Q * NUM_MMA_D * NUM_WARPS_Q * NUM_WARPS_KV * WARP_SIZE * 8);
    // o: [num_warps, NUM_MMA_Q, NUM_MMA_D, WARP_SIZE(32), 8]
    // md: [num_warps, NUM_MMA_Q, 2, WARP_SIZE(32), 2 (m/d)]
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D; ++mma_d) {
        vec_t<float, 8>::memcpy(
            smem_workspace +
                (((warp_idx * NUM_MMA_Q + mma_q) * NUM_MMA_D + mma_d) * WARP_SIZE + lane_idx) * 8,
            o_frag[mma_q][mma_d]);
      }
    }

    if constexpr (variant.use_softmax) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t j = 0; j < 1; ++j) {
          smem_md[((warp_idx * NUM_MMA_Q + mma_q) * 2 + j) * WARP_SIZE + lane_idx] =
              make_float2(float(m[mma_q][j]), d[mma_q][j]);
        }
      }

      // synchronize m,d first
      __syncthreads();
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
        float o_scale[2][NUM_WARPS_KV];
#pragma unroll
        for (uint32_t j = 0; j < 1; ++j) {
          float m_new = -math::inf, d_new = 1.f;
#pragma unroll
          for (uint32_t i = 0; i < NUM_WARPS_KV; ++i) {
            float2 md = smem_md[(((i * NUM_WARPS_Q + get_warp_idx_q<NUM_WARPS_Q, NUM_WARPS_KV>()) *
                                      NUM_MMA_Q +
                                  mma_q) *
                                     2 +
                                 j) *
                                    WARP_SIZE +
                                lane_idx];
            float m_prev = m_new, d_prev = d_new;
            m_new = max(m_new, md.x);
            d_new = d_prev * math::ptx_exp2(m_prev - m_new) + md.y * math::ptx_exp2(md.x - m_new);
          }

#pragma unroll
          for (uint32_t i = 0; i < NUM_WARPS_KV; ++i) {
            float2 md = smem_md[(((i * NUM_WARPS_Q + get_warp_idx_q<NUM_WARPS_Q, NUM_WARPS_KV>()) *
                                      NUM_MMA_Q +
                                  mma_q) *
                                     2 +
                                 j) *
                                    WARP_SIZE +
                                lane_idx];
            float mi = md.x;
            o_scale[j][i] = math::ptx_exp2(float(mi - m_new));
          }
          m[mma_q][j] = DTypeQKAccum(m_new);
          d[mma_q][j] = d_new;
        }

#pragma unroll
        for (uint32_t mma_d = 0; mma_d < NUM_MMA_D; ++mma_d) {
          vec_t<float, 8> o_new;
          o_new.fill(0.f);
#pragma unroll
          for (uint32_t i = 0; i < NUM_WARPS_KV; ++i) {
            vec_t<float, 8> oi;
            oi.load(smem_workspace +
                    ((((i * NUM_WARPS_Q + get_warp_idx_q<NUM_WARPS_Q, NUM_WARPS_KV>()) * NUM_MMA_Q +
                       mma_q) *
                          NUM_MMA_D +
                      mma_d) *
                         WARP_SIZE +
                     lane_idx) *
                        8);
#pragma unroll
            for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
              o_new[reg_id] += oi[reg_id] * o_scale[(reg_id % 4) / 2][i];
            }
          }
          o_new.store(o_frag[mma_q][mma_d]);
        }
      }
    } else {
      // synchronize m,d first
      __syncthreads();
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < NUM_MMA_D; ++mma_d) {
          vec_t<float, 8> o_new;
          o_new.fill(0.f);
#pragma unroll
          for (uint32_t i = 0; i < NUM_WARPS_KV; ++i) {
            vec_t<float, 8> oi;
            oi.load(smem_workspace +
                    ((((i * NUM_WARPS_Q + get_warp_idx_q<NUM_WARPS_Q, NUM_WARPS_KV>()) * NUM_MMA_Q +
                       mma_q) *
                          NUM_MMA_D +
                      mma_d) *
                         WARP_SIZE +
                     lane_idx) *
                        8);
#pragma unroll
            for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
              o_new[reg_id] += oi[reg_id];
            }
          }
          o_new.store(o_frag[mma_q][mma_d]);
        }
      }
    }
  }
}

template <uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV, uint32_t NUM_MMA_Q, uint32_t NUM_MMA_D,
          SwizzleMode swizzle_mode, typename DTypeO>
__device__ __forceinline__ void write_o_reg_gmem(
    float (*o_frag)[NUM_MMA_D][4], smem_t<swizzle_mode>* o_smem, DTypeO* o_ptr_base,
    const uint32_t o_packed_idx_base, const uint32_t qo_upper_bound, const uint32_t o_stride_n,
    const uint32_t o_stride_h, const uint_fastdiv group_size) {
  constexpr uint32_t head_dim = NUM_MMA_D * 16;
  constexpr uint32_t channel_size_128b_out = head_dim / num_elems_per_128b<DTypeO>();
  const uint32_t warp_idx_x = get_warp_idx_q<NUM_WARPS_Q, NUM_WARPS_KV>();
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t real_lane_idx = threadIdx.x;

  if (get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>() == 0) {
#if 0 // TODO: use shared memory here to enable 128bit global memory coalesce writing
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D; ++mma_d) {
        uint32_t o_frag_f16[4];
        //vec_cast<DTypeO, float>::cast<8>((DTypeO*)o_frag_f16, o_frag[mma_q][mma_d]);
//hipFIXED
        if constexpr(std::is_same<DTypeO, __half>::value)
        vec_cast<__half, float>::cast<8>((DTypeO*)o_frag_f16, o_frag[mma_q][mma_d]);
#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
        uint32_t o_smem_offset_w = o_smem->template get_permuted_offset<channel_size_128b_out>(
            (warp_idx_x * NUM_MMA_Q + mma_q) * 16 + lane_idx % 16, mma_d * 2 + lane_idx / 16);
        o_smem->stmatrix_m8n8x4(o_smem_offset_w, o_frag_f16);
#else
        uint32_t o_smem_offset_w = o_smem->template get_permuted_offset<channel_size_128b_out>(
            (warp_idx_x * NUM_MMA_Q + mma_q) * 16 + lane_idx / 4, mma_d * 2);
        ((uint32_t*)(o_smem->base + o_smem_offset_w))[lane_idx % 4] = o_frag_f16[0];
        ((uint32_t*)(o_smem->base + o_smem_offset_w + 8 * channel_size_128b_out))[lane_idx % 4] =
            o_frag_f16[1];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[lane_idx % 4] = o_frag_f16[2];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) +
                     8 * channel_size_128b_out))[lane_idx % 4] = o_frag_f16[3];
#endif
      }
    }

    uint32_t o_smem_offset_w = o_smem->template get_permuted_offset<channel_size_128b_out>(
        warp_idx_x * NUM_MMA_Q * 16 + lane_idx / 8, lane_idx % 8);
#endif
    // FIXME: use more efficient way to write o_frag. for now, we are using its register layout.
    //        to change the layout, make sure we also update the o_ptr_base arg.
    DTypeO o_frag_f16[NUM_MMA_Q][NUM_MMA_D][4];
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_do = 0; mma_do < NUM_MMA_D; ++mma_do) {
        vec_cast<DTypeO, float>::template cast<4>(o_frag_f16[mma_q][mma_do], o_frag[mma_q][mma_do]);
      }
    }

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
      uint32_t q, r;
      group_size.divmod(o_packed_idx_base + real_lane_idx % 16 + mma_q * 16, q, r);
      const uint32_t o_idx = q;
      DTypeO* o_ptr = o_ptr_base + q * o_stride_n + r * o_stride_h;
#pragma unroll
      for (uint32_t mma_do = 0; mma_do < NUM_MMA_D; ++mma_do) {
        if (o_idx < qo_upper_bound) {
          memcpy(o_ptr, &o_frag_f16[mma_q][mma_do], sizeof(**o_frag_f16));
        }
        o_ptr += 16;
      }
    }
  }
}

}  // namespace

/*!
 * \brief FlashAttention prefill CUDA kernel for a single request.
 * \tparam partition_kv Whether to split kv_len into chunks.
 * \tparam mask_mode The mask mode used in the attention operation.
 * \tparam POS_ENCODING_MODE The positional encoding mode.
 * \tparam NUM_MMA_Q The number of fragments in x dimension.
 * \tparam NUM_MMA_D The number of fragments in y dimension.
 * \tparam NUM_MMA_KV The number of fragments in z dimension.
 * \tparam num_warps The number of warps in the threadblock.
 * \tparam DTypeQ The data type of the query tensor.
 * \tparam DTypeKV The data type of the key/value tensor.
 * \tparam DTypeO The data type of the output tensor.
 * \param q The query tensor.
 * \param k The key tensor.
 * \param v The value tensor.
 * \param o The output tensor.
 * \param tmp The temporary buffer (used when partition_kv is true).
 * \param lse The logsumexp value.
 * \param log2_rope_rcp_scale log2(1/(rope_scale)), where rope_scale is the scaling
 *   factor used in RoPE interpolation.
 * \param log2_rope_rcp_theta log2(1/(rope_theta)), where rope_theta is the theta
 *   used in RoPE.
 */
template <MaskMode MASK_MODE, flashinfer::PosEncodingMode POS_ENCODING_MODE, uint32_t NUM_MMA_Q,
          uint32_t NUM_MMA_D, uint32_t NUM_MMA_KV, uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV,
          typename DTypeQKAccum, typename AttentionVariant>
__global__
__launch_bounds__(NUM_WARPS_Q* NUM_WARPS_KV* WARP_SIZE) void SinglePrefillWithKVCacheKernel(
    const uint_fastdiv group_size,
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
    const typename AttentionVariant::ParamsT params) {
#else
    const __grid_constant__ typename AttentionVariant::ParamsT params) {
#endif
  using DTypeQ = typename AttentionVariant::DTypeQ;
#if (__CUDA_ARCH__ < 800)
  if constexpr (std::is_same_v<DTypeQ, gpu_bfloat16>) {
    FLASHINFER_RUNTIME_ASSERT("Prefill kernels do not support bf16 on sm75.");
  } else {
#endif
    using DTypeKV = typename AttentionVariant::DTypeKV;
    using DTypeO = typename AttentionVariant::DTypeO;
    DTypeQ* q = params.q;
    DTypeKV* k = params.k;
    DTypeKV* v = params.v;
    DTypeO* o = params.o;
    float* lse = params.lse;
    const uint32_t qo_len = params.qo_len;
    const uint32_t kv_len = params.kv_len;
    const bool partition_kv = params.partition_kv;
    const uint32_t q_stride_n = params.q_stride_n;
    const uint32_t q_stride_h = params.q_stride_h;
    const uint32_t kv_stride_n = params.kv_stride_n;
    const uint32_t kv_stride_h = params.kv_stride_h;
    const int32_t maybe_window_left = params.window_left;

    static_assert(sizeof(DTypeQ) == 2);
    static_assert(sizeof(DTypeO) == 2);
    const uint32_t lane_idx = threadIdx.x % CUDA_WARP_SIZE, 
                   real_lane_idx = threadIdx.x,
                   warp_idx = get_warp_idx<NUM_WARPS_Q, NUM_WARPS_KV>();
    const uint32_t bx = blockIdx.x, chunk_idx = blockIdx.y, kv_head_idx = blockIdx.z;
    const uint32_t num_kv_heads = gridDim.z, num_qo_heads = num_kv_heads * group_size;
    constexpr uint32_t num_rows_per_cta = NUM_MMA_Q * NUM_WARPS_Q * 16;
    const tensor_info_t qkv_info(qo_len, kv_len, num_qo_heads, num_kv_heads, q_stride_n, q_stride_h,
                                 kv_stride_n, kv_stride_h, /*head_dim=*/NUM_MMA_D * 16);

    const uint32_t num_chunks = gridDim.y;
    const uint32_t max_chunk_size = partition_kv ? ceil_div(kv_len, num_chunks) : kv_len;
    const uint32_t chunk_start = partition_kv ? chunk_idx * max_chunk_size : 0;
    const uint32_t chunk_end =
        partition_kv ? min((chunk_idx + 1) * max_chunk_size, kv_len) : kv_len;
    const uint32_t chunk_size = chunk_end - chunk_start;

    auto block = cg::this_thread_block();
    extern __shared__ uint8_t smem[];
    AttentionVariant variant(params, /*batch_idx=*/0, smem);
    const uint32_t window_left = variant.window_left;

    constexpr uint32_t head_dim = NUM_MMA_D * 16;
    constexpr uint32_t channel_size_128b_q = head_dim / num_elems_per_128b<DTypeQ>();
    constexpr uint32_t channel_size_128b_kv = head_dim / num_elems_per_128b<DTypeKV>();
    constexpr uint32_t channel_size_128b_out = head_dim / num_elems_per_128b<DTypeO>();

    DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][8];
    alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D][8];
    DTypeQKAccum m[NUM_MMA_Q][2];
    float d[NUM_MMA_Q][2];
    float rope_freq[NUM_MMA_D / 2][4];
    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      const float log2_rope_rcp_scale = params.log2_rope_rcp_scale;
      const float log2_rope_rcp_theta = params.log2_rope_rcp_theta;
      init_rope_freq<NUM_MMA_D>(rope_freq, log2_rope_rcp_scale, log2_rope_rcp_theta);
    }
    init_states<NUM_MMA_Q, NUM_MMA_D>(variant, o_frag, m, d);

    // cooperative fetch q fragment from gmem to reg
    const uint32_t qo_packed_idx_base =
        (bx * NUM_WARPS_Q + get_warp_idx_q<NUM_WARPS_Q, NUM_WARPS_KV>()) * NUM_MMA_Q * 16;
    constexpr SwizzleMode swizzle_mode_q = SwizzleMode::k128B;
    smem_t<swizzle_mode_q> qo_smem(smem);
    DTypeQ* q_ptr_base =
        q + qkv_info.get_q_elem_offset(0, kv_head_idx * group_size,
                                       (lane_idx % 8) * num_elems_per_128b<DTypeQ>());
    DTypeO* o_ptr_base =
        partition_kv
            ? o + chunk_idx * num_qo_heads * head_dim +
                  qkv_info.get_o_elem_offset(0, kv_head_idx * group_size,
                                             (lane_idx % 8) * num_elems_per_128b<DTypeO>())
            : o + qkv_info.get_o_elem_offset(0, kv_head_idx * group_size,
                                             (lane_idx % 8) * num_elems_per_128b<DTypeO>());

    uint32_t q_smem_offset_r = qo_smem.get_permuted_offset<channel_size_128b_q>(
        get_warp_idx_q<NUM_WARPS_Q, NUM_WARPS_KV>() * NUM_MMA_Q * 16 + lane_idx % 16,
        lane_idx / 16);

    load_q_global_smem<NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_Q, NUM_MMA_D>(
        qo_packed_idx_base, qo_len, q_ptr_base, q_stride_n, q_stride_h, group_size, &qo_smem);

    cp_async::commit_group();
    cp_async::wait_group<0>();
    block.sync();

    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      q_smem_inplace_apply_rotary<NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_Q, NUM_MMA_D, swizzle_mode_q,
                                  DTypeQ>(qo_packed_idx_base, qo_len, kv_len, group_size, &qo_smem,
                                          &q_smem_offset_r, rope_freq);
      block.sync();
    }
    q_smem_inplace_transform<NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_Q, NUM_MMA_D, swizzle_mode_q>(
        params, variant, &qo_smem);

    constexpr SwizzleMode swizzle_mode_kv =
        (sizeof(DTypeKV) == 1 && head_dim == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
    constexpr uint32_t kv_frag_rows = swizzle_mode_kv == SwizzleMode::k128B ? 4 : 8;
    constexpr uint32_t kv_frag_cols = swizzle_mode_kv == SwizzleMode::k128B ? 8 : 4;
    smem_t<swizzle_mode_kv> k_smem(smem +
                                   (NUM_WARPS_Q * NUM_MMA_Q * sizeof(DTypeQ)) * 16 * head_dim),
        v_smem(smem + (NUM_WARPS_Q * NUM_MMA_Q * sizeof(DTypeQ) +
                       NUM_WARPS_KV * NUM_MMA_KV * sizeof(DTypeKV)) *
                          16 * head_dim);

    const uint32_t num_iterations = ceil_div(
        MASK_MODE == MaskMode::kCausal
            ? min(chunk_size,
                  sub_if_greater_or_zero(
                      kv_len - qo_len + ((bx + 1) * num_rows_per_cta) / group_size, chunk_start))
            : chunk_size,
        16 * NUM_WARPS_KV * NUM_MMA_KV);

    const uint32_t window_iteration =
        ceil_div(sub_if_greater_or_zero(kv_len + (bx + 1) * num_rows_per_cta,
                                        qo_len + window_left + chunk_start),
                 (16 * NUM_WARPS_KV * NUM_MMA_KV));

    const uint32_t mask_iteration =
        (MASK_MODE == MaskMode::kCausal
             ? min(chunk_size,
                   sub_if_greater_or_zero(kv_len + (bx * num_rows_per_cta) / group_size - qo_len,
                                          chunk_start))
             : chunk_size) /
        (16 * NUM_WARPS_KV * NUM_MMA_KV);

    DTypeKV* k_ptr =
        k + qkv_info.get_kv_elem_offset(
                chunk_start + warp_idx * kv_frag_rows + lane_idx / kv_frag_cols, kv_head_idx,
                (lane_idx % kv_frag_cols) * num_elems_per_128b<DTypeKV>());
    DTypeKV* v_ptr =
        v + qkv_info.get_kv_elem_offset(
                chunk_start + warp_idx * kv_frag_rows + lane_idx / kv_frag_cols, kv_head_idx,
                (lane_idx % kv_frag_cols) * num_elems_per_128b<DTypeKV>());

    uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<channel_size_128b_kv>(
                 get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>() * NUM_MMA_KV * 16 +
                     8 * (lane_idx / 16) + lane_idx % 8,
                 (lane_idx % 16) / 8),
             v_smem_offset_r = v_smem.template get_permuted_offset<channel_size_128b_kv>(
                 get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>() * NUM_MMA_KV * 16 + lane_idx % 16,
                 lane_idx / 16),
             kv_smem_offset_w = k_smem.template get_permuted_offset<channel_size_128b_kv>(
                 warp_idx * kv_frag_rows + lane_idx / kv_frag_cols, lane_idx % kv_frag_cols);
    produce_kv<SharedMemFillMode::kNoFill, NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_D, NUM_MMA_KV>(
        k_smem, &kv_smem_offset_w, &k_ptr, kv_stride_n, 0, chunk_size);
    cp_async::commit_group();
    produce_kv<SharedMemFillMode::kFillZero, NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_D, NUM_MMA_KV>(
        v_smem, &kv_smem_offset_w, &v_ptr, kv_stride_n, 0, chunk_size);
    cp_async::commit_group();

#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      cp_async::wait_group<1>();
      block.sync();

      if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
        k_smem_inplace_apply_rotary<NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_D, NUM_MMA_KV,
                                    swizzle_mode_kv, DTypeKV>(
            chunk_start + iter * 16 * NUM_WARPS_KV * NUM_MMA_KV, &k_smem, &k_smem_offset_r,
            rope_freq);
        block.sync();
      }

      // compute attention score
      compute_qk<NUM_MMA_Q, NUM_MMA_D, NUM_MMA_KV, swizzle_mode_q, swizzle_mode_kv, DTypeQ,
                 DTypeKV>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);

      logits_transform<NUM_MMA_Q, NUM_MMA_D, NUM_MMA_KV>(
          params, variant, /*batch_idx=*/0, qo_packed_idx_base,
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>()) *
                            NUM_MMA_KV * 16,
          qo_len, kv_len, group_size, s_frag);

      // apply mask
      if (MASK_MODE == MaskMode::kCustom || (iter >= mask_iteration || iter < window_iteration)) {
        logits_mask<MASK_MODE, NUM_MMA_Q, NUM_MMA_D, NUM_MMA_KV>(
            params, variant, /*batch_idx=*/0, qo_packed_idx_base,
            chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>()) *
                              NUM_MMA_KV * 16,
            qo_len, kv_len, chunk_end, group_size, s_frag);
      }

      // compute m,d states in online softmax
      update_mdo_states<NUM_MMA_Q, NUM_MMA_D, NUM_MMA_KV>(variant, s_frag, o_frag, m, d);

      block.sync();
      produce_kv<SharedMemFillMode::kNoFill, NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_D, NUM_MMA_KV>(
          k_smem, &kv_smem_offset_w, &k_ptr, kv_stride_n,
          (iter + 1) * 16 * NUM_WARPS_KV * NUM_MMA_KV, chunk_size);
      cp_async::commit_group();
      cp_async::wait_group<1>();
      block.sync();

      // compute sfm*v
      compute_sfm_v<NUM_MMA_Q, NUM_MMA_D, NUM_MMA_KV, swizzle_mode_kv, DTypeQ, DTypeKV>(
          variant, &v_smem,
          get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>() * NUM_MMA_KV * 16 + (real_lane_idx / 16) * 4,
          (lane_idx % 16) / 8,
          s_frag, o_frag, d);

      block.sync();
      produce_kv<SharedMemFillMode::kFillZero, NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_D, NUM_MMA_KV>(
          v_smem, &kv_smem_offset_w, &v_ptr, kv_stride_n,
          (iter + 1) * 16 * NUM_WARPS_KV * NUM_MMA_KV, chunk_size);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    block.sync();

    // threadblock synchronization
    threadblock_sync_mdo_states<NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_Q, NUM_MMA_D, DTypeQKAccum>(
        variant, o_frag, (float*)smem, m, d, warp_idx, lane_idx);

    // normalize d
    normalize_d<NUM_MMA_Q, NUM_MMA_D>(variant, o_frag, m, d);

    // write back
    write_o_reg_gmem<NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_Q, NUM_MMA_D>(
        o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
        /*o_stride_n=*/
        partition_kv ? num_qo_heads * head_dim * num_chunks : num_qo_heads * head_dim,
        /*o_stride_h=*/head_dim, group_size);

    // write lse
    if constexpr (variant.use_softmax) {
      if (lse != nullptr || partition_kv) {
        if (get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>() == 0) {
#pragma unroll
          for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
            for (uint32_t j = 0; j < 1; ++j) {
              uint32_t q, r;
              group_size.divmod(qo_packed_idx_base + lane_idx / 4 + j * 8 + mma_q * 16, q, r);
              const uint32_t qo_head_idx = kv_head_idx * group_size + r;
              const uint32_t qo_idx = q;
              if (qo_idx < qo_len) {
                if (partition_kv) {
                  lse[(qo_idx * num_chunks + chunk_idx) * num_qo_heads + qo_head_idx] =
                      math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                } else {
                  lse[qo_idx * num_qo_heads + qo_head_idx] =
                      math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                }
              }
            }
          }
        }
      }
    }
#if (__CUDA_ARCH__ < 800)
  }
#endif
}

template <uint32_t HEAD_DIM, flashinfer::PosEncodingMode POS_ENCODING_MODE, bool ALLOW_FP16_QK_REDUCTION,
          MaskMode MASK_MODE, typename AttentionVariant>
gpuError_t SinglePrefillWithKVCacheDispatched(typename AttentionVariant::ParamsT params,
                                              typename AttentionVariant::DTypeO* tmp,
                                              gpuStream_t stream) {
  using DTypeQ = typename AttentionVariant::DTypeQ;
  using DTypeKV = typename AttentionVariant::DTypeKV;
  using DTypeO = typename AttentionVariant::DTypeO;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;
  const uint32_t qo_len = params.qo_len;
  const uint32_t kv_len = params.kv_len;
  if (kv_len < qo_len && MASK_MODE == MaskMode::kCausal) {
    std::ostringstream err_msg;
    err_msg << "When mask_mode is set to MaskMode::kCausal, kv_len must be greater than or equal "
               "to qo_len, got kv_len"
            << kv_len << " and qo_len " << qo_len;
    FLASHINFER_ERROR(err_msg.str());
  }

  const uint32_t group_size = num_qo_heads / num_kv_heads;
  const uint_fastdiv group_size_fastdiv(group_size);
  constexpr uint32_t NUM_MMA_D = HEAD_DIM / 16;
  uint32_t cta_tile_q = 0;
  int64_t unpacked_qo_len = qo_len * group_size;
  if (unpacked_qo_len > 64 && HEAD_DIM < 256) {
    cta_tile_q = 128;
  } else {
    auto compute_capacity = GetCudaComputeCapability();
    if (compute_capacity.first >= 8) {
      // Ampere or newer
      if (unpacked_qo_len > 16) {
        // avg_packed_qo_len <= 64
        cta_tile_q = 64;
      } else {
        // avg_packed_qo_len <= 16
        cta_tile_q = 16;
      }
    } else {
      // NOTE(Zihao): not enough shared memory on Turing for 1x4 warp layout
      cta_tile_q = 64;
    }
  }

  DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
    constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
    constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);
    constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);

    using DTypeQKAccum =
        typename std::conditional<ALLOW_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
                                  float>::type;

    int dev_id = 0;
    FLASHINFER_CUDA_CALL(gpuGetDevice(&dev_id));
    int max_smem_per_sm = 0;
    FLASHINFER_CUDA_CALL(gpuDeviceGetAttribute(
        &max_smem_per_sm, gpuDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
    // we expect each sm execute two threadblocks
    // TODO(Zihao): fix the following computation
    const int num_ctas_per_sm = max_smem_per_sm > (16 * HEAD_DIM * sizeof(DTypeQ) * 16) ? 2 : 1;
    const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

    const uint32_t max_num_mma_kv_reg =
        (HEAD_DIM >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
         !ALLOW_FP16_QK_REDUCTION)
            ? 2
            : (8 / NUM_MMA_Q);
    // TODO(Zihao): fix the following computation
    const uint32_t max_num_mma_kv_smem =
        (max_smem_per_threadblock / (16 * HEAD_DIM * sizeof(DTypeQ)) - NUM_MMA_Q * NUM_WARPS_Q) /
        (2 * NUM_WARPS_KV);

    // control NUM_MMA_KV for maximum warp occupancy
    DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
      if constexpr (is_invalid_configuration<POS_ENCODING_MODE, DTypeKV, DTypeQKAccum>(
                        NUM_MMA_Q, NUM_MMA_D, NUM_MMA_KV, NUM_WARPS_Q, NUM_WARPS_KV)) {
        // Invalid configuration, skip
        std::ostringstream err_msg;
        err_msg << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=" << NUM_MMA_Q
                << " NUM_MMA_D=" << NUM_MMA_D << " NUM_MMA_KV=" << NUM_MMA_KV
                << " NUM_WARPS_Q=" << NUM_WARPS_Q << " NUM_WARPS_KV=" << NUM_WARPS_KV
                << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                   " and report the issue to the developers.";
        FLASHINFER_ERROR(err_msg.str());
      } else {
        constexpr uint32_t num_threads = (NUM_WARPS_Q * NUM_WARPS_KV) * WARP_SIZE;
        constexpr uint32_t num_rows_per_cta = NUM_MMA_Q * NUM_WARPS_Q * 16;
        auto kernel = SinglePrefillWithKVCacheKernel<MASK_MODE, POS_ENCODING_MODE, NUM_MMA_Q,
                                                     NUM_MMA_D, NUM_MMA_KV, NUM_WARPS_Q,
                                                     NUM_WARPS_KV, DTypeQKAccum, AttentionVariant>;
        // TODO(Zihao): fix the following computation
        uint32_t smem_size = (NUM_MMA_Q * NUM_WARPS_Q * sizeof(DTypeQ) +
                              NUM_MMA_KV * NUM_WARPS_KV * 2 * sizeof(DTypeQ)) *
                             16 * HEAD_DIM;
        FLASHINFER_CUDA_CALL(
            gpuFuncSetAttribute(reinterpret_cast<const void*>(kernel), gpuFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        int num_blocks_per_sm = 0;
        int num_sm = 0;
        FLASHINFER_CUDA_CALL(
            gpuDeviceGetAttribute(&num_sm, gpuDevAttrMultiProcessorCount, dev_id));
        FLASHINFER_CUDA_CALL(gpuOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, kernel, num_threads, smem_size));
        uint32_t max_num_kv_chunks =
            (num_blocks_per_sm * num_sm) /
            (num_kv_heads * ceil_div(qo_len * group_size, num_rows_per_cta));
        uint32_t num_chunks;
        if (max_num_kv_chunks > 0) {
          uint32_t chunk_size = max(ceil_div(kv_len, max_num_kv_chunks), 256);
          num_chunks = ceil_div(kv_len, chunk_size);
        } else {
          num_chunks = 0;
        }

        if (num_chunks <= 1 || tmp == nullptr) {
          // Enough parallelism, do not split-kv
          params.partition_kv = false;
          void* args[] = {(void*)&group_size_fastdiv, (void*)&params};
          dim3 nblks(ceil_div(qo_len * group_size, num_rows_per_cta), 1, num_kv_heads);
          dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);
          FLASHINFER_CUDA_CALL(
              gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        } else {
          // Use cooperative groups to increase occupancy
          params.partition_kv = true;
          float* tmp_lse = (float*)(tmp + num_chunks * qo_len * num_qo_heads * HEAD_DIM);
          auto o = params.o;
          auto lse = params.lse;
          params.o = tmp;
          params.lse = tmp_lse;
          void* args[] = {(void*)&group_size_fastdiv, (void*)&params};
          dim3 nblks(ceil_div(qo_len * group_size, num_rows_per_cta), num_chunks, num_kv_heads);
          dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);
          FLASHINFER_CUDA_CALL(
              gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
          if constexpr (AttentionVariant::use_softmax) {
            FLASHINFER_CUDA_CALL(MergeStates(tmp, tmp_lse, o, lse, num_chunks, qo_len, num_qo_heads,
                                             HEAD_DIM, stream));
          } else {
            FLASHINFER_CUDA_CALL(
                AttentionSum(tmp, o, num_chunks, qo_len, num_qo_heads, HEAD_DIM, stream));
          }
        }
      }
    })
  });
  return gpuSuccess;
}

template <MaskMode MASK_MODE, PosEncodingMode POS_ENCODING_MODE, uint32_t NUM_MMA_Q,
          uint32_t NUM_MMA_D, uint32_t NUM_MMA_KV, uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV,
          typename DTypeQKAccum, typename AttentionVariant>
__global__
__launch_bounds__(NUM_WARPS_Q* NUM_WARPS_KV* WARP_SIZE) void BatchPrefillWithRaggedKVCacheKernel(
    const uint_fastdiv group_size,
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
    const typename AttentionVariant::ParamsT params) {
#else
    const __grid_constant__ typename AttentionVariant::ParamsT params) {
#endif
}

template <MaskMode MASK_MODE, PosEncodingMode POS_ENCODING_MODE, uint32_t NUM_MMA_Q,
          uint32_t NUM_MMA_D, uint32_t NUM_MMA_KV, uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_KV,
          typename DTypeQKAccum, typename AttentionVariant>
__global__
__launch_bounds__(NUM_WARPS_Q* NUM_WARPS_KV* WARP_SIZE) void BatchPrefillWithPagedKVCacheKernel(
    const uint_fastdiv group_size,
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
    const typename AttentionVariant::ParamsT params) {
#else
    const __grid_constant__ typename AttentionVariant::ParamsT params) {
#endif
  using DTypeQ = typename AttentionVariant::DTypeQ;

#if !(defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__))
  #if (__CUDA_ARCH__ < 800)
    if constexpr (std::is_same_v<DTypeQ, gpu_bfloat16>) {
      FLASHINFER_RUNTIME_ASSERT("Prefill kernels do not support bf16 on sm75.");
    } else {
  #endif
#endif
    using DTypeKV = typename AttentionVariant::DTypeKV;
    using DTypeO = typename AttentionVariant::DTypeO;
    using IdType = typename AttentionVariant::IdType;
    IdType* request_indices = params.request_indices;
    IdType* qo_tile_indices = params.qo_tile_indices;
    IdType* kv_tile_indices = params.kv_tile_indices;
    DTypeQ* q = params.q;
    IdType* q_indptr = params.q_indptr;
    IdType* q_offset = params.q_offset;
    IdType* o_indptr = params.o_indptr;
    DTypeO* o = params.o;
    float* lse = params.lse;
    bool* block_valid_mask = params.block_valid_mask;
    const paged_kv_t<DTypeKV, IdType>& paged_kv = params.paged_kv;
    const bool partition_kv = params.partition_kv;
    const int32_t maybe_window_left = params.window_left;

    static_assert(sizeof(DTypeQ) == 2);
    static_assert(sizeof(DTypeO) == 2);
    auto block = cg::this_thread_block();
    const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);

    const uint32_t bx = blockIdx.x, lane_idx = threadIdx.x % CUDA_WARP_SIZE,
                   real_lane_idx = threadIdx.x,
                   warp_idx = get_warp_idx<NUM_WARPS_Q, NUM_WARPS_KV>(), kv_head_idx = blockIdx.z;
    if (block_valid_mask && !block_valid_mask[bx]) {
      return;
    }
    const uint32_t num_kv_heads = gridDim.z, num_qo_heads = num_kv_heads * group_size;

    const uint32_t request_idx = request_indices[bx], qo_tile_idx = qo_tile_indices[bx],
                   kv_tile_idx = kv_tile_indices[bx];
    constexpr uint32_t num_rows_per_cta = NUM_MMA_Q * NUM_WARPS_Q * 16;
    extern __shared__ uint8_t smem[];
    AttentionVariant variant(params, /*batch_idx=*/request_idx, smem);
    const uint32_t qo_len = variant.qo_len, kv_len = variant.kv_len,
                   window_left = variant.window_left;
    const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;
    const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
    const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
    const uint32_t chunk_end =
        partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
    const uint32_t chunk_size = chunk_end - chunk_start;
    const uint32_t qo_upper_bound =
        min(qo_len, ceil_div((qo_tile_idx + 1) * num_rows_per_cta, group_size));

    constexpr uint32_t head_dim = NUM_MMA_D * 16;
    constexpr uint32_t channel_size_128b_q = head_dim / num_elems_per_128b<DTypeQ>();
    constexpr uint32_t channel_size_128b_kv = head_dim / num_elems_per_128b<DTypeKV>();
    constexpr uint32_t channel_size_128b_out = head_dim / num_elems_per_128b<DTypeO>();

    DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][4];
    alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D][4];
    DTypeQKAccum m[NUM_MMA_Q][1];
    float d[NUM_MMA_Q][1];
    float rope_freq[NUM_MMA_D / 2][4];

    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      const float log2_rope_rcp_scale = params.log2_rope_rcp_scale;
      const float log2_rope_rcp_theta = params.log2_rope_rcp_theta;
      init_rope_freq<NUM_MMA_D>(rope_freq, log2_rope_rcp_scale, log2_rope_rcp_theta);
    }
    init_states<NUM_MMA_Q, NUM_MMA_D>(variant, o_frag, m, d);

    const uint32_t qo_packed_idx_base =
        (qo_tile_idx * NUM_WARPS_Q + get_warp_idx_q<NUM_WARPS_Q, NUM_WARPS_KV>()) * NUM_MMA_Q * 16;
    const uint32_t q_stride_n = params.q_stride_n, q_stride_h = params.q_stride_h;
    constexpr SwizzleMode swizzle_mode_q = SwizzleMode::k128B;
    smem_t<swizzle_mode_q> qo_smem(smem);
    DTypeQ* q_ptr_base = q + get_elem_offset_impl(q_indptr[request_idx], kv_head_idx * group_size,
                                                  (lane_idx % 8) * num_elems_per_128b<DTypeQ>(),
                                                  q_stride_n, q_stride_h);
    DTypeO* o_ptr_base =
        partition_kv ? o + kv_tile_idx * num_qo_heads * head_dim +
                           get_elem_offset_impl(o_indptr[request_idx], kv_head_idx * group_size,
                                                real_lane_idx / 16 * 4,
                                                num_qo_heads * head_dim, head_dim)
                     : o + get_elem_offset_impl(o_indptr[request_idx], kv_head_idx * group_size,
                                                real_lane_idx / 16 * 4,
                                                num_qo_heads * head_dim, head_dim);
    uint32_t q_smem_offset_r = qo_smem.get_permuted_offset<channel_size_128b_q>(
        get_warp_idx_q<NUM_WARPS_Q, NUM_WARPS_KV>() * NUM_MMA_Q * 16 + lane_idx % 16,
        real_lane_idx / 16 / 2);

    load_q_global_smem<NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_Q, NUM_MMA_D>(
        qo_packed_idx_base, qo_upper_bound, q_ptr_base, q_stride_n, q_stride_h, group_size,
        &qo_smem);

    cp_async::commit_group();
    cp_async::wait_group<0>();
    block.sync();

    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      if (q_offset == nullptr) {
        q_smem_inplace_apply_rotary<NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_Q, NUM_MMA_D, swizzle_mode_q,
                                    DTypeQ>(qo_packed_idx_base, qo_len, kv_len, group_size,
                                            &qo_smem, &q_smem_offset_r, rope_freq);
      } else {
        q_smem_inplace_apply_rotary_with_pos<NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_Q, NUM_MMA_D,
                                             swizzle_mode_q, DTypeQ>(
            qo_packed_idx_base, q_offset + q_indptr[request_idx], &qo_smem, group_size,
            &q_smem_offset_r, rope_freq);
      }
      block.sync();
    }

    q_smem_inplace_transform<NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_Q, NUM_MMA_D, swizzle_mode_q>(
      params, variant, &qo_smem);

    constexpr SwizzleMode swizzle_mode_kv =
        (sizeof(DTypeKV) == 1 && head_dim == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
    constexpr uint32_t kv_frag_rows = swizzle_mode_kv == SwizzleMode::k128B ? 4 : 8;
    constexpr uint32_t kv_frag_cols = swizzle_mode_kv == SwizzleMode::k128B ? 8 : 4;
    smem_t<swizzle_mode_kv> k_smem(smem +
                                   (NUM_WARPS_Q * NUM_MMA_Q * sizeof(DTypeQ)) * 16 * head_dim),
        v_smem(smem + (NUM_WARPS_Q * NUM_MMA_Q * sizeof(DTypeQ) +
                       NUM_WARPS_KV * NUM_MMA_KV * sizeof(DTypeKV)) *
                          16 * head_dim);
    size_t kv_offset[NUM_MMA_KV * (swizzle_mode_kv == SwizzleMode::k128B ? 4 : 2) / NUM_WARPS_Q];

    uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<channel_size_128b_kv>(
                 get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>() * NUM_MMA_KV * 16 + lane_idx % 16,
                 real_lane_idx / 16 / 2),
             v_smem_offset_r = v_smem.template get_permuted_offset<channel_size_128b_kv>(
                 get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>() * NUM_MMA_KV * 16 + lane_idx % 16,
                 real_lane_idx / 16 / 2),
             kv_smem_offset_w = k_smem.template get_permuted_offset<channel_size_128b_kv>(
                 warp_idx * kv_frag_rows + lane_idx / kv_frag_cols, lane_idx % kv_frag_cols);
    const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

    uint32_t packed_page_iter_base =
        paged_kv.indptr[request_idx] * paged_kv.page_size + chunk_start;
#pragma unroll
    for (uint32_t i = 0;
         i < NUM_MMA_KV * (swizzle_mode_kv == SwizzleMode::k128B ? 4 : 2) / NUM_WARPS_Q; ++i) {
      uint32_t page_iter, entry_idx;
      paged_kv.page_size.divmod(packed_page_iter_base + warp_idx * kv_frag_rows +
                                    lane_idx / kv_frag_cols +
                                    kv_frag_rows * NUM_WARPS_Q * NUM_WARPS_KV * i,
                                page_iter, entry_idx);
      kv_offset[i] = paged_kv.protective_get_kv_offset(
          page_iter, kv_head_idx, entry_idx,
          (lane_idx % kv_frag_cols) * num_elems_per_128b<DTypeKV>(), last_indptr);
    }
    page_produce_kv<false, NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_D, NUM_MMA_KV>(
        k_smem, &kv_smem_offset_w, paged_kv, 0, kv_offset, chunk_size);
    cp_async::commit_group();
    page_produce_kv<true, NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_D, NUM_MMA_KV>(
        v_smem, &kv_smem_offset_w, paged_kv, 0, kv_offset, chunk_size);
    cp_async::commit_group();

    const uint32_t num_iterations = ceil_div(
        (MASK_MODE == MaskMode::kCausal
             ? min(chunk_size,
                   sub_if_greater_or_zero(
                       kv_len - qo_len + ((qo_tile_idx + 1) * num_rows_per_cta) / group_size,
                       chunk_start))
             : chunk_size),
        16 * NUM_WARPS_KV * NUM_MMA_KV);

    const uint32_t window_iteration =
        ceil_div(sub_if_greater_or_zero(kv_len + (qo_tile_idx + 1) * num_rows_per_cta,
                                        qo_len + window_left + chunk_start),
                 (16 * NUM_WARPS_KV * NUM_MMA_KV));

    const uint32_t mask_iteration =
        (MASK_MODE == MaskMode::kCausal
             ? min(chunk_size, sub_if_greater_or_zero(
                                   kv_len + (qo_tile_idx * num_rows_per_cta) / group_size - qo_len,
                                   chunk_start))
             : chunk_size) /
        (16 * NUM_WARPS_KV * NUM_MMA_KV);

#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      packed_page_iter_base += 16 * NUM_WARPS_KV * NUM_MMA_KV;
#pragma unroll
      for (uint32_t i = 0;
           i < NUM_MMA_KV * (swizzle_mode_kv == SwizzleMode::k128B ? 4 : 2) / NUM_WARPS_Q; ++i) {
        uint32_t page_iter, entry_idx;
        paged_kv.page_size.divmod(packed_page_iter_base + warp_idx * kv_frag_rows +
                                      lane_idx / kv_frag_cols +
                                      kv_frag_rows * NUM_WARPS_Q * NUM_WARPS_KV * i,
                                  page_iter, entry_idx);
        kv_offset[i] = paged_kv.protective_get_kv_offset(
            page_iter, kv_head_idx, entry_idx,
            (lane_idx % kv_frag_cols) * num_elems_per_128b<DTypeKV>(), last_indptr);
      }
      cp_async::wait_group<1>();
      block.sync();

      if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
        k_smem_inplace_apply_rotary<NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_D, NUM_MMA_KV,
                                    swizzle_mode_kv, DTypeKV>(
            (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[request_idx]) +
                chunk_start + iter * 16 * NUM_WARPS_KV * NUM_MMA_KV,
            &k_smem, &k_smem_offset_r, rope_freq);
        block.sync();
      }

      // compute attention score
      __syncthreads();
      compute_qk<NUM_MMA_Q, NUM_MMA_D, NUM_MMA_KV, swizzle_mode_q, swizzle_mode_kv, DTypeQ,
                 DTypeKV>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);

      logits_transform<NUM_MMA_Q, NUM_MMA_D, NUM_MMA_KV>(
          params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>()) *
                            NUM_MMA_KV * 16,
          qo_len, kv_len, group_size, s_frag);

      // apply mask
      if (MASK_MODE == MaskMode::kCustom || (iter >= mask_iteration || iter < window_iteration)) {
        logits_mask<MASK_MODE, NUM_MMA_Q, NUM_MMA_D, NUM_MMA_KV>(
            params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
            chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>()) *
                              NUM_MMA_KV * 16,
            qo_len, kv_len, chunk_end, group_size, s_frag);
      }

      // compute m,d states in online softmax
      update_mdo_states<NUM_MMA_Q, NUM_MMA_D, NUM_MMA_KV>(variant, s_frag, o_frag, m, d);

      block.sync();
      page_produce_kv<false, NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_D, NUM_MMA_KV>(
          k_smem, &kv_smem_offset_w, paged_kv, (iter + 1) * 16 * NUM_WARPS_KV * NUM_MMA_KV,
          kv_offset, chunk_size);
      cp_async::commit_group();
      cp_async::wait_group<1>();
      block.sync();

      // compute sfm*v
      compute_sfm_v<NUM_MMA_Q, NUM_MMA_D, NUM_MMA_KV, swizzle_mode_kv, DTypeQ, DTypeKV>(
          variant, &v_smem,
          get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>() * NUM_MMA_KV * 16 + (real_lane_idx / 16) * 4,
          (lane_idx % 16) / 8,
          s_frag, o_frag, d);

      block.sync();
      page_produce_kv<true, NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_D, NUM_MMA_KV>(
          v_smem, &kv_smem_offset_w, paged_kv, (iter + 1) * 16 * NUM_WARPS_KV * NUM_MMA_KV,
          kv_offset, chunk_size);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    block.sync();

    // threadblock synchronization
    threadblock_sync_mdo_states<NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_Q, NUM_MMA_D, DTypeQKAccum>(
        variant, o_frag, (float*)smem, m, d, warp_idx, lane_idx);

    // normalize d
    normalize_d<NUM_MMA_Q, NUM_MMA_D>(variant, o_frag, m, d);

    const uint32_t num_kv_chunks = (kv_len_safe + kv_chunk_size - 1) / kv_chunk_size;

    // write_back
    write_o_reg_gmem<NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_Q, NUM_MMA_D>(
        o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
        /*o_stride_n=*/
        partition_kv ? num_qo_heads * head_dim * num_kv_chunks : num_qo_heads * head_dim,
        /*o_stride_h=*/head_dim, group_size);

    // write lse
    if constexpr (variant.use_softmax) {
      if (lse != nullptr) {
        if (get_warp_idx_kv<NUM_WARPS_Q, NUM_WARPS_KV>() == 0) {
#pragma unroll
          for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
            for (uint32_t j = 0; j < 1; ++j) {
              uint32_t q, r;
              group_size.divmod(qo_packed_idx_base + real_lane_idx % 16 + mma_q * 16, q, r);
              const uint32_t qo_head_idx = kv_head_idx * group_size + r;
              const uint32_t qo_idx = q;
              if (qo_idx < qo_upper_bound) {
                if (partition_kv) {
                  lse[(o_indptr[request_idx] + qo_idx * num_kv_chunks + kv_tile_idx) *
                          num_qo_heads +
                      qo_head_idx] = math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                } else {
                  lse[(o_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
                      math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                }
              }
            }
          }
        }
      }
    }
#if !(defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__))
  #if (__CUDA_ARCH__ < 800)
    }
  #endif
#endif
}

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
          bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename AttentionVariant>
gpuError_t BatchPrefillWithRaggedKVCacheDispatched(typename AttentionVariant::ParamsT params,
                                                   typename AttentionVariant::DTypeO* tmp_v,
                                                   float* tmp_s, gpuStream_t stream) {
  using DTypeQ = typename AttentionVariant::DTypeQ;
  using DTypeKV = typename AttentionVariant::DTypeKV;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;
  const uint_fastdiv group_size_fastdiv(num_qo_heads / num_kv_heads);
  constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);

  if (padded_batch_size == 0) {
    // No request, skip
    // this won't happen in CUDAGraph mode because we fixed the padded_batch_size
    return gpuSuccess;
  }

  dim3 nblks(padded_batch_size, 1, num_kv_heads);
  dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);
  constexpr uint32_t NUM_MMA_D = HEAD_DIM / 16;
  using DTypeQKAccum =
      typename std::conditional<ALLOW_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
                                float>::type;

  int dev_id = 0;
  FLASHINFER_CUDA_CALL(gpuGetDevice(&dev_id));
  int max_smem_per_sm = 0;
  FLASHINFER_CUDA_CALL(gpuDeviceGetAttribute(&max_smem_per_sm,
                                             gpuDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
  // we expect each sm execute two threadblocks
  // TODO(Zihao): fix the following computation
  const int num_ctas_per_sm = max_smem_per_sm > (16 * HEAD_DIM * sizeof(DTypeQ) * 16) ? 2 : 1;
  const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

  const uint32_t max_num_mma_kv_reg =
      (HEAD_DIM >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !ALLOW_FP16_QK_REDUCTION)
          ? 2
          : (8 / NUM_MMA_Q);
  // TODO(Zihao): fix the following computation
  const uint32_t max_num_mma_kv_smem =
      (max_smem_per_threadblock / (16 * HEAD_DIM * sizeof(DTypeQ)) - NUM_MMA_Q * NUM_WARPS_Q) /
      (2 * NUM_WARPS_KV);

  DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
    if constexpr (is_invalid_configuration<POS_ENCODING_MODE, DTypeKV, DTypeQKAccum>(
                      NUM_MMA_Q, NUM_MMA_D, NUM_MMA_KV, NUM_WARPS_Q, NUM_WARPS_KV)) {
      // Invalid configuration, skip
      std::ostringstream err_msg;
      err_msg << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=" << NUM_MMA_Q
              << " NUM_MMA_D=" << NUM_MMA_D << " NUM_MMA_KV=" << NUM_MMA_KV
              << " NUM_WARPS_Q=" << NUM_WARPS_Q << " NUM_WARPS_KV=" << NUM_WARPS_KV
              << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                 " and report the issue to the developers.";
      FLASHINFER_ERROR(err_msg.str());
    } else {
      // TODO(Zihao): fix the following computation
      uint32_t smem_size = (NUM_MMA_Q * NUM_WARPS_Q * sizeof(DTypeQ) +
                            NUM_MMA_KV * NUM_WARPS_KV * 2 * sizeof(DTypeQ)) *
                           16 * HEAD_DIM;
      auto kernel =
          BatchPrefillWithRaggedKVCacheKernel<MASK_MODE, POS_ENCODING_MODE, NUM_MMA_Q, NUM_MMA_D,
                                              NUM_MMA_KV, NUM_WARPS_Q, NUM_WARPS_KV, DTypeQKAccum,
                                              AttentionVariant>;
      FLASHINFER_CUDA_CALL(
          gpuFuncSetAttribute(reinterpret_cast<const void*>(kernel), gpuFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      if (tmp_v == nullptr) {
        // do not partition kv
        params.partition_kv = false;
        void* args[] = {(void*)&group_size_fastdiv, (void*)&params};
        FLASHINFER_CUDA_CALL(
            gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      } else {
        // partition kv
        params.partition_kv = true;
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp_v;
        params.lse = tmp_s;
        void* args[] = {(void*)&group_size_fastdiv, (void*)&params};
        FLASHINFER_CUDA_CALL(
            gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
              tmp_v, tmp_s, params.merge_indptr, o, lse, params.max_total_num_rows,
              params.total_num_rows, num_qo_heads, HEAD_DIM, stream));
        } else {
          FLASHINFER_CUDA_CALL(
              VariableLengthAttentionSum(tmp_v, params.merge_indptr, o, params.max_total_num_rows,
                                         params.total_num_rows, num_qo_heads, HEAD_DIM, stream));
        }
      }
    }
  });
  return gpuSuccess;
}

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
          bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename AttentionVariant>
gpuError_t BatchPrefillWithPagedKVCacheDispatched(typename AttentionVariant::ParamsT params,
                                                  typename AttentionVariant::DTypeO* tmp_v,
                                                  float* tmp_s, gpuStream_t stream) {
  using DTypeQ = typename AttentionVariant::DTypeQ;
  using DTypeKV = typename AttentionVariant::DTypeKV;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.paged_kv.num_heads;
  const uint_fastdiv group_size_fastdiv(num_qo_heads / num_kv_heads);
  constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);

  if (padded_batch_size == 0) {
    // No request, skip
    // this won't happen in CUDAGraph mode because we fixed the padded_batch_size
    return gpuSuccess;
  }

  dim3 nblks(padded_batch_size, 1, num_kv_heads);
  dim3 nthrs(WARP_SIZE, NUM_WARPS_Q, NUM_WARPS_KV);

  constexpr uint32_t NUM_MMA_D = HEAD_DIM / 16;
  using DTypeQKAccum =
      typename std::conditional<ALLOW_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
                                float>::type;

  int dev_id = 0;
  FLASHINFER_CUDA_CALL(gpuGetDevice(&dev_id));
  int max_smem_per_sm = 0;
  FLASHINFER_CUDA_CALL(gpuDeviceGetAttribute(&max_smem_per_sm,
                                             gpuDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
  // we expect each sm execute two threadblocks
  // TODO(Zihao): fix the following computation
  const int num_ctas_per_sm = max_smem_per_sm > (16 * HEAD_DIM * sizeof(DTypeQ) * 16) ? 2 : 1;
  const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

  const uint32_t max_num_mma_kv_reg =
      (HEAD_DIM >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !ALLOW_FP16_QK_REDUCTION)
          ? 2
          : (8 / NUM_MMA_Q);
  // TODO(Zihao): fix the following computation
  const uint32_t max_num_mma_kv_smem =
      (max_smem_per_threadblock / (16 * HEAD_DIM * sizeof(DTypeQ)) - NUM_MMA_Q * NUM_WARPS_Q) /
      (2 * NUM_WARPS_KV);

  DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
    if constexpr (is_invalid_configuration<POS_ENCODING_MODE, DTypeKV, DTypeQKAccum>(
                      NUM_MMA_Q, NUM_MMA_D, NUM_MMA_KV, NUM_WARPS_Q, NUM_WARPS_KV)) {
      // Invalid configuration, skip
      std::ostringstream err_msg;
      err_msg << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=" << NUM_MMA_Q
              << " NUM_MMA_D=" << NUM_MMA_D << " NUM_MMA_KV=" << NUM_MMA_KV
              << " NUM_WARPS_Q=" << NUM_WARPS_Q << " NUM_WARPS_KV=" << NUM_WARPS_KV
              << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                 " and report the issue to the developers.";
      FLASHINFER_ERROR(err_msg.str());
    } else {
      // TODO(Zihao): fix the following computation
      uint32_t smem_size = (NUM_MMA_Q * NUM_WARPS_Q * sizeof(DTypeQ) +
                            NUM_MMA_KV * NUM_WARPS_KV * 2 * sizeof(DTypeQ)) *
                           16 * HEAD_DIM;
      auto kernel =
          BatchPrefillWithPagedKVCacheKernel<MASK_MODE, POS_ENCODING_MODE, NUM_MMA_Q, NUM_MMA_D,
                                             NUM_MMA_KV, NUM_WARPS_Q, NUM_WARPS_KV, DTypeQKAccum,
                                             AttentionVariant>;
      FLASHINFER_CUDA_CALL(
          gpuFuncSetAttribute(reinterpret_cast<const void*>(kernel), gpuFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      if (tmp_v == nullptr) {
        // do not partition kv
        params.partition_kv = false;
        void* args[] = {(void*)&group_size_fastdiv, (void*)&params};
        FLASHINFER_CUDA_CALL(
            gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      } else {
        params.partition_kv = true;
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp_v;
        params.lse = tmp_s;
        void* args[] = {(void*)&group_size_fastdiv, (void*)&params};
        FLASHINFER_CUDA_CALL(
            gpuLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
              tmp_v, tmp_s, params.merge_indptr, o, lse, params.max_total_num_rows,
              params.total_num_rows, num_qo_heads, HEAD_DIM, stream));
        } else {
          FLASHINFER_CUDA_CALL(
              VariableLengthAttentionSum(tmp_v, params.merge_indptr, o, params.max_total_num_rows,
                                         params.total_num_rows, num_qo_heads, HEAD_DIM, stream));
        }
      }
      gpuDeviceSynchronize();
    }
  });
  return gpuSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_CUH_
