/*
 * Copyright (c) 2024, The vLLM team.
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

// Vendored for AgentKernelArena from vLLM csrc/rocm/attention.cu at commit
// edbc4969a76b25c531f7b9bb16b79984e65ef023 (Apache-2.0). Modifications:
//  - Only the gfx9 (MI300/MI355) kernels and their host-pass stubs are kept;
//    the gfx11/gfx12 kernels and the Navi launcher were removed.
//  - The FP8 KV-cache, FP8-MFMA and FP8-output branches and their helpers
//    were removed: this scenario uses a bfloat16 cache and output
//    (kv_cache_dtype "auto", mfma_type "f16"). The BF16 code path is unchanged.
//  - WARP_SIZE is the gfx9 wavefront size (csrc/cuda_compat.h queries it at
//    run time) and Fp8KVCacheDataType is copied from
//    csrc/attention/dtype_fp8.cuh.
//  - paged_attention() and its dtype/block/head dispatch are replaced by
//    run(), which allocates the per-partition scratch as vLLM's
//    chunked_prefill_paged_decode does and calls
//    paged_attention_custom_launcher on the current HIP stream. The launcher
//    takes ATen tensors and plain pointers instead of the optional arguments
//    this scenario does not use, and its GQA switch keeps the scenario's ratio.

// vLLM builds its ROCm extension with -U__HIP_NO_HALF_OPERATORS__
// -U__HIP_NO_HALF_CONVERSIONS__; torch's extension builder defines both.
#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <torch/library.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <type_traits>

// csrc/cuda_compat.h: the wavefront size of gfx9 GPUs.
#define WARP_SIZE 64

// csrc/attention/dtype_fp8.cuh
namespace vllm {

enum class Fp8KVCacheDataType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
};

}  // namespace vllm

#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__GFX9__
#endif

#if defined(NDEBUG)
  #undef NDEBUG
  #include <assert.h>
  #define UNREACHABLE_CODE assert(false);
  #define NDEBUG
#else
  #define UNREACHABLE_CODE assert(false);
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

enum class MFMAType {
  F16 = 0,
  Fp8 = 1,
  Fp4 = 2,
};

#if defined(__HIP__GFX9__)

  #define GCN_MFMA_INSTR1 __builtin_amdgcn_mfma_f32_16x16x4f32
  #define GCN_MFMA_INSTR __builtin_amdgcn_mfma_f32_4x4x4f16

using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float16x4 =
    __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
typedef float16x4 _Half4;
using float16x2 =
    __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
typedef float16x2 _Half2;
typedef struct _Half8 {
  _Half4 xy[2];
} _Half8;

using bit16_t = uint16_t;
using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
typedef bit16x4 _B16x4;
typedef struct _B16x8 {
  _B16x4 xy[2];
} _B16x8;

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma4x4x4_instr(const _B16x4& inpA,
                                                       const _B16x4& inpB,
                                                       const floatx4& inpC) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return __builtin_amdgcn_mfma_f32_4x4x4f16(inpA, inpB, inpC, absz, cbid,
                                              blgp);
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(inpA, inpB, inpC, absz, cbid,
                                                  blgp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x16_instr(const _B16x4& inpA,
                                                          const _B16x4& inpB,
                                                          const floatx4& inpC) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(inpA, inpB, inpC, absz, cbid,
                                                 blgp);
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(inpA, inpB, inpC, absz,
                                                     cbid, blgp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ float to_float(const T& inp) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return (float)inp;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __bfloat162float(inp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ T from_float(const float& inp) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return (_Float16)inp;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __float2bfloat16(inp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x4 from_floatx4(const floatx4& inp) {
  _B16x4 ret;
  if constexpr (std::is_same<T, _Float16>::value) {
    union h2cvt {
      __half2 h2[2];
      _B16x4 b16x4;
    } u;
    u.h2[0] = __float22half2_rn(make_float2(inp[0], inp[1]));
    u.h2[1] = __float22half2_rn(make_float2(inp[2], inp[3]));
    return u.b16x4;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    for (int i = 0; i < 4; i++) {
      union fcvt {
        uint32_t u32;
        float f32;
      } u;
      u.f32 = inp[i];
      u.u32 += 0x7fff + ((u.u32 >> 16) & 1);  // BF16 RNE with no nan/inf check
      ret[i] = uint16_t(u.u32 >> 16);
    }
    return ret;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x4 addx4(const _B16x4& inp1,
                                        const _B16x4& inp2) {
  _B16x4 ret;
  if constexpr (std::is_same<T, _Float16>::value) {
    union h2cvt {
      _B16x4 b16x4;
      __half2 h2[2];
    } u1, u2, s;
    u1.b16x4 = inp1;
    u2.b16x4 = inp2;
    s.h2[0] = u1.h2[0] + u2.h2[0];
    s.h2[1] = u1.h2[1] + u2.h2[1];
    return s.b16x4;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    for (int i = 0; i < 4; i++) {
      union fcvt {
        float f32;
        uint32_t i32;
      } u1, u2, s;
      u1.i32 = uint32_t(inp1[i]) << 16;
      u2.i32 = uint32_t(inp2[i]) << 16;
      s.f32 = u1.f32 + u2.f32;
      ret[i] = uint16_t(s.i32 >> 16);
    }
    return ret;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x4 from_floatx4_rtz(const floatx4& inp) {
  _B16x4 ret;
  if constexpr (std::is_same<T, _Float16>::value) {
    union h2cvt {
      _Half2 h2[2];
      _B16x4 b16x4;
    } u;
    u.h2[0] = __builtin_amdgcn_cvt_pkrtz(inp[0], inp[1]);
    u.h2[1] = __builtin_amdgcn_cvt_pkrtz(inp[2], inp[3]);
    return u.b16x4;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    for (int i = 0; i < 4; i++) {
      union fcvt {
        uint32_t i32;
        float f32;
      } u;
      u.f32 = inp[i];
      ret[i] = uint16_t(u.i32 >> 16);
    }
    return ret;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

// grid (num_seqs, num_partitions,num_kv_heads)
// block (256)
// clang-format off
template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED, int GQA_RATIO, MFMAType MFMA_TYPE>
__global__
__launch_bounds__(NUM_THREADS, 5) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,    // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,    // [num_blocks, num_kv_heads, head_size, block_size]
    const int num_kv_heads,   
    const float scale,    
    const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,   // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,   // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes, // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    float* __restrict__ exp_sums,           // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,         // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ out,             // [num_seqs, num_heads, max_num_partitions, head_size]
    OUTT* __restrict__ final_out,           // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, const float* k_scale, const float* v_scale) {
  // clang-format on
  constexpr int NWARPS = NUM_THREADS / WARP_SIZE;
  const auto warpid = threadIdx.x / WARP_SIZE;
  const auto laneid = threadIdx.x % WARP_SIZE;
  const int lane4id = laneid % 4;
  const int lane16id = laneid % 16;
  const int rowid = laneid / 16;

  const auto seq_idx = blockIdx.x;
  // NOTE queries with sequence len > 1 are prefills and taken care by another
  // kernel.
  if (query_start_loc_ptr != nullptr &&
      (query_start_loc_ptr[seq_idx + 1] - query_start_loc_ptr[seq_idx]) != 1) {
    return;
  }

  const auto partition_idx = blockIdx.y;

  constexpr int T_PAR_SIZE = 256;  // token partition size set to 256

  const auto max_num_partitions = gridDim.y;

  const int seq_len = seq_lens[seq_idx];

  const int partition_start_token_idx =
      partition_idx * T_PAR_SIZE;  // partition_size;
  // exit if partition is out of context for seq
  if (partition_start_token_idx >= seq_len) {
    return;
  }

  constexpr int GQA_RATIO4 = DIVIDE_ROUND_UP(GQA_RATIO, 4);

  // shared_logits is used for multiple purposes
  __shared__ _B16x4 shared_logits[NWARPS][4][16][4];

  // for QK mfma16x16, layout is QHead/Tokenx16 across every 16 lanes, 16 Bytes
  // HeadElements in each lane, 4x16B HeadElements across 4 rows of warp
  constexpr int ROWS_PER_WARP =
      WARP_SIZE / 16;  // rows refers to 16 lanes; refer DDP (Data Parallel
                       // Processing) terminology
  constexpr int CONTIGUOUS_KV_ELEMS_16B_LOAD =
      16 / sizeof(cache_t);  // 8 for 16 bit cache type, 16 for 8 bit types
  constexpr int QKHE_PER_FETCH =
      CONTIGUOUS_KV_ELEMS_16B_LOAD *
      ROWS_PER_WARP;  // each fetch across a warp fetches these many elements
  constexpr int QK_SIZE_RATIO =
      sizeof(scalar_t) /
      sizeof(cache_t);  // 1 for 16bit types, 2 for 8bit types
  constexpr int QKHELOOP = HEAD_SIZE / QKHE_PER_FETCH;  // 4xQKHE_16B across
                                                        // warp

  _B16x8 Qlocal[QKHELOOP]
               [QK_SIZE_RATIO];  // note that 16 contiguous elements of Q should
                                 // be fetched per lane for 8 bit cache types :
                                 // QK_SIZE_RATIO changes for this

  constexpr int CONTIGUOUS_SCALAR_ELEMS_16B = 16 / sizeof(scalar_t);

  constexpr int TOKENS_PER_WARP =
      T_PAR_SIZE /
      NWARPS;  // sub partition of tokens per warp for qk calculation
  constexpr int TLOOP =
      TOKENS_PER_WARP /
      16;  // each mfma16x16x16 instruction processes 16 tokens

  // can be interpreted as B8x16 for 8 bit types
  _B16x8 Klocal[TLOOP][QKHELOOP];

  const auto wg_start_head_idx = blockIdx.z * GQA_RATIO;
  const auto wg_start_kv_head_idx = blockIdx.z;
  const auto total_num_heads = gridDim.z * GQA_RATIO;

  // for QK mfma, tokens in multiples of TOKENS_PER_WARP are spread across warps
  // each mfma takes QH16xT16x16HE across warp
  // repeat mfmas across QKHELOOP dimension
  // output layout from QKmfma : QH16xT4x4 16 qheads across 16 lanes, 16 tokens
  // across 4 rows x 4 tokens per lane

  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int last_seq_block = num_seq_blocks - 1;

  const int* block_table_seq = block_tables + seq_idx * max_num_blocks_per_seq;

  int kphysical_block_number[TLOOP];

  // fetch k physical block numbers
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    const int klocal_token_idx =
        TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
    const int kglobal_token_idx = partition_start_token_idx + klocal_token_idx;
    const int kblock_idx = (kglobal_token_idx < seq_len)
                               ? kglobal_token_idx / BLOCK_SIZE
                               : last_seq_block;
    kphysical_block_number[token_depth] = block_table_seq[kblock_idx];
  }

  // fetch Q in shared across warps and then write to registers
  const int local_qhead_idx = 4 * warpid + rowid;
  const int global_qhead_idx = wg_start_head_idx + local_qhead_idx;
  const int64_t query_start_off = static_cast<int64_t>(
      query_start_loc_ptr ? query_start_loc_ptr[seq_idx] : seq_idx);
  const scalar_t* q_ptr =
      q + query_start_off * q_stride + global_qhead_idx * HEAD_SIZE;

  const int qhead_element = lane16id * CONTIGUOUS_SCALAR_ELEMS_16B;
  if ((local_qhead_idx < GQA_RATIO) && (qhead_element < HEAD_SIZE)) {
    const scalar_t* q_fetch_ptr = q_ptr + qhead_element;
    const _B16x8* q_fetch_ptr_16B =
        reinterpret_cast<const _B16x8*>(q_fetch_ptr);
    _B16x8 tmp = *q_fetch_ptr_16B;
    if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
      const int offset1 =
          lane16id /
          4;  // 16 contiguous chunks of head elems are spread across 4x4lanes
      shared_logits[offset1][lane4id][local_qhead_idx][0] = tmp.xy[0];
      shared_logits[offset1][lane4id][local_qhead_idx][1] = tmp.xy[1];
    }
  }
  __syncthreads();
  for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
    for (int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++) {
      for (int i = 0; i < 2; i++) {
        Qlocal[qkhe_depth][qkratio].xy[i] =
            shared_logits[qkhe_depth][rowid][lane16id % GQA_RATIO]
                         [2 * qkratio + i];
      }
    }
  }

  constexpr int KX =
      16 / sizeof(cache_t);  // vLLM defines x as 16 Bytes of kv cache elements
  const cache_t* k_ptr = k_cache + wg_start_kv_head_idx * kv_head_stride;

  const int row_head_elem = rowid * CONTIGUOUS_KV_ELEMS_16B_LOAD;
  // fetch K values
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    const int64_t kblock_number =
        static_cast<int64_t>(kphysical_block_number[token_depth]);
    const cache_t* k_ptr2 = k_ptr + kblock_number * kv_block_stride;
    const int klocal_token_idx =
        TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
    const int kphysical_block_offset = klocal_token_idx % BLOCK_SIZE;
    const cache_t* k_ptr3 = k_ptr2 + kphysical_block_offset * KX;

    for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
      const int head_elem = row_head_elem + qkhe_depth * QKHE_PER_FETCH;
      const int offset1 = head_elem / KX;
      const int offset2 = head_elem % KX;
      const cache_t* k_fetch_ptr = k_ptr3 + offset1 * BLOCK_SIZE * KX + offset2;
      const _B16x8* k_fetch_ptr_16B =
          reinterpret_cast<const _B16x8*>(k_fetch_ptr);
      Klocal[token_depth][qkhe_depth] = *k_fetch_ptr_16B;
    }
  }

  float alibi_slope;
  if constexpr (ALIBI_ENABLED) {
    const int alibi_head_idx = wg_start_head_idx + lane16id;
    alibi_slope = (lane16id < GQA_RATIO) ? alibi_slopes[alibi_head_idx] : 0.f;
  }

  constexpr int VTOKENS_PER_LANE =
      TOKENS_PER_WARP / ROWS_PER_WARP;  // 64/4 = 16 contiguous vtokens per lane
  constexpr int VBLOCKS_PER_LANE =
      1;  // assumes block size >=16, each lane can correspond to 1 block only
  constexpr int VTLOOP = NWARPS;  // corresponds to tokens across warps
  constexpr int VTLANELOOP = DIVIDE_ROUND_UP(
      VTOKENS_PER_LANE,
      CONTIGUOUS_KV_ELEMS_16B_LOAD);  // optimized for 16B fetches; assumes
                                      // minimum block size is 16
  constexpr int VHELOOP = HEAD_SIZE / 16 / NWARPS;

  int vphysical_block_number[VTLOOP][VBLOCKS_PER_LANE];

  // fetch v physical block numbers
  for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
    for (int vblock_depth = 0; vblock_depth < VBLOCKS_PER_LANE;
         vblock_depth++) {
      const int vlocal_token_idx =
          vtoken_depth * VTOKENS_PER_LANE * ROWS_PER_WARP +
          rowid * VTOKENS_PER_LANE + vblock_depth * BLOCK_SIZE;
      // Safe to use an int32_t here assuming we are working with < 2 billion
      // tokens
      const int vglobal_token_idx =
          partition_start_token_idx + vlocal_token_idx;
      const int vblock_idx = (vglobal_token_idx < seq_len)
                                 ? vglobal_token_idx / BLOCK_SIZE
                                 : last_seq_block;
      vphysical_block_number[vtoken_depth][vblock_depth] =
          block_table_seq[vblock_idx];
    }
  }

  _B16x8 Vlocal[VTLOOP][VHELOOP][VTLANELOOP];  // this could be B8x16 too

  const cache_t* v_ptr = v_cache + wg_start_kv_head_idx * kv_head_stride +
                         ((rowid * VTOKENS_PER_LANE) % BLOCK_SIZE);

  // v fetches are 16head elems across lanes x 16 tokens per lane
  for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
    const int vhead_elem = vhe_depth * NWARPS * 16 + warpid * 16 + lane16id;
    const cache_t* v_ptr2 = v_ptr + vhead_elem * BLOCK_SIZE;

    for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
      for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++) {
        const int vblock_depth = 0;
        const int64_t vblock_number = static_cast<int64_t>(
            vphysical_block_number[vtoken_depth][vblock_depth]);
        const cache_t* v_ptr3 = v_ptr2 + (vblock_number * kv_block_stride);

        const cache_t* v_fetch_ptr =
            v_ptr3 + vfetch_depth * CONTIGUOUS_KV_ELEMS_16B_LOAD;
        const _B16x8* v_fetch_ptr_16B =
            reinterpret_cast<const _B16x8*>(v_fetch_ptr);
        Vlocal[vtoken_depth][vhe_depth][vfetch_depth] = *v_fetch_ptr_16B;
      }
    }
  }

  // calculate post qk mfma scale
  float scale2 = scale;

  floatx4 d_out[TLOOP];
  // qk mfma
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    d_out[token_depth] = {0};
    for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
      if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
        for (int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++) {
          for (int i = 0; i < 2; i++) {
            d_out[token_depth] = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(
                Klocal[token_depth][qkhe_depth].xy[i],
                Qlocal[qkhe_depth][qkratio].xy[i], d_out[token_depth]);
          }
        }
      }
    }
    d_out[token_depth] *= scale2;
  }

  const int qkout_token_idx =
      partition_start_token_idx + TOKENS_PER_WARP * warpid + rowid * 4;

  // apply alibi
  if constexpr (ALIBI_ENABLED) {
    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
      const int local_token_idx = qkout_token_idx + token_depth * 16;
      const int alibi_offset = local_token_idx - seq_len + 1;
      for (int i = 0; i < 4; i++) {
        d_out[token_depth][i] += alibi_slope * (alibi_offset + i);
      }
    }
  }

  // calculate qk_max and exp_sum per warp and write to shared memory
  float qk_max = -FLT_MAX;
  float exp_sum = 0.0f;

  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    const int local_token_idx = qkout_token_idx + token_depth * 16;
    for (int i = 0; i < 4; i++) {
      const float tmp =
          (local_token_idx + i < seq_len) ? d_out[token_depth][i] : -FLT_MAX;
      qk_max = fmaxf(qk_max, tmp);
    }
  }

  for (int mask = WARP_SIZE / 2; mask >= 16; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor(qk_max, mask));
  }

  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    const int local_token_idx = qkout_token_idx + token_depth * 16;
    for (int i = 0; i < 4; i++) {
      const float tmp = (local_token_idx + i < seq_len)
                            ? __expf(d_out[token_depth][i] - qk_max)
                            : 0.0f;
      d_out[token_depth][i] = tmp;
      exp_sum += tmp;
    }
  }

  for (int mask = WARP_SIZE / 2; mask >= 16; mask /= 2) {
    exp_sum += __shfl_xor(exp_sum, mask);
  }

  __syncthreads();  // sync before writing to shared mem

  float* shared_mem = reinterpret_cast<float*>(shared_logits);
  if (laneid < 16) {
    const int qk_max_offset = warpid * 16 + lane16id;
    shared_mem[qk_max_offset] = qk_max;
    const int exp_sum_offset = NWARPS * 16 + qk_max_offset;
    shared_mem[exp_sum_offset] = exp_sum;
  }

  __syncthreads();

  // calculate partition qk_max and exp_sum
  float partition_qk_max = -FLT_MAX;
  float warp_qk_max_exp[NWARPS];
  float partition_exp_sum = 0.0f;

  for (int w = 0; w < NWARPS; w++) {
    warp_qk_max_exp[w] = shared_mem[w * 16 + lane16id];
    partition_qk_max = fmaxf(partition_qk_max, warp_qk_max_exp[w]);
  }

  for (int w = 0; w < NWARPS; w++) {
    warp_qk_max_exp[w] = __expf(warp_qk_max_exp[w] - partition_qk_max);
    partition_exp_sum +=
        shared_mem[NWARPS * 16 + w * 16 + lane16id] * warp_qk_max_exp[w];
  }

  const float inv_sum_scale =
      __fdividef(1.f, partition_exp_sum + 1e-6f) * warp_qk_max_exp[warpid];

  __syncthreads();

  // disable rtz conversion due to its impact on accuracy.
  constexpr bool LOGITS_RTZ_CONVERSION = false;

  // write logits to shared mem
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    d_out[token_depth] *= inv_sum_scale;
    if constexpr (MFMA_TYPE != MFMAType::Fp8) {
      if constexpr (LOGITS_RTZ_CONVERSION) {
        // use rtz conversion for better performance, with negligible impact on
        // accuracy
        shared_logits[warpid][token_depth][lane16id][rowid] =
            from_floatx4_rtz<scalar_t>(d_out[token_depth]);
      } else {
        shared_logits[warpid][token_depth][lane16id][rowid] =
            from_floatx4<scalar_t>(d_out[token_depth]);
      }
    }
  }

  // write out partition max_logits and exp_sum
  if (threadIdx.x < GQA_RATIO) {
    const int qhead_idx = lane16id;
    const int64_t offset = static_cast<int64_t>(seq_idx) *
                               static_cast<int64_t>(total_num_heads) *
                               static_cast<int64_t>(max_num_partitions) +
                           (static_cast<int64_t>(wg_start_head_idx) +
                            static_cast<int64_t>(qhead_idx)) *
                               static_cast<int64_t>(max_num_partitions) +
                           static_cast<int64_t>(partition_idx);
    max_logits[offset] = partition_qk_max;
    exp_sums[offset] = partition_exp_sum;
  }

  __syncthreads();

  constexpr int ELEMS8_ELEMS4_RATIO = 8 / 4;

  _B16x4 outelems[VHELOOP];
  // Softmax V mfma
  // v layout: 16he across lanes x 16 tokens per lane
  for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
    floatx4 tmp_out = {0};

    for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
      if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
        for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++) {
          for (int i = 0; i < ELEMS8_ELEMS4_RATIO; i++) {
            const int offset = rowid * VTLANELOOP * ELEMS8_ELEMS4_RATIO +
                               vfetch_depth * ELEMS8_ELEMS4_RATIO + i;
            const int offset1 = offset % ROWS_PER_WARP;
            const int offset2 = offset / ROWS_PER_WARP;
            // output format is 16 qheads across 16 lanes, 16 head elems spread
            // across 4 rows
            tmp_out = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(
                Vlocal[vtoken_depth][vhe_depth][vfetch_depth].xy[i],
                shared_logits[vtoken_depth][offset2][lane16id][offset1],
                tmp_out);
          }
        }
      }
    }
    outelems[vhe_depth] = from_floatx4<scalar_t>(tmp_out);
  }

  __syncthreads();

  // store Softmax-V mfma output to shared mem
  for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
    // lane16 id head dimension; rowid head element dimension
    shared_logits[warpid][vhe_depth][lane16id][rowid] = outelems[vhe_depth];
  }

  __syncthreads();

  // write to tmp_out with coalesced writes after reading from shared mem
  if (warpid == 0) {
    _B16x8 vout[GQA_RATIO4];
    // each lane writes out 16Bytes of tmp_out along head elem dimension
    const int head_elem_idx = lane16id * 8;
    if (head_elem_idx < HEAD_SIZE) {
      for (int h = 0; h < GQA_RATIO4; h++) {
        const int local_head_idx = 4 * h + rowid;
        const int offset1 = (head_elem_idx / 16) % 4;
        const int offset2 = head_elem_idx / 16 / NWARPS;
        const int offset3 = (head_elem_idx / 4) % 4;
        for (int i = 0; i < 2; i++) {
          vout[h].xy[i] =
              shared_logits[offset1][offset2][local_head_idx][offset3 + i];
        }
      }

      const int64_t hsz_maxp_mult =
          static_cast<int64_t>(HEAD_SIZE * max_num_partitions);
      scalar_t* out_ptr = out + seq_idx * total_num_heads * hsz_maxp_mult +
                          partition_idx * HEAD_SIZE;
      for (int h = 0; h < GQA_RATIO4; h++) {
        const int local_head_idx = 4 * h + rowid;
        if (local_head_idx < GQA_RATIO) {
          const int64_t out_head_idx =
              static_cast<int64_t>(wg_start_head_idx + local_head_idx);
          scalar_t* out_ptr2 = out_ptr + out_head_idx * hsz_maxp_mult;
          scalar_t* out_ptr3 = out_ptr2 + head_elem_idx;
          _B16x8* out_ptr_B16x8 = reinterpret_cast<_B16x8*>(out_ptr3);
          *out_ptr_B16x8 = vout[h];
        }
      }
    }
  }
}

// grid (num_seqs, num_partitions, num_kv_heads)
// block (256 : partition size)
// each WG handles 1 partition per sequence
// clang-format off
template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED,
          int GQA_RATIO>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma4_kernel(
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,    // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,    // [num_blocks, num_kv_heads, head_size, block_size]
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,   // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,   // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes, // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    float* __restrict__ exp_sums,           // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,         // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ out,             // [num_seqs, num_heads, max_num_partitions, head_size]
    OUTT* __restrict__ final_out,           // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, const float* k_scale, const float* v_scale) {
  // clang-format on
  constexpr int NWARPS = NUM_THREADS / WARP_SIZE;
  const auto warpid = threadIdx.x / WARP_SIZE;
  const auto laneid = threadIdx.x % WARP_SIZE;
  const int lane4id = laneid % 4;

  const auto seq_idx = blockIdx.x;
  // NOTE queries with sequence len > 1 are prefills and taken care by another
  // kernel.
  if (query_start_loc_ptr != nullptr &&
      (query_start_loc_ptr[seq_idx + 1] - query_start_loc_ptr[seq_idx] != 1)) {
    return;
  }
  const auto partition_idx = blockIdx.y;
  const auto partition_size = blockDim.x;
  const auto max_num_partitions = gridDim.y;

  const int seq_len = seq_lens[seq_idx];
  const int partition_start_token_idx = partition_idx * partition_size;
  // exit if partition is out of context for seq
  if (partition_start_token_idx >= seq_len) {
    return;
  }
  // every 4 lanes fetch 4 different qheads
  // qhloop = num loops over qhead dimension
  constexpr int QHLOOP = DIVIDE_ROUND_UP(GQA_RATIO, 4);
  constexpr int GQA_RATIO4 = 4 * QHLOOP;
  __shared__ float shared_qk_max[NWARPS][GQA_RATIO4 + 1];
  __shared__ float shared_exp_sum[NWARPS][GQA_RATIO4 + 1];
  _B16x8 Qlocal[QHLOOP];
  constexpr int x = 16 / sizeof(scalar_t);
  // kheloop = num loops over head_size for 16Bytes of Q/dequantized K elements
  constexpr int KHELOOP = HEAD_SIZE / x;
  _B16x8 Klocal[KHELOOP];
  // for SoftMax-V Gemm, V head_size dimension is distributed across warp
  // vheloop = num loops to cover v head size dimension
  constexpr int VHELOOP = HEAD_SIZE / WARP_SIZE;
  // softmax out has warp_size tokens across warp
  // vtloop = num loops to cover warp_size(64) tokens with 16Bytes of
  // dequantized V elements
  constexpr int VTLOOP = WARP_SIZE / 8;
  // num vblocks to cover warp_size(64) v elements
  constexpr int VBLOCKS = 8 * VTLOOP / BLOCK_SIZE;
  int vphysical_blocks[VBLOCKS];
  _B16x8 Vlocal[VHELOOP][VTLOOP];
  floatx4 d_out[QHLOOP];
  float qk_max[QHLOOP];

  __shared__ _B16x4 vout_shared[QHLOOP][VHELOOP][WARP_SIZE][NWARPS + 1];

  for (int h = 0; h < QHLOOP; h++) {
    d_out[h] = {0};
    qk_max[h] = -FLT_MAX;
  }

  const auto wg_start_head_idx = blockIdx.z * GQA_RATIO;
  const auto wg_start_kv_head_idx = blockIdx.z;

  const int warp_start_token_idx =
      partition_start_token_idx + warpid * WARP_SIZE;

  if (warp_start_token_idx >= seq_len) {  // warp out of context
  #pragma unroll
    for (int h = 0; h < GQA_RATIO4; h++) {
      shared_qk_max[warpid][h] = -FLT_MAX;
      shared_exp_sum[warpid][h] = 0.0f;
    }
  } else {  // warp within context

    const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
    const int last_seq_block = num_seq_blocks - 1;

    const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
    // token id within partition
    const auto local_token_idx = threadIdx.x;
    // token id within sequence
    const int global_token_idx = partition_start_token_idx + local_token_idx;

    // fetch block number for k
    const int block_idx = (global_token_idx < seq_len)
                              ? global_token_idx / BLOCK_SIZE
                              : last_seq_block;

    // fetch k physical block number
    //  int32 physical_block_number leads to overflow when multiplied with
    //  kv_block_stride
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // fetch vphysical block numbers up front
    const int warp_start_block_idx = warp_start_token_idx / BLOCK_SIZE;
    for (int b = 0; b < VBLOCKS; b++) {
      const int vblock_idx = warp_start_block_idx + b;
      const int vblock_idx_ctx =
          (vblock_idx <= last_seq_block) ? vblock_idx : last_seq_block;
      vphysical_blocks[b] = block_table[vblock_idx_ctx];
    }

    // fetch q elements
    // every 4 lanes fetch 8 elems, so warp fetches 8*16 = 128 elemsc
    const int64_t query_start_off = static_cast<int64_t>(
        query_start_loc_ptr ? query_start_loc_ptr[seq_idx] : seq_idx);
    const scalar_t* q_ptr =
        q + query_start_off * q_stride + wg_start_head_idx * HEAD_SIZE;
    const _B16x8* q_ptrh8 = reinterpret_cast<const _B16x8*>(q_ptr);
    const int qhead_elemh8 = MIN(laneid / 4, HEAD_SIZE / 8 - 1);

    for (int h = 0; h < QHLOOP - 1; h++) {
      const int qhead_idx = h * 4 + lane4id;
      Qlocal[h] = q_ptrh8[qhead_idx * HEAD_SIZE / 8 + qhead_elemh8];
    }
    const int final_qhead_idx = 4 * (QHLOOP - 1) + lane4id;
    if (final_qhead_idx < GQA_RATIO) {
      Qlocal[QHLOOP - 1] =
          q_ptrh8[final_qhead_idx * HEAD_SIZE / 8 + qhead_elemh8];
    } else {
      Qlocal[QHLOOP - 1].xy[0] = {0};
      Qlocal[QHLOOP - 1].xy[1] = {0};
    }

    // fetch k elements
    const cache_t* k_ptr = k_cache + physical_block_number * kv_block_stride +
                           wg_start_kv_head_idx * kv_head_stride;

    // physical_block_offset is already cast in terms of _B16x8
    const int physical_block_offset = local_token_idx % BLOCK_SIZE;

    // each K fetch is for 8 elements of cache_t which are later dequantized to
    // scalar_t for fp8
    if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
      const _B16x8* k_ptrh8 = reinterpret_cast<const _B16x8*>(k_ptr);
      for (int d = 0; d < KHELOOP; d++) {
        Klocal[d] = k_ptrh8[d * BLOCK_SIZE + physical_block_offset];
      }
    }

    // optional alibi fetch
    float alibi_slope[QHLOOP];
    if constexpr (ALIBI_ENABLED) {
      for (int h = 0; h < QHLOOP; h++) {
        const int qhead_idx = h * 4 + lane4id;
        alibi_slope[h] = (qhead_idx < GQA_RATIO)
                             ? alibi_slopes[wg_start_head_idx + qhead_idx]
                             : 0.f;
      }
    }

    const cache_t* v_ptr = v_cache + wg_start_kv_head_idx * kv_head_stride;
    // fetch vcache in kv cache auto case
    if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
      const _B16x8* v_ptrh8 = reinterpret_cast<const _B16x8*>(v_ptr);
      // iterate over each v block
      for (int b = 0; b < VBLOCKS; b++) {
        // int32 physical_block_number leads to overflow when multiplied with
        // kv_block_stride
        const int64_t vphysical_block_number =
            static_cast<int64_t>(vphysical_blocks[b]);
        const _B16x8* v_ptrh8b =
            v_ptrh8 + (vphysical_block_number * kv_block_stride) / 8;
        // iterate over each head elem (within head_size)
        for (int h = 0; h < VHELOOP; h++) {
          const int head_size_elem = h * WARP_SIZE + laneid;
          const _B16x8* v_ptrh8be = v_ptrh8b + head_size_elem * BLOCK_SIZE / 8;
          // iterate over all velems within block
          for (int d = 0; d < BLOCK_SIZE / 8; d++) {
            Vlocal[h][b * BLOCK_SIZE / 8 + d] = v_ptrh8be[d];
          }
        }
      }
    }  // if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto)

  #define QK_mfma(x)                                             \
    for (int h = 0; h < QHLOOP; h++) {                           \
      d_out[h] = gcn_mfma4x4x4_instr<scalar_t, 4, x, 0>(         \
          Qlocal[h].xy[0], Klocal[x].xy[0], d_out[h]);           \
      d_out[h] = gcn_mfma4x4x4_instr<scalar_t, 4, x, 0>(         \
          Qlocal[h].xy[1], Klocal[x].xy[1], d_out[h]);           \
    }
    // QK mfma with Q mfma block broadcast
    // Q values across head_size dimension stored across lanes
    // K values across head_size dimension are stored depthwise within lane
    // Q broadcast with absz, cbid of mfma instruction
    QK_mfma(0);
    QK_mfma(1);
    QK_mfma(2);
    QK_mfma(3);
    QK_mfma(4);
    QK_mfma(5);
    QK_mfma(6);
    QK_mfma(7);
    // below only needed for head size 128
    if constexpr (KHELOOP > 8) {
      QK_mfma(8);
      QK_mfma(9);
      QK_mfma(10);
      QK_mfma(11);
      QK_mfma(12);
      QK_mfma(13);
      QK_mfma(14);
      QK_mfma(15);
    }
  #undef QK_mfma

    float scale2 = scale;

    for (int h = 0; h < QHLOOP; h++) {
      d_out[h] *= scale2;
    }

    // transpose d_out so that 4 token ids are in each lane, and 4 heads are
    // across 4 lanes
    for (int h = 0; h < QHLOOP; h++) {
      floatx4 tmp = {0};
      for (int i = 0; i < 4; i++) {
        const float B = (lane4id == i) ? 1.0f : 0.0f;
        tmp = __builtin_amdgcn_mfma_f32_4x4x1f32(d_out[h][i], B, tmp, 0, 0, 0);
      }
      d_out[h] = tmp;
    }

    const int lane4_token_idx = 4 * (global_token_idx >> 2);

    if constexpr (ALIBI_ENABLED) {
      const int alibi_offset = lane4_token_idx - seq_len + 1;
      for (int h = 0; h < QHLOOP; h++) {
        for (int i = 0; i < 4; i++) {
          d_out[h][i] += alibi_slope[h] * (alibi_offset + i);
        }
      }
    }

    const int bpermute_mask = 4 * (16 * ((laneid >> 2) % 4) + lane4id);

    for (int h = 0; h < QHLOOP; h++) {
      qk_max[h] = -FLT_MAX;
      for (int i = 0; i < 4; i++) {
        qk_max[h] = (lane4_token_idx + i < seq_len)
                        ? fmaxf(qk_max[h], d_out[h][i])
                        : qk_max[h];
      }

      // for (int mask = WARP_SIZE / 2; mask >= 4; mask /= 2) {
      //   qk_max[h] = fmaxf(qk_max[h], __shfl_xor(qk_max[h], mask));
      // }
      // faster version of above code with dpp
      asm("v_nop\n v_nop\n v_max_f32_dpp %0, %1, %2 row_ror:4"
          : "=v"(qk_max[h])
          : "v"(qk_max[h]), "v"(qk_max[h]));
      asm("v_nop\n v_nop\n v_max_f32_dpp %0, %1, %2 row_ror:8"
          : "=v"(qk_max[h])
          : "v"(qk_max[h]), "v"(qk_max[h]));

      auto tmp = __builtin_amdgcn_ds_bpermute(
          bpermute_mask, *reinterpret_cast<int*>(&qk_max[h]));
      qk_max[h] = *reinterpret_cast<float*>(&tmp);
      asm("v_nop\n v_nop\n v_max_f32_dpp %0, %1, %2 row_ror:4"
          : "=v"(qk_max[h])
          : "v"(qk_max[h]), "v"(qk_max[h]));
      asm("v_nop\n v_nop\n v_max_f32_dpp %0, %1, %2 row_ror:8"
          : "=v"(qk_max[h])
          : "v"(qk_max[h]), "v"(qk_max[h]));
    }

    float exp_sum[QHLOOP];
    for (int h = 0; h < QHLOOP; h++) {
      exp_sum[h] = 0.0f;
      for (int i = 0; i < 4; i++) {
        d_out[h][i] = (lane4_token_idx + i < seq_len)
                          ? __expf(d_out[h][i] - qk_max[h])
                          : 0.0f;
        exp_sum[h] += d_out[h][i];
      }
      // for (int mask = WARP_SIZE / 2; mask >= 4; mask /= 2) {
      //   exp_sum[h] += __shfl_xor(exp_sum[h], mask);
      // }
      // faster version of above code with dpp
      asm("v_nop\n v_nop\n v_add_f32_dpp %0, %1, %2 row_ror:4"
          : "=v"(exp_sum[h])
          : "v"(exp_sum[h]), "v"(exp_sum[h]));
      asm("v_nop\n v_nop\n v_add_f32_dpp %0, %1, %2 row_ror:8"
          : "=v"(exp_sum[h])
          : "v"(exp_sum[h]), "v"(exp_sum[h]));

      auto tmp = __builtin_amdgcn_ds_bpermute(
          bpermute_mask, *reinterpret_cast<int*>(&exp_sum[h]));
      exp_sum[h] = *reinterpret_cast<float*>(&tmp);
      asm("v_nop\n v_nop\n v_add_f32_dpp %0, %1, %2 row_ror:4"
          : "=v"(exp_sum[h])
          : "v"(exp_sum[h]), "v"(exp_sum[h]));
      asm("v_nop\n v_nop\n v_add_f32_dpp %0, %1, %2 row_ror:8"
          : "=v"(exp_sum[h])
          : "v"(exp_sum[h]), "v"(exp_sum[h]));
    }

    if (laneid < 4) {
      for (int h = 0; h < QHLOOP; h++) {
        const int head_idx = 4 * h + lane4id;
        shared_qk_max[warpid][head_idx] = qk_max[h];
        shared_exp_sum[warpid][head_idx] = exp_sum[h];
      }
    }
  }  // warp within context

  __syncthreads();

  const auto num_heads = gridDim.z * GQA_RATIO;
  float* max_logits_ptr =
      max_logits + seq_idx * num_heads * max_num_partitions + partition_idx;
  float* exp_sums_ptr =
      exp_sums + seq_idx * num_heads * max_num_partitions + partition_idx;
  // calculate qk_max and exp_sums for partition
  for (int h = 0; h < QHLOOP; h++) {
    float global_qk_max = -FLT_MAX;
    float warp_qk_max[NWARPS];
    const int head_idx = 4 * h + lane4id;
    for (int w = 0; w < NWARPS; w++) {
      warp_qk_max[w] = shared_qk_max[w][head_idx];
      global_qk_max = fmaxf(global_qk_max, warp_qk_max[w]);
    }
    float global_exp_sum = 0.0f;
    for (int w = 0; w < NWARPS; w++) {
      global_exp_sum +=
          shared_exp_sum[w][head_idx] * __expf(warp_qk_max[w] - global_qk_max);
    }
    if (head_idx < GQA_RATIO) {
      max_logits_ptr[(wg_start_head_idx + head_idx) * max_num_partitions] =
          global_qk_max;
      exp_sums_ptr[(wg_start_head_idx + head_idx) * max_num_partitions] =
          global_exp_sum;
    }
    const float global_inv_sum_scale = __fdividef(1.f, global_exp_sum + 1e-6f) *
                                       __expf(qk_max[h] - global_qk_max);
    d_out[h] *= global_inv_sum_scale;
  }
  constexpr bool LOGITS_RTZ_CONVERSION = false;
  // logits[h] -> every 4 lanes hold 4 heads, each lane holds 4 tokens, there
  // are 4x16 tokens across warp
  _B16x4 logits[QHLOOP];
  for (int h = 0; h < QHLOOP; h++) {
    if constexpr (LOGITS_RTZ_CONVERSION) {
      // use rtz for faster performance with no perceivable accuracy loss
      logits[h] = from_floatx4_rtz<scalar_t>(d_out[h]);
    } else {
      logits[h] = from_floatx4<scalar_t>(d_out[h]);
    }
  }

  if (warp_start_token_idx >= seq_len) {  // warp out of context
    for (int qh = 0; qh < QHLOOP; qh++) {
      for (int vh = 0; vh < VHELOOP; vh++) {
        vout_shared[qh][vh][laneid][warpid] = {0};
      }
    }
  } else {  // warp in context
  #define SV_mfma(x)                                                  \
    for (int qh = 0; qh < QHLOOP; qh++) {                             \
      acc[qh] = gcn_mfma4x4x4_instr<scalar_t, 4, 2 * x, 0>(           \
          logits[qh], Vlocal[vh][x].xy[0], acc[qh]);                  \
      acc[qh] = gcn_mfma4x4x4_instr<scalar_t, 4, 2 * x + 1, 0>(       \
          logits[qh], Vlocal[vh][x].xy[1], acc[qh]);                  \
    }

    for (int vh = 0; vh < VHELOOP; vh++) {
      floatx4 acc[QHLOOP];
      for (int qh = 0; qh < QHLOOP; qh++) {
        acc[qh] = {0};
      }
      // SoftMax-V calculation
      // logits -> token dimension is distributed across lanes
      // Vlocal -> token dimension is depthwise within lane
      // uses mfma instruction block broadcast for logits
      SV_mfma(0);
      SV_mfma(1);
      SV_mfma(2);
      SV_mfma(3);
      SV_mfma(4);
      SV_mfma(5);
      SV_mfma(6);
      SV_mfma(7);

      for (int qh = 0; qh < QHLOOP; qh++) {
        vout_shared[qh][vh][laneid][warpid] = from_floatx4<scalar_t>(acc[qh]);
      }
    }

  #undef SV_mfma
  }  // warp in context

  __syncthreads();

  // final write to tmp_out after vout accumulation
  if (warpid == 0) {
    _B16x4 vout[QHLOOP][VHELOOP];
    // iterate across heads
    for (int qh = 0; qh < QHLOOP; qh++) {
      // iterate over each v head elem (within head_size)
      for (int vh = 0; vh < VHELOOP; vh++) {
        vout[qh][vh] = {0};
        for (int w = 0; w < NWARPS; w++) {
          vout[qh][vh] =
              addx4<scalar_t>(vout[qh][vh], vout_shared[qh][vh][laneid][w]);
        }
      }
    }

    scalar_t* out_ptr = out +
                        seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                        partition_idx * HEAD_SIZE;
    const int out_num_partitions = max_num_partitions;
    bit16_t* out_ptr_b16 = reinterpret_cast<bit16_t*>(out_ptr);
    for (int qh = 0; qh < QHLOOP; qh++) {
      for (int vh = 0; vh < VHELOOP; vh++) {
        const int head_size_elem = vh * WARP_SIZE + laneid;
        for (int i = 0; i < 4; i++) {
          const int head_idx = 4 * qh + i;
          if (head_idx < GQA_RATIO) {
            out_ptr_b16[(wg_start_head_idx + head_idx) * out_num_partitions *
                            HEAD_SIZE +
                        head_size_elem] = vout[qh][vh][i];
          }
        }
      }
    }
  }  // warpid == 0
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, typename OUTT, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE, int NPAR_LOOPS>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads,
                                           // max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                           // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads,
                                           // max_num_partitions, head_size]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,  // [num_seqs]
    const int max_num_partitions, const float* __restrict__ fp8_out_scale_ptr) {
  const auto num_heads = gridDim.x;
  const auto head_idx = blockIdx.x;
  const auto seq_idx = blockIdx.y;

  // NOTE queries with sequence len > 1 are prefills and taken care by another
  // kernel.
  if (query_start_loc_ptr != nullptr &&
      (query_start_loc_ptr[seq_idx + 1] - query_start_loc_ptr[seq_idx] != 1)) {
    return;
  }

  const int seq_len = seq_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);
  const auto warpid = threadIdx.x / WARP_SIZE;

  __shared__ float shared_global_exp_sum;
  // max num partitions supported is warp_size * NPAR_LOOPS
  __shared__ float shared_exp_sums[NPAR_LOOPS * WARP_SIZE];

  if (warpid == 0) {
    const float* max_logits_ptr = max_logits +
                                  seq_idx * num_heads * max_num_partitions +
                                  head_idx * max_num_partitions;

    // valid partition is the last valid partition in case threadid > num
    // partitions
    int valid_partition[NPAR_LOOPS];
    float reg_max_logit[NPAR_LOOPS];
    const int last_valid_partition = num_partitions - 1;

  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const auto partition_no = i * WARP_SIZE + threadIdx.x;
      valid_partition[i] =
          (partition_no < num_partitions) ? partition_no : last_valid_partition;
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      reg_max_logit[i] = max_logits_ptr[valid_partition[i]];
    }
    float max_logit = reg_max_logit[0];
  #pragma unroll
    for (int i = 1; i < NPAR_LOOPS; i++) {
      max_logit = fmaxf(max_logit, reg_max_logit[i]);
    }

  #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      max_logit = fmaxf(max_logit, __shfl_xor(max_logit, mask));
    }

    const float* exp_sums_ptr = exp_sums +
                                seq_idx * num_heads * max_num_partitions +
                                head_idx * max_num_partitions;

    float rescaled_exp_sum[NPAR_LOOPS];
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      rescaled_exp_sum[i] = exp_sums_ptr[valid_partition[i]];
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const auto partition_no = i * WARP_SIZE + threadIdx.x;
      rescaled_exp_sum[i] *= (partition_no < num_partitions)
                                 ? expf(reg_max_logit[i] - max_logit)
                                 : 0.0f;
    }
    float global_exp_sum = rescaled_exp_sum[0];
  #pragma unroll
    for (int i = 1; i < NPAR_LOOPS; i++) {
      global_exp_sum += rescaled_exp_sum[i];
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const auto partition_no = i * WARP_SIZE + threadIdx.x;
      shared_exp_sums[partition_no] = rescaled_exp_sum[i];
    }

  #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      global_exp_sum += __shfl_xor(global_exp_sum, mask);
    }
    if (threadIdx.x == 0) {
      shared_global_exp_sum = global_exp_sum;
    }
  }  // warpid == 0
  const scalar_t* tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE + threadIdx.x;
  constexpr int MAX_NPAR = 64;
  scalar_t tmps[MAX_NPAR];
  const float dzero = 0.0f;
  #pragma unroll
  for (int j = 0; j < MAX_NPAR; j++) {
    tmps[j] = from_float<scalar_t>(dzero);
  }
  const int last_partition_offset = (num_partitions - 1) * HEAD_SIZE;
  const int num_partition_offset = (num_partitions)*HEAD_SIZE;
  int idx = 0;

  constexpr int JCHUNK = 16;

  #pragma unroll
  for (int j = 0; j < JCHUNK * HEAD_SIZE; j += HEAD_SIZE) {
    // lastj is last valid partition
    const int lastj_offset =
        (j < num_partition_offset) ? j : last_partition_offset;
    tmps[idx] = tmp_out_ptr[lastj_offset];
    idx++;
  }
  __syncthreads();

  if (num_partitions > JCHUNK) {
  #pragma unroll
    for (int j = JCHUNK * HEAD_SIZE; j < 2 * JCHUNK * HEAD_SIZE;
         j += HEAD_SIZE) {
      const int lastj_offset =
          (j < num_partition_offset) ? j : last_partition_offset;
      tmps[idx] = tmp_out_ptr[lastj_offset];
      idx++;
    }

    if (num_partitions > 2 * JCHUNK) {
  #pragma unroll
      for (int j = 2 * JCHUNK * HEAD_SIZE; j < MAX_NPAR * HEAD_SIZE;
           j += HEAD_SIZE) {
        const int lastj_offset =
            (j < num_partition_offset) ? j : last_partition_offset;
        tmps[idx] = tmp_out_ptr[lastj_offset];
        idx++;
      }
    }
  }  // num_partitions > JCHUNK

  // Aggregate tmp_out to out.
  float acc = 0.0f;
  #pragma unroll
  for (int j = 0; j < JCHUNK; j++) {
    acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
  }
  if (num_partitions > JCHUNK) {
  #pragma unroll
    for (int j = JCHUNK; j < 2 * JCHUNK; j++) {
      acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
    }
    if (num_partitions > 2 * JCHUNK) {
  #pragma unroll
      for (int j = 2 * JCHUNK; j < MAX_NPAR; j++) {
        acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
      }
    }
  }

  for (int p = 1; p < NPAR_LOOPS; p++) {
    if (num_partitions > p * MAX_NPAR) {
      idx = 0;
  #pragma unroll
      for (int j = p * MAX_NPAR * HEAD_SIZE; j < (p + 1) * MAX_NPAR * HEAD_SIZE;
           j += HEAD_SIZE) {
        // lastj is last valid partition
        const int lastj_offset =
            (j < num_partition_offset) ? j : last_partition_offset;
        tmps[idx] = tmp_out_ptr[lastj_offset];
        idx++;
      }

  #pragma unroll
      for (int j = 0; j < MAX_NPAR; j++) {
        acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j + p * MAX_NPAR];
      }
    }
  }

  const float inv_global_exp_sum =
      __fdividef(1.0f, shared_global_exp_sum + 1e-6f);
  const float out_scale =
      (fp8_out_scale_ptr != nullptr) ? 1.0f / (*fp8_out_scale_ptr) : 1.0f;
  acc *= inv_global_exp_sum;
  acc *= out_scale;
  const int64_t query_start_off = static_cast<int64_t>(
      query_start_loc_ptr ? query_start_loc_ptr[seq_idx] : seq_idx);
  OUTT* out_ptr = out + query_start_off * num_heads * HEAD_SIZE +
                  static_cast<int64_t>(head_idx) * HEAD_SIZE;
  out_ptr[threadIdx.x] = from_float<scalar_t>(acc);
}

#else

// clang-format off
template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED,
          int GQA_RATIO, MFMAType MFMA_TYPE>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,    // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,    // [num_blocks, num_kv_heads, head_size, block_size]
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,    // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,    // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    float* __restrict__ exp_sums,             // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,           // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ out,               // [num_seqs, num_heads, max_num_partitions, head_size]
    OUTT* __restrict__ final_out,             // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, const float* k_scale, const float* v_scale) {
  UNREACHABLE_CODE
}

template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED,
          int GQA_RATIO>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma4_kernel(
    const scalar_t* __restrict__ q,          // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,     // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,     // [num_blocks, num_kv_heads, head_size, block_size]
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,    // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,    // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    float* __restrict__ exp_sums,            // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,          // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ out,              // [num_seqs, num_heads, max_num_partitions, head_size]
    OUTT* __restrict__ final_out,            // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, const float* k_scale, const float* v_scale) {
  UNREACHABLE_CODE
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, typename OUTT, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE, int NPAR_LOOPS>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads, max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads, max_num_partitions, head_size]
    const int* __restrict__ seq_lens,  // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,  // [num_seqs]
    const int max_num_partitions, const float* __restrict__ fp8_out_scale_ptr) {
  UNREACHABLE_CODE
}
// clang-format on

#endif

#define LAUNCH_CUSTOM_ATTENTION_MFMA16(GQA_RATIO)                              \
  paged_attention_ll4mi_QKV_mfma16_kernel<T, KVT, KV_DTYPE, OUTT, BLOCK_SIZE,  \
                                          HEAD_SIZE, NTHR, ALIBI_ENABLED,      \
                                          GQA_RATIO, MFMA_TYPE>                \
      <<<grid, block, 0, stream>>>(                                            \
          query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, scale,      \
          block_tables_ptr, seq_lens_ptr, query_start_loc_ptr,                 \
          max_num_blocks_per_seq, alibi_slopes_ptr, q_stride, kv_block_stride, \
          kv_head_stride, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, out_ptr,  \
          max_ctx_blocks, k_scale_ptr, v_scale_ptr);

#define LAUNCH_CUSTOM_ATTENTION_MFMA4(GQA_RATIO)                               \
  paged_attention_ll4mi_QKV_mfma4_kernel<T, KVT, KV_DTYPE, OUTT, BLOCK_SIZE,   \
                                         HEAD_SIZE, NTHR, ALIBI_ENABLED,       \
                                         GQA_RATIO>                            \
      <<<grid, block, 0, stream>>>(                                            \
          query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, scale,      \
          block_tables_ptr, seq_lens_ptr, query_start_loc_ptr,                 \
          max_num_blocks_per_seq, alibi_slopes_ptr, q_stride, kv_block_stride, \
          kv_head_stride, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, out_ptr,  \
          max_ctx_blocks, k_scale_ptr, v_scale_ptr);

#define LAUNCH_CUSTOM_REDUCTION(NPAR_LOOPS)                                 \
  paged_attention_ll4mi_reduce_kernel<T, OUTT, HEAD_SIZE, HEAD_SIZE,        \
                                      PARTITION_SIZE, NPAR_LOOPS>           \
      <<<reduce_grid, reduce_block, 0, stream>>>(                           \
          out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, seq_lens_ptr, \
          query_start_loc_ptr, max_num_partitions, fp8_out_scale_ptr);

template <typename T, typename KVT, vllm::Fp8KVCacheDataType KV_DTYPE,
          int BLOCK_SIZE, int HEAD_SIZE, typename OUTT, int PARTITION_SIZE_OLD,
          bool ALIBI_ENABLED, MFMAType MFMA_TYPE>
void paged_attention_custom_launcher(
    const at::Tensor& out, const at::Tensor& exp_sums,
    const at::Tensor& max_logits, const at::Tensor& tmp_out,
    const at::Tensor& query, const at::Tensor& key_cache,
    const at::Tensor& value_cache, const int num_kv_heads, float scale,
    const at::Tensor& block_tables, const at::Tensor& seq_lens,
    int max_seq_len, hipStream_t stream) {
  static_assert(KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto &&
                    MFMA_TYPE == MFMAType::F16 && !ALIBI_ENABLED,
                "this copy keeps only the BF16-cache path without ALiBi");
  int num_seqs = block_tables.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  // Decode only: every sequence has one query token, so the kernels index
  // queries by sequence (query_start_loc is optional and absent here).
  const int* query_start_loc_ptr = nullptr;

  // No ALiBi.
  const float* alibi_slopes_ptr = nullptr;

  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  KVT* key_cache_ptr = reinterpret_cast<KVT*>(key_cache.data_ptr());
  KVT* value_cache_ptr = reinterpret_cast<KVT*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();
  // BF16 cache and output: no KV-cache or output scales.
  const float* k_scale_ptr = nullptr;
  const float* v_scale_ptr = nullptr;
  const float* fp8_out_scale_ptr = nullptr;
  OUTT* out_ptr = reinterpret_cast<OUTT*>(out.data_ptr());

  const int max_ctx_blocks = DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE);

  // partition size is fixed at 256 since both mfma4 and mfma16 kernels support
  // it mfma4 kernel also supports partition size 512
  constexpr int PARTITION_SIZE = 256;
  const int max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);
  const int gqa_ratio = num_heads / num_kv_heads;
  assert(num_heads % num_kv_heads == 0);
  assert(head_size == HEAD_SIZE);

  constexpr int NTHR = 256;
  dim3 grid(num_seqs, max_num_partitions, num_kv_heads);
  dim3 block(NTHR);

  // mfma4 kernel is faster than mfma16 for gqa_ratio <= 4
  switch (gqa_ratio) {
    case 4:
      LAUNCH_CUSTOM_ATTENTION_MFMA4(4);
      break;
    default:
      TORCH_CHECK(false, "Unsupported gqa ratio: ", gqa_ratio);
      break;
  }

  dim3 reduce_grid(num_heads, num_seqs);
  dim3 reduce_block(head_size);
  const int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, WARP_SIZE);
  // reduction kernel supports upto 8 NPAR_loops * 64 (warp_size) * 256
  // (partition size) = 128K context length
  switch (npar_loops) {
    case 1:
      LAUNCH_CUSTOM_REDUCTION(1);
      break;
    case 2:
      LAUNCH_CUSTOM_REDUCTION(2);
      break;
    case 3:
      LAUNCH_CUSTOM_REDUCTION(3);
      break;
    case 4:
      LAUNCH_CUSTOM_REDUCTION(4);
      break;
    case 5:
      LAUNCH_CUSTOM_REDUCTION(5);
      break;
    case 6:
      LAUNCH_CUSTOM_REDUCTION(6);
      break;
    case 7:
      LAUNCH_CUSTOM_REDUCTION(7);
      break;
    case 8:
      LAUNCH_CUSTOM_REDUCTION(8);
      break;
    default:
      TORCH_CHECK(false, "Unsupported npar_loops: ", npar_loops);
      break;
  }
}

// AgentKernelArena entry point: one decode step of paged attention, launched
// as vLLM's chunked_prefill_paged_decode launches it on ROCm.
void run(const at::Tensor& query, const at::Tensor& key_cache,
         const at::Tensor& value_cache, const at::Tensor& block_table,
         const at::Tensor& seq_lens, double scale, at::Tensor& out) {
  constexpr int kBlockSize = 16;
  constexpr int kHeadSize = 128;
  constexpr int kPartitionSize = 256;  // _PARTITION_SIZE_ROCM
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 &&
                  key_cache.scalar_type() == at::kBFloat16 &&
                  value_cache.scalar_type() == at::kBFloat16 &&
                  out.scalar_type() == at::kBFloat16,
              "query, key_cache, value_cache and out must be bfloat16");
  TORCH_CHECK(block_table.scalar_type() == at::kInt &&
                  seq_lens.scalar_type() == at::kInt,
              "block_table and seq_lens must be int32");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == kHeadSize &&
                  key_cache.dim() == 5 && value_cache.dim() == 4 &&
                  value_cache.size(3) == kBlockSize && block_table.dim() == 2 &&
                  block_table.size(0) == query.size(0),
              "unexpected tensor shapes");
  TORCH_CHECK(query.is_contiguous() && key_cache.is_contiguous() &&
                  value_cache.is_contiguous() && block_table.is_contiguous() &&
                  seq_lens.is_contiguous() && out.is_contiguous(),
              "tensors must be contiguous");
  const int64_t num_seqs = query.size(0);
  const int64_t num_heads = query.size(1);
  const int num_kv_heads = static_cast<int>(key_cache.size(1));
  // vLLM passes the longest context of the batch as max_seq_len. The block
  // table holds ceil(max_seq_len / block_size) pages per sequence, and its
  // width in tokens gives the same partition count.
  const int max_seq_len = static_cast<int>(block_table.size(1)) * kBlockSize;
  const int64_t max_num_partitions =
      DIVIDE_ROUND_UP(max_seq_len, kPartitionSize);
  // Per-partition scratch, allocated as chunked_prefill_paged_decode does.
  at::Tensor tmp_out = at::empty(
      {num_seqs, num_heads, max_num_partitions, kHeadSize}, query.options());
  at::Tensor exp_sums = at::empty({num_seqs, num_heads, max_num_partitions},
                                  query.options().dtype(at::kFloat));
  at::Tensor max_logits = at::empty_like(exp_sums);
  hipStream_t stream = c10::hip::getCurrentHIPStream();
  paged_attention_custom_launcher<__hip_bfloat16, __hip_bfloat16,
                                  vllm::Fp8KVCacheDataType::kAuto, kBlockSize,
                                  kHeadSize, __hip_bfloat16, kPartitionSize,
                                  false, MFMAType::F16>(
      out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,
      num_kv_heads, static_cast<float>(scale), block_table, seq_lens,
      max_seq_len, stream);
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def(
      "run(Tensor query, Tensor key_cache, Tensor value_cache, "
      "Tensor block_table, Tensor seq_lens, float scale, Tensor(a!) out) -> ()",
      &run);
}
