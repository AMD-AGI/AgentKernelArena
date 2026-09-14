// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Baseline hand-written HIP MLA (multi-latent attention) decode kernel
// for AMD CDNA-3 / CDNA-4 (gfx942 / gfx950).
//
// Naive but correct. The body is intentionally simple to leave headroom
// for an optimization pass:
//   - One workgroup per (batch, head_group). One wave per workgroup.
//   - Q tile (BLOCK_H heads x LK) staged into LDS once.
//   - Per token: lane-distributed FP32 QK dot (each lane handles 9 of LK)
//     followed by warp-reduce, online softmax update, per-lane PV
//     accumulator across 8 owned output dims.
//   - No MFMA, no transposed LDS reads, no persistent grid, no K-split.
//
// Shape contract:
//   - NHEAD = 128, BLOCK_H = 16, HEAD_GROUPS = 8
//   - LK = 576 (= 512 NoPE + 64 RoPE), LV = 512
//   - decode_qlen = 1, page_size = 1
//   - Q dtype: bf16, KV dtype: fp8 e4m3fn (saturating, bias=7), O dtype: bf16
//
// The protected native driver owns the workload dimensions, input generation,
// correctness reference, launch geometry, and timing.  This editable source is
// deliberately limited to the target kernel and its implementation helpers.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>

using bf16 = __hip_bfloat16;

namespace {

constexpr int NHEAD         = 128;
constexpr int BLOCK_H       = 16;
constexpr int HEAD_GROUPS   = NHEAD / BLOCK_H;  // 8
constexpr int LK            = 576;
constexpr int LV            = 512;

constexpr int WARP_SIZE     = 64;
constexpr int BLOCK_THREADS = WARP_SIZE;        // single wave per workgroup
constexpr int DIMS_PER_LANE = LV / WARP_SIZE;   // 8 (output dim partition)
constexpr int LK_PER_LANE   = LK / WARP_SIZE;   // 9 (QK dot partition)
static_assert(LV  % WARP_SIZE == 0, "LV must divide WARP_SIZE");
static_assert(LK  % WARP_SIZE == 0, "LK must divide WARP_SIZE");

}  // namespace

// ---------------------------------------------------------------------------
// fp8 e4m3fn helpers (bias = 7). Match torch.float8_e4m3fn semantics.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float fp8_e4m3fn_to_f32(uint8_t b) {
    if (b == 0x80 || b == 0x00) return 0.0f;
    int sign = (b & 0x80) ? -1 : 1;
    int e = (b >> 3) & 0xF;
    int m = b & 0x7;
    float val;
    if (e == 0) {
        val = static_cast<float>(m) / 8.0f / 64.0f;  // subnormal: m * 2^-9
    } else {
        float mantissa = 1.0f + static_cast<float>(m) / 8.0f;
        val = mantissa * exp2f(static_cast<float>(e) - 7);
    }
    return sign * val;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_max(float x) {
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        float y = __shfl_xor(x, off);
        x = fmaxf(x, y);
    }
    return x;
}

__device__ __forceinline__ float warp_reduce_sum(float x) {
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        x += __shfl_xor(x, off);
    }
    return x;
}

extern "C" __global__ void mla_decode_kernel(
    const bf16*    __restrict__ q,             // [batch, NHEAD, LK]
    const uint8_t* __restrict__ kv,            // [num_tokens, 1, LK] fp8 e4m3fn
    const int32_t* __restrict__ req_to_token,  // [batch, max_ctx]
    const int32_t* __restrict__ b_seq_len,     // [batch]
    float sm_scale,
    bf16*  __restrict__ o,                     // [batch, NHEAD, LV]
    float* __restrict__ lse,                   // [batch, NHEAD] (optional)
    int stride_r2t_b,                          // elements
    int /*max_ctx*/
) {
    const int batch_id = blockIdx.x;
    const int hg_id    = blockIdx.y;
    const int lane_id  = threadIdx.x;

    const int head_base = hg_id * BLOCK_H;
    const int seq_len   = b_seq_len[batch_id];

    extern __shared__ float smem_buf[];
    float* q_lds = smem_buf;  // BLOCK_H * LK floats = 36 KB

    {
        const bf16* q_head = q + (batch_id * NHEAD + head_base) * LK;
        for (int i = lane_id; i < BLOCK_H * LK; i += BLOCK_THREADS) {
            q_lds[i] = __bfloat162float(q_head[i]);
        }
    }
    __syncthreads();

    float e_max[BLOCK_H];
    float e_sum[BLOCK_H];
    float acc[BLOCK_H][DIMS_PER_LANE];

    #pragma unroll
    for (int h = 0; h < BLOCK_H; ++h) {
        e_max[h] = -INFINITY;
        e_sum[h] = 0.0f;
        #pragma unroll
        for (int d = 0; d < DIMS_PER_LANE; ++d) acc[h][d] = 0.0f;
    }

    for (int tok = 0; tok < seq_len; ++tok) {
        const int phys = req_to_token[batch_id * stride_r2t_b + tok];
        const uint8_t* k_row = kv + static_cast<size_t>(phys) * LK;

        // Each lane computes its slice of the LK dot (9 elements), then we
        // warp-reduce-sum to broadcast the full QK score to every lane.
        float qk[BLOCK_H];
        #pragma unroll
        for (int h = 0; h < BLOCK_H; ++h) qk[h] = 0.0f;

        #pragma unroll
        for (int i = 0; i < LK_PER_LANE; ++i) {
            const int d = lane_id * LK_PER_LANE + i;
            float k_v = fp8_e4m3fn_to_f32(k_row[d]);
            #pragma unroll
            for (int h = 0; h < BLOCK_H; ++h) {
                qk[h] = fmaf(q_lds[h * LK + d], k_v, qk[h]);
            }
        }
        #pragma unroll
        for (int h = 0; h < BLOCK_H; ++h) {
            qk[h] = warp_reduce_sum(qk[h]) * sm_scale;
        }

        #pragma unroll
        for (int h = 0; h < BLOCK_H; ++h) {
            float new_max  = fmaxf(e_max[h], qk[h]);
            float old_scale = __expf(e_max[h] - new_max);
            float p        = __expf(qk[h] - new_max);

            e_sum[h] = e_sum[h] * old_scale + p;
            e_max[h] = new_max;
            #pragma unroll
            for (int d = 0; d < DIMS_PER_LANE; ++d) acc[h][d] *= old_scale;

            #pragma unroll
            for (int d = 0; d < DIMS_PER_LANE; ++d) {
                int v_idx = lane_id * DIMS_PER_LANE + d;
                float v = fp8_e4m3fn_to_f32(k_row[v_idx]);
                acc[h][d] = fmaf(p, v, acc[h][d]);
            }
        }
    }

    #pragma unroll
    for (int h = 0; h < BLOCK_H; ++h) {
        const int head = head_base + h;
        const float inv_sum = 1.0f / e_sum[h];
        #pragma unroll
        for (int d = 0; d < DIMS_PER_LANE; ++d) {
            int v_idx = lane_id * DIMS_PER_LANE + d;
            float out_val = acc[h][d] * inv_sum;
            o[(batch_id * NHEAD + head) * LV + v_idx] = __float2bfloat16(out_val);
        }
        if (lane_id == 0 && lse != nullptr) {
            lse[batch_id * NHEAD + head] = e_max[h] + __logf(e_sum[h]);
        }
    }
}
