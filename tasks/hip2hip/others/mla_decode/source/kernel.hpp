// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#pragma once
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
