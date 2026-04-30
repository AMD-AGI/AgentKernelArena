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
// CLI:  ./applications_mla_decode --batch B --ctx CTX [--mode {full,check,bench}]
//   --mode full  (default): prints both Check: and Perf: lines.
//   --mode check          : prints only Check: line; exit 0/1 on PASS/FAIL.
//   --mode bench          : prints only Perf: line (skips host fp32 ref).

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
    std::exit(1);} } while(0)

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

__host__ __device__ __forceinline__ float fp8_e4m3fn_to_f32(uint8_t b) {
    if (b == 0x80 || b == 0x00) return 0.0f;
    int sign = (b & 0x80) ? -1 : 1;
    int e = (b >> 3) & 0xF;
    int m = b & 0x7;
    float val;
    if (e == 0) {
        val = static_cast<float>(m) / 8.0f / 64.0f;  // subnormal: m * 2^-9
    } else {
        float mantissa = 1.0f + static_cast<float>(m) / 8.0f;
#ifdef __HIP_DEVICE_COMPILE__
        val = mantissa * exp2f(static_cast<float>(e) - 7);
#else
        val = mantissa * std::exp2f(static_cast<float>(e) - 7);
#endif
    }
    return sign * val;
}

// Saturating round to e4m3fn. Used only on the host to set up the KV cache.
static uint8_t f32_to_fp8_e4m3fn(float x) {
    if (x == 0.0f || !std::isfinite(x)) return 0x00;
    uint32_t sign_bit = (x < 0) ? 0x80 : 0x00;
    float ax = std::fabs(x);
    // e4m3fn finite max = 1.75 * 2^8 = 448.
    const float max_e4m3 = 448.0f;
    if (ax >= max_e4m3) return static_cast<uint8_t>(sign_bit | 0x7E);  // saturate to ±max
    int e_unbiased = static_cast<int>(std::floor(std::log2(ax)));
    int e_biased = e_unbiased + 7;
    if (e_biased <= 0) {
        // Subnormal: m * 2^-9
        int m = static_cast<int>(std::round(ax * 512.0f));
        if (m > 7) m = 7;
        if (m == 0) return static_cast<uint8_t>(sign_bit | 0x00);
        return static_cast<uint8_t>(sign_bit | (m & 0x7));
    }
    if (e_biased > 15) e_biased = 15;
    float mantissa = ax / std::ldexp(1.0f, e_unbiased) - 1.0f;
    int m = static_cast<int>(std::round(mantissa * 8.0f));
    if (m == 8) { ++e_biased; m = 0; if (e_biased > 15) { e_biased = 15; m = 7; } }
    if (m < 0) m = 0;
    if (m > 7) m = 7;
    return static_cast<uint8_t>(sign_bit | ((e_biased & 0xF) << 3) | (m & 0x7));
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

// ---------------------------------------------------------------------------
// Host helpers: random fill, fp32 reference, max_diff
// ---------------------------------------------------------------------------

static void fill_random_bf16(std::vector<bf16>& buf, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& v : buf) v = __float2bfloat16(dist(rng));
}

static void fill_kv_fp8(std::vector<uint8_t>& kv, uint32_t seed) {
    std::mt19937 rng(seed);
    // Match the kernel's natural input range (typical KV-cache magnitudes).
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& byte : kv) byte = f32_to_fp8_e4m3fn(dist(rng));
}

static void host_reference(
    std::vector<bf16>& out,
    const std::vector<bf16>& q,
    const std::vector<uint8_t>& kv,
    const std::vector<int32_t>& req_to_token,
    const std::vector<int32_t>& b_seq_len,
    int batch, int max_ctx, float sm_scale
) {
    // out: [batch, NHEAD, LV]
    // For each (b, h): scores[k] = sum_d q[b,h,d] * dequant(kv[r2t[b,k]][d]) * sm_scale
    //                  out[b,h,:] = softmax(scores) @ v_b[:, :LV]
    out.assign(static_cast<size_t>(batch) * NHEAD * LV, __float2bfloat16(0.0f));
#if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
#endif
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < NHEAD; ++h) {
            const int seq = b_seq_len[b];
            std::vector<float> scores(seq, 0.0f);
            for (int t = 0; t < seq; ++t) {
                const int phys = req_to_token[static_cast<size_t>(b) * max_ctx + t];
                const uint8_t* k_row = kv.data() + static_cast<size_t>(phys) * LK;
                float s = 0.0f;
                for (int d = 0; d < LK; ++d) {
                    float qd = __bfloat162float(q[(static_cast<size_t>(b) * NHEAD + h) * LK + d]);
                    float kd = fp8_e4m3fn_to_f32(k_row[d]);
                    s += qd * kd;
                }
                scores[t] = s * sm_scale;
            }
            float m = -INFINITY;
            for (int t = 0; t < seq; ++t) m = std::fmax(m, scores[t]);
            double sum = 0.0;
            for (int t = 0; t < seq; ++t) {
                scores[t] = std::exp(scores[t] - m);
                sum += scores[t];
            }
            const float inv_sum = static_cast<float>(1.0 / sum);
            std::vector<float> ovec(LV, 0.0f);
            for (int t = 0; t < seq; ++t) {
                const int phys = req_to_token[static_cast<size_t>(b) * max_ctx + t];
                const uint8_t* v_row = kv.data() + static_cast<size_t>(phys) * LK;
                const float p = scores[t] * inv_sum;
                for (int d = 0; d < LV; ++d) {
                    ovec[d] += p * fp8_e4m3fn_to_f32(v_row[d]);
                }
            }
            for (int d = 0; d < LV; ++d) {
                out[(static_cast<size_t>(b) * NHEAD + h) * LV + d] = __float2bfloat16(ovec[d]);
            }
        }
    }
}

static void max_diff(
    const std::vector<bf16>& a, const std::vector<bf16>& b,
    double& max_abs, double& max_rel
) {
    max_abs = 0.0;
    max_rel = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = static_cast<double>(__bfloat162float(a[i]));
        const double y = static_cast<double>(__bfloat162float(b[i]));
        const double d = std::fabs(x - y);
        max_abs = std::fmax(max_abs, d);
        const double denom = std::fmax(1e-8, std::fmax(std::fabs(x), std::fabs(y)));
        max_rel = std::fmax(max_rel, d / denom);
    }
}

// ---------------------------------------------------------------------------
// Bench
// ---------------------------------------------------------------------------

template <typename Fn>
static float time_kernel_ms(Fn launch, int warmup, int iters) {
    hipEvent_t s, t;
    HIP_CHECK(hipEventCreate(&s));
    HIP_CHECK(hipEventCreate(&t));
    for (int i = 0; i < warmup; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(s));
    for (int i = 0; i < iters; ++i) launch();
    HIP_CHECK(hipEventRecord(t));
    HIP_CHECK(hipEventSynchronize(t));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, s, t));
    HIP_CHECK(hipEventDestroy(s));
    HIP_CHECK(hipEventDestroy(t));
    return ms / iters;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int batch = 4;
    int ctx   = 1024;
    enum class Mode { Full, Check, Bench };
    Mode mode = Mode::Full;
    for (int i = 1; i + 1 < argc; ++i) {
        std::string a(argv[i]);
        if      (a == "--batch") batch = std::atoi(argv[++i]);
        else if (a == "--ctx")   ctx   = std::atoi(argv[++i]);
        else if (a == "--mode") {
            std::string m(argv[++i]);
            if      (m == "full")  mode = Mode::Full;
            else if (m == "check") mode = Mode::Check;
            else if (m == "bench") mode = Mode::Bench;
            else { fprintf(stderr, "unknown --mode %s\n", m.c_str()); return 2; }
        }
    }
    if (batch <= 0 || ctx <= 0) {
        fprintf(stderr, "invalid --batch %d or --ctx %d\n", batch, ctx);
        return 2;
    }
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(LK));

    const size_t q_e   = static_cast<size_t>(batch) * NHEAD * LK;
    const size_t o_e   = static_cast<size_t>(batch) * NHEAD * LV;
    const size_t r2t_e = static_cast<size_t>(batch) * ctx;
    const size_t kv_tokens = static_cast<size_t>(batch) * ctx + 1024;
    const size_t kv_e  = kv_tokens * LK;

    std::vector<bf16>    h_q(q_e);
    std::vector<bf16>    h_o(o_e);
    std::vector<bf16>    h_ref(o_e);
    std::vector<uint8_t> h_kv(kv_e);
    std::vector<int32_t> h_r2t(r2t_e);
    std::vector<int32_t> h_bsl(batch, ctx);

    fill_random_bf16(h_q,  -1.0f, 1.0f, /*seed=*/42 + batch * 31 + ctx);
    fill_kv_fp8     (h_kv,                   /*seed=*/123 + batch * 17 + ctx);
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < ctx; ++t) {
            h_r2t[static_cast<size_t>(b) * ctx + t] = b * ctx + t;
        }
    }

    bf16    *d_q = nullptr,  *d_o = nullptr;
    uint8_t *d_kv = nullptr;
    int32_t *d_r2t = nullptr, *d_bsl = nullptr;
    HIP_CHECK(hipMalloc(&d_q,   q_e   * sizeof(bf16)));
    HIP_CHECK(hipMalloc(&d_o,   o_e   * sizeof(bf16)));
    HIP_CHECK(hipMalloc(&d_kv,  kv_e  * sizeof(uint8_t)));
    HIP_CHECK(hipMalloc(&d_r2t, r2t_e * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_bsl, batch * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(d_q,   h_q.data(),   q_e   * sizeof(bf16),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_kv,  h_kv.data(),  kv_e  * sizeof(uint8_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_r2t, h_r2t.data(), r2t_e * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsl, h_bsl.data(), batch * sizeof(int32_t), hipMemcpyHostToDevice));

    dim3 grid(batch, HEAD_GROUPS);
    dim3 block(BLOCK_THREADS);
    const size_t smem = BLOCK_H * LK * sizeof(float);

    auto launch = [&]() {
        hipLaunchKernelGGL(
            mla_decode_kernel, grid, block, smem, /*stream=*/0,
            d_q, d_kv, d_r2t, d_bsl, sm_scale,
            d_o, /*lse=*/static_cast<float*>(nullptr),
            /*stride_r2t_b=*/ctx,
            /*max_ctx=*/ctx
        );
    };

    bool check_ok = true;

    // Correctness (full or check mode).
    if (mode == Mode::Full || mode == Mode::Check) {
        launch();
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_o.data(), d_o, o_e * sizeof(bf16), hipMemcpyDeviceToHost));
        host_reference(h_ref, h_q, h_kv, h_r2t, h_bsl, batch, ctx, sm_scale);

        double max_abs = 0.0, max_rel = 0.0;
        max_diff(h_o, h_ref, max_abs, max_rel);
        // bf16 + fp8 dequant noise: a few percent absolute error on a softmax-
        // normalized output is fine. Mirror the silu task's tolerance shape.
        const double atol = 5e-2, rtol = 1e-1;
        check_ok = (max_abs <= atol) || (max_rel <= rtol);
        printf("Check: max_abs=%.4g max_rel=%.4g -> %s\n",
               max_abs, max_rel, check_ok ? "PASS" : "FAIL");
    }

    // Performance (full or bench mode).
    if (mode == Mode::Full || mode == Mode::Bench) {
        const float us = time_kernel_ms(launch, /*warmup=*/5, /*iters=*/50) * 1000.0f;
        const double bytes_kv = static_cast<double>(batch) * ctx * LK;       // kv reads
        const double bytes_q  = static_cast<double>(batch) * NHEAD * LK * 2; // bf16 q reads
        const double bytes_o  = static_cast<double>(batch) * NHEAD * LV * 2; // bf16 o writes
        const double gbs = ((bytes_kv + bytes_q + bytes_o) / (us * 1e-6)) / 1e9;
        printf("Perf: %.3f us/launch | ~BW: %.1f GB/s\n", us, gbs);
    }

    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_o));
    HIP_CHECK(hipFree(d_kv));
    HIP_CHECK(hipFree(d_r2t));
    HIP_CHECK(hipFree(d_bsl));
    return check_ok ? 0 : 1;
}
