// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// Non-local attention block (dot-product / ELU variant) on a float32 feature map, written as two
// kernels: a fused 1x1-conv projection of q, k and v, and a tiled attention kernel that never
// materializes the HW x HW energy matrix and applies the output 1x1 conv (gamma) in its epilogue.
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

constexpr int C = 64;            // channels
constexpr int CK = 16;           // query / key channels (C / 4)
constexpr int OC = 2 * CK + C;   // fused projection output channels: q | k | v

constexpr int PB = 64;           // positions per projection block
constexpr int BJ = 64;           // output positions per attention block
constexpr int BI = 32;           // summed (row) positions per attention step

// qkv[b, o, p] = bias[o] + sum_c W[o, c] * x[b, c, p], with W = [wq; wk; wv].
__global__ __launch_bounds__(256) void qkv_projection_kernel(
    const float* __restrict__ x, const float* __restrict__ wq, const float* __restrict__ bq,
    const float* __restrict__ wk, const float* __restrict__ bk, const float* __restrict__ wv,
    const float* __restrict__ bv, float* __restrict__ qkv, int N)
{
    __shared__ float xs[C][PB];
    __shared__ float ws[OC][C + 1];
    const int t = threadIdx.x;
    const size_t b = blockIdx.y;
    const int p0 = blockIdx.x * PB;

    for (int idx = t; idx < C * PB; idx += 256) {
        const int c = idx / PB, p = idx % PB;
        xs[c][p] = (p0 + p < N) ? x[(b * C + c) * N + p0 + p] : 0.0f;
    }
    for (int idx = t; idx < OC * C; idx += 256) {
        const int o = idx / C, c = idx % C;
        const float* w = o < CK ? wq : (o < 2 * CK ? wk : wv);
        const int row = o < CK ? o : (o < 2 * CK ? o - CK : o - 2 * CK);
        ws[o][c] = w[row * C + c];
    }
    __syncthreads();

    const int tx = t % 32;      // positions tx and tx + 32
    const int o0 = (t / 32) * 12;  // 12 output channels
    float acc[12][2];
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        const int o = o0 + k;
        const float bias = o < CK ? bq[o] : (o < 2 * CK ? bk[o - CK] : bv[o - 2 * CK]);
        acc[k][0] = bias;
        acc[k][1] = bias;
    }
    for (int c = 0; c < C; ++c) {
        const float x0 = xs[c][tx], x1 = xs[c][tx + 32];
        #pragma unroll
        for (int k = 0; k < 12; ++k) {
            const float w = ws[o0 + k][c];
            acc[k][0] += w * x0;
            acc[k][1] += w * x1;
        }
    }
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int p = p0 + tx + 32 * r;
            if (p < N) qkv[(b * OC + o0 + k) * N + p] = acc[k][r];
        }
    }
}

// out[b, :, j] = gamma_w * (sum_i v[b, :, i] * elu(q[b, :, i] . k[b, :, j]) / N) + gamma_b.
__global__ __launch_bounds__(256) void nonlocal_dot_attention_kernel(
    const float* __restrict__ qkv, const float* __restrict__ gw, const float* __restrict__ gb,
    float* __restrict__ out, int N)
{
    __shared__ float ks[CK][BJ];
    __shared__ float qs[CK][BI];
    __shared__ float vs[C][BI + 1];
    __shared__ float es[BI][BJ];
    __shared__ float os[C][BJ];
    const int t = threadIdx.x;
    const size_t b = blockIdx.y;
    const int j0 = blockIdx.x * BJ;
    const float* q = qkv + b * OC * N;
    const float* k = q + (size_t)CK * N;
    const float* v = k + (size_t)CK * N;
    const float inv_n = 1.0f / (float)N;

    for (int idx = t; idx < CK * BJ; idx += 256) {
        const int d = idx / BJ, jj = idx % BJ;
        ks[d][jj] = (j0 + jj < N) ? k[(size_t)d * N + j0 + jj] : 0.0f;
    }

    const int cb = (t / 16) * 4;   // output channels cb..cb+3
    const int jb = (t % 16) * 4;   // output positions jb..jb+3
    float acc[4][4];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        #pragma unroll
        for (int m = 0; m < 4; ++m) acc[a][m] = 0.0f;
    }

    for (int i0 = 0; i0 < N; i0 += BI) {
        __syncthreads();
        for (int idx = t; idx < CK * BI; idx += 256) {
            const int d = idx / BI, ii = idx % BI;
            qs[d][ii] = (i0 + ii < N) ? q[(size_t)d * N + i0 + ii] : 0.0f;
        }
        for (int idx = t; idx < C * BI; idx += 256) {
            const int c = idx / BI, ii = idx % BI;
            vs[c][ii] = (i0 + ii < N) ? v[(size_t)c * N + i0 + ii] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int r = 0; r < BI / 16; ++r) {
            const int il = t / 16 + 16 * r;
            float e[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int d = 0; d < CK; ++d) {
                const float qv = qs[d][il];
                #pragma unroll
                for (int m = 0; m < 4; ++m) e[m] += qv * ks[d][jb + m];
            }
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                const float elu = e[m] > 0.0f ? e[m] : expm1f(e[m]);
                es[il][jb + m] = (i0 + il < N) ? elu * inv_n : 0.0f;
            }
        }
        __syncthreads();

        for (int il = 0; il < BI; ++il) {
            float ev[4];
            #pragma unroll
            for (int m = 0; m < 4; ++m) ev[m] = es[il][jb + m];
            #pragma unroll
            for (int a = 0; a < 4; ++a) {
                const float vv = vs[cb + a][il];
                #pragma unroll
                for (int m = 0; m < 4; ++m) acc[a][m] += vv * ev[m];
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        #pragma unroll
        for (int m = 0; m < 4; ++m) os[cb + a][jb + m] = acc[a][m];
    }
    __syncthreads();

    float res[4][4];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        #pragma unroll
        for (int m = 0; m < 4; ++m) res[a][m] = gb[cb + a];
    }
    for (int c2 = 0; c2 < C; ++c2) {
        float ov[4];
        #pragma unroll
        for (int m = 0; m < 4; ++m) ov[m] = os[c2][jb + m];
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            const float g = gw[(cb + a) * C + c2];
            #pragma unroll
            for (int m = 0; m < 4; ++m) res[a][m] += g * ov[m];
        }
    }
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (j0 + jb + m < N) out[(b * C + cb + a) * N + j0 + jb + m] = res[a][m];
        }
    }
}

static void check_tensor(const at::Tensor& t, const char* name, int64_t d0, int64_t d1, int64_t d2 = -1)
{
    TORCH_CHECK(t.scalar_type() == at::kFloat && t.is_contiguous(), name, " must be contiguous float32");
    TORCH_CHECK(t.size(0) == d0 && t.size(1) == d1 && (d2 < 0 || t.size(2) == d2), name, " has the wrong shape");
}

void run(const at::Tensor& x, const at::Tensor& query_weight, const at::Tensor& query_bias,
         const at::Tensor& key_weight, const at::Tensor& key_bias, const at::Tensor& value_weight,
         const at::Tensor& value_bias, const at::Tensor& gamma_weight, const at::Tensor& gamma_bias,
         at::Tensor& out)
{
    TORCH_CHECK(x.dim() == 3 && x.size(1) == C, "x must be [batch, 64, hw]");
    const int64_t B = x.size(0);
    const int64_t N = x.size(2);
    check_tensor(x, "x", B, C, N);
    check_tensor(out, "out", B, C, N);
    check_tensor(query_weight, "query_weight", CK, C);
    check_tensor(key_weight, "key_weight", CK, C);
    check_tensor(value_weight, "value_weight", C, C);
    check_tensor(gamma_weight, "gamma_weight", C, C);
    TORCH_CHECK(query_bias.numel() == CK && key_bias.numel() == CK && value_bias.numel() == C &&
                gamma_bias.numel() == C, "bias size mismatch");
    TORCH_CHECK(N >= 1 && B >= 1 && B * OC * N < (int64_t(1) << 40), "unsupported size");

    auto stream = c10::hip::getCurrentHIPStream();
    auto qkv = at::empty({B, OC, N}, x.options());

    dim3 projection_grid(static_cast<unsigned>((N + PB - 1) / PB), static_cast<unsigned>(B));
    hipLaunchKernelGGL(qkv_projection_kernel, projection_grid, dim3(256), 0, stream,
                       x.data_ptr<float>(), query_weight.data_ptr<float>(), query_bias.data_ptr<float>(),
                       key_weight.data_ptr<float>(), key_bias.data_ptr<float>(),
                       value_weight.data_ptr<float>(), value_bias.data_ptr<float>(),
                       qkv.data_ptr<float>(), static_cast<int>(N));

    dim3 attention_grid(static_cast<unsigned>((N + BJ - 1) / BJ), static_cast<unsigned>(B));
    hipLaunchKernelGGL(nonlocal_dot_attention_kernel, attention_grid, dim3(256), 0, stream,
                       qkv.data_ptr<float>(), gamma_weight.data_ptr<float>(), gamma_bias.data_ptr<float>(),
                       out.data_ptr<float>(), static_cast<int>(N));
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor x, Tensor query_weight, Tensor query_bias, Tensor key_weight, Tensor key_bias, "
        "Tensor value_weight, Tensor value_bias, Tensor gamma_weight, Tensor gamma_bias, Tensor(a!) out) -> ()", &run);
}
