// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// Transformer feed-forward block in float32: out = gelu(x @ w1^T + b1) @ w2^T + b2, as two tiled GEMM
// kernels with the bias and the exact (erf) GELU fused into the epilogue of the first one.
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>
#include <cstdint>

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int PAD = 4;
constexpr int THREADS = 256;

__device__ __forceinline__ float gelu_erf(float v) { return 0.5f * v * (1.0f + erff(v * 0.7071067811865475244f)); }

// C[M, N] = act(A[M, K] @ B[N, K]^T + bias[N]); every thread owns a 4x4 block of C.
template <bool GELU>
__global__ void __launch_bounds__(THREADS) gemm_bias_act_kernel(const float* __restrict__ A,
                                                                const float* __restrict__ B,
                                                                const float* __restrict__ bias,
                                                                float* __restrict__ C,
                                                                int M, int N, int K) {
    __shared__ __attribute__((aligned(16))) float As[BK][BM + PAD];
    __shared__ __attribute__((aligned(16))) float Bs[BK][BN + PAD];

    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int i = 0; i < (BM * BK) / THREADS; ++i) {
            const int idx = tid + i * THREADS;
            const int r = idx / BK;
            const int kk = idx % BK;
            const int gk = k0 + kk;
            const int ga = m0 + r;
            const int gb = n0 + r;
            As[kk][r] = (ga < M && gk < K) ? A[static_cast<int64_t>(ga) * K + gk] : 0.0f;
            Bs[kk][r] = (gb < N && gk < K) ? B[static_cast<int64_t>(gb) * K + gk] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
            const float4 b = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
            const float av[4] = {a.x, a.y, a.z, a.w};
            const float bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = m0 + ty * 4 + i;
        if (r >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = n0 + tx * 4 + j;
            if (c >= N) continue;
            float v = acc[i][j] + bias[c];
            if (GELU) v = gelu_erf(v);
            C[static_cast<int64_t>(r) * N + c] = v;
        }
    }
}

template <bool GELU>
static void launch_gemm(const at::Tensor& A, const at::Tensor& B, const at::Tensor& bias, at::Tensor& C,
                        hipStream_t stream) {
    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(0));
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    hipLaunchKernelGGL((gemm_bias_act_kernel<GELU>), grid, dim3(THREADS), 0, stream, A.data_ptr<float>(),
                       B.data_ptr<float>(), bias.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

void run(const at::Tensor& x, const at::Tensor& conv_weight, const at::Tensor& conv_bias,
         const at::Tensor& linear_weight, const at::Tensor& linear_bias, at::Tensor& out) {
    TORCH_CHECK(x.is_contiguous() && conv_weight.is_contiguous() && conv_bias.is_contiguous() &&
                    linear_weight.is_contiguous() && linear_bias.is_contiguous() && out.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kFloat && conv_weight.scalar_type() == at::kFloat &&
                    conv_bias.scalar_type() == at::kFloat && linear_weight.scalar_type() == at::kFloat &&
                    linear_bias.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat,
                "float32 tensors expected");
    const int64_t tokens = x.size(0);
    const int64_t hidden = x.size(1);
    const int64_t filter = conv_weight.size(0);
    TORCH_CHECK(conv_weight.size(1) == hidden && linear_weight.size(0) == hidden && linear_weight.size(1) == filter,
                "weight shapes do not match the input width");
    TORCH_CHECK(out.size(0) == tokens && out.size(1) == hidden, "out must be [tokens, hidden]");
    if (tokens == 0) return;

    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    at::Tensor mid = at::empty({tokens, filter}, x.options());
    launch_gemm<true>(x, conv_weight, conv_bias, mid, stream);
    launch_gemm<false>(mid, linear_weight, linear_bias, out, stream);
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor x, Tensor conv_weight, Tensor conv_bias, Tensor linear_weight, Tensor linear_bias, "
        "Tensor(a!) out) -> ()", &run);
}
