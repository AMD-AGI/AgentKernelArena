// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>
#include <cmath>

// Tile sizes (kept within 1024 threads per block)
constexpr int TM = 32; // rows per block (M)
constexpr int TN = 32; // cols per block (N)
constexpr int TK = 16; // K step

// First GEMM: Y[M,H] = ReLU( X[M,D] * W1^T[D,H] + b1[H] )
// X row-major [M,D]
// W1 stored as [H,D] row-major, but we need W1^T without materializing: indexing W1[h, d]
// Y row-major [M,H]
template <typename scalar_t>
__global__ void gemm_bias_relu_kernel(
    const scalar_t* __restrict__ X,    // [M, D]
    const scalar_t* __restrict__ W1,   // [H, D]
    const scalar_t* __restrict__ b1,   // [H]
    scalar_t* __restrict__ Y,          // [M, H]
    int M, int D, int H)
{
    __shared__ scalar_t As[TM][TK];  // tile from X: [TM, TK]
    __shared__ scalar_t Bs[TK][TN];  // tile from W1^T: [TK, TN] where TN maps to H

    int block_m = blockIdx.y * TM;
    int block_n = blockIdx.x * TN;

    int ty = threadIdx.y; // [0, TM)
    int tx = threadIdx.x; // [0, TN)

    int m = block_m + ty; // row index in M
    int n = block_n + tx; // col index in H

    scalar_t acc = scalar_t(0);

    for (int k0 = 0; k0 < D; k0 += TK) {
        // Load X tile: rows m, cols k0..k0+TK-1
        if (m < M) {
            int a_col = k0 + tx; // reuse tx up to TK with condition
            if (tx < TK && a_col < D) {
                As[ty][tx] = X[m * D + a_col];
            }
        }
        // Load W1^T tile: rows k0..k0+TK-1 (D), cols n (H)
        if (tx < TN) {
            int b_row = k0 + ty; // use ty up to TK with condition
            int b_col = block_n + tx; // n
            if (ty < TK && b_row < D && b_col < H) {
                // W1 is [H, D], element for W1^T[b_row, b_col] = W1[b_col, b_row]
                Bs[ty][tx] = W1[b_col * D + b_row];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TK; ++k) {
            if (k0 + k < D && m < M && n < H) {
                acc += As[ty][k] * Bs[k][tx];
            }
        }
        __syncthreads();
    }

    if (m < M && n < H) {
        acc += b1[n];
        acc = acc > scalar_t(0) ? acc : scalar_t(0);
        Y[m * H + n] = acc;
    }
}

// Second GEMM: Z[M,D] = Y[M,H] * W2^T[H,D] + b2[D]
// Y row-major [M,H]
// W2 stored as [D,H], need W2^T without materializing: W2^T[h,d] = W2[d,h]
// Z row-major [M,D]
template <typename scalar_t>
__global__ void gemm_bias_kernel(
    const scalar_t* __restrict__ Y,    // [M, H]
    const scalar_t* __restrict__ W2,   // [D, H]
    const scalar_t* __restrict__ b2,   // [D]
    scalar_t* __restrict__ Z,          // [M, D]
    int M, int H, int D)
{
    __shared__ scalar_t As[TM][TK];  // tile from Y: [TM, TK]
    __shared__ scalar_t Bs[TK][TN];  // tile from W2^T: [TK, TN] where TN maps to D

    int block_m = blockIdx.y * TM;
    int block_n = blockIdx.x * TN;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int m = block_m + ty; // row in M
    int n = block_n + tx; // col in D

    scalar_t acc = scalar_t(0);

    for (int k0 = 0; k0 < H; k0 += TK) {
        // Load Y tile
        if (m < M) {
            int a_col = k0 + tx; // use tx < TK
            if (tx < TK && a_col < H) {
                As[ty][tx] = Y[m * H + a_col];
            }
        }
        // Load W2^T tile
        if (tx < TN) {
            int b_row = k0 + ty; // use ty < TK
            int b_col = block_n + tx; // n in D
            if (ty < TK && b_row < H && b_col < D) {
                // W2 is [D, H], W2^T[b_row, b_col] = W2[b_col, b_row]
                Bs[ty][tx] = W2[b_col * H + b_row];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TK; ++k) {
            if (k0 + k < H && m < M && n < D) {
                acc += As[ty][k] * Bs[k][tx];
            }
        }
        __syncthreads();
    }

    if (m < M && n < D) {
        acc += b2[n];
        Z[m * D + n] = acc;
    }
}

// Residual add + LayerNorm over last dim (D) with eps, affine gamma/beta (dropout is the identity in eval mode)
// One block per row m, up to 256 threads per block
// Uses Welford's algorithm for numerical stability
template <typename scalar_t>
__global__ void add_layernorm_kernel(
    const scalar_t* __restrict__ residual, // x [M, D]
    const scalar_t* __restrict__ in,       // out after dropout [M, D]
    const scalar_t* __restrict__ gamma,    // [D]
    const scalar_t* __restrict__ beta,     // [D]
    scalar_t* __restrict__ out,            // [M, D]
    int M, int D, float eps)
{
    int m = blockIdx.x;        // row index
    int tid = threadIdx.x;
    int nthreads = blockDim.x; // <= 256

    // Accumulate partial Welford over this thread's strided elements
    float mean = 0.0f;
    float m2 = 0.0f;
    int count = 0;

    const int base = m * D;

    for (int d = tid; d < D; d += nthreads) {
        float v = (float)in[base + d] + (float)residual[base + d]; // residual add first
        // Welford online update
        count += 1;
        float delta = v - mean;
        mean += delta / (float)count;
        float delta2 = v - mean;
        m2 += delta * delta2;
    }

    __shared__ float mean_shared[256];
    __shared__ float m2_shared[256];
    __shared__ int   cnt_shared[256];

    mean_shared[tid] = mean;
    m2_shared[tid]   = m2;
    cnt_shared[tid]  = count;
    __syncthreads();

    // Pairwise reduction to combine Welford states
    for (int offset = nthreads >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float mean_a = mean_shared[tid];
            float m2_a   = m2_shared[tid];
            int   cnt_a  = cnt_shared[tid];

            float mean_b = mean_shared[tid + offset];
            float m2_b   = m2_shared[tid + offset];
            int   cnt_b  = cnt_shared[tid + offset];

            int cnt = cnt_a + cnt_b;
            float mean_new = (cnt > 0) ? ((mean_a * (float)cnt_a + mean_b * (float)cnt_b) / (float)cnt) : 0.0f;
            float delta = mean_b - mean_a;
            float m2_new = m2_a + m2_b + ((float)cnt_a * (float)cnt_b / (cnt > 0 ? (float)cnt : 1.0f)) * delta * delta;

            mean_shared[tid] = mean_new;
            m2_shared[tid]   = m2_new;
            cnt_shared[tid]  = cnt;
        }
        __syncthreads();
    }

    float final_mean = mean_shared[0];
    float final_var  = (D > 0) ? (m2_shared[0] / (float)D) : 0.0f; // unbiased=False
    float inv_std = rsqrtf(final_var + eps);

    for (int d = tid; d < D; d += nthreads) {
        float v = (float)in[base + d] + (float)residual[base + d];
        float nrm = (v - final_mean) * inv_std;
        float g = (float)gamma[d];
        float b = (float)beta[d];
        out[base + d] = (scalar_t)(nrm * g + b);
    }
}

// Launcher: out = LayerNorm(W2 relu(W1 x + b1) + b2 + x), writing the preallocated out.
void run(const at::Tensor& x, const at::Tensor& w1_weight, const at::Tensor& w1_bias,
         const at::Tensor& w2_weight, const at::Tensor& w2_bias,
         const at::Tensor& ln_gamma, const at::Tensor& ln_beta, double eps, at::Tensor& out)
{
    TORCH_CHECK(x.is_contiguous() && w1_weight.is_contiguous() && w1_bias.is_contiguous()
                && w2_weight.is_contiguous() && w2_bias.is_contiguous() && ln_gamma.is_contiguous()
                && ln_beta.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat, "float32 tensors expected");
    TORCH_CHECK(x.dim() == 2, "x must be [tokens, d_model]");

    const int64_t M = x.size(0);
    const int64_t D = x.size(1);
    const int64_t H = w1_weight.size(0);
    TORCH_CHECK(w1_weight.size(1) == D, "w1_weight must be [H, D]");
    TORCH_CHECK(w2_weight.size(0) == D && w2_weight.size(1) == H, "w2_weight must be [D, H]");
    if (M == 0) return;

    auto options = x.options();
    auto out1 = at::empty({M, H}, options);
    auto out2 = at::empty({M, D}, options);

    hipStream_t stream = c10::hip::getCurrentHIPStream();

    // First GEMM: [M,D] x W1^T -> [M,H], with bias+ReLU
    {
        dim3 block((unsigned)TN, (unsigned)TM, 1); // 32x32 = 1024
        dim3 grid((unsigned)((H + TN - 1) / TN), (unsigned)((M + TM - 1) / TM), 1);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_bias_relu_kernel<float>),
            grid, block, 0, stream,
            x.data_ptr<float>(), w1_weight.data_ptr<float>(), w1_bias.data_ptr<float>(),
            out1.data_ptr<float>(), (int)M, (int)D, (int)H);
    }

    // Second GEMM: [M,H] x W2^T -> [M,D], with bias
    {
        dim3 block((unsigned)TN, (unsigned)TM, 1);
        dim3 grid((unsigned)((D + TN - 1) / TN), (unsigned)((M + TM - 1) / TM), 1);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_bias_kernel<float>),
            grid, block, 0, stream,
            out1.data_ptr<float>(), w2_weight.data_ptr<float>(), w2_bias.data_ptr<float>(),
            out2.data_ptr<float>(), (int)M, (int)H, (int)D);
    }

    // Residual add + LayerNorm per row
    {
        int threads = 1;
        while (threads < D && threads < 256) threads <<= 1; // up to 256
        if (threads < 32) threads = 32;
        dim3 grid((unsigned)M);
        dim3 block((unsigned)threads);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(add_layernorm_kernel<float>),
            grid, block, 0, stream,
            x.data_ptr<float>(), out2.data_ptr<float>(), ln_gamma.data_ptr<float>(),
            ln_beta.data_ptr<float>(), out.data_ptr<float>(), (int)M, (int)D, (float)eps);
    }
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor x, Tensor w1_weight, Tensor w1_bias, Tensor w2_weight, Tensor w2_bias, "
        "Tensor ln_gamma, Tensor ln_beta, float eps, Tensor(a!) out) -> ()", &run);
}
