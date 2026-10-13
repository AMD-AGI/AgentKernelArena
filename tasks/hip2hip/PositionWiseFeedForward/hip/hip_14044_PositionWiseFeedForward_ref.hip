// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <cmath>

#ifndef CHECK_HIP
#define CHECK_HIP(x) do { hipError_t err = (x); if (err != hipSuccess) { printf("HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); abort(); } } while(0)
#endif

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

// Residual add + LayerNorm over last dim (D) with eps, affine gamma/beta
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

// Public launcher matching requested signature
// Returns tensor of same shape as x
torch::Tensor positionwise_ffn_forward(
    torch::Tensor x,                // [..., D]
    torch::Tensor w1_weight,        // [H, D]
    torch::Tensor w1_bias,          // [H]
    torch::Tensor w2_weight,        // [D, H]
    torch::Tensor w2_bias,          // [D]
    torch::Tensor ln_gamma,         // [D]
    torch::Tensor ln_beta,          // [D]
    float dropout_p = 0.5f,
    float eps = 1e-5f)
{
    TORCH_CHECK(x.is_cuda(), "x must be on HIP/CUDA device (ROCm)");
    TORCH_CHECK(w1_weight.is_cuda() && w1_bias.is_cuda() && w2_weight.is_cuda() && w2_bias.is_cuda() && ln_gamma.is_cuda() && ln_beta.is_cuda(), "All tensors must be on device");

    x = x.contiguous();
    w1_weight = w1_weight.contiguous();
    w1_bias = w1_bias.contiguous();
    w2_weight = w2_weight.contiguous();
    w2_bias = w2_bias.contiguous();
    ln_gamma = ln_gamma.contiguous();
    ln_beta = ln_beta.contiguous();

    const int64_t D = x.size(-1);
    TORCH_CHECK(w1_weight.size(1) == D, "w1_weight in_features must match D");
    const int64_t H = w1_weight.size(0);
    TORCH_CHECK(w2_weight.size(0) == D && w2_weight.size(1) == H, "w2_weight must be [D, H]");
    TORCH_CHECK(w1_bias.numel() == H && w2_bias.numel() == D && ln_gamma.numel() == D && ln_beta.numel() == D, "bias/gamma/beta sizes must match");

    // Flatten leading dims to M
    int64_t M = x.numel() / D;
    auto options = x.options();

    auto x2d = x.view({M, D});
    auto out1 = torch::empty({M, H}, options);
    auto out2 = torch::empty({M, D}, options);

    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Launch first GEMM: [M,D] x W1^T -> [M,H], with bias+ReLU
    {
        dim3 block((unsigned)TN, (unsigned)TM, 1); // 32x32 = 1024
        dim3 grid((unsigned)((H + TN - 1) / TN), (unsigned)((M + TM - 1) / TM), 1);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gemm_bias_relu_kernel", [&] {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemm_bias_relu_kernel<scalar_t>),
                grid, block, 0, stream,
                x2d.data_ptr<scalar_t>(),
                w1_weight.data_ptr<scalar_t>(),
                w1_bias.data_ptr<scalar_t>(),
                out1.data_ptr<scalar_t>(),
                (int)M, (int)D, (int)H);
        });
        CHECK_HIP(hipGetLastError());
    }

    // Second GEMM: [M,H] x W2^T -> [M,D], with bias
    {
        dim3 block((unsigned)TN, (unsigned)TM, 1);
        dim3 grid((unsigned)((D + TN - 1) / TN), (unsigned)((M + TM - 1) / TM), 1);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gemm_bias_kernel", [&] {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemm_bias_kernel<scalar_t>),
                grid, block, 0, stream,
                out1.data_ptr<scalar_t>(),
                w2_weight.data_ptr<scalar_t>(),
                w2_bias.data_ptr<scalar_t>(),
                out2.data_ptr<scalar_t>(),
                (int)M, (int)H, (int)D);
        });
        CHECK_HIP(hipGetLastError());
    }

    // Dropout using PyTorch's implementation to guarantee identical RNG behavior
    // training=True to match F.dropout in module_fn
    torch::Tensor dropped = at::dropout(out2, dropout_p, /*train=*/true);

    // Residual add + LayerNorm per row M
    auto y = torch::empty_like(dropped);
    {
        int threads = 1;
        while (threads < D && threads < 256) threads <<= 1; // up to 256
        threads = std::max(threads, 32); // at least 32 threads for small D
        threads = std::min(threads, 256);
        dim3 grid((unsigned)M);
        dim3 block((unsigned)threads);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "add_layernorm_kernel", [&] {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(add_layernorm_kernel<scalar_t>),
                grid, block, 0, stream,
                x2d.data_ptr<scalar_t>(),
                dropped.data_ptr<scalar_t>(),
                ln_gamma.data_ptr<scalar_t>(),
                ln_beta.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                (int)M, (int)D, (float)eps);
        });
        CHECK_HIP(hipGetLastError());
    }

    return y.view(x.sizes());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &positionwise_ffn_forward,
          py::arg("x"),
          py::arg("w1_weight"),
          py::arg("w1_bias"),
          py::arg("w2_weight"),
          py::arg("w2_bias"),
          py::arg("ln_gamma"),
          py::arg("ln_beta"),
          py::arg("dropout_p") = 0.5f,
          py::arg("eps") = 1e-5f,
          "Position-wise FeedForward forward (HIP/ROCm)");
}

