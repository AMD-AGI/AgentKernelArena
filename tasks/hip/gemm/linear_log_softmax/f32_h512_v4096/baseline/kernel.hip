// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// Float32 linear layer fused with log_softmax: out = log_softmax(x @ weight^T + bias, dim=-1).
// A transpose kernel first builds weight^T; then one block per row computes the logits three times
// (row max, sum of exponentials, output) so no intermediate logits are stored.
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>
#include <cstdint>
#include <limits>

// Block-wide reductions; blockDim.x must be a power of two (<= 1024).
__device__ inline float block_allreduce_max(float val) {
    extern __shared__ unsigned char smem_[];
    float* smem = reinterpret_cast<float*>(smem_);
    const int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float v = smem[tid + s];
            smem[tid] = v > smem[tid] ? v : smem[tid];
        }
        __syncthreads();
    }
    float result = smem[0];
    __syncthreads();
    return result;
}

__device__ inline float block_allreduce_sum(float val) {
    extern __shared__ unsigned char smem_[];
    float* smem = reinterpret_cast<float*>(smem_);
    const int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float result = smem[0];
    __syncthreads();
    return result;
}

// Transpose kernel: input A [rows, cols] -> output B [cols, rows].
__global__ void transpose2d_kernel(const float* __restrict__ A, float* __restrict__ B, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // [0, cols)
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // [0, rows)
    if (row < rows && col < cols) {
        B[col * rows + row] = A[row * cols + col];
    }
}

// Fused kernel: y = log_softmax(x @ W^T + b, dim=-1).
// input_2d: [N, in_features], weight_T: [in_features, out_features], bias: [out_features],
// output: [N, out_features]. One block processes one row; threads stride over out_features.
__global__ void fused_linear_logsoftmax_kernel(const float* __restrict__ input_2d,
                                               const float* __restrict__ weight_T,
                                               const float* __restrict__ bias,
                                               float* __restrict__ output,
                                               int N, int in_features, int out_features) {
    int n = blockIdx.x;
    if (n >= N) return;

    const float* xrow = input_2d + static_cast<long long>(n) * in_features;

    float thread_max = -std::numeric_limits<float>::infinity();
    for (int col = threadIdx.x; col < out_features; col += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            acc += xrow[k] * weight_T[static_cast<long long>(k) * out_features + col];
        }
        acc += bias[col];
        thread_max = acc > thread_max ? acc : thread_max;
    }
    float row_max = block_allreduce_max(thread_max);

    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < out_features; col += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            acc += xrow[k] * weight_T[static_cast<long long>(k) * out_features + col];
        }
        acc += bias[col];
        thread_sum += expf(acc - row_max);
    }
    float row_sum = block_allreduce_sum(thread_sum);

    float lse = row_max + logf(row_sum);

    for (int col = threadIdx.x; col < out_features; col += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            acc += xrow[k] * weight_T[static_cast<long long>(k) * out_features + col];
        }
        acc += bias[col];
        output[static_cast<long long>(n) * out_features + col] = acc - lse;
    }
}

void run(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias, at::Tensor& out) {
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && bias.is_contiguous() && out.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat &&
                    bias.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat,
                "float32 tensors expected");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && bias.dim() == 1, "x [N,H], weight [V,H], bias [V] expected");

    const int64_t N = x.size(0);
    const int64_t H = x.size(1);
    const int64_t V = weight.size(0);
    TORCH_CHECK(weight.size(1) == H, "weight hidden dim mismatch");
    TORCH_CHECK(bias.size(0) == V, "bias size mismatch");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == N && out.size(1) == V, "out must be [N, V]");
    if (N == 0) return;

    // Transposed weight scratch: [H, V].
    at::Tensor weight_T = at::empty({H, V}, weight.options());

    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    const int TILE_X = 32;
    const int TILE_Y = 8;
    dim3 tpb(TILE_X, TILE_Y);
    dim3 gridT((H + TILE_X - 1) / TILE_X, (V + TILE_Y - 1) / TILE_Y);
    hipLaunchKernelGGL(transpose2d_kernel, gridT, tpb, 0, stream, weight.data_ptr<float>(),
                       weight_T.data_ptr<float>(), static_cast<int>(V), static_cast<int>(H));

    const int block = 256;  // must be a power of two for the reduction helpers
    dim3 gridF(static_cast<unsigned int>(N));
    size_t smem = block * sizeof(float);
    hipLaunchKernelGGL(fused_linear_logsoftmax_kernel, gridF, dim3(block), smem, stream, x.data_ptr<float>(),
                       weight_T.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(),
                       static_cast<int>(N), static_cast<int>(H), static_cast<int>(V));
    TORCH_CHECK(hipGetLastError() == hipSuccess, "kernel launch failed");
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor x, Tensor weight, Tensor bias, Tensor(a!) out) -> ()", &run);
}
