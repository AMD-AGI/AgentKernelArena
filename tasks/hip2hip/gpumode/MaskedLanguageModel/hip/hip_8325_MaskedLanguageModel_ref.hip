// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>
#include <c10/hip/HIPStream.h>

// Helpers for type-specific math on device
template <typename T>
__device__ inline T device_exp(T x);

template <>
__device__ inline float device_exp<float>(float x) { return expf(x); }

template <>
__device__ inline double device_exp<double>(double x) { return exp(x); }

template <typename T>
__device__ inline T device_log(T x);

template <>
__device__ inline float device_log<float>(float x) { return logf(x); }

template <>
__device__ inline double device_log<double>(double x) { return log(x); }

// Simple block-wide reduction for max and sum
// Assumes blockDim.x is a power of two up to 1024

template <typename T>
__device__ inline T block_allreduce_max(T val) {
    extern __shared__ unsigned char smem_[];
    T* smem = reinterpret_cast<T*>(smem_);
    const int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            T v = smem[tid + s];
            smem[tid] = v > smem[tid] ? v : smem[tid];
        }
        __syncthreads();
    }
    return smem[0];
}

template <typename T>
__device__ inline T block_allreduce_sum(T val) {
    extern __shared__ unsigned char smem_[];
    T* smem = reinterpret_cast<T*>(smem_);
    const int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    return smem[0];
}

// Transpose kernel: input A [rows, cols] -> output B [cols, rows]
// Launched with 2D grid and block.
template <typename scalar_t>
__global__ void transpose2d_kernel(const scalar_t* __restrict__ A,
                                   scalar_t* __restrict__ B,
                                   int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0, cols)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0, rows)
    if (row < rows && col < cols) {
        // A[row, col] -> B[col, row]
        B[col * rows + row] = A[row * cols + col];
    }
}

// Fused kernel: y = log_softmax(x @ W^T + b, dim=-1)
// input_2d: [N, in_features]
// weight_T: [in_features, out_features] (transposed)
// bias: [out_features]
// output: [N, out_features]
// One block processes one row (n). Threads iterate across out_features.
template <typename scalar_t>
__global__ void fused_linear_logsoftmax_kernel(
    const scalar_t* __restrict__ input_2d,
    const scalar_t* __restrict__ weight_T,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int N,
    int in_features,
    int out_features)
{
    int n = blockIdx.x; // row index
    if (n >= N) return;

    const scalar_t* xrow = input_2d + static_cast<long long>(n) * in_features;

    // Pass 1: compute per-thread max over its column stride
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    for (int col = threadIdx.x; col < out_features; col += blockDim.x) {
        // dot(xrow, W[col,:]) where W is [out_features, in_features]
        // with weight_T layout: weight_T[k, col] => weight_T[k*out_features + col]
        scalar_t acc = scalar_t(0);
        for (int k = 0; k < in_features; ++k) {
            acc += xrow[k] * weight_T[static_cast<long long>(k) * out_features + col];
        }
        acc += bias[col];
        thread_max = acc > thread_max ? acc : thread_max;
    }

    // Reduce to get row_max
    scalar_t row_max = block_allreduce_max(thread_max);

    // Pass 2: compute per-thread sum of exp(acc - row_max)
    scalar_t thread_sum = scalar_t(0);
    for (int col = threadIdx.x; col < out_features; col += blockDim.x) {
        scalar_t acc = scalar_t(0);
        for (int k = 0; k < in_features; ++k) {
            acc += xrow[k] * weight_T[static_cast<long long>(k) * out_features + col];
        }
        acc += bias[col];
        thread_sum += device_exp(acc - row_max);
    }

    scalar_t row_sum = block_allreduce_sum(thread_sum);

    // Compute logsumexp = row_max + log(row_sum)
    scalar_t lse = row_max + device_log(row_sum);

    // Pass 3: write outputs
    for (int col = threadIdx.x; col < out_features; col += blockDim.x) {
        scalar_t acc = scalar_t(0);
        for (int k = 0; k < in_features; ++k) {
            acc += xrow[k] * weight_T[static_cast<long long>(k) * out_features + col];
        }
        acc += bias[col];
        output[static_cast<long long>(n) * out_features + col] = acc - lse;
    }
}

// Public forward: x [B,S1,S2,H], weight [V,H], bias [V] -> [B,S1,S2,V]
torch::Tensor forward(
    torch::Tensor input,   // [B, S1, S2, hidden]
    torch::Tensor weight,  // [vocab_size, hidden]
    torch::Tensor bias     // [vocab_size]
) {
    TORCH_CHECK(input.is_cuda(), "input must be on HIP device");
    TORCH_CHECK(weight.is_cuda(), "weight must be on HIP device");
    TORCH_CHECK(bias.is_cuda(), "bias must be on HIP device");
    TORCH_CHECK(input.dim() == 4, "input must be 4D [B,S1,S2,H]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [V,H]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [V]");

    auto B = input.size(0);
    auto S1 = input.size(1);
    auto S2 = input.size(2);
    auto H = input.size(3);
    auto V = weight.size(0);
    TORCH_CHECK(weight.size(1) == H, "weight hidden dim mismatch");
    TORCH_CHECK(bias.size(0) == V, "bias size mismatch");

    // Flatten to [N, H]
    long long N = B * S1 * S2;

    auto input_2d = input.contiguous().view({N, H});
    auto out_2d = torch::empty({N, V}, input.options());

    // Prepare transposed weight: [H, V]
    auto weight_T = torch::empty({H, V}, weight.options());

    hipStream_t stream = c10::hip::getCurrentHIPStream();

    // Launch transpose: A [V,H] -> B [H,V]
    const int TILE_X = 32;
    const int TILE_Y = 8;
    dim3 tpb(TILE_X, TILE_Y);
    dim3 gridT((V + TILE_X - 1) / TILE_X, (H + TILE_Y - 1) / TILE_Y);

    AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "transpose2d_kernel", [&] {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(transpose2d_kernel<scalar_t>),
            gridT, tpb, 0, stream,
            weight.data_ptr<scalar_t>(),
            weight_T.data_ptr<scalar_t>(),
            static_cast<int>(V),
            static_cast<int>(H)
        );
    });

    // Fused linear + log_softmax
    const int block = 256; // must be power of two for our reduction helpers
    dim3 gridF(N);
    size_t smem_bytes = block * sizeof(float); // upper bound; we'll set per dtype below

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_linear_logsoftmax_kernel", [&] {
        size_t smem = block * sizeof(scalar_t);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(fused_linear_logsoftmax_kernel<scalar_t>),
            gridF, dim3(block), smem, stream,
            input_2d.data_ptr<scalar_t>(),
            weight_T.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            out_2d.data_ptr<scalar_t>(),
            static_cast<int>(N),
            static_cast<int>(H),
            static_cast<int>(V)
        );
    });

    // Reshape back to [B,S1,S2,V]
    return out_2d.view({B, S1, S2, V});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaskedLanguageModel forward (HIP fused linear+log_softmax)");
}

