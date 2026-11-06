// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// Fused bias + leaky ReLU + scale (StyleGAN2 FusedLeakyReLU) on NCHW float32.
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

// Branchless leaky ReLU helpers
__device__ __forceinline__ float leaky_relu_scaled_f(float x, float neg_slope, float scale) {
    float pos = fmaxf(x, 0.0f);
    float neg = fminf(x, 0.0f);
    return pos * scale + neg * (neg_slope * scale);
}

// Kernel: 2D grid over (spatial, N*C)
__global__ void fused_leaky_relu_2d_kernel(
    const float* __restrict__ input,   // [N, C, H, W]
    const float* __restrict__ bias,    // [C]
    float* __restrict__ output,        // [N, C, H, W]
    int N, int C, int H, int W,
    float negative_slope,
    float scale)
{
    const int S = H * W;
    const int slice = blockIdx.y; // 0..(N*C-1)
    if (slice >= N * C) return;
    const int n = slice / C;
    const int c = slice - n * C;

    const int slice_offset = (n * C + c) * S;

    // Load bias[c] once
    const float b = bias[c];

    // grid-stride over spatial
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (; idx < S; idx += stride) {
        const int off = slice_offset + idx;
        float x = input[off];
        float v = x + b;
        output[off] = leaky_relu_scaled_f(v, negative_slope, scale);
    }
}

// Fallback 1D kernel (used only if grid.y limit exceeded)
__global__ void fused_leaky_relu_1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    float negative_slope,
    float scale)
{
    const long total = static_cast<long>(N) * C * H * W;
    const int S = H * W;

    for (long linear = blockIdx.x * (long)blockDim.x + threadIdx.x; linear < total; linear += (long)blockDim.x * gridDim.x) {
        // layout: n*(C*H*W) + c*(H*W) + s
        int tmp = static_cast<int>(linear / S);
        int c = tmp % C;
        float x = input[linear];
        float b = bias[c];
        float v = x + b;
        output[linear] = leaky_relu_scaled_f(v, negative_slope, scale);
    }
}

void run(const at::Tensor& x, const at::Tensor& bias, double negative_slope, double scale, at::Tensor& out)
{
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && bias.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [N, C, H, W]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [C]");
    TORCH_CHECK(x.scalar_type() == at::kFloat && bias.scalar_type() == at::kFloat
                && out.scalar_type() == at::kFloat, "float32 tensors expected");

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    TORCH_CHECK(bias.size(0) >= C, "bias length must be >= C");

    const int S = static_cast<int>(H * W);
    const int64_t slices = N * C;
    const int threads = 256;
    auto stream = c10::hip::getCurrentHIPStream();

    // Prefer 2D launch when grid.y limits permit
    const bool use2D = (slices > 0 && slices <= 65535);

    if (use2D) {
        int blocks_x = (S + threads - 1) / threads;
        blocks_x = blocks_x > 0 ? blocks_x : 1;
        blocks_x = blocks_x > 65535 ? 65535 : blocks_x;
        dim3 grid(blocks_x, static_cast<unsigned int>(slices), 1);
        dim3 block(threads);
        hipLaunchKernelGGL(
            fused_leaky_relu_2d_kernel,
            grid, block, 0, stream,
            x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(),
            static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
            static_cast<float>(negative_slope), static_cast<float>(scale));
    } else {
        // Fallback 1D kernel when N*C exceeds gridDim.y limit
        const long total = N * C * H * W;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        if (blocks <= 0) blocks = 1;
        if (blocks > 65535) blocks = 65535;
        dim3 grid(blocks);
        dim3 block(threads);
        hipLaunchKernelGGL(
            fused_leaky_relu_1d_kernel,
            grid, block, 0, stream,
            x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(),
            static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
            static_cast<float>(negative_slope), static_cast<float>(scale));
    }
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor x, Tensor bias, float negative_slope, float scale, Tensor(a!) out) -> ()", &run);
}
