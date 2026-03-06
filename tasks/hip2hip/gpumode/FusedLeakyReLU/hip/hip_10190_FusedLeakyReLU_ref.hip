// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Branchless leaky ReLU helpers
__device__ __forceinline__ float leaky_relu_scaled_f(float x, float neg_slope, float scale) {
    float pos = fmaxf(x, 0.0f);
    float neg = fminf(x, 0.0f);
    return pos * scale + neg * (neg_slope * scale);
}

// Kernel: 2D grid over (spatial, N*C) for float/double
template <typename scalar_t>
__global__ void fused_leaky_relu_2d_kernel(
    const scalar_t* __restrict__ input,   // [N, C, H, W]
    const scalar_t* __restrict__ bias,    // [C]
    scalar_t* __restrict__ output,        // [N, C, H, W]
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
    const scalar_t bc = bias[c];

    // grid-stride over spatial
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (; idx < S; idx += stride) {
        // linear spatial index
        const int off = slice_offset + idx;
        float x = static_cast<float>(input[off]);
        float b = static_cast<float>(bc);
        float v = x + b;
        float y = leaky_relu_scaled_f(v, negative_slope, scale);
        output[off] = static_cast<scalar_t>(y);
    }
}

// Half specialization kernel using __half loads/stores with float math
__global__ void fused_leaky_relu_2d_kernel_half(
    const __half* __restrict__ input,   // [N, C, H, W]
    const __half* __restrict__ bias,    // [C]
    __half* __restrict__ output,        // [N, C, H, W]
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

    const float b = __half2float(bias[c]);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (; idx < S; idx += stride) {
        const int off = slice_offset + idx;
        float x = __half2float(input[off]);
        float v = x + b;
        float y = leaky_relu_scaled_f(v, negative_slope, scale);
        output[off] = __float2half(y);
    }
}

// Fallback 1D kernel (used only if grid.y limit exceeded)
template <typename scalar_t>
__global__ void fused_leaky_relu_1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
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
        float x = static_cast<float>(input[linear]);
        float b = static_cast<float>(bias[c]);
        float v = x + b;
        float y = leaky_relu_scaled_f(v, negative_slope, scale);
        output[linear] = static_cast<scalar_t>(y);
    }
}

__global__ void fused_leaky_relu_1d_kernel_half(
    const __half* __restrict__ input,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int N, int C, int H, int W,
    float negative_slope,
    float scale)
{
    const long total = static_cast<long>(N) * C * H * W;
    const int S = H * W;

    for (long linear = blockIdx.x * (long)blockDim.x + threadIdx.x; linear < total; linear += (long)blockDim.x * gridDim.x) {
        int tmp = static_cast<int>(linear / S);
        int c = tmp % C;
        float x = __half2float(input[linear]);
        float b = __half2float(bias[c]);
        float v = x + b;
        float y = leaky_relu_scaled_f(v, negative_slope, scale);
        output[linear] = __float2half(y);
    }
}

// Launcher: returns only the output tensor to match PyTorch fused function
at::Tensor fused_leaky_relu_forward(
    at::Tensor input,      // [N, C, H, W]
    at::Tensor bias,       // [C]
    double negative_slope,
    double scale)
{
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D [N, C, H, W]");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D [C]");

    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);

    TORCH_CHECK(bias.size(0) >= C, "Bias length must be >= C to match PyTorch slicing [:C]");

    auto output = at::empty_like(input);

    const int S = static_cast<int>(H * W);
    const int64_t slices = N * C;

    const int threads = 256;

    // Prefer 2D launch when grid.y limits permit
    const bool use2D = (slices > 0 && slices <= 65535);

    if (use2D) {
        int blocks_x = (S + threads - 1) / threads;
        blocks_x = blocks_x > 0 ? blocks_x : 1;
        blocks_x = blocks_x > 65535 ? 65535 : blocks_x;
        dim3 grid(blocks_x, static_cast<unsigned int>(slices), 1);
        dim3 block(threads);

        auto stream = c10::hip::getCurrentHIPStream();

        if (input.scalar_type() == at::kHalf) {
            const __half* in_ptr = reinterpret_cast<const __half*>(input.data_ptr<at::Half>());
            const __half* b_ptr  = reinterpret_cast<const __half*>(bias.data_ptr<at::Half>());
            __half* out_ptr      = reinterpret_cast<__half*>(output.data_ptr<at::Half>());
            hipLaunchKernelGGL(
                fused_leaky_relu_2d_kernel_half,
                grid, block, 0, stream,
                in_ptr, b_ptr, out_ptr,
                static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
                static_cast<float>(negative_slope), static_cast<float>(scale)
            );
        } else {
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_leaky_relu_2d_kernel", [&] {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(fused_leaky_relu_2d_kernel<scalar_t>),
                    grid, block, 0, stream,
                    input.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
                    static_cast<float>(negative_slope), static_cast<float>(scale)
                );
            });
        }
    } else {
        // Fallback 1D kernel when N*C exceeds gridDim.y limit
        const long total = N * C * H * W;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        if (blocks <= 0) blocks = 1;
        if (blocks > 65535) blocks = 65535;
        dim3 grid(blocks);
        dim3 block(threads);
        auto stream = c10::hip::getCurrentHIPStream();

        if (input.scalar_type() == at::kHalf) {
            const __half* in_ptr = reinterpret_cast<const __half*>(input.data_ptr<at::Half>());
            const __half* b_ptr  = reinterpret_cast<const __half*>(bias.data_ptr<at::Half>());
            __half* out_ptr      = reinterpret_cast<__half*>(output.data_ptr<at::Half>());
            hipLaunchKernelGGL(
                fused_leaky_relu_1d_kernel_half,
                grid, block, 0, stream,
                in_ptr, b_ptr, out_ptr,
                static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
                static_cast<float>(negative_slope), static_cast<float>(scale)
            );
        } else {
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_leaky_relu_1d_kernel", [&] {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(fused_leaky_relu_1d_kernel<scalar_t>),
                    grid, block, 0, stream,
                    input.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
                    static_cast<float>(negative_slope), static_cast<float>(scale)
                );
            });
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_leaky_relu_forward, "FusedLeakyReLU forward (HIP, ROCm)");
}

