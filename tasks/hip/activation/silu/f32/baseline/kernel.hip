// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SiLU, out = x * sigmoid(x), float32 only (HIP).
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>

// Use precise exp to match PyTorch behavior
__device__ __forceinline__ float d_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Scalar grid-stride kernel: y[i] = x[i] * sigmoid(x[i])
__global__ void silu_kernel_scalar(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * (long long)gridDim.x;
    for (long long i = idx; i < n; i += stride) {
        float v = x[i];
        y[i] = v * d_sigmoid(v);
    }
}

// Vectorized grid-stride kernel; processes 4 elements per thread iteration
__global__ void silu_kernel_vectorized(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       long long n_packed) {
    const float4* __restrict__ xv = reinterpret_cast<const float4*>(x);
    float4* __restrict__ yv = reinterpret_cast<float4*>(y);

    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * (long long)gridDim.x;

    for (long long i = idx; i < n_packed; i += stride) {
        float4 vin = xv[i];
        float4 vout;
        vout.x = vin.x * d_sigmoid(vin.x);
        vout.y = vin.y * d_sigmoid(vin.y);
        vout.z = vin.z * d_sigmoid(vin.z);
        vout.w = vin.w * d_sigmoid(vin.w);
        yv[i] = vout;
    }
}

void run(const at::Tensor& x, at::Tensor& out) {
    TORCH_CHECK(x.is_cuda() && out.is_cuda(), "tensors must be on the HIP device");
    TORCH_CHECK(x.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat, "float32 only");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");

    const long long numel = x.numel();
    if (numel == 0) return;

    const int threads = 256;
    // Cap the grid; the kernels are grid-stride loops.
    const int maxBlocks = 65535;
    hipStream_t stream = c10::hip::getCurrentHIPStream();

    const float* x_ptr = x.data_ptr<float>();
    float* y_ptr = out.data_ptr<float>();

    uintptr_t x_addr = reinterpret_cast<uintptr_t>(x_ptr);
    uintptr_t y_addr = reinterpret_cast<uintptr_t>(y_ptr);
    const bool can_vectorize = (x_addr % sizeof(float4) == 0) && (y_addr % sizeof(float4) == 0);

    if (can_vectorize && (numel % 4 == 0)) {
        long long n_packed = numel / 4;
        int blocks = static_cast<int>((n_packed + threads - 1) / threads);
        blocks = blocks > maxBlocks ? maxBlocks : blocks;
        if (blocks < 1) blocks = 1;
        hipLaunchKernelGGL(silu_kernel_vectorized, dim3(blocks), dim3(threads), 0, stream,
                           x_ptr, y_ptr, n_packed);
    } else {
        int blocks = static_cast<int>((numel + threads - 1) / threads);
        blocks = blocks > maxBlocks ? maxBlocks : blocks;
        if (blocks < 1) blocks = 1;
        hipLaunchKernelGGL(silu_kernel_scalar, dim3(blocks), dim3(threads), 0, stream,
                           x_ptr, y_ptr, numel);
    }
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor x, Tensor(a!) out) -> ()", &run);
}
