// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>

// out = x.transpose(1, 2).contiguous() for a 4-D fp32 tensor.
// Each thread maps a contiguous output index to the input offset by decomposing it with the
// output's contiguous strides and re-composing it with the input strides permuted into
// output-dimension order. The stride arrays travel by value as kernel arguments, so the launch
// needs no host-to-device copy and can be captured in a graph.

struct Strides4 {
    int64_t v[4];
};

__global__ void transpose_nd_contig_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    const Strides4 out_strides,
    const Strides4 in_strides_perm,
    const int64_t numel) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t linear = tid; linear < numel; linear += step) {
        int64_t tmp = linear;
        int64_t in_offset = 0;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            int64_t os = out_strides.v[d];
            int64_t q = tmp / os;
            tmp -= q * os;
            in_offset += q * in_strides_perm.v[d];
        }
        out[linear] = in[in_offset];
    }
}

void run(const torch::Tensor& x, torch::Tensor& out) {
    TORCH_CHECK(x.dim() == 4 && x.scalar_type() == torch::kFloat, "x must be a 4-D float32 tensor");
    TORCH_CHECK(out.dim() == 4 && out.scalar_type() == torch::kFloat, "out must be a 4-D float32 tensor");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "x and out must be contiguous");

    // Swap dimensions 1 and 2.
    const int perm[4] = {0, 2, 1, 3};
    Strides4 out_strides;
    Strides4 in_strides_perm;
    int64_t s = 1;
    for (int d = 3; d >= 0; --d) {
        out_strides.v[d] = s;
        s *= std::max<int64_t>(x.size(perm[d]), 1);
        in_strides_perm.v[d] = x.stride(perm[d]);
    }
    const int64_t numel = x.numel();
    TORCH_CHECK(out.numel() == numel, "out has the wrong number of elements");
    if (numel == 0) {
        return;
    }

    const int threads = 256;
    const int64_t blocks64 = (numel + threads - 1) / threads;
    const int blocks = static_cast<int>(std::min<int64_t>(blocks64, 65535));
    hipStream_t stream = at::hip::getCurrentHIPStream();
    hipLaunchKernelGGL(transpose_nd_contig_kernel, dim3(blocks), dim3(threads), 0, stream,
                       x.data_ptr<float>(), out.data_ptr<float>(), out_strides, in_strides_perm, numel);
    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("run(Tensor x, Tensor(a!) out) -> ()", &run);
}
