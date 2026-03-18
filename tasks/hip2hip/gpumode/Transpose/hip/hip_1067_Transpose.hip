// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdint>

// Kernel: map contiguous output indices to input using permuted input strides
// out_strides: contiguous strides of the output tensor (row-major for permuted sizes)
// in_strides_perm: input strides indexed in output-dimension order (i.e., in_strides[perm[d]])
// Both stride arrays are int64_t and length ndim

template <typename scalar_t>
__global__ void transpose_nd_contig_kernel(
    const scalar_t* __restrict__ in,
    scalar_t* __restrict__ out,
    const int64_t* __restrict__ out_strides,
    const int64_t* __restrict__ in_strides_perm,
    const int64_t ndim,
    const int64_t numel) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t linear = tid; linear < numel; linear += step) {
        int64_t tmp = linear;
        int64_t in_offset = 0;
        // Decompose linear index in terms of output contiguous strides
        // and accumulate input offset using permuted input strides
        for (int64_t d = 0; d < ndim; ++d) {
            int64_t os = out_strides[d];
            int64_t q = (os > 0) ? (tmp / os) : 0;
            tmp -= q * os;
            in_offset += q * in_strides_perm[d];
        }
        out[linear] = in[in_offset];
    }
}

// Host launcher that mirrors: input.transpose(dim1, dim2).contiguous()
torch::Tensor transpose_4d_hip(torch::Tensor input, int64_t dim1, int64_t dim2) {
    // CPU fallback for non-HIP tensors to ensure functional correctness
    if (!input.device().is_hip()) {
        return input.transpose(dim1, dim2).contiguous();
    }

    TORCH_CHECK(input.dim() >= 0, "Input must have non-negative number of dimensions");

    const int64_t ndim = input.dim();

    auto norm_dim = [&](int64_t d) -> int64_t {
        if (d < 0) d += ndim;
        TORCH_CHECK(d >= 0 && d < ndim, "Dimension out of range");
        return d;
    };

    int64_t d1 = norm_dim(dim1);
    int64_t d2 = norm_dim(dim2);

    // Build permutation that swaps d1 and d2
    std::vector<int64_t> perm(ndim);
    for (int64_t i = 0; i < ndim; ++i) perm[i] = i;
    if (d1 != d2) std::swap(perm[d1], perm[d2]);

    // Gather input sizes/strides
    std::vector<int64_t> in_sizes = input.sizes().vec();
    std::vector<int64_t> in_strides = input.strides().vec();

    // Output sizes after transpose
    std::vector<int64_t> out_sizes(ndim);
    for (int64_t d = 0; d < ndim; ++d) out_sizes[d] = in_sizes[perm[d]];

    // Output contiguous strides
    std::vector<int64_t> out_strides(ndim);
    {
        int64_t s = 1;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            out_strides[d] = s;
            s *= std::max<int64_t>(out_sizes[d], 1);
        }
    }

    // Allocate output (contiguous)
    auto output = torch::empty(out_sizes, input.options());

    const int64_t numel = output.numel();
    if (numel == 0) {
        return output;
    }

    // Prepare metadata for kernel
    std::vector<int64_t> in_strides_perm(ndim);
    for (int64_t d = 0; d < ndim; ++d) in_strides_perm[d] = in_strides[perm[d]];

    // Copy stride arrays to device
    auto dev_i64 = torch::TensorOptions().device(input.device()).dtype(torch::kInt64);
    auto h_out_strides = torch::from_blob(out_strides.data(), {ndim}, torch::TensorOptions().dtype(torch::kInt64));
    auto h_in_strides_perm = torch::from_blob(in_strides_perm.data(), {ndim}, torch::TensorOptions().dtype(torch::kInt64));
    auto d_out_strides = h_out_strides.to(dev_i64);
    auto d_in_strides_perm = h_in_strides_perm.to(dev_i64);

    // If dtype not supported by our kernel, fallback to PyTorch op on device
    auto dtype = input.scalar_type();
    bool supported =
        dtype == torch::kFloat || dtype == torch::kDouble ||
        dtype == torch::kHalf  || dtype == torch::kBFloat16;
    if (!supported) {
        return input.transpose(d1, d2).contiguous();
    }

    int threads = 256;
    int64_t blocks64 = (numel + threads - 1) / threads;
    int blocks = static_cast<int>(std::min<int64_t>(blocks64, 65535));
    if (blocks <= 0) blocks = 1;

    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, input.scalar_type(), "transpose_nd_contig_hip", [&] {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(transpose_nd_contig_kernel<scalar_t>),
            dim3(blocks), dim3(threads), 0, 0,
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            d_out_strides.data_ptr<int64_t>(),
            d_in_strides_perm.data_ptr<int64_t>(),
            ndim,
            numel);
    });

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &transpose_4d_hip, "ND Transpose between two dims followed by contiguous (HIP)");
}

