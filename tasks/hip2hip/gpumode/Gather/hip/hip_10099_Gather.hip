// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <tuple>
#include <algorithm>

// Utility: product of sizes in [start, end)
static inline int64_t prod_range(const at::IntArrayRef sizes, int64_t start, int64_t end) {
    int64_t v = 1;
    for (int64_t d = start; d < end; ++d) v *= sizes[d];
    return v;
}

static inline int roundup_pow2_64(int x) {
    // next power-of-two up to 64
    if (x <= 1) return 1;
    int p = 1;
    while (p < x && p < 64) p <<= 1;
    return std::min(p, 64);
}

// Small-inner specialized kernel: one wavefront per (outer_idx, idx_pos)
// blockDim.x <= 64, each thread copies at most one element (or strided if inner_size > blockDim.x but <=64)
template <typename scalar_t>
__global__ void gather_small_inner_kernel(
    const scalar_t* __restrict__ input,
    const int64_t* __restrict__ indices,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t inner_size,
    int64_t in_dim,
    int64_t idx_elems)
{
    const int64_t pair_id = static_cast<int64_t>(blockIdx.x);
    if (pair_id >= outer_size * idx_elems) return;

    const int64_t outer_idx = pair_id / idx_elems;
    const int64_t idx_pos   = pair_id % idx_elems;

    const int64_t gather_src = indices[idx_pos];
    // Assuming valid indices as typical for gather

    const int64_t in_row_base  = (outer_idx * in_dim + gather_src) * inner_size;
    const int64_t out_row_base = (outer_idx * idx_elems + idx_pos) * inner_size;

    for (int64_t inner = threadIdx.x; inner < inner_size; inner += blockDim.x) {
        output[out_row_base + inner] = input[in_row_base + inner];
    }
}

// Vectorized 1D element-parallel kernel for sizeof(scalar_t)==4 and V in {2,4}
template <typename scalar_t, int V>
__global__ void gather_vec1d_kernel(
    const scalar_t* __restrict__ input,
    const int64_t* __restrict__ indices,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t inner_size,
    int64_t in_dim,
    int64_t idx_elems)
{
    constexpr int64_t vecW = V;
    const int64_t inner_vec = inner_size / vecW;
    const int64_t total_vec_elems = outer_size * idx_elems * inner_vec;

    for (int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         tid < total_vec_elems;
         tid += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t inner_v = tid % inner_vec;
        const int64_t tmp = tid / inner_vec;
        const int64_t idx_pos = tmp % idx_elems;
        const int64_t outer_idx = tmp / idx_elems;

        const int64_t gather_src = indices[idx_pos];

        const int64_t in_base  = (outer_idx * in_dim + gather_src) * inner_size + inner_v * vecW;
        const int64_t out_base = (outer_idx * idx_elems + idx_pos) * inner_size + inner_v * vecW;

        #pragma unroll
        for (int i = 0; i < V; ++i) {
            output[out_base + i] = input[in_base + i];
        }
    }
}

// Scalar 1D element-parallel kernel: generic types
template <typename scalar_t>
__global__ void gather_scalar1d_kernel(
    const scalar_t* __restrict__ input,
    const int64_t* __restrict__ indices,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t inner_size,
    int64_t in_dim,
    int64_t idx_elems)
{
    const int64_t total = outer_size * idx_elems * inner_size;

    for (int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         tid < total;
         tid += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t inner = tid % inner_size;
        const int64_t tmp = tid / inner_size;
        const int64_t idx_pos = tmp % idx_elems;
        const int64_t outer_idx = tmp / idx_elems;

        const int64_t gather_src = indices[idx_pos];

        const int64_t in_off  = (outer_idx * in_dim + gather_src) * inner_size + inner;
        const int64_t out_off = tid; // already linearized order matches output layout
        output[out_off] = input[in_off];
    }
}

static inline void validate_inputs(const at::Tensor& input, const at::Tensor& indices, int64_t dim) {
    TORCH_CHECK(input.defined(), "input must be defined");
    TORCH_CHECK(indices.defined(), "indices must be defined");
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor on ROCm");
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA/HIP tensor on ROCm");
    TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64 (Long)");
    TORCH_CHECK(input.layout() == c10::kStrided, "input must be strided");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "dim out of range");
}

// Host entry: mirrors Python module_fn/gather_fn semantics
at::Tensor forward(
    at::Tensor input,
    at::Tensor indices,
    int64_t dim)
{
    // Normalize dim to [0, ndim)
    const int64_t ndim = input.dim();
    if (dim < 0) dim += ndim;

    input = input.contiguous();
    indices = indices.contiguous();

    validate_inputs(input, indices, dim);

    const auto sizes = input.sizes();
    const int64_t in_dim = sizes[dim];

    // Optional bounds check (device-to-host scalar reads). Keep lightweight and robust.
    if (indices.numel() > 0) {
        auto t = at::aminmax(indices);
        const int64_t min_idx = std::get<0>(t).item<int64_t>();
        const int64_t max_idx = std::get<1>(t).item<int64_t>();
        TORCH_CHECK(min_idx >= 0 && max_idx < in_dim, "gather index out of range");
    }

    const int64_t outer_size = prod_range(sizes, 0, dim);
    const int64_t inner_size = prod_range(sizes, dim + 1, ndim);
    const int64_t idx_elems  = indices.numel();

    // Output shape: replace size[dim] with indices.numel()
    std::vector<int64_t> out_sizes(sizes.begin(), sizes.end());
    out_sizes[dim] = idx_elems;
    at::Tensor output = at::empty(out_sizes, input.options());

    if (outer_size == 0 || inner_size == 0 || idx_elems == 0) {
        return output;
    }

    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    // Choose launch config
    auto pick_block = [&](int64_t work)->int {
        if (work >= 512) return 512; // multiple of 64
        if (work >= 256) return 256;
        if (work >= 128) return 128;
        return 64;
    };

    // Special small-inner kernel path (<= 64)
    if (inner_size <= 64) {
        const int block_x = roundup_pow2_64(static_cast<int>(inner_size));
        const int64_t pairs = outer_size * idx_elems;
        const int grid_x = static_cast<int>(std::min<int64_t>(pairs, 2147483647));
        AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "gather_small_inner_kernel", [&] {
            hipLaunchKernelGGL(
                (gather_small_inner_kernel<scalar_t>),
                dim3(grid_x, 1, 1), dim3(block_x, 1, 1), 0, stream,
                input.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(),
                outer_size, inner_size, in_dim, idx_elems);
        });
        hipError_t err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP kernel launch error: ", hipGetErrorString(err));
        return output;
    }

    // Try vectorized path only for 4-byte types and when inner_size divisible by 4 or 2
    const bool is4B = (input.element_size() == 4);

    if (is4B && (inner_size % 4 == 0)) {
        const int64_t inner_vec = inner_size / 4;
        const int64_t total_vec = outer_size * idx_elems * inner_vec;
        const int block_x = pick_block(total_vec);
        const int grid_x = static_cast<int>((total_vec + block_x - 1) / block_x);
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kInt, at::kLong, input.scalar_type(), "gather_vec1d_kernel_V4", [&] {
            using scalar4_t = scalar_t; // same scalar_t; vectorization via manual unroll
            hipLaunchKernelGGL(
                (gather_vec1d_kernel<scalar4_t, 4>),
                dim3(grid_x, 1, 1), dim3(block_x, 1, 1), 0, stream,
                input.data_ptr<scalar4_t>(),
                indices.data_ptr<int64_t>(),
                output.data_ptr<scalar4_t>(),
                outer_size, inner_size, in_dim, idx_elems);
        });
        hipError_t err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP kernel launch error: ", hipGetErrorString(err));
        return output;
    }

    if (is4B && (inner_size % 2 == 0)) {
        const int64_t inner_vec = inner_size / 2;
        const int64_t total_vec = outer_size * idx_elems * inner_vec;
        const int block_x = pick_block(total_vec);
        const int grid_x = static_cast<int>((total_vec + block_x - 1) / block_x);
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kInt, at::kLong, input.scalar_type(), "gather_vec1d_kernel_V2", [&] {
            using scalar2_t = scalar_t;
            hipLaunchKernelGGL(
                (gather_vec1d_kernel<scalar2_t, 2>),
                dim3(grid_x, 1, 1), dim3(block_x, 1, 1), 0, stream,
                input.data_ptr<scalar2_t>(),
                indices.data_ptr<int64_t>(),
                output.data_ptr<scalar2_t>(),
                outer_size, inner_size, in_dim, idx_elems);
        });
        hipError_t err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP kernel launch error: ", hipGetErrorString(err));
        return output;
    }

    // Fallback: generic scalar 1D kernel
    {
        const int64_t total = outer_size * idx_elems * inner_size;
        const int block_x = pick_block(total);
        const int grid_x = static_cast<int>((total + block_x - 1) / block_x);
        AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "gather_scalar1d_kernel", [&] {
            hipLaunchKernelGGL(
                (gather_scalar1d_kernel<scalar_t>),
                dim3(grid_x, 1, 1), dim3(block_x, 1, 1), 0, stream,
                input.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(),
                outer_size, inner_size, in_dim, idx_elems);
        });
        hipError_t err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP kernel launch error: ", hipGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Gather along a dimension (HIP, optimized 1D & small-inner)");
}

