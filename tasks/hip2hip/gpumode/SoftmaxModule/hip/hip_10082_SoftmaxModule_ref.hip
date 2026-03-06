// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <limits>
#include <cstdint>

// Device exp helpers
template <typename T>
__device__ __forceinline__ T device_exp(T x);

template <>
__device__ __forceinline__ float device_exp<float>(float x) { return expf(x); }

template <>
__device__ __forceinline__ double device_exp<double>(double x) { return exp(x); }

// Wavefront (warp) shuffle reductions (AMD wavefront size = 64)
// Reduce within a single wavefront
template <typename T, typename Op>
__device__ __forceinline__ T warp_reduce(T v, Op op) {
    // Assume warpSize is 64 on AMD
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        T other = __shfl_down(v, offset);
        v = op(v, other);
    }
    return v;
}

struct MaxOpF {
    __device__ __forceinline__ float operator()(float a, float b) const { return a > b ? a : b; }
};
struct MaxOpD {
    __device__ __forceinline__ double operator()(double a, double b) const { return a > b ? a : b; }
};
struct SumOpF {
    __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
};
struct SumOpD {
    __device__ __forceinline__ double operator()(double a, double b) const { return a + b; }
};

// Block-wide reduction using wavefront shuffles + one scalar per wavefront in shared memory
template <typename T, typename Op, typename MaxIdentity>
__device__ __forceinline__ T block_reduce(T val, Op op, MaxIdentity identity, T* smem_per_warp) {
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp_id = threadIdx.x / warpSize;
    // Intra-warp reduce
    val = warp_reduce<T, Op>(val, op);
    // Write one value per warp
    if (lane == 0) {
        smem_per_warp[warp_id] = val;
    }
    __syncthreads();
    // Final reduce by warp 0 over the number of warps
    T out = identity;
    if (warp_id == 0) {
        out = (lane < (blockDim.x + warpSize - 1) / warpSize) ? smem_per_warp[lane] : identity;
        out = warp_reduce<T, Op>(out, op);
        if (lane == 0) smem_per_warp[0] = out; // broadcast via shared memory
    }
    __syncthreads();
    return smem_per_warp[0];
}

// Kernel: one block processes multiple rows via grid-stride over rows; threads stride across softmax dim
// Pass 1: compute row-wise max
// Pass 2: compute exp(x - max), store to output, reduce sum
// Pass 3: normalize output in-place

template <typename scalar_t>
__global__ void softmax_rowwise_shuffle_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t softmax_size)
{
    extern __shared__ unsigned char smem_raw[];
    scalar_t* smem_warps = reinterpret_cast<scalar_t*>(smem_raw); // size = num_warps

    for (int64_t row = blockIdx.x; row < outer_size; row += gridDim.x) {
        const int64_t base = row * softmax_size;

        // 1) Reduce max
        scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
        for (int64_t i = threadIdx.x; i < softmax_size; i += blockDim.x) {
            scalar_t v = input[base + i];
            local_max = v > local_max ? v : local_max;
        }
        // Select appropriate op for dtype
        scalar_t row_max;
        if constexpr (std::is_same<scalar_t, float>::value) {
            row_max = block_reduce<scalar_t>(local_max, MaxOpF{}, -std::numeric_limits<float>::infinity(), smem_warps);
        } else {
            row_max = block_reduce<scalar_t>(local_max, MaxOpD{}, -std::numeric_limits<double>::infinity(), smem_warps);
        }

        // 2) Compute exp(x - max), write numerators to output, reduce sum
        scalar_t local_sum = static_cast<scalar_t>(0);
        for (int64_t i = threadIdx.x; i < softmax_size; i += blockDim.x) {
            scalar_t e = device_exp<scalar_t>(input[base + i] - row_max);
            output[base + i] = e;
            local_sum += e;
        }
        scalar_t row_sum;
        if constexpr (std::is_same<scalar_t, float>::value) {
            row_sum = block_reduce<scalar_t>(local_sum, SumOpF{}, static_cast<float>(0), smem_warps);
        } else {
            row_sum = block_reduce<scalar_t>(local_sum, SumOpD{}, static_cast<double>(0), smem_warps);
        }

        scalar_t inv_sum = static_cast<scalar_t>(1) / row_sum;
        // 3) Normalize in-place
        for (int64_t i = threadIdx.x; i < softmax_size; i += blockDim.x) {
            output[base + i] = output[base + i] * inv_sum;
        }
        __syncthreads();
    }
}

static inline void hipCheck(const char* msg) {
    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, msg, ": ", hipGetErrorString(err));
}

// Helper to compute launch parameters tuned for AMD wavefronts
static inline int choose_block_size(int64_t softmax_size) {
    // Prefer multiples of 64. Heuristic across sizes.
    if (softmax_size >= 512) return 512;
    if (softmax_size >= 256) return 256;
    if (softmax_size >= 128) return 128;
    if (softmax_size >= 64) return 64;
    // For very small sizes, still use one wavefront
    return 64;
}

// Core entry: matches requested API
torch::Tensor softmax_forward(torch::Tensor input, int64_t axis) {
    TORCH_CHECK(input.is_cuda(), "Input must be on a HIP device (ROCm)");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");
    auto dtype = input.scalar_type();
    TORCH_CHECK(at::isFloatingType(dtype), "Only floating point tensors are supported");

    const int64_t ndim = input.dim();
    TORCH_CHECK(axis >= 0 && axis < ndim, "Invalid axis");

    // Fast path: if axis is last and input is contiguous, avoid permute
    bool fast_path = (axis == ndim - 1) && input.is_contiguous();

    at::Tensor in_contig;
    std::vector<int64_t> perm;
    std::vector<int64_t> inv_perm(ndim);

    if (fast_path) {
        in_contig = input;
    } else {
        perm.reserve(ndim);
        for (int64_t i = 0; i < ndim; ++i) if (i != axis) perm.push_back(i);
        perm.push_back(axis);
        in_contig = input.permute(perm).contiguous();
        for (int64_t i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;
    }

    // Compute outer_size and softmax_size from the layout where axis is last
    int64_t softmax_size = fast_path ? input.size(ndim - 1) : in_contig.size(ndim - 1);
    int64_t outer_size = 1;
    if (fast_path) {
        for (int64_t i = 0; i < ndim - 1; ++i) outer_size *= input.size(i);
    } else {
        for (int64_t i = 0; i < ndim - 1; ++i) outer_size *= in_contig.size(i);
    }

    // View as 2D [outer, softmax]
    auto in2d = (fast_path ? in_contig : in_contig).view({outer_size, softmax_size});
    auto out2d = at::empty_like(in2d);

    int threads = choose_block_size(softmax_size);
    int max_grid = 65535; // HIP grid x-dim practical limit
    int blocks = static_cast<int>(std::min<int64_t>(outer_size, max_grid));
    size_t num_warps = (threads + warpSize - 1) / warpSize; // warpSize from HIP

    AT_DISPATCH_FLOATING_TYPES(in2d.scalar_type(), "softmax_rowwise_shuffle_kernel", [&] {
        size_t shmem_bytes = num_warps * sizeof(scalar_t);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(softmax_rowwise_shuffle_kernel<scalar_t>),
            dim3(blocks), dim3(threads), shmem_bytes, 0,
            in2d.data_ptr<scalar_t>(),
            out2d.data_ptr<scalar_t>(),
            outer_size,
            softmax_size);
    });
    hipCheck("softmax_rowwise_shuffle_kernel launch failed");

    // Restore original shape
    at::Tensor out;
    if (fast_path) {
        out = out2d.view(input.sizes());
    } else {
        // Build permuted shape
        std::vector<int64_t> perm_sizes(ndim);
        for (int64_t i = 0; i < ndim; ++i) perm_sizes[i] = in_contig.size(i);
        auto out_perm = out2d.view(perm_sizes);
        out = out_perm.permute(inv_perm).contiguous();
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_forward, "Softmax forward (HIP, ROCm)");
}

