// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cmath>

// Device-side sigmoid helper
template <typename T>
__device__ __forceinline__ T d_sigmoid(T x) {
    // Use precise exp to match PyTorch behavior
    return T(1) / (T(1) + exp(-x));
}

// Scalar grid-stride kernel: y[i] = x[i] * sigmoid(x[i])
template <typename scalar_t>
__global__ void silu_kernel_scalar(const scalar_t* __restrict__ x,
                                   scalar_t* __restrict__ y,
                                   long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * (long long)gridDim.x;
    for (long long i = idx; i < n; i += stride) {
        scalar_t v = x[i];
        y[i] = v * d_sigmoid(v);
    }
}

// Vectorized types
template <typename T> struct VecType { using type = T; static constexpr int pack = 1; };
template <> struct VecType<float>  { using type = float4;  static constexpr int pack = 4; };
template <> struct VecType<double> { using type = double2; static constexpr int pack = 2; };

// Helpers to load/store vector elements generically
__device__ __forceinline__ float  get_elem(const float4& v, int i){ return reinterpret_cast<const float*>(&v)[i]; }
__device__ __forceinline__ void   set_elem(float4& v, int i, float x){ reinterpret_cast<float*>(&v)[i] = x; }
__device__ __forceinline__ double get_elem(const double2& v, int i){ return reinterpret_cast<const double*>(&v)[i]; }
__device__ __forceinline__ void   set_elem(double2& v, int i, double x){ reinterpret_cast<double*>(&v)[i] = x; }

// Vectorized grid-stride kernel; processes Pack elements per thread iteration
template <typename scalar_t>
__global__ void silu_kernel_vectorized(const scalar_t* __restrict__ x,
                                       scalar_t* __restrict__ y,
                                       long long n_packed) {
    using V = typename VecType<scalar_t>::type;
    constexpr int Pack = VecType<scalar_t>::pack;

    const V* __restrict__ xv = reinterpret_cast<const V*>(x);
    V* __restrict__ yv = reinterpret_cast<V*>(y);

    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * (long long)gridDim.x;

    for (long long i = idx; i < n_packed; i += stride) {
        V vin = xv[i];
        V vout;
#pragma unroll
        for (int k = 0; k < Pack; ++k) {
            scalar_t val = get_elem(vin, k);
            set_elem(vout, k, val * d_sigmoid(val));
        }
        yv[i] = vout;
    }
}

// Kernel launcher matching required signature
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA (HIP) tensor on ROCm");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto y = torch::empty_like(x);
    const long long numel = x.numel();
    if (numel == 0) return y;

    const int threads = 256;
    // Use a reasonable grid size; cap to avoid oversubscription
    const int maxBlocks = 65535;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "silu_kernel_launch", [&] {
        const scalar_t* x_ptr = x.data_ptr<scalar_t>();
        scalar_t* y_ptr = y.data_ptr<scalar_t>();

        constexpr int Pack = VecType<scalar_t>::pack;
        bool can_vectorize = (Pack > 1);
        if (can_vectorize) {
            // Check alignment for vectorized access
            uintptr_t x_addr = reinterpret_cast<uintptr_t>(x_ptr);
            uintptr_t y_addr = reinterpret_cast<uintptr_t>(y_ptr);
            size_t align = sizeof(typename VecType<scalar_t>::type);
            can_vectorize = (x_addr % align == 0) && (y_addr % align == 0);
        }

        if (can_vectorize && (numel % Pack == 0)) {
            long long n_packed = numel / Pack;
            int blocks = static_cast<int>((n_packed + threads - 1) / threads);
            blocks = blocks > maxBlocks ? maxBlocks : blocks;
            if (blocks < 1) blocks = 1;
            hipLaunchKernelGGL(
                (silu_kernel_vectorized<scalar_t>),
                dim3(blocks), dim3(threads), 0, 0,
                x_ptr, y_ptr, n_packed
            );
        } else {
            int blocks = static_cast<int>((numel + threads - 1) / threads);
            blocks = blocks > maxBlocks ? maxBlocks : blocks;
            if (blocks < 1) blocks = 1;
            hipLaunchKernelGGL(
                (silu_kernel_scalar<scalar_t>),
                dim3(blocks), dim3(threads), 0, 0,
                x_ptr, y_ptr, numel
            );
        }
    });

    hipDeviceSynchronize();
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "SiLU activation forward (HIP, optimized)");
}

