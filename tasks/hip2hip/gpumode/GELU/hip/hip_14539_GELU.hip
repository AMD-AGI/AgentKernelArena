// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>
#include <type_traits>
#include <cstdlib>

// ---------- Utilities ----------
static inline int64_t ceil_div_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }

static inline int get_cached_cu_count() {
    static int cached = -1;
    if (cached < 0) {
        int dev = 0;
        (void)hipGetDevice(&dev);
        hipDeviceProp_t prop{};
        (void)hipGetDeviceProperties(&prop, dev);
        cached = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 80;
    }
    return cached;
}

// Read integer env var, return def if unset/invalid
static inline int read_env_int(const char* name, int def) {
    const char* v = std::getenv(name);
    if (!v) return def;
    int val = std::atoi(v);
    return (val > 0) ? val : def;
}

// ---------- NaN-preserving ReLU ----------
template <typename T>
__device__ __forceinline__ T relu_nan_select(T v) {
    T z = static_cast<T>(0);
    T m = v > z ? v : z; // clamp with zero (does not propagate NaN)
    bool isn = !(v == v); // true if NaN
    return isn ? v : m;   // propagate NaN
}

// Vector lane helpers
__device__ __forceinline__ void relu_apply_vec(float4 &v) {
    v.x = relu_nan_select<float>(v.x);
    v.y = relu_nan_select<float>(v.y);
    v.z = relu_nan_select<float>(v.z);
    v.w = relu_nan_select<float>(v.w);
}

__device__ __forceinline__ void relu_apply_vec(double2 &v) {
    v.x = relu_nan_select<double>(v.x);
    v.y = relu_nan_select<double>(v.y);
}

// ---------- Scalar kernel ----------
template <typename scalar_t>
__global__ void relu_inplace_scalar_kernel_nan(scalar_t* __restrict__ data, int64_t N) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < N; i += stride * 2) {
        if (i < N) data[i] = relu_nan_select<scalar_t>(data[i]);
        int64_t j = i + stride;
        if (j < N) data[j] = relu_nan_select<scalar_t>(data[j]);
    }
}

// ---------- Vectorization traits ----------
template <typename T> struct VecTraits { static constexpr int Width = 1; using VecT = T; };
template <> struct VecTraits<float>  { static constexpr int Width = 4; using VecT = float4; };
template <> struct VecTraits<double> { static constexpr int Width = 2; using VecT = double2; };

// ---------- Fused vector kernel (host precomputes regions) ----------
template <typename T, int EPT>
__global__ void relu_inplace_fused_vec_kernel(T* __restrict__ data,
                                              int64_t N,
                                              int64_t prefix_elems,
                                              int64_t vecCount,
                                              int64_t processed) {
    using Traits = VecTraits<T>;
    using VecT = typename Traits::VecT; // float4 or double2

    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // 1) Prologue [0, prefix_elems)
    for (int64_t i = tid; i < prefix_elems; i += stride) {
        data[i] = relu_nan_select<T>(data[i]);
    }

    // 2) Vectorized bulk over vecCount VecT packets starting at data + prefix_elems
    VecT* vptr = reinterpret_cast<VecT*>(data + prefix_elems);
    for (int64_t p = tid * EPT; p < vecCount; p += stride * EPT) {
        #pragma unroll
        for (int k = 0; k < EPT; ++k) {
            int64_t q = p + k;
            if (q < vecCount) {
                VecT vv = vptr[q];
                relu_apply_vec(vv);
                vptr[q] = vv;
            }
        }
    }

    // 3) Epilogue [processed, N)
    for (int64_t i = processed + tid; i < N; i += stride) {
        data[i] = relu_nan_select<T>(data[i]);
    }
}

// ---------- Host helpers ----------
template <typename T>
static inline void compute_vector_regions(T* ptr, int64_t N,
                                          int64_t& prefix_elems,
                                          int64_t& vecCount,
                                          int64_t& processed) {
    using Traits = VecTraits<T>;
    constexpr int VEC = Traits::Width; // 4 for float, 2 for double

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    const size_t alignB = 16; // target vector alignment
    size_t misalign = addr % alignB;

    int64_t elems_to_align = 0;
    if (misalign != 0) {
        size_t bytes_to_align = alignB - misalign;
        elems_to_align = static_cast<int64_t>(bytes_to_align / sizeof(T));
        if (bytes_to_align % sizeof(T)) elems_to_align += 1; // ceil to element
        if (elems_to_align > N) elems_to_align = N;
    }

    int64_t vecStart = elems_to_align;
    int64_t bulk_elems = N > vecStart ? (N - vecStart) : 0;
    int64_t bulk_vec = (bulk_elems / VEC) * VEC;
    int64_t vecEnd = vecStart + bulk_vec;

    prefix_elems = vecStart;
    vecCount = bulk_vec / VEC;       // number of VecT packets
    processed = vecEnd;              // first scalar tail index
}

static inline int choose_threads_per_block(int64_t N) {
    int tpb_env = read_env_int("HIP_RELU_TPB", -1);
    if (tpb_env > 0) return ((tpb_env + 63) / 64) * 64; // round to multiple of 64
    // Heuristic: 256 for most, 512 for very large tensors
    return (N >= (1LL << 24)) ? 512 : 256;
}

static inline int choose_ept(int64_t N) {
    int ept_env = read_env_int("HIP_RELU_EPT", -1);
    if (ept_env > 0) return ept_env;
    if (N < 100000) return 1;       // small: minimal regs/overhead
    if (N < 10000000) return 2;     // medium: balanced ILP
    return 4;                        // large: higher ILP (ensure occupancy acceptable)
}

static inline int compute_blocks_capped(int64_t work_items, int threads) {
    int cap_per_cu = read_env_int("HIP_RELU_BPCU", 24); // blocks-per-CU cap
    int cu = get_cached_cu_count();
    int cap = cu * cap_per_cu;
    int base = static_cast<int>(ceil_div_i64(work_items, threads));
    if (base < 1) base = 1;
    if (cap > 0 && base > cap) base = cap;
    return base;
}

// ---------- Launcher ----------
torch::Tensor forward(torch::Tensor x) {
    // In-place on contiguous storage; if non-contiguous, materialize contiguous copy
    torch::Tensor y = x.is_contiguous() ? x : x.contiguous();
    TORCH_CHECK(y.is_cuda(), "Input must be a HIP tensor under ROCm.");

    int64_t N = y.numel();
    if (N == 0) return y;

    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    // Tiny fast path: scalar kernel with minimal blocks
    if (N <= 4096) {
        int threads = choose_threads_per_block(N);
        int blocks = static_cast<int>(ceil_div_i64(N, threads));
        if (blocks < 1) blocks = 1;
        AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "relu_inplace_scalar_kernel_nan_tiny", ([&] {
            auto* ptr = y.data_ptr<scalar_t>();
            hipLaunchKernelGGL((relu_inplace_scalar_kernel_nan<scalar_t>), dim3(blocks), dim3(threads), 0, stream, ptr, N);
        }));
        return y;
    }

    int threads = choose_threads_per_block(N);
    auto st = y.scalar_type();

    if (st == at::kFloat) {
        float* ptr = y.data_ptr<float>();
        int64_t prefix = 0, vecCount = 0, processed = 0;
        compute_vector_regions<float>(ptr, N, prefix, vecCount, processed);

        int ept = choose_ept(N);
        int64_t work_items = (vecCount > 0) ? vecCount : N; // size-aware for bulk; ensure >= 1
        int blocks = compute_blocks_capped(work_items, threads);

        if (ept <= 1) {
            hipLaunchKernelGGL((relu_inplace_fused_vec_kernel<float,1>), dim3(blocks), dim3(threads), 0, stream,
                               ptr, N, prefix, vecCount, processed);
        } else if (ept == 2) {
            hipLaunchKernelGGL((relu_inplace_fused_vec_kernel<float,2>), dim3(blocks), dim3(threads), 0, stream,
                               ptr, N, prefix, vecCount, processed);
        } else { // ept >= 3 -> use 4
            hipLaunchKernelGGL((relu_inplace_fused_vec_kernel<float,4>), dim3(blocks), dim3(threads), 0, stream,
                               ptr, N, prefix, vecCount, processed);
        }
    } else if (st == at::kDouble) {
        double* ptr = y.data_ptr<double>();
        int64_t prefix = 0, vecCount = 0, processed = 0;
        compute_vector_regions<double>(ptr, N, prefix, vecCount, processed);

        int ept = choose_ept(N);
        int64_t work_items = (vecCount > 0) ? vecCount : N;
        int blocks = compute_blocks_capped(work_items, threads);

        if (ept <= 1) {
            hipLaunchKernelGGL((relu_inplace_fused_vec_kernel<double,1>), dim3(blocks), dim3(threads), 0, stream,
                               ptr, N, prefix, vecCount, processed);
        } else if (ept == 2) {
            hipLaunchKernelGGL((relu_inplace_fused_vec_kernel<double,2>), dim3(blocks), dim3(threads), 0, stream,
                               ptr, N, prefix, vecCount, processed);
        } else { // ept >= 3 -> use 4
            hipLaunchKernelGGL((relu_inplace_fused_vec_kernel<double,4>), dim3(blocks), dim3(threads), 0, stream,
                               ptr, N, prefix, vecCount, processed);
        }
    } else {
        // Fallback scalar kernel for other floating types
        int blocks = compute_blocks_capped(N, threads);
        AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "relu_inplace_scalar_kernel_nan", ([&] {
            auto* ptr = y.data_ptr<scalar_t>();
            hipLaunchKernelGGL((relu_inplace_scalar_kernel_nan<scalar_t>), dim3(blocks), dim3(threads), 0, stream, ptr, N);
        }));
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "In-place ReLU forward (HIP, fused, NaN-preserving, vectorized, tuned)");
}

