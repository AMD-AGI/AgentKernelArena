// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// Exact (erf) GELU on a flat float32 tensor: out = 0.5 * x * (1 + erf(x / sqrt(2))).
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>
#include <cstdint>

static inline int64_t ceil_div_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }

__device__ __forceinline__ float gelu_act(float v) { return 0.5f * v * (1.0f + erff(v * 0.7071067811865475244f)); }

__device__ __forceinline__ void gelu_apply_vec(float4 &v) {
    v.x = gelu_act(v.x);
    v.y = gelu_act(v.y);
    v.z = gelu_act(v.z);
    v.w = gelu_act(v.w);
}

// ---------- Scalar kernel (tiny inputs) ----------
__global__ void gelu_scalar_kernel(const float* __restrict__ in, float* __restrict__ out, int64_t N) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < N; i += stride * 2) {
        if (i < N) out[i] = gelu_act(in[i]);
        int64_t j = i + stride;
        if (j < N) out[j] = gelu_act(in[j]);
    }
}

// ---------- Fused vector kernel (host precomputes regions) ----------
template <int EPT>
__global__ void gelu_fused_vec_kernel(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int64_t N,
                                      int64_t prefix_elems,
                                      int64_t vecCount,
                                      int64_t processed) {
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // 1) Prologue [0, prefix_elems)
    for (int64_t i = tid; i < prefix_elems; i += stride) {
        out[i] = gelu_act(in[i]);
    }

    // 2) Vectorized bulk over vecCount float4 packets starting at prefix_elems
    const float4* vin = reinterpret_cast<const float4*>(in + prefix_elems);
    float4* vout = reinterpret_cast<float4*>(out + prefix_elems);
    for (int64_t p = tid * EPT; p < vecCount; p += stride * EPT) {
        #pragma unroll
        for (int k = 0; k < EPT; ++k) {
            int64_t q = p + k;
            if (q < vecCount) {
                float4 vv = vin[q];
                gelu_apply_vec(vv);
                vout[q] = vv;
            }
        }
    }

    // 3) Epilogue [processed, N)
    for (int64_t i = processed + tid; i < N; i += stride) {
        out[i] = gelu_act(in[i]);
    }
}

// ---------- Host helpers ----------
// Elements before the 16-byte aligned bulk, the bulk size in float4 packets and the first tail index.
// Input and output must share their alignment for the vector path; otherwise everything is scalar.
static inline void compute_vector_regions(const float* in, const float* out, int64_t N,
                                          int64_t& prefix_elems, int64_t& vecCount, int64_t& processed) {
    const uintptr_t in_addr = reinterpret_cast<uintptr_t>(in);
    const uintptr_t out_addr = reinterpret_cast<uintptr_t>(out);
    if (((in_addr | out_addr) & 3) != 0 || ((in_addr ^ out_addr) & 15) != 0) {
        prefix_elems = N;
        vecCount = 0;
        processed = N;
        return;
    }
    int64_t elems_to_align = 0;
    const size_t misalign = out_addr % 16;
    if (misalign != 0) {
        elems_to_align = static_cast<int64_t>((16 - misalign) / sizeof(float));
        if (elems_to_align > N) elems_to_align = N;
    }
    const int64_t bulk_elems = N > elems_to_align ? (N - elems_to_align) : 0;
    const int64_t bulk_vec = (bulk_elems / 4) * 4;
    prefix_elems = elems_to_align;
    vecCount = bulk_vec / 4;
    processed = elems_to_align + bulk_vec;
}

static inline int choose_threads_per_block(int64_t N) {
    // Heuristic: 256 for most, 512 for very large tensors
    return (N >= (1LL << 24)) ? 512 : 256;
}

static inline int choose_ept(int64_t N) {
    if (N < 100000) return 1;       // small: minimal regs/overhead
    if (N < 10000000) return 2;     // medium: balanced ILP
    return 4;                       // large: higher ILP
}

static inline int compute_blocks_capped(int64_t work_items, int threads) {
    const int cap_per_cu = 24;      // blocks-per-CU cap
    int dev = 0;
    (void)hipGetDevice(&dev);
    int cu = 0;
    (void)hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, dev);
    if (cu <= 0) cu = 80;
    const int cap = cu * cap_per_cu;
    int base = static_cast<int>(ceil_div_i64(work_items, threads));
    if (base < 1) base = 1;
    if (base > cap) base = cap;
    return base;
}

// ---------- Launcher ----------
void run(const at::Tensor& x, at::Tensor& out) {
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat, "float32 tensors expected");
    TORCH_CHECK(x.numel() == out.numel(), "x and out must have the same number of elements");

    const int64_t N = x.numel();
    if (N == 0) return;

    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    const float* in_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int threads = choose_threads_per_block(N);

    // Tiny fast path: scalar kernel with minimal blocks
    if (N <= 4096) {
        int blocks = static_cast<int>(ceil_div_i64(N, threads));
        if (blocks < 1) blocks = 1;
        hipLaunchKernelGGL(gelu_scalar_kernel, dim3(blocks), dim3(threads), 0, stream, in_ptr, out_ptr, N);
        return;
    }

    int64_t prefix = 0, vecCount = 0, processed = 0;
    compute_vector_regions(in_ptr, out_ptr, N, prefix, vecCount, processed);

    const int ept = choose_ept(N);
    const int64_t work_items = (vecCount > 0) ? vecCount : N;
    const int blocks = compute_blocks_capped(work_items, threads);

    if (ept <= 1) {
        hipLaunchKernelGGL((gelu_fused_vec_kernel<1>), dim3(blocks), dim3(threads), 0, stream,
                           in_ptr, out_ptr, N, prefix, vecCount, processed);
    } else if (ept == 2) {
        hipLaunchKernelGGL((gelu_fused_vec_kernel<2>), dim3(blocks), dim3(threads), 0, stream,
                           in_ptr, out_ptr, N, prefix, vecCount, processed);
    } else {
        hipLaunchKernelGGL((gelu_fused_vec_kernel<4>), dim3(blocks), dim3(threads), 0, stream,
                           in_ptr, out_ptr, N, prefix, vecCount, processed);
    }
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor x, Tensor(a!) out) -> ()", &run);
}
