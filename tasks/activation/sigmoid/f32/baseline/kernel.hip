// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// Scaled sigmoid, out = sigmoid(a * x) * max_value, float32 only (HIP).
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

// Sigmoid matching PyTorch default formula (no fast-math)
__device__ __forceinline__ float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

constexpr int kThreads = 512;     // threads per block
constexpr int kBlocksPerCu = 24;  // grid cap, in blocks per compute unit
constexpr int kPack = 4;          // float4
constexpr int kUnroll = 4;

// Fused kernel: scalar head -> vectorized bulk -> scalar tail
template <int Pack, int UNROLL, int TB>
__global__ __launch_bounds__(TB, 2)
void sigmoid_mul_fused_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int64_t n,
    int64_t head,
    int64_t n_packs,
    int64_t tail,
    float a,
    float maxv)
{
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t gstride = (int64_t)blockDim.x * gridDim.x;

    // 1) Scalar head: unroll by 4, consecutive
    for (int64_t base = tid * 4; base < head; base += gstride * 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int64_t i = base + k;
            if (i < head) { out[i] = sigmoid_f32(a * in[i]) * maxv; }
        }
    }

    // 2) Vectorized bulk over packs
    if (n_packs > 0) {
        const float4* __restrict__ vin = reinterpret_cast<const float4*>(in + head);
        float4* __restrict__ vout = reinterpret_cast<float4*>(out + head);

        for (int64_t p = tid * UNROLL; p < n_packs; p += gstride * UNROLL) {
            #pragma unroll
            for (int k = 0; k < UNROLL; ++k) {
                int64_t q = p + k;
                if (q >= n_packs) break;
                float4 vv = vin[q];
                float4 rr;
                rr.x = sigmoid_f32(a * vv.x) * maxv;
                rr.y = sigmoid_f32(a * vv.y) * maxv;
                rr.z = sigmoid_f32(a * vv.z) * maxv;
                rr.w = sigmoid_f32(a * vv.w) * maxv;
                vout[q] = rr;
            }
        }
    }

    // 3) Scalar tail: unroll by 4, consecutive
    if (tail > 0) {
        const int64_t tail_off = head + n_packs * Pack;
        for (int64_t base = tid * 4; base < tail; base += gstride * 4) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int64_t t = base + k;
                if (t < tail) { int64_t idx = tail_off + t; out[idx] = sigmoid_f32(a * in[idx]) * maxv; }
            }
        }
    }
}

// Scalar fallback kernel with unroll-by-4
template <int TB>
__global__ __launch_bounds__(TB, 2)
void sigmoid_mul_scalar_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int64_t n,
    float a,
    float maxv)
{
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t gstride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t base = tid * 4; base < n; base += gstride * 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int64_t i = base + k;
            if (i < n) { out[i] = sigmoid_f32(a * in[i]) * maxv; }
        }
    }
}

static inline int compute_blocks(int64_t work_items, int threads, int unroll) {
    if (work_items <= 0) return 1;
    int dev = 0;
    hipGetDevice(&dev);
    int cu = 1;
    hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, dev);
    if (cu < 1) cu = 1;
    int64_t need = (work_items + (int64_t)threads * unroll - 1) / ((int64_t)threads * unroll);
    int64_t cap = (int64_t)cu * kBlocksPerCu;
    if (need > cap) need = cap;
    return (int)(need > 0 ? need : 1);
}

// Compute a head in [0, Pack-1] that makes both in and out 16-byte aligned.
static inline bool compute_dual_alignment_head(const float* in, const float* out, int64_t n, int64_t& head_out) {
    constexpr int align = 16;
    if (n <= 0) { head_out = 0; return false; }
    uintptr_t ia = reinterpret_cast<uintptr_t>(in);
    uintptr_t oa = reinterpret_cast<uintptr_t>(out);
    for (int h = 0; h < kPack && h < n; ++h) {
        uintptr_t ip = ia + (uintptr_t)h * sizeof(float);
        uintptr_t op = oa + (uintptr_t)h * sizeof(float);
        if ((ip % align == 0u) && (op % align == 0u)) { head_out = h; return true; }
    }
    head_out = 0;
    return false;
}

void run(const at::Tensor& x, double a, double max_value, at::Tensor& out) {
    TORCH_CHECK(x.is_cuda() && out.is_cuda(), "tensors must be on the HIP device");
    TORCH_CHECK(x.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat, "float32 only");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
    const int64_t n = x.numel();
    if (n == 0) return;

    hipStream_t stream = c10::hip::getCurrentHIPStream();
    const float* in_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const float af = static_cast<float>(a);
    const float mf = static_cast<float>(max_value);

    int64_t head = 0;
    bool can_vec = compute_dual_alignment_head(in_ptr, out_ptr, n, head);
    int64_t n_packs = 0, tail = 0;
    if (can_vec) {
        int64_t eligible = n - head;
        n_packs = eligible / kPack;
        tail = eligible - n_packs * kPack;
    }

    if (can_vec && n_packs > 0) {
        int blocks = compute_blocks(n_packs, kThreads, kUnroll);
        hipLaunchKernelGGL((sigmoid_mul_fused_kernel<kPack, kUnroll, kThreads>),
            dim3(blocks), dim3(kThreads), 0, stream,
            in_ptr, out_ptr, n, head, n_packs, tail, af, mf);
    } else {
        int blocks = compute_blocks(n, kThreads, 4);
        hipLaunchKernelGGL((sigmoid_mul_scalar_kernel<kThreads>),
            dim3(blocks), dim3(kThreads), 0, stream,
            in_ptr, out_ptr, n, af, mf);
    }
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor x, float a, float max_value, Tensor(a!) out) -> ()", &run);
}
