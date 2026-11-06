// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>
#include <cstdint>

// Tunable knobs (can be adjusted at build time)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef BLOCKS_PER_CU
#define BLOCKS_PER_CU 8
#endif
#ifndef VEC_UNROLL
#define VEC_UNROLL 2
#endif

// Exact activation to match PyTorch tanh for float/double
__device__ __forceinline__ float act_f32(float v, float a, float maxv) {
    return ::tanhf(a * v) * maxv;
}

// Compute alignment plan for float32: head (scalar), vec4 (float4), vec2 (float2), tail (scalar)
static inline void plan_alignment_f32(const float* in_ptr, const float* out_ptr, int64_t numel,
                                      int64_t& head, int64_t& vec4, int64_t& vec2, int64_t& tail) {
    constexpr size_t A16 = 16;
    constexpr size_t A8  = 8;
    uintptr_t in_addr  = reinterpret_cast<uintptr_t>(in_ptr);
    uintptr_t out_addr = reinterpret_cast<uintptr_t>(out_ptr);

    head = 0; vec4 = 0; vec2 = 0; tail = 0;

    auto same_mod = [](uintptr_t a, uintptr_t b, size_t M) {
        return (a % M) == (b % M);
    };

    if (same_mod(in_addr, out_addr, A16)) {
        size_t mis_bytes = (A16 - (in_addr % A16)) & (A16 - 1);
        int64_t head_elems = static_cast<int64_t>(mis_bytes / sizeof(float));
        if (head_elems > numel) head_elems = numel;
        head = head_elems;
        int64_t remain = numel - head;
        vec4 = remain / 4;
        int64_t rem_after4 = remain - (vec4 * 4);
        // float2 fallback for remaining even elements (same_mod16 implies same_mod8)
        vec2 = rem_after4 / 2;
        tail = rem_after4 - (vec2 * 2);
    } else if (same_mod(in_addr, out_addr, A8)) {
        size_t mis_bytes = (A8 - (in_addr % A8)) & (A8 - 1);
        int64_t head_elems = static_cast<int64_t>(mis_bytes / sizeof(float));
        if (head_elems > numel) head_elems = numel;
        head = head_elems;
        int64_t remain = numel - head;
        vec4 = 0;
        vec2 = remain / 2;
        tail = remain - (vec2 * 2);
    } else {
        // No common alignment: all scalar
        head = numel;
        vec4 = 0;
        vec2 = 0;
        tail = 0;
    }
}

// Fused kernel for float32: head scalar, float4 body, float2 fallback body, tail scalar
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void tanh_scale_fused_float32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t head_elems,
    int64_t vec4_count,
    int64_t vec2_count,
    int64_t tail_elems,
    float a,
    float max_val)
{
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // Head scalar region
    for (int64_t i = tid; i < head_elems; i += stride) {
        float v = input[i];
        output[i] = act_f32(v, a, max_val);
    }

    // Body float4 vectorized region
    const int64_t body4_start = head_elems;
    const float4* __restrict__ in4 = reinterpret_cast<const float4*>(input + body4_start);
    float4* __restrict__ out4      = reinterpret_cast<float4*>(output + body4_start);

    for (int64_t i4 = tid; i4 < vec4_count; i4 += stride * VEC_UNROLL) {
        #pragma unroll
        for (int u = 0; u < VEC_UNROLL; ++u) {
            int64_t idx4 = i4 + static_cast<int64_t>(u) * stride;
            if (idx4 >= vec4_count) break;
            float4 v = in4[idx4];
            v.x = act_f32(v.x, a, max_val);
            v.y = act_f32(v.y, a, max_val);
            v.z = act_f32(v.z, a, max_val);
            v.w = act_f32(v.w, a, max_val);
            out4[idx4] = v;
        }
    }

    // Body float2 vectorized region (fallback when 16B not fully usable)
    const int64_t body2_start = body4_start + (vec4_count << 2);
    const float2* __restrict__ in2 = reinterpret_cast<const float2*>(input + body2_start);
    float2* __restrict__ out2      = reinterpret_cast<float2*>(output + body2_start);

    for (int64_t i2 = tid; i2 < vec2_count; i2 += stride * VEC_UNROLL) {
        #pragma unroll
        for (int u = 0; u < VEC_UNROLL; ++u) {
            int64_t idx2 = i2 + static_cast<int64_t>(u) * stride;
            if (idx2 >= vec2_count) break;
            float2 v = in2[idx2];
            v.x = act_f32(v.x, a, max_val);
            v.y = act_f32(v.y, a, max_val);
            out2[idx2] = v;
        }
    }

    // Tail scalar region
    const int64_t tail_start = body2_start + (vec2_count << 1);
    for (int64_t t = tid; t < tail_elems; t += stride) {
        int64_t idx = tail_start + t;
        float v = input[idx];
        output[idx] = act_f32(v, a, max_val);
    }
}

static inline int choose_blocks_from_work(int64_t work_units, int blockSize) {
    if (work_units <= 0) return 1;
    // Try to cap by device CUs to avoid oversubscription
    int device = -1;
    hipGetDevice(&device);
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device);
    int cu = props.multiProcessorCount > 0 ? props.multiProcessorCount : 80; // reasonable default
    int maxBlocksByCU = cu * BLOCKS_PER_CU;

    int64_t blocks_by_work = (work_units + blockSize - 1) / blockSize;
    int64_t blocks = blocks_by_work;
    if (blocks > maxBlocksByCU) blocks = maxBlocksByCU;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
    return static_cast<int>(blocks);
}

// Entry point: out = tanh(a * x) * scale for contiguous float32 x and out.
void run(const at::Tensor& x, double a, double scale, at::Tensor& out)
{
    TORCH_CHECK(x.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat, "float32 tensors expected");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && x.numel() == out.numel(), "contiguous, equal sizes");

    const int64_t numel = x.numel();
    if (numel == 0) return;

    hipStream_t stream = c10::hip::getCurrentHIPStream();
    const float* in_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    int64_t head = 0, vec4 = 0, vec2 = 0, tail = 0;
    plan_alignment_f32(in_ptr, out_ptr, numel, head, vec4, vec2, tail);

    int blockSize = BLOCK_SIZE;
    // Choose grid based on the largest work unit among segments (counts not elements for vector segments)
    int64_t maxWork = head;
    if (vec4 > maxWork) maxWork = vec4;
    if (vec2 > maxWork) maxWork = vec2;
    if (tail > maxWork) maxWork = tail;
    if (maxWork == 0) maxWork = 1;
    int gridSize = choose_blocks_from_work(maxWork, blockSize);

    hipLaunchKernelGGL(
        tanh_scale_fused_float32,
        dim3(gridSize), dim3(blockSize), 0, stream,
        in_ptr, out_ptr, head, vec4, vec2, tail,
        static_cast<float>(a), static_cast<float>(scale));
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("run(Tensor x, float a, float scale, Tensor(a!) out) -> ()", &run);
}
