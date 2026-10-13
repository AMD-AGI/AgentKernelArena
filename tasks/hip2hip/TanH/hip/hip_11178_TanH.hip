// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <c10/util/Half.h>
#include <type_traits>
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
__device__ __forceinline__ double act_f64(double v, double a, double maxv) {
    return ::tanh(a * v) * maxv;
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

// Compute alignment plan for float64: head (scalar), vec2 (double2), tail (scalar)
static inline void plan_alignment_f64(const double* in_ptr, const double* out_ptr, int64_t numel,
                                      int64_t& head, int64_t& vec2, int64_t& tail) {
    constexpr size_t A16 = 16; // 16B alignment for double2
    uintptr_t in_addr  = reinterpret_cast<uintptr_t>(in_ptr);
    uintptr_t out_addr = reinterpret_cast<uintptr_t>(out_ptr);

    head = 0; vec2 = 0; tail = 0;

    if ((in_addr % A16) == (out_addr % A16)) {
        size_t mis_bytes = (A16 - (in_addr % A16)) & (A16 - 1);
        int64_t head_elems = static_cast<int64_t>(mis_bytes / sizeof(double));
        if (head_elems > numel) head_elems = numel;
        head = head_elems;
        int64_t remain = numel - head;
        vec2 = remain / 2;
        tail = remain - (vec2 * 2);
    } else {
        head = numel;
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

// Fused kernel for float64: head scalar, double2 body, tail scalar
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void tanh_scale_fused_float64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int64_t head_elems,
    int64_t vec2_count,
    int64_t tail_elems,
    double a,
    double max_val)
{
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // Head scalar
    for (int64_t i = tid; i < head_elems; i += stride) {
        double v = input[i];
        output[i] = act_f64(v, a, max_val);
    }

    // Body double2
    const int64_t body_start = head_elems;
    const double2* __restrict__ in2 = reinterpret_cast<const double2*>(input + body_start);
    double2* __restrict__ out2      = reinterpret_cast<double2*>(output + body_start);

    for (int64_t i2 = tid; i2 < vec2_count; i2 += stride * VEC_UNROLL) {
        #pragma unroll
        for (int u = 0; u < VEC_UNROLL; ++u) {
            int64_t idx2 = i2 + static_cast<int64_t>(u) * stride;
            if (idx2 >= vec2_count) break;
            double2 v = in2[idx2];
            v.x = act_f64(v.x, a, max_val);
            v.y = act_f64(v.y, a, max_val);
            out2[idx2] = v;
        }
    }

    // Tail scalar
    const int64_t tail_start = body_start + (vec2_count << 1);
    for (int64_t t = tid; t < tail_elems; t += stride) {
        int64_t idx = tail_start + t;
        double v = input[idx];
        output[idx] = act_f64(v, a, max_val);
    }
}

// Scalar fallback kernel for other floating types (half/bfloat16)
template <typename scalar_t>
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void tanh_scale_scalar_fallback(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t numel,
    float a,
    float max_val)
{
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // Small unroll to improve ILP
    int64_t i = tid;
    for (; i + 3 * stride < numel; i += 4 * stride) {
        float v0 = static_cast<float>(input[i + 0 * stride]);
        float v1 = static_cast<float>(input[i + 1 * stride]);
        float v2 = static_cast<float>(input[i + 2 * stride]);
        float v3 = static_cast<float>(input[i + 3 * stride]);
        output[i + 0 * stride] = static_cast<scalar_t>(::tanhf(a * v0) * max_val);
        output[i + 1 * stride] = static_cast<scalar_t>(::tanhf(a * v1) * max_val);
        output[i + 2 * stride] = static_cast<scalar_t>(::tanhf(a * v2) * max_val);
        output[i + 3 * stride] = static_cast<scalar_t>(::tanhf(a * v3) * max_val);
    }
    for (; i < numel; i += stride) {
        float v = static_cast<float>(input[i]);
        output[i] = static_cast<scalar_t>(::tanhf(a * v) * max_val);
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

// Entry point with the required signature
torch::Tensor forward(
    torch::Tensor input,
    float a,
    float max_val)
{
    TORCH_CHECK(input.is_floating_point(), "Input must be a floating point tensor");
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    const int64_t numel = input.numel();
    auto output = torch::empty_like(input);
    if (numel == 0) return output;

    hipStream_t stream = c10::hip::getCurrentHIPStream();

    if (input.scalar_type() == at::kFloat) {
        const float* in_ptr = input.data_ptr<float>();
        float* out_ptr = output.data_ptr<float>();

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
            in_ptr,
            out_ptr,
            head,
            vec4,
            vec2,
            tail,
            static_cast<float>(a),
            static_cast<float>(max_val)
        );
        return output;
    }

    if (input.scalar_type() == at::kDouble) {
        const double* in_ptr = input.data_ptr<double>();
        double* out_ptr = output.data_ptr<double>();

        int64_t head = 0, vec2 = 0, tail = 0;
        plan_alignment_f64(in_ptr, out_ptr, numel, head, vec2, tail);

        int blockSize = BLOCK_SIZE;
        int64_t maxWork = head;
        if (vec2 > maxWork) maxWork = vec2;
        if (tail > maxWork) maxWork = tail;
        if (maxWork == 0) maxWork = 1;
        int gridSize = choose_blocks_from_work(maxWork, blockSize);

        hipLaunchKernelGGL(
            tanh_scale_fused_float64,
            dim3(gridSize), dim3(blockSize), 0, stream,
            in_ptr,
            out_ptr,
            head,
            vec2,
            tail,
            static_cast<double>(a),
            static_cast<double>(max_val)
        );
        return output;
    }

    // Fallback for other floating types (half, bfloat16)
    int blockSize = BLOCK_SIZE;
    int gridSize = choose_blocks_from_work(numel, blockSize);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "tanh_scale_scalar_fallback", [&] {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(tanh_scale_scalar_fallback<scalar_t>),
            dim3(gridSize), dim3(blockSize), 0, stream,
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<int64_t>(numel),
            a,
            max_val
        );
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Scaled Tanh activation (HIP, fused with float4/float2/double2 vectorization)");
}

