// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <cfloat>

namespace {

constexpr int kWaveSize = 64;
constexpr int kOptimizedChannels = 10;

// The task tensors are NCHW. Assigning a lane to (n, h, w), instead of to a
// channel, makes every channel-plane access contiguous across a Wave64.
// C=10 is known for all benchmark shapes, so retain the scaled logits in VGPRs
// and avoid the two input rereads performed by the reference kernel.
__global__ __launch_bounds__(kWaveSize) void kd_loss_c10_kernel(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ out_sum,
    int spatial_size,
    float inv_temp,
    float loss_scale) {
    const int hw = static_cast<int>(blockIdx.x) * kWaveSize
                 + static_cast<int>(threadIdx.x);
    const int n = static_cast<int>(blockIdx.y);

    float loss = 0.0f;
    if (hw < spatial_size) {
        const int batch_offset = n * kOptimizedChannels * spatial_size + hw;
        const float* input_ptr = input + batch_offset;
        const float* target_ptr = target + batch_offset;

        float logits[kOptimizedChannels];
        float max_logit = -INFINITY;
#pragma unroll
        for (int c = 0; c < kOptimizedChannels; ++c) {
            const float value = input_ptr[c * spatial_size] * inv_temp;
            logits[c] = value;
            max_logit = fmaxf(max_logit, value);
        }

        float exp_sum = 0.0f;
#pragma unroll
        for (int c = 0; c < kOptimizedChannels; ++c) {
            exp_sum += __expf(logits[c] - max_logit);
        }
        const float log_sum_exp =
            max_logit + __logf(fmaxf(exp_sum, FLT_MIN));

#pragma unroll
        for (int c = 0; c < kOptimizedChannels; ++c) {
            const float t = target_ptr[c * spatial_size];
            // Match xlogy(target, target): in particular, 0*log(0) is zero.
            if (t != 0.0f) {
                loss += t * (__logf(t) - logits[c] + log_sum_exp);
            }
        }
    }

    // A workgroup is exactly one Wave64, so no LDS or workgroup barrier is
    // required. Scaling once per wave also removes the finalization kernel.
#pragma unroll
    for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
        loss += __shfl_down(loss, offset, kWaveSize);
    }
    if (threadIdx.x == 0) {
        atomicAdd(out_sum, loss * loss_scale);
    }
}

// Preserve the public function's general channel behavior. This path keeps
// the same coalesced spatial mapping, but rereads logits because C is dynamic.
__global__ __launch_bounds__(kWaveSize) void kd_loss_generic_kernel(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ out_sum,
    int channels,
    int spatial_size,
    float inv_temp,
    float loss_scale) {
    const int hw = static_cast<int>(blockIdx.x) * kWaveSize
                 + static_cast<int>(threadIdx.x);
    const int n = static_cast<int>(blockIdx.y);

    float loss = 0.0f;
    if (hw < spatial_size) {
        const int batch_offset = n * channels * spatial_size + hw;
        const float* input_ptr = input + batch_offset;
        const float* target_ptr = target + batch_offset;

        float max_logit = -INFINITY;
        for (int c = 0; c < channels; ++c) {
            max_logit = fmaxf(
                max_logit, input_ptr[c * spatial_size] * inv_temp);
        }

        float exp_sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            const float value = input_ptr[c * spatial_size] * inv_temp;
            exp_sum += __expf(value - max_logit);
        }
        const float log_sum_exp =
            max_logit + __logf(fmaxf(exp_sum, FLT_MIN));

        for (int c = 0; c < channels; ++c) {
            const float t = target_ptr[c * spatial_size];
            if (t != 0.0f) {
                const float value = input_ptr[c * spatial_size] * inv_temp;
                loss += t * (__logf(t) - value + log_sum_exp);
            }
        }
    }

#pragma unroll
    for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
        loss += __shfl_down(loss, offset, kWaveSize);
    }
    if (threadIdx.x == 0) {
        atomicAdd(out_sum, loss * loss_scale);
    }
}

} // namespace

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor target,
    float temp_factor) {
    TORCH_CHECK(input.is_cuda(), "input must be a HIP/CUDA tensor on device");
    TORCH_CHECK(target.is_cuda(), "target must be a HIP/CUDA tensor on device");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(target.scalar_type() == at::kFloat,
                "Only float32 supported for target");
    TORCH_CHECK(input.sizes() == target.sizes(),
                "input and target must have the same shape");
    TORCH_CHECK(input.dim() == 4,
                "Expected input/target to be 4D [N,C,H,W]");

    input = input.contiguous();
    target = target.contiguous();

    const auto sizes = input.sizes();
    const int N = static_cast<int>(sizes[0]);
    const int C = static_cast<int>(sizes[1]);
    const int H = static_cast<int>(sizes[2]);
    const int W = static_cast<int>(sizes[3]);
    const int spatial_size = H * W;

    auto out_sum = torch::zeros({}, input.options());
    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    const dim3 block(kWaveSize);
    const dim3 grid((spatial_size + kWaveSize - 1) / kWaveSize, N);
    const float inv_temp = 1.0f / temp_factor;
    float loss_scale = temp_factor;
    loss_scale *= temp_factor;
    loss_scale /= static_cast<float>(N);

    if (C == kOptimizedChannels) {
        hipLaunchKernelGGL(
            kd_loss_c10_kernel,
            grid, block, 0, stream,
            input.data_ptr<float>(),
            target.data_ptr<float>(),
            out_sum.data_ptr<float>(),
            spatial_size,
            inv_temp,
            loss_scale);
    } else {
        hipLaunchKernelGGL(
            kd_loss_generic_kernel,
            grid, block, 0, stream,
            input.data_ptr<float>(),
            target.data_ptr<float>(),
            out_sum.data_ptr<float>(),
            C,
            spatial_size,
            inv_temp,
            loss_scale);
    }

    const hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess,
                "HIP kernel launch failed: ", hipGetErrorString(err));
    return out_sum;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "KD Loss forward (HIP, fused Wave64)");
}
