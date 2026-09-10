// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/relu.h>
#include <hip/hip_runtime.h>
#if defined(USE_ROCM)
#include <ATen/hip/HIPContext.h>
#endif

#include <cstdint>

namespace {

constexpr int kInput = 4;
constexpr int kN1 = 4096;
constexpr int kN2 = 2048;
constexpr int kN3 = 512;
constexpr int kN4 = 128;
constexpr int kN5 = 64;
constexpr int kN6 = 32;
constexpr int kOutput = 4;

// The first layer has K=4, which is too small to amortize a library GEMM.
// Eight workgroups per batch row expose enough parallelism even for B=4.
__global__ __launch_bounds__(256)
void first_layer_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output)
{
    __shared__ float x_shared[kInput];
    const int tid = static_cast<int>(threadIdx.x);
    const int batch = static_cast<int>(blockIdx.y);

    if (tid < kInput) {
        x_shared[tid] = input[static_cast<size_t>(batch) * kInput + tid];
    }
    __syncthreads();

    // Two coalesced groups of 256 output neurons per workgroup. Each float4
    // load is one complete, naturally aligned weight row.
    int n = static_cast<int>(blockIdx.x) * 512 + tid;
#pragma unroll
    for (int item = 0; item < 2; ++item, n += 256) {
        const float4 w = *reinterpret_cast<const float4*>(
            weight + static_cast<size_t>(n) * kInput);
        float sum = 0.0f;
        sum = fmaf(x_shared[0], w.x, sum);
        sum = fmaf(x_shared[1], w.y, sum);
        sum = fmaf(x_shared[2], w.z, sum);
        sum = fmaf(x_shared[3], w.w, sum);
        sum += bias[n];
        output[static_cast<size_t>(batch) * kN1 + n] =
            sum > 0.0f ? sum : 0.0f;
    }
}

// Layer two's GEMM produces the dot products. Fold its bias and activation
// into one bandwidth-efficient pass before the next GEMM consumes the tensor.
__global__ __launch_bounds__(256)
void bias_relu_2048_kernel(
    float* __restrict__ values,
    const float* __restrict__ bias,
    int elements)
{
    const int index = static_cast<int>(blockIdx.x) * 256
                    + static_cast<int>(threadIdx.x);
    if (index < elements) {
        const float value = values[index] + bias[index & (kN2 - 1)];
        values[index] = value > 0.0f ? value : 0.0f;
    }
}

// Finish the layer-three epilogue and all four small tail layers in one
// workgroup per batch row. Keeping activations in LDS eliminates six global
// intermediate tensor transfers and three launch boundaries.
__global__ __launch_bounds__(128)
void fused_tail_kernel(
    const float* __restrict__ layer3_dot,
    const float* __restrict__ b3,
    const float* __restrict__ w4,
    const float* __restrict__ b4,
    const float* __restrict__ w5,
    const float* __restrict__ b5,
    const float* __restrict__ w6,
    const float* __restrict__ b6,
    const float* __restrict__ w7,
    const float* __restrict__ b7,
    float* __restrict__ output)
{
    __shared__ float wide[kN3];
    __shared__ float narrow[kN4];

    const int tid = static_cast<int>(threadIdx.x);
    const int batch = static_cast<int>(blockIdx.x);
    const float* layer3_row = layer3_dot + static_cast<size_t>(batch) * kN3;

    // Layer 3 bias + ReLU.
#pragma unroll
    for (int i = tid; i < kN3; i += 128) {
        const float value = layer3_row[i] + b3[i];
        wide[i] = value > 0.0f ? value : 0.0f;
    }
    __syncthreads();

    // Layer 4: 512 -> 128, bias + ReLU.
    if (tid < kN4) {
        const float* weight_row = w4 + static_cast<size_t>(tid) * kN3;
        float sum = 0.0f;
#pragma unroll 8
        for (int k = 0; k < kN3; ++k) {
            sum = fmaf(wide[k], weight_row[k], sum);
        }
        sum += b4[tid];
        narrow[tid] = sum > 0.0f ? sum : 0.0f;
    }
    __syncthreads();

    // Layer 5: 128 -> 64, bias + ReLU. Reuse the wide LDS buffer.
    if (tid < kN5) {
        const float* weight_row = w5 + static_cast<size_t>(tid) * kN4;
        float sum = 0.0f;
#pragma unroll 8
        for (int k = 0; k < kN4; ++k) {
            sum = fmaf(narrow[k], weight_row[k], sum);
        }
        sum += b5[tid];
        wide[tid] = sum > 0.0f ? sum : 0.0f;
    }
    __syncthreads();

    // Layer 6: 64 -> 32, bias + ReLU. Reuse the narrow LDS buffer.
    if (tid < kN6) {
        const float* weight_row = w6 + static_cast<size_t>(tid) * kN5;
        float sum = 0.0f;
#pragma unroll 8
        for (int k = 0; k < kN5; ++k) {
            sum = fmaf(wide[k], weight_row[k], sum);
        }
        sum += b6[tid];
        narrow[tid] = sum > 0.0f ? sum : 0.0f;
    }
    __syncthreads();

    // Output layer: 32 -> 4, bias only.
    if (tid < kOutput) {
        const float* weight_row = w7 + static_cast<size_t>(tid) * kN6;
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < kN6; ++k) {
            sum = fmaf(narrow[k], weight_row[k], sum);
        }
        output[static_cast<size_t>(batch) * kOutput + tid] = sum + b7[tid];
    }
}

inline bool is_canonical_shape(
    const at::Tensor& input,
    const at::Tensor& w1, const at::Tensor& b1,
    const at::Tensor& w2, const at::Tensor& b2,
    const at::Tensor& w3, const at::Tensor& b3,
    const at::Tensor& w4, const at::Tensor& b4,
    const at::Tensor& w5, const at::Tensor& b5,
    const at::Tensor& w6, const at::Tensor& b6,
    const at::Tensor& w7, const at::Tensor& b7)
{
    return input.numel() == input.size(0) * kInput
        && w1.dim() == 2 && w1.size(0) == kN1 && w1.size(1) == kInput && b1.numel() == kN1
        && w2.dim() == 2 && w2.size(0) == kN2 && w2.size(1) == kN1 && b2.numel() == kN2
        && w3.dim() == 2 && w3.size(0) == kN3 && w3.size(1) == kN2 && b3.numel() == kN3
        && w4.dim() == 2 && w4.size(0) == kN4 && w4.size(1) == kN3 && b4.numel() == kN4
        && w5.dim() == 2 && w5.size(0) == kN5 && w5.size(1) == kN4 && b5.numel() == kN5
        && w6.dim() == 2 && w6.size(0) == kN6 && w6.size(1) == kN5 && b6.numel() == kN6
        && w7.dim() == 2 && w7.size(0) == kOutput && w7.size(1) == kN6 && b7.numel() == kOutput;
}

// Preserve the source implementation's general shape behavior outside the
// benchmark's fixed topology.
torch::Tensor generic_forward(
    const torch::Tensor& input,
    const torch::Tensor& w1, const torch::Tensor& b1,
    const torch::Tensor& w2, const torch::Tensor& b2,
    const torch::Tensor& w3, const torch::Tensor& b3,
    const torch::Tensor& w4, const torch::Tensor& b4,
    const torch::Tensor& w5, const torch::Tensor& b5,
    const torch::Tensor& w6, const torch::Tensor& b6,
    const torch::Tensor& w7, const torch::Tensor& b7)
{
    auto value = input.view({input.size(0), -1});
    value = at::relu(at::linear(value, w1, b1));
    value = at::relu(at::linear(value, w2, b2));
    value = at::relu(at::linear(value, w3, b3));
    value = at::relu(at::linear(value, w4, b4));
    value = at::relu(at::linear(value, w5, b5));
    value = at::relu(at::linear(value, w6, b6));
    return at::linear(value, w7, b7);
}

} // namespace

// Forward signature and Python binding are the task's public contract.
torch::Tensor mlp_forward(
    torch::Tensor input,
    torch::Tensor w1, torch::Tensor b1,
    torch::Tensor w2, torch::Tensor b2,
    torch::Tensor w3, torch::Tensor b3,
    torch::Tensor w4, torch::Tensor b4,
    torch::Tensor w5, torch::Tensor b5,
    torch::Tensor w6, torch::Tensor b6,
    torch::Tensor w7, torch::Tensor b7)
{
    TORCH_CHECK(input.is_cuda(), "Input must be on HIP device");
    c10::DeviceGuard guard(input.device());

    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(w1.scalar_type() == at::kFloat && b1.scalar_type() == at::kFloat, "dtype mismatch w1/b1");
    TORCH_CHECK(w2.scalar_type() == at::kFloat && b2.scalar_type() == at::kFloat, "dtype mismatch w2/b2");
    TORCH_CHECK(w3.scalar_type() == at::kFloat && b3.scalar_type() == at::kFloat, "dtype mismatch w3/b3");
    TORCH_CHECK(w4.scalar_type() == at::kFloat && b4.scalar_type() == at::kFloat, "dtype mismatch w4/b4");
    TORCH_CHECK(w5.scalar_type() == at::kFloat && b5.scalar_type() == at::kFloat, "dtype mismatch w5/b5");
    TORCH_CHECK(w6.scalar_type() == at::kFloat && b6.scalar_type() == at::kFloat, "dtype mismatch w6/b6");
    TORCH_CHECK(w7.scalar_type() == at::kFloat && b7.scalar_type() == at::kFloat, "dtype mismatch w7/b7");

    auto x = input.contiguous();
    w1 = w1.contiguous(); b1 = b1.contiguous();
    w2 = w2.contiguous(); b2 = b2.contiguous();
    w3 = w3.contiguous(); b3 = b3.contiguous();
    w4 = w4.contiguous(); b4 = b4.contiguous();
    w5 = w5.contiguous(); b5 = b5.contiguous();
    w6 = w6.contiguous(); b6 = b6.contiguous();
    w7 = w7.contiguous(); b7 = b7.contiguous();

    TORCH_CHECK(x.dim() >= 1 && x.size(0) > 0, "Input batch must be nonempty");
    if (!is_canonical_shape(x, w1, b1, w2, b2, w3, b3, w4, b4,
                            w5, b5, w6, b6, w7, b7)) {
        return generic_forward(x, w1, b1, w2, b2, w3, b3, w4, b4,
                               w5, b5, w6, b6, w7, b7);
    }

    const int batch = static_cast<int>(x.size(0));
    x = x.view({batch, kInput});

#if defined(USE_ROCM)
    hipStream_t stream = at::hip::getCurrentHIPStream();
#else
    hipStream_t stream = nullptr;
#endif

    auto y1 = torch::empty({batch, kN1}, x.options());
    const dim3 first_block(256, 1, 1);
    const dim3 first_grid(kN1 / 512, batch, 1);
    hipLaunchKernelGGL(first_layer_kernel, first_grid, first_block, 0, stream,
                       x.data_ptr<float>(), w1.data_ptr<float>(), b1.data_ptr<float>(),
                       y1.data_ptr<float>());

    // These two GEMMs dominate the FLOP count. ATen dispatches them to the
    // architecture-tuned ROCm BLAS implementation without materializing W^T.
    auto y2 = at::mm(y1, w2.t());
    const int y2_elements = batch * kN2;
    const dim3 epilogue_block(256, 1, 1);
    const dim3 epilogue_grid((y2_elements + 255) / 256, 1, 1);
    hipLaunchKernelGGL(bias_relu_2048_kernel, epilogue_grid, epilogue_block, 0, stream,
                       y2.data_ptr<float>(), b2.data_ptr<float>(), y2_elements);

    auto y3 = at::mm(y2, w3.t());
    auto output = torch::empty({batch, kOutput}, x.options());
    const dim3 tail_block(128, 1, 1);
    const dim3 tail_grid(batch, 1, 1);
    hipLaunchKernelGGL(fused_tail_kernel, tail_grid, tail_block, 0, stream,
                       y3.data_ptr<float>(), b3.data_ptr<float>(),
                       w4.data_ptr<float>(), b4.data_ptr<float>(),
                       w5.data_ptr<float>(), b5.data_ptr<float>(),
                       w6.data_ptr<float>(), b6.data_ptr<float>(),
                       w7.data_ptr<float>(), b7.data_ptr<float>(),
                       output.data_ptr<float>());

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mlp_forward, "Optimized 7-layer MLP forward (HIP/ROCm)");
}
