// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>
#if defined(USE_ROCM)
#include <ATen/hip/HIPContext.h>
#endif
#include <cstdint>
#include <algorithm>

// Kernel: computes Y = A[M,K] * Wt[K,N] + bias[N], with optional ReLU.
// Mapping: one block per row m, threads cover N with per-thread micro-tile RN.
// Dtype: float (ROCm-friendly, consistent with provided test).
template <bool ApplyRelu, int RN>
__global__ void linear_bias_epilogue_kernel(
    const float* __restrict__ A,   // [M,K]
    const float* __restrict__ Wt,  // [K,N] = W^T contiguous
    const float* __restrict__ bias,// [N]
    float* __restrict__ Y,         // [M,N]
    int M, int N, int K)
{
    int m = blockIdx.y; // one block per row
    if (m >= M) return;

    int t = threadIdx.x;
    int threads = blockDim.x;

    // Each thread computes RN consecutive columns starting at n0
    int n0 = (blockIdx.x * threads + t) * RN;
    if (n0 >= N) return;

    // Accumulators
    float acc[RN];
#pragma unroll
    for (int i = 0; i < RN; ++i) acc[i] = 0.0f;

    const float* a_row = A + (size_t)m * K;

    // Iterate K and accumulate
    for (int k = 0; k < K; ++k) {
        float a = a_row[k];
        const float* w_row = Wt + (size_t)k * N + n0;
#pragma unroll
        for (int i = 0; i < RN; ++i) {
            int n = n0 + i;
            if (n < N) {
                float w = w_row[i];
                acc[i] = fmaf(a, w, acc[i]);
            }
        }
    }

    // Epilogue: add bias and optional ReLU, then store
    float* y_row = Y + (size_t)m * N + n0;
#pragma unroll
    for (int i = 0; i < RN; ++i) {
        int n = n0 + i;
        if (n < N) {
            float v = acc[i] + bias[n];
            if (ApplyRelu) v = v > 0.0f ? v : 0.0f;
            y_row[i] = v;
        }
    }
}

static inline void launch_linear_bias(
    const at::Tensor& A,     // [M,K]
    const at::Tensor& Wt,    // [K,N]
    const at::Tensor& bias,  // [N]
    at::Tensor& Y,           // [M,N]
    bool apply_relu)
{
    TORCH_CHECK(A.scalar_type() == at::kFloat && Wt.scalar_type() == at::kFloat && bias.scalar_type() == at::kFloat && Y.scalar_type() == at::kFloat,
                "All tensors must be float32 for this kernel");
    TORCH_CHECK(A.is_cuda() && Wt.is_cuda() && bias.is_cuda() && Y.is_cuda(), "All tensors must be on HIP device");
    TORCH_CHECK(A.is_contiguous() && Wt.is_contiguous() && bias.is_contiguous() && Y.is_contiguous(), "All tensors must be contiguous");

    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int N = static_cast<int>(Wt.size(1));
    TORCH_CHECK(Wt.size(0) == K, "Wt shape must be [K,N]");
    TORCH_CHECK(bias.numel() == N, "bias size mismatch");
    TORCH_CHECK(Y.size(0) == M && Y.size(1) == N, "Output shape mismatch");

#if defined(USE_ROCM)
    hipStream_t stream = at::hip::getCurrentHIPStream();
#else
    hipStream_t stream = nullptr;
#endif

    constexpr int RN = 4;            // per-thread micro-tile in N
    const int threads = 256;         // multiple of 64 (AMD wavefront)
    int tiles_n = (N + RN - 1) / RN; // number of RN-wide tiles in N
    dim3 block(threads, 1, 1);
    dim3 grid((tiles_n + threads - 1) / threads, M, 1);

    const float* A_ptr = A.data_ptr<float>();
    const float* Wt_ptr = Wt.data_ptr<float>();
    const float* b_ptr = bias.data_ptr<float>();
    float* Y_ptr = Y.data_ptr<float>();

    if (apply_relu) {
        hipLaunchKernelGGL((linear_bias_epilogue_kernel<true, RN>), grid, block, 0, stream,
                           A_ptr, Wt_ptr, b_ptr, Y_ptr, M, N, K);
    } else {
        hipLaunchKernelGGL((linear_bias_epilogue_kernel<false, RN>), grid, block, 0, stream,
                           A_ptr, Wt_ptr, b_ptr, Y_ptr, M, N, K);
    }
}

// Forward pass: mirrors the given PyTorch functional implementation exactly
// Layers: 6 hidden with ReLU, final linear without ReLU
// Inputs/weights must be float32 on HIP device for this implementation.
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

    // Enforce float32 for robust compilation and exact matching with tests
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(w1.scalar_type() == at::kFloat && b1.scalar_type() == at::kFloat, "dtype mismatch w1/b1");
    TORCH_CHECK(w2.scalar_type() == at::kFloat && b2.scalar_type() == at::kFloat, "dtype mismatch w2/b2");
    TORCH_CHECK(w3.scalar_type() == at::kFloat && b3.scalar_type() == at::kFloat, "dtype mismatch w3/b3");
    TORCH_CHECK(w4.scalar_type() == at::kFloat && b4.scalar_type() == at::kFloat, "dtype mismatch w4/b4");
    TORCH_CHECK(w5.scalar_type() == at::kFloat && b5.scalar_type() == at::kFloat, "dtype mismatch w5/b5");
    TORCH_CHECK(w6.scalar_type() == at::kFloat && b6.scalar_type() == at::kFloat, "dtype mismatch w6/b6");
    TORCH_CHECK(w7.scalar_type() == at::kFloat && b7.scalar_type() == at::kFloat, "dtype mismatch w7/b7");

    // Make tensors contiguous
    auto x = input.contiguous();
    w1 = w1.contiguous(); b1 = b1.contiguous();
    w2 = w2.contiguous(); b2 = b2.contiguous();
    w3 = w3.contiguous(); b3 = b3.contiguous();
    w4 = w4.contiguous(); b4 = b4.contiguous();
    w5 = w5.contiguous(); b5 = b5.contiguous();
    w6 = w6.contiguous(); b6 = b6.contiguous();
    w7 = w7.contiguous(); b7 = b7.contiguous();

    // Flatten input: xb.view(xb.size(0), -1)
    const auto B = x.size(0);
    const int64_t in_flat = x.numel() / B;
    x = x.view({B, in_flat});

    // Dim checks align with PyTorch's nn.Linear usage
    TORCH_CHECK(w1.size(1) == in_flat && b1.numel() == w1.size(0), "w1/b1 size mismatch");
    TORCH_CHECK(w2.size(1) == w1.size(0) && b2.numel() == w2.size(0), "w2/b2 size mismatch");
    TORCH_CHECK(w3.size(1) == w2.size(0) && b3.numel() == w3.size(0), "w3/b3 size mismatch");
    TORCH_CHECK(w4.size(1) == w3.size(0) && b4.numel() == w4.size(0), "w4/b4 size mismatch");
    TORCH_CHECK(w5.size(1) == w4.size(0) && b5.numel() == w5.size(0), "w5/b5 size mismatch");
    TORCH_CHECK(w6.size(1) == w5.size(0) && b6.numel() == w6.size(0), "w6/b6 size mismatch");
    TORCH_CHECK(w7.size(1) == w6.size(0) && b7.numel() == w7.size(0), "w7/b7 size mismatch");

    auto opts = x.options();

    // Pretranspose weights for coalesced access: [K,N]
    auto W1t = w1.t().contiguous();
    auto W2t = w2.t().contiguous();
    auto W3t = w3.t().contiguous();
    auto W4t = w4.t().contiguous();
    auto W5t = w5.t().contiguous();
    auto W6t = w6.t().contiguous();
    auto W7t = w7.t().contiguous();

    // Allocate intermediates
    const int64_t N1 = w1.size(0);
    const int64_t N2 = w2.size(0);
    const int64_t N3 = w3.size(0);
    const int64_t N4 = w4.size(0);
    const int64_t N5 = w5.size(0);
    const int64_t N6 = w6.size(0);
    const int64_t N7 = w7.size(0);

    auto y1 = torch::empty({B, N1}, opts);
    auto y2 = torch::empty({B, N2}, opts);
    auto y3 = torch::empty({B, N3}, opts);
    auto y4 = torch::empty({B, N4}, opts);
    auto y5 = torch::empty({B, N5}, opts);
    auto y6 = torch::empty({B, N6}, opts);
    auto y7 = torch::empty({B, N7}, opts);

    // Layer 1: ReLU
    launch_linear_bias(x, W1t, b1, y1, true);
    // Layer 2: ReLU
    launch_linear_bias(y1, W2t, b2, y2, true);
    // Layer 3: ReLU
    launch_linear_bias(y2, W3t, b3, y3, true);
    // Layer 4: ReLU
    launch_linear_bias(y3, W4t, b4, y4, true);
    // Layer 5: ReLU
    launch_linear_bias(y4, W5t, b5, y5, true);
    // Layer 6: ReLU
    launch_linear_bias(y5, W6t, b6, y6, true);
    // Output layer: no ReLU
    launch_linear_bias(y6, W7t, b7, y7, false);

    return y7;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mlp_forward, "7-layer MLP forward (HIP/ROCm) with transposed-weight, micro-tiled kernel and fused epilogue");
}

