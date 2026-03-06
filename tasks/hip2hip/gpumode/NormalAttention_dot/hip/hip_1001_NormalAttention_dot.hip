// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <optional>
#include <cmath>

// HIP kernel: ELU + divide by factor
__global__ void elu_div_kernel(float* __restrict__ x, int N, float div)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = x[idx];
        float e = (v > 0.0f) ? v : (expf(v) - 1.0f);
        x[idx] = e / div;
    }
}

static inline void hipblas_check(hipblasStatus_t stat, const char* msg) {
    if (stat != HIPBLAS_STATUS_SUCCESS) {
        TORCH_CHECK(false, "hipBLAS error: ", msg, ", status=", (int)stat);
    }
}

// Forward implementation
torch::Tensor normal_attention_dot_forward(
    torch::Tensor input,           // (B, C, H, W)
    torch::Tensor query_weight,    // (Ck, C, 1, 1)
    torch::Tensor query_bias,      // (Ck)
    torch::Tensor key_weight,      // (Ck, C, 1, 1)
    torch::Tensor key_bias,        // (Ck)
    torch::Tensor value_weight,    // (C, C, 1, 1)
    torch::Tensor value_bias,      // (C)
    torch::Tensor gamma_weight,    // (C, C, 1, 1)
    torch::Tensor gamma_bias       // (C)
) {
    TORCH_CHECK(input.is_cuda(), "Input must be on HIP device (ROCm)");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(query_weight.scalar_type() == at::kFloat && key_weight.scalar_type() == at::kFloat && value_weight.scalar_type() == at::kFloat && gamma_weight.scalar_type() == at::kFloat, "Weights must be float32");
    TORCH_CHECK(query_bias.scalar_type() == at::kFloat && key_bias.scalar_type() == at::kFloat && value_bias.scalar_type() == at::kFloat && gamma_bias.scalar_type() == at::kFloat, "Biases must be float32");

    input = input.contiguous();
    query_weight = query_weight.contiguous();
    key_weight   = key_weight.contiguous();
    value_weight = value_weight.contiguous();
    gamma_weight = gamma_weight.contiguous();
    query_bias = query_bias.contiguous();
    key_bias   = key_bias.contiguous();
    value_bias = value_bias.contiguous();
    gamma_bias = gamma_bias.contiguous();

    const int64_t B = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);
    const int64_t HW = H * W;

    TORCH_CHECK(query_weight.size(2) == 1 && query_weight.size(3) == 1, "query_weight must be 1x1");
    TORCH_CHECK(key_weight.size(2) == 1 && key_weight.size(3) == 1, "key_weight must be 1x1");
    TORCH_CHECK(value_weight.size(2) == 1 && value_weight.size(3) == 1, "value_weight must be 1x1");
    TORCH_CHECK(gamma_weight.size(2) == 1 && gamma_weight.size(3) == 1, "gamma_weight must be 1x1");

    const int64_t Ck = query_weight.size(0);
    TORCH_CHECK(Ck == key_weight.size(0), "query/key out channels mismatch");
    TORCH_CHECK(query_weight.size(1) == C && key_weight.size(1) == C && value_weight.size(0) == C && value_weight.size(1) == C && gamma_weight.size(0) == C && gamma_weight.size(1) == C, "Channel mismatch");
    TORCH_CHECK(query_bias.numel() == Ck && key_bias.numel() == Ck && value_bias.numel() == C && gamma_bias.numel() == C, "Bias size mismatch");

    auto options = input.options();

    // 1x1 convolutions with bias (exact semantics of PyTorch conv2d)
    auto proj_query = at::conv2d(input, query_weight, std::optional<at::Tensor>(query_bias), at::IntArrayRef({1, 1}), at::IntArrayRef({0, 0}), at::IntArrayRef({1, 1}), 1).contiguous(); // (B, Ck, H, W)
    auto proj_key   = at::conv2d(input, key_weight,   std::optional<at::Tensor>(key_bias),   at::IntArrayRef({1, 1}), at::IntArrayRef({0, 0}), at::IntArrayRef({1, 1}), 1).contiguous(); // (B, Ck, H, W)
    auto proj_value = at::conv2d(input, value_weight, std::optional<at::Tensor>(value_bias), at::IntArrayRef({1, 1}), at::IntArrayRef({0, 0}), at::IntArrayRef({1, 1}), 1).contiguous(); // (B, C,  H, W)

    // Reshape for batched GEMM
    auto Aq = proj_query.view({B, Ck, HW}).permute({0, 2, 1}).contiguous(); // (B, M=HW, K=Ck)
    auto Bk = proj_key.view({B, Ck, HW}).contiguous();                      // (B, K=Ck, N=HW)
    auto Av = proj_value.view({B, C,  HW}).contiguous();                     // (B, M=C,  K=HW)

    // energy = Aq x Bk -> (B, HW, HW)
    auto energy = torch::empty({B, HW, HW}, options);

    // Setup hipBLAS
    hipblasHandle_t handle;
    hipblas_check(hipblasCreate(&handle), "hipblasCreate");
    hipStream_t stream = at::hip::getCurrentHIPStream();
    hipblas_check(hipblasSetStream(handle, stream), "hipblasSetStream");

    const float alpha = 1.0f;
    const float beta0 = 0.0f;

    // Map row-major GEMM to hipBLAS column-major using transpose trick:
    // For C_rm = A_rm(MxK) * B_rm(KxN), call hipBLAS on C_cm(NxM) = B_cm(NxK) * A_cm(KxM)
    // energy: M=HW, K=Ck, N=HW
    {
        int m_cm = static_cast<int>(HW);   // rows of C_cm = N
        int n_cm = static_cast<int>(HW);   // cols of C_cm = M
        int k_cm = static_cast<int>(Ck);   // K

        int lda = m_cm; // for A_cm (B_rm) size N x K -> lda = N = HW
        int ldb = k_cm; // for B_cm (A_rm) size K x M -> ldb = K = Ck (but note: B here is A_rm) -> rows=K, so ldb=K=Ck
        int ldc = m_cm; // for C_cm size N x M -> ldc = N = HW

        long long strideA = static_cast<long long>(Ck) * static_cast<long long>(HW); // B_rm stride: K*N
        long long strideB = static_cast<long long>(HW) * static_cast<long long>(Ck); // A_rm stride: M*K
        long long strideC = static_cast<long long>(HW) * static_cast<long long>(HW); // C_rm stride: M*N

        hipblas_check(
            hipblasSgemmStridedBatched(
                handle,
                HIPBLAS_OP_N, HIPBLAS_OP_N,
                m_cm, n_cm, k_cm,
                &alpha,
                // A = Bk (B_rm) treated as column-major N x K
                Bk.data_ptr<float>(), lda, strideA,
                // B = Aq (A_rm) treated as column-major K x M
                Aq.data_ptr<float>(), ldb, strideB,
                &beta0,
                // C = energy (C_rm) treated as column-major N x M
                energy.data_ptr<float>(), ldc, strideC,
                static_cast<int>(B)
            ),
            "hipblasSgemmStridedBatched (energy)"
        );
    }

    // Apply ELU and divide by HW
    {
        int64_t Ne = B * HW * HW;
        int threads = 256;
        int blocks = static_cast<int>((Ne + threads - 1) / threads);
        float div = static_cast<float>(HW);
        hipLaunchKernelGGL(elu_div_kernel, dim3(blocks), dim3(threads), 0, stream,
                           energy.data_ptr<float>(), static_cast<int>(Ne), div);
    }

    // out2d = Av (B, C, HW) x energy (B, HW, HW) -> (B, C, HW)
    auto out2d = torch::empty({B, C, HW}, options);
    {
        // M=C, K=HW, N=HW
        int m_cm = static_cast<int>(HW);  // rows of C_cm = N
        int n_cm = static_cast<int>(C);   // cols of C_cm = M
        int k_cm = static_cast<int>(HW);  // K

        int lda = m_cm; // A_cm (energy) size N x K -> lda = N = HW
        int ldb = k_cm; // B_cm (Av) size K x M -> ldb = K = HW
        int ldc = m_cm; // C_cm size N x M -> ldc = N = HW

        long long strideA = static_cast<long long>(HW) * static_cast<long long>(HW); // energy stride: K*N
        long long strideB = static_cast<long long>(C) * static_cast<long long>(HW);  // Av stride: M*K
        long long strideC = static_cast<long long>(C) * static_cast<long long>(HW);  // out2d stride: M*N

        hipblas_check(
            hipblasSgemmStridedBatched(
                handle,
                HIPBLAS_OP_N, HIPBLAS_OP_N,
                m_cm, n_cm, k_cm,
                &alpha,
                // A = energy (right multiplicand), treated as column-major N x K
                energy.data_ptr<float>(), lda, strideA,
                // B = Av (left multiplicand), treated as column-major K x M
                Av.data_ptr<float>(), ldb, strideB,
                &beta0,
                // C = out2d, treated as column-major N x M
                out2d.data_ptr<float>(), ldc, strideC,
                static_cast<int>(B)
            ),
            "hipblasSgemmStridedBatched (out2d)"
        );
    }

    // Destroy hipBLAS handle
    hipblas_check(hipblasDestroy(handle), "hipblasDestroy");

    // Reshape to (B, C, H, W)
    auto out = out2d.view({B, C, H, W});

    // Final 1x1 conv: gamma
    auto out_gamma = at::conv2d(out, gamma_weight, std::optional<at::Tensor>(gamma_bias), at::IntArrayRef({1, 1}), at::IntArrayRef({0, 0}), at::IntArrayRef({1, 1}), 1);

    return out_gamma;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &normal_attention_dot_forward, "NormalAttention_dot forward (HIP/ROCm, hipBLAS optimized)");
}
