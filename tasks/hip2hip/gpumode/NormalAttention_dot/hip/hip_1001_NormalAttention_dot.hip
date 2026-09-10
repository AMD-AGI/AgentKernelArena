// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <optional>
#include <cmath>

namespace {

constexpr int kAttentionThreads = 256;

// Specialized path for the task's C=4, Ck=1 layout.  A 256-element spatial
// tile is staged at a time so that query/value projection results are shared
// by all output positions in the workgroup.  This directly forms V*ELU(Q*K)
// and therefore avoids materializing the O(B*(H*W)^2) attention matrix.
__global__ __launch_bounds__(kAttentionThreads)
void fused_attention_c4_ck1_kernel(
    const float* __restrict__ input,
    const float* __restrict__ query_weight,
    const float* __restrict__ query_bias,
    const float* __restrict__ key_weight,
    const float* __restrict__ key_bias,
    const float* __restrict__ value_weight,
    const float* __restrict__ value_bias,
    const float* __restrict__ gamma_weight,
    const float* __restrict__ gamma_bias,
    float* __restrict__ output,
    int spatial_size)
{
    __shared__ float staged_query[kAttentionThreads];
    __shared__ float staged_value[4][kAttentionThreads];
    __shared__ float q_weight[4];
    __shared__ float k_weight[4];
    __shared__ float v_weight[16];
    __shared__ float g_weight[16];
    __shared__ float v_bias[4];
    __shared__ float g_bias[4];
    __shared__ float q_bias;
    __shared__ float k_bias;

    const int tid = static_cast<int>(threadIdx.x);
    if (tid < 4) {
        q_weight[tid] = query_weight[tid];
        k_weight[tid] = key_weight[tid];
        v_bias[tid] = value_bias[tid];
        g_bias[tid] = gamma_bias[tid];
    }
    if (tid < 16) {
        v_weight[tid] = value_weight[tid];
        g_weight[tid] = gamma_weight[tid];
    }
    if (tid == 0) {
        q_bias = query_bias[0];
        k_bias = key_bias[0];
    }
    __syncthreads();

    const int batch = static_cast<int>(blockIdx.x);
    const int out_pos = static_cast<int>(blockIdx.y) * kAttentionThreads + tid;
    const size_t batch_base = static_cast<size_t>(batch) * 4 * spatial_size;

    float key = 0.0f;
    if (out_pos < spatial_size) {
        const float x0 = input[batch_base + out_pos];
        const float x1 = input[batch_base + spatial_size + out_pos];
        const float x2 = input[batch_base + 2 * spatial_size + out_pos];
        const float x3 = input[batch_base + 3 * spatial_size + out_pos];
        key = k_bias;
        key = fmaf(k_weight[0], x0, key);
        key = fmaf(k_weight[1], x1, key);
        key = fmaf(k_weight[2], x2, key);
        key = fmaf(k_weight[3], x3, key);
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int tile = 0; tile < spatial_size; tile += kAttentionThreads) {
        const int in_pos = tile + tid;
        if (in_pos < spatial_size) {
            const float x0 = input[batch_base + in_pos];
            const float x1 = input[batch_base + spatial_size + in_pos];
            const float x2 = input[batch_base + 2 * spatial_size + in_pos];
            const float x3 = input[batch_base + 3 * spatial_size + in_pos];

            float query = q_bias;
            query = fmaf(q_weight[0], x0, query);
            query = fmaf(q_weight[1], x1, query);
            query = fmaf(q_weight[2], x2, query);
            query = fmaf(q_weight[3], x3, query);
            staged_query[tid] = query;

            float value0 = v_bias[0];
            value0 = fmaf(v_weight[0], x0, value0);
            value0 = fmaf(v_weight[1], x1, value0);
            value0 = fmaf(v_weight[2], x2, value0);
            staged_value[0][tid] = fmaf(v_weight[3], x3, value0);

            float value1 = v_bias[1];
            value1 = fmaf(v_weight[4], x0, value1);
            value1 = fmaf(v_weight[5], x1, value1);
            value1 = fmaf(v_weight[6], x2, value1);
            staged_value[1][tid] = fmaf(v_weight[7], x3, value1);

            float value2 = v_bias[2];
            value2 = fmaf(v_weight[8], x0, value2);
            value2 = fmaf(v_weight[9], x1, value2);
            value2 = fmaf(v_weight[10], x2, value2);
            staged_value[2][tid] = fmaf(v_weight[11], x3, value2);

            float value3 = v_bias[3];
            value3 = fmaf(v_weight[12], x0, value3);
            value3 = fmaf(v_weight[13], x1, value3);
            value3 = fmaf(v_weight[14], x2, value3);
            staged_value[3][tid] = fmaf(v_weight[15], x3, value3);
        }
        __syncthreads();

        if (out_pos < spatial_size) {
            const int tile_count = min(kAttentionThreads, spatial_size - tile);
            #pragma unroll 4
            for (int i = 0; i < tile_count; ++i) {
                const float score = staged_query[i] * key;
                const float attention = score > 0.0f ? score : expf(score) - 1.0f;
                acc0 = fmaf(staged_value[0][i], attention, acc0);
                acc1 = fmaf(staged_value[1][i], attention, acc1);
                acc2 = fmaf(staged_value[2][i], attention, acc2);
                acc3 = fmaf(staged_value[3][i], attention, acc3);
            }
        }
        __syncthreads();
    }

    if (out_pos < spatial_size) {
        const float inv_spatial = 1.0f / static_cast<float>(spatial_size);
        acc0 *= inv_spatial;
        acc1 *= inv_spatial;
        acc2 *= inv_spatial;
        acc3 *= inv_spatial;

        float out0 = g_bias[0];
        out0 = fmaf(g_weight[0], acc0, out0);
        out0 = fmaf(g_weight[1], acc1, out0);
        out0 = fmaf(g_weight[2], acc2, out0);
        out0 = fmaf(g_weight[3], acc3, out0);

        float out1 = g_bias[1];
        out1 = fmaf(g_weight[4], acc0, out1);
        out1 = fmaf(g_weight[5], acc1, out1);
        out1 = fmaf(g_weight[6], acc2, out1);
        out1 = fmaf(g_weight[7], acc3, out1);

        float out2 = g_bias[2];
        out2 = fmaf(g_weight[8], acc0, out2);
        out2 = fmaf(g_weight[9], acc1, out2);
        out2 = fmaf(g_weight[10], acc2, out2);
        out2 = fmaf(g_weight[11], acc3, out2);

        float out3 = g_bias[3];
        out3 = fmaf(g_weight[12], acc0, out3);
        out3 = fmaf(g_weight[13], acc1, out3);
        out3 = fmaf(g_weight[14], acc2, out3);
        out3 = fmaf(g_weight[15], acc3, out3);

        output[batch_base + out_pos] = out0;
        output[batch_base + spatial_size + out_pos] = out1;
        output[batch_base + 2 * spatial_size + out_pos] = out2;
        output[batch_base + 3 * spatial_size + out_pos] = out3;
    }
}

} // namespace

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

    // The benchmarked module fixes C=4 and k=4, hence Ck=1.  Use the fused
    // implementation for that contract and retain the general path below for
    // other valid channel configurations.
    if (C == 4 && Ck == 1 && HW > 0 && B > 0) {
        auto output = torch::empty_like(input);
        hipStream_t stream = at::hip::getCurrentHIPStream();
        const dim3 block(kAttentionThreads);
        const dim3 grid(
            static_cast<unsigned int>(B),
            static_cast<unsigned int>((HW + kAttentionThreads - 1) / kAttentionThreads));
        hipLaunchKernelGGL(
            fused_attention_c4_ck1_kernel,
            grid,
            block,
            0,
            stream,
            input.data_ptr<float>(),
            query_weight.data_ptr<float>(),
            query_bias.data_ptr<float>(),
            key_weight.data_ptr<float>(),
            key_bias.data_ptr<float>(),
            value_weight.data_ptr<float>(),
            value_bias.data_ptr<float>(),
            gamma_weight.data_ptr<float>(),
            gamma_bias.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(HW));
        return output;
    }

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
