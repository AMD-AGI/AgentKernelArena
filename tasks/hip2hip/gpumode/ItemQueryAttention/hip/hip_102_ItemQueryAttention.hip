// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(err) \
    if ((err) != hipSuccess) { \
        printf("HIP error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(-1); \
    }

// Kernel: Linear transform along last dimension for a 3D tensor [A, B, F] -> [A, B, H]
// y[n, h] = sum_f x[n, f] * W[h, f] + b[h]
__global__ void linear_lastdim_kernel(
    const float* __restrict__ x,   // [A*B, F]
    const float* __restrict__ W,   // [H, F]
    const float* __restrict__ b,   // [H]
    float* __restrict__ y,         // [A*B, H]
    int AB, int F, int H)
{
    int n = blockIdx.x; // 0..AB-1
    int tid = threadIdx.x;
    for (int h = tid; h < H; h += blockDim.x) {
        float acc = b ? b[h] : 0.0f;
        const float* w_row = W + (size_t)h * F;
        const float* x_row = x + (size_t)n * F;
        // Unrolled loop chunks for better ILP
        int f = 0;
        for (; f + 3 < F; f += 4) {
            float x0 = x_row[f + 0];
            float x1 = x_row[f + 1];
            float x2 = x_row[f + 2];
            float x3 = x_row[f + 3];
            acc += x0 * w_row[f + 0] + x1 * w_row[f + 1] + x2 * w_row[f + 2] + x3 * w_row[f + 3];
        }
        for (; f < F; ++f) {
            acc += x_row[f] * w_row[f];
        }
        y[(size_t)n * H + h] = acc;
    }
}

// Fused attention + softmax + weighted sum kernel.
// Computes out[q, s, iq, f] = sum_j softmax_j( sum_h tanh(qs_lin[q, iq, h] * hs_lin[s, j, h]) ) * hs_raw[s, j, f]
// Grid: (q_size, s_size, seq_size), Block: threads over feature dimension F
extern "C" __global__ void weighted_context_kernel(
    const float* __restrict__ qs_lin,   // [q, seq, H]
    const float* __restrict__ hs_lin,   // [s, seq, H]
    const float* __restrict__ hs_raw,   // [s, seq, F]
    float* __restrict__ out,            // [q, s, seq, F]
    int q_size,
    int s_size,
    int seq_size,
    int H,
    int F)
{
    int q = blockIdx.x;
    int s = blockIdx.y;
    int iq = blockIdx.z; // query time index
    int tid = threadIdx.x;

    // Shared memory for probabilities over j
    extern __shared__ float shmem[]; // size >= seq_size
    float* probs = shmem;

    // Pointers
    const float* vq = qs_lin + ((size_t)q * seq_size + iq) * H; // [H]

    // Compute attention scores over j, then softmax
    // We do it in thread 0 to avoid complex reductions; seq_size is usually small.
    float maxv = -1e30f;
    if (tid == 0) {
        // First pass: compute raw scores and max
        for (int j = 0; j < seq_size; ++j) {
            const float* vs = hs_lin + ((size_t)s * seq_size + j) * H; // [H]
            float sum = 0.0f;
            int h = 0;
            for (; h + 3 < H; h += 4) {
                float a0 = vq[h + 0] * vs[h + 0];
                float a1 = vq[h + 1] * vs[h + 1];
                float a2 = vq[h + 2] * vs[h + 2];
                float a3 = vq[h + 3] * vs[h + 3];
                sum += tanhf(a0) + tanhf(a1) + tanhf(a2) + tanhf(a3);
            }
            for (; h < H; ++h) {
                sum += tanhf(vq[h] * vs[h]);
            }
            probs[j] = sum; // temporarily store raw score
            if (sum > maxv) maxv = sum;
        }
        // Second pass: exp and sum
        float denom = 0.0f;
        for (int j = 0; j < seq_size; ++j) {
            float e = expf(probs[j] - maxv);
            probs[j] = e; // store exp
            denom += e;
        }
        // Normalize
        float inv_denom = denom > 0 ? 1.0f / denom : 0.0f;
        for (int j = 0; j < seq_size; ++j) {
            probs[j] *= inv_denom;
        }
    }
    __syncthreads();

    // Weighted sum across j for feature dimension f
    for (int f = tid; f < F; f += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < seq_size; ++j) {
            const float* hs_row = hs_raw + ((size_t)s * seq_size + j) * F;
            acc += probs[j] * hs_row[f];
        }
        float* out_row = out + (((size_t)q * s_size + s) * seq_size + iq) * F;
        out_row[f] = acc;
    }
}

// Forward entry
torch::Tensor forward(
    torch::Tensor qs,      // [q_size, seq_size, feature_size]
    torch::Tensor hs,      // [s_size, seq_size, feature_size]
    torch::Tensor weight,  // [hidden_size, feature_size]
    torch::Tensor bias     // [hidden_size]
) {
    TORCH_CHECK(qs.is_cuda(), "qs must be on CUDA/HIP device");
    TORCH_CHECK(hs.is_cuda(), "hs must be on CUDA/HIP device");
    TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA/HIP device");
    TORCH_CHECK(bias.is_cuda(), "bias must be on CUDA/HIP device");

    TORCH_CHECK(qs.scalar_type() == at::kFloat, "qs must be float32");
    TORCH_CHECK(hs.scalar_type() == at::kFloat, "hs must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");

    qs = qs.contiguous();
    hs = hs.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    int q_size = static_cast<int>(qs.size(0));
    int seq_size = static_cast<int>(qs.size(1));
    int feature_size = static_cast<int>(qs.size(2));

    int s_size = static_cast<int>(hs.size(0));
    TORCH_CHECK(static_cast<int>(hs.size(1)) == seq_size, "qs and hs must have same seq_size");
    TORCH_CHECK(static_cast<int>(hs.size(2)) == feature_size, "qs and hs must have same feature_size");

    int hidden_size = static_cast<int>(weight.size(0));
    TORCH_CHECK(static_cast<int>(weight.size(1)) == feature_size, "weight shape must be [hidden_size, feature_size]");
    TORCH_CHECK(static_cast<int>(bias.size(0)) == hidden_size, "bias shape must be [hidden_size]");

    auto options = qs.options();

    // Linear transforms: qs_lin [q, seq, H], hs_lin [s, seq, H]
    at::Tensor qs_flat = qs.view({q_size * seq_size, feature_size});
    at::Tensor hs_flat = hs.view({s_size * seq_size, feature_size});

    at::Tensor qs_lin_flat = at::empty({q_size * seq_size, hidden_size}, options);
    at::Tensor hs_lin_flat = at::empty({s_size * seq_size, hidden_size}, options);

    // Launch linear kernels
    int AB_q = q_size * seq_size;
    int AB_s = s_size * seq_size;
    int threads_lin = std::min(256, std::max(32, hidden_size));

    hipLaunchKernelGGL(linear_lastdim_kernel, dim3(AB_q), dim3(threads_lin), 0, 0,
        qs_flat.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), qs_lin_flat.data_ptr<float>(), AB_q, feature_size, hidden_size);
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(linear_lastdim_kernel, dim3(AB_s), dim3(threads_lin), 0, 0,
        hs_flat.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), hs_lin_flat.data_ptr<float>(), AB_s, feature_size, hidden_size);
    HIP_CHECK(hipGetLastError());

    at::Tensor qs_lin = qs_lin_flat.view({q_size, seq_size, hidden_size}).contiguous();
    at::Tensor hs_lin = hs_lin_flat.view({s_size, seq_size, hidden_size}).contiguous();

    // Output tensor: [q, s, seq, feature]
    at::Tensor out = at::empty({q_size, s_size, seq_size, feature_size}, options);

    // Launch fused attention kernel
    int threads_att = std::min(256, std::max(32, feature_size));
    dim3 grid(q_size, s_size, seq_size);
    size_t shmem_bytes = static_cast<size_t>(seq_size) * sizeof(float);

    hipLaunchKernelGGL(weighted_context_kernel, grid, dim3(threads_att), shmem_bytes, 0,
        qs_lin.data_ptr<float>(),
        hs_lin.data_ptr<float>(),
        hs.data_ptr<float>(),
        out.data_ptr<float>(),
        q_size, s_size, seq_size, hidden_size, feature_size);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipDeviceSynchronize());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ItemQueryAttention HIP forward (ROCm)");
}

