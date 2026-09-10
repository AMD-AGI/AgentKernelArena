// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
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

// The task's hot path has four input features and four projected features.  A
// wave owns one query position and its lanes own support positions, which turns
// both softmax reductions and the four weighted sums into wave reductions.  A
// workgroup contains four independent waves and needs no LDS or workgroup
// barriers.
__device__ __forceinline__ float wave_sum_64(float value)
{
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, 64);
    }
    return value;
}

__device__ __forceinline__ float wave_max_64(float value)
{
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down(value, offset, 64));
    }
    return value;
}

__device__ __forceinline__ float dot4_bias(
    const float4 x, const float4 w, const float bias)
{
    return bias + (x.x * w.x + x.y * w.y + x.z * w.z + x.w * w.w);
}

__global__ __launch_bounds__(256) void item_query_attention_f4_h4_kernel(
    const float* __restrict__ qs,      // [q, seq, 4]
    const float* __restrict__ hs,      // [s, seq, 4]
    const float* __restrict__ weight,  // [4, 4]
    const float* __restrict__ bias,    // [4]
    float* __restrict__ out,           // [q, s, seq, 4]
    int s_size,
    int seq_size)
{
    constexpr int wave_size = 64;
    constexpr int waves_per_block = 4;

    const int q = blockIdx.x;
    const int s = blockIdx.y;
    const int lane = threadIdx.x & (wave_size - 1);
    const int wave = threadIdx.x >> 6;

    // All tensor rows are 16-byte aligned for the specialized four-float path.
    const float4 w0 = *reinterpret_cast<const float4*>(weight + 0);
    const float4 w1 = *reinterpret_cast<const float4*>(weight + 4);
    const float4 w2 = *reinterpret_cast<const float4*>(weight + 8);
    const float4 w3 = *reinterpret_cast<const float4*>(weight + 12);
    const float4 bv = *reinterpret_cast<const float4*>(bias);

    // Each lane caches at most two support positions (seq_size <= 128).  These
    // values and projections are reused for every query position handled by the
    // wave, avoiding materialized projection tensors and repeated support work.
    const int j0 = lane;
    const int j1 = lane + wave_size;

    float4 hs0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 hs1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float hp00 = 0.0f, hp01 = 0.0f, hp02 = 0.0f, hp03 = 0.0f;
    float hp10 = 0.0f, hp11 = 0.0f, hp12 = 0.0f, hp13 = 0.0f;

    const float* hs_base = hs + (size_t)s * seq_size * 4;
    if (j0 < seq_size) {
        hs0 = *reinterpret_cast<const float4*>(hs_base + (size_t)j0 * 4);
        hp00 = dot4_bias(hs0, w0, bv.x);
        hp01 = dot4_bias(hs0, w1, bv.y);
        hp02 = dot4_bias(hs0, w2, bv.z);
        hp03 = dot4_bias(hs0, w3, bv.w);
    }
    if (j1 < seq_size) {
        hs1 = *reinterpret_cast<const float4*>(hs_base + (size_t)j1 * 4);
        hp10 = dot4_bias(hs1, w0, bv.x);
        hp11 = dot4_bias(hs1, w1, bv.y);
        hp12 = dot4_bias(hs1, w2, bv.z);
        hp13 = dot4_bias(hs1, w3, bv.w);
    }

    const float* qs_base = qs + (size_t)q * seq_size * 4;
    for (int iq = wave; iq < seq_size; iq += waves_per_block) {
        const float4 qv = *reinterpret_cast<const float4*>(qs_base + (size_t)iq * 4);
        const float qp0 = dot4_bias(qv, w0, bv.x);
        const float qp1 = dot4_bias(qv, w1, bv.y);
        const float qp2 = dot4_bias(qv, w2, bv.z);
        const float qp3 = dot4_bias(qv, w3, bv.w);

        float score0 = -1.0e30f;
        float score1 = -1.0e30f;
        if (j0 < seq_size) {
            score0 = tanhf(qp0 * hp00) + tanhf(qp1 * hp01)
                   + tanhf(qp2 * hp02) + tanhf(qp3 * hp03);
        }
        if (j1 < seq_size) {
            score1 = tanhf(qp0 * hp10) + tanhf(qp1 * hp11)
                   + tanhf(qp2 * hp12) + tanhf(qp3 * hp13);
        }

        float max_score = wave_max_64(fmaxf(score0, score1));
        max_score = __shfl(max_score, 0, 64);

        const float e0 = j0 < seq_size ? expf(score0 - max_score) : 0.0f;
        const float e1 = j1 < seq_size ? expf(score1 - max_score) : 0.0f;
        float denom = wave_sum_64(e0 + e1);
        denom = __shfl(denom, 0, 64);
        const float inv_denom = 1.0f / denom;
        const float p0 = e0 * inv_denom;
        const float p1 = e1 * inv_denom;

        float out0 = p0 * hs0.x + p1 * hs1.x;
        float out1 = p0 * hs0.y + p1 * hs1.y;
        float out2 = p0 * hs0.z + p1 * hs1.z;
        float out3 = p0 * hs0.w + p1 * hs1.w;

        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1) {
            out0 += __shfl_down(out0, offset, 64);
            out1 += __shfl_down(out1, offset, 64);
            out2 += __shfl_down(out2, offset, 64);
            out3 += __shfl_down(out3, offset, 64);
        }

        if (lane == 0) {
            float* out_row = out + (((size_t)q * s_size + s) * seq_size + iq) * 4;
            *reinterpret_cast<float4*>(out_row) = make_float4(out0, out1, out2, out3);
        }
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
    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    // Output tensor: [q, s, seq, feature]
    at::Tensor out = at::empty({q_size, s_size, seq_size, feature_size}, options);

    // All benchmark shapes use this path.  Keep the original general kernels
    // below as a fallback so the public forward interface remains generic.
    if (feature_size == 4 && hidden_size == 4 && seq_size > 0 && seq_size <= 128) {
        dim3 grid(q_size, s_size);
        hipLaunchKernelGGL(item_query_attention_f4_h4_kernel, grid, dim3(256), 0, stream,
            qs.data_ptr<float>(),
            hs.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            s_size, seq_size);
        HIP_CHECK(hipGetLastError());
        return out;
    }

    // Linear transforms: qs_lin [q, seq, H], hs_lin [s, seq, H]
    at::Tensor qs_flat = qs.view({q_size * seq_size, feature_size});
    at::Tensor hs_flat = hs.view({s_size * seq_size, feature_size});

    at::Tensor qs_lin_flat = at::empty({q_size * seq_size, hidden_size}, options);
    at::Tensor hs_lin_flat = at::empty({s_size * seq_size, hidden_size}, options);

    // Launch linear kernels
    int AB_q = q_size * seq_size;
    int AB_s = s_size * seq_size;
    int threads_lin = std::min(256, std::max(32, hidden_size));

    hipLaunchKernelGGL(linear_lastdim_kernel, dim3(AB_q), dim3(threads_lin), 0, stream,
        qs_flat.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), qs_lin_flat.data_ptr<float>(), AB_q, feature_size, hidden_size);
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(linear_lastdim_kernel, dim3(AB_s), dim3(threads_lin), 0, stream,
        hs_flat.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), hs_lin_flat.data_ptr<float>(), AB_s, feature_size, hidden_size);
    HIP_CHECK(hipGetLastError());

    at::Tensor qs_lin = qs_lin_flat.view({q_size, seq_size, hidden_size}).contiguous();
    at::Tensor hs_lin = hs_lin_flat.view({s_size, seq_size, hidden_size}).contiguous();

    // Launch fused attention kernel
    int threads_att = std::min(256, std::max(32, feature_size));
    dim3 grid(q_size, s_size, seq_size);
    size_t shmem_bytes = static_cast<size_t>(seq_size) * sizeof(float);

    hipLaunchKernelGGL(weighted_context_kernel, grid, dim3(threads_att), shmem_bytes, stream,
        qs_lin.data_ptr<float>(),
        hs_lin.data_ptr<float>(),
        hs.data_ptr<float>(),
        out.data_ptr<float>(),
        q_size, s_size, seq_size, hidden_size, feature_size);
    HIP_CHECK(hipGetLastError());

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ItemQueryAttention HIP forward (ROCm)");
}
