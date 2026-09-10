// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <cmath>
#include <cstdint>

static inline int div_up(int a, int b) { return (a + b - 1) / b; }

// Broadcasted mask application: scores [B,H,S,S], mask [B,S,S]
// Where mask==0, set scores to a large negative value (-1e9) to zero after softmax

template <typename scalar_t>
__global__ void apply_mask_kernel(
    scalar_t* __restrict__ scores,    // [B,H,S,S]
    const uint8_t* __restrict__ mask, // [B,S,S]
    int B, int H, int S)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)B * H * S * S;
    if (idx >= total) return;

    int s2 = (int)(idx % S);
    long long t1 = idx / S;
    int s1 = (int)(t1 % S);
    long long t2 = t1 / S;
    // int h = (int)(t2 % H);
    int b = (int)(t2 / H);

    long long midx = ((long long)b * S + s1) * S + s2;
    if (mask[midx] == 0) {
        scores[idx] = (scalar_t)(-1.0e9f);
    }
}

// The benchmark's D=4 case is too small for a sequence of BLAS calls to be
// efficient.  Fuse projection, attention, softmax, value accumulation, and the
// output projection into one workgroup per query token.  Each Wave64 owns one
// of the two heads, so all attention reductions stay within a wavefront.
__global__ __launch_bounds__(128) void fused_mha_d4_h2_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const float* __restrict__ v_weight,
    const float* __restrict__ out_weight,
    const float* __restrict__ out_bias,
    float* __restrict__ output,
    int S)
{
    const int tid = (int)threadIdx.x;
    const int head = tid >> 6;
    const int lane = tid & 63;
    const int token = (int)blockIdx.x;
    const int batch = token / S;
    const int query_pos = token - batch * S;

    __shared__ float q_projected[4];
    __shared__ float attended[4];

    // Four lanes cooperatively project the query once for both wavefronts.
    if (tid < 4) {
        const float4 qv = *reinterpret_cast<const float4*>(
            q + ((long long)batch * S + query_pos) * 4);
        const int col = tid;
        float x = qv.x * q_weight[col];
        x = fmaf(qv.y, q_weight[4 + col], x);
        x = fmaf(qv.z, q_weight[8 + col], x);
        x = fmaf(qv.w, q_weight[12 + col], x);
        q_projected[col] = x;
    }
    __syncthreads();

    const int col0 = head * 2;
    const int col1 = col0 + 1;
    float score = -INFINITY;
    float4 vv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (lane < S) {
        const long long row = ((long long)batch * S + lane) * 4;
        const float4 kv = *reinterpret_cast<const float4*>(k + row);
        vv = *reinterpret_cast<const float4*>(v + row);

        float kp0 = kv.x * k_weight[col0];
        kp0 = fmaf(kv.y, k_weight[4 + col0], kp0);
        kp0 = fmaf(kv.z, k_weight[8 + col0], kp0);
        kp0 = fmaf(kv.w, k_weight[12 + col0], kp0);

        float kp1 = kv.x * k_weight[col1];
        kp1 = fmaf(kv.y, k_weight[4 + col1], kp1);
        kp1 = fmaf(kv.z, k_weight[8 + col1], kp1);
        kp1 = fmaf(kv.w, k_weight[12 + col1], kp1);

        score = fmaf(q_projected[col1], kp1,
                     q_projected[col0] * kp0) * 0.7071067811865475244f;
    }

    float row_max = score;
#pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        row_max = fmaxf(row_max, __shfl_down(row_max, offset));
    }
    row_max = __shfl(row_max, 0);

    const float probability_numerator = expf(score - row_max);
    float row_sum = probability_numerator;
#pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        row_sum += __shfl_down(row_sum, offset);
    }
    row_sum = __shfl(row_sum, 0);
    const float probability = probability_numerator / row_sum;

    float vp0 = 0.0f;
    float vp1 = 0.0f;
    if (lane < S) {
        vp0 = vv.x * v_weight[col0];
        vp0 = fmaf(vv.y, v_weight[4 + col0], vp0);
        vp0 = fmaf(vv.z, v_weight[8 + col0], vp0);
        vp0 = fmaf(vv.w, v_weight[12 + col0], vp0);

        vp1 = vv.x * v_weight[col1];
        vp1 = fmaf(vv.y, v_weight[4 + col1], vp1);
        vp1 = fmaf(vv.z, v_weight[8 + col1], vp1);
        vp1 = fmaf(vv.w, v_weight[12 + col1], vp1);
    }

    float acc0 = probability * vp0;
    float acc1 = probability * vp1;
#pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        acc0 += __shfl_down(acc0, offset);
        acc1 += __shfl_down(acc1, offset);
    }
    if (lane == 0) {
        attended[col0] = acc0;
        attended[col1] = acc1;
    }
    __syncthreads();

    // F.linear uses output = input @ out_weight.T + out_bias.
    if (tid < 4) {
        const int out_col = tid;
        float x = fmaf(attended[0], out_weight[out_col * 4], out_bias[out_col]);
        x = fmaf(attended[1], out_weight[out_col * 4 + 1], x);
        x = fmaf(attended[2], out_weight[out_col * 4 + 2], x);
        x = fmaf(attended[3], out_weight[out_col * 4 + 3], x);
        output[(long long)token * 4 + out_col] = x;
    }
}

// Main entry matching Python call order and semantics
// q,k,v: arbitrary leading dims but last dim = d_model; flatten middle dims to S_total
// q_linear1, k_linear1, v_linear1: [D,D]; out_weight: [D,D]; out_bias: [D]

torch::Tensor forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor q_linear1,
    torch::Tensor k_linear1,
    torch::Tensor v_linear1,
    torch::Tensor out_weight,
    torch::Tensor out_bias,
    int d_model,
    c10::optional<torch::Tensor> mask_opt = c10::nullopt,
    c10::optional<torch::Tensor> /*dropout_opt*/ = c10::nullopt,
    int heads = 4)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA/HIP tensors");
    TORCH_CHECK(q_linear1.is_cuda() && k_linear1.is_cuda() && v_linear1.is_cuda(), "proj weights must be CUDA/HIP tensors");
    TORCH_CHECK(out_weight.is_cuda() && out_bias.is_cuda(), "out weight/bias must be CUDA/HIP tensors");

    auto q0 = q.contiguous();
    auto k0 = k.contiguous();
    auto v0 = v.contiguous();

    TORCH_CHECK(q0.size(-1) == d_model, "Last dim of q must equal d_model");
    TORCH_CHECK(k0.size(-1) == d_model && v0.size(-1) == d_model, "Last dim of k/v must equal d_model");

    const int64_t B = q0.size(0);
    const int64_t D = d_model;

    // The Python logic uses seq_len = q.size(1) prior to flattening
    int64_t seq_len_logic = (q0.dim() >= 2) ? q0.size(1) : 0;

    // Flatten all middle dims into S_total
    int64_t elems = q0.numel();
    TORCH_CHECK(elems % (B * D) == 0, "Invalid q shape for flattening");
    int64_t S_total = elems / (B * D);

    auto options = q0.options();

    // Dynamic heads selection matching effective Python condition tree
    int H;
    if (seq_len_logic > 230) {
        H = 8;
    } else if (seq_len_logic <= 138 && seq_len_logic > 0) {
        H = 2;
    } else {
        H = heads;
    }
    TORCH_CHECK(D % H == 0, "d_model must be divisible by number of heads");
    int d_k = (int)(D / H);

    // Specialize the task's hot path.  The fallback below retains the full
    // original behavior for masks, other dtypes, and general dimensions.
    const bool no_mask = !mask_opt.has_value() || !mask_opt.value().defined();
    const bool same_shape = k0.numel() == elems && v0.numel() == elems;
    const bool all_float =
        q0.scalar_type() == at::kFloat && k0.scalar_type() == at::kFloat &&
        v0.scalar_type() == at::kFloat &&
        q_linear1.scalar_type() == at::kFloat &&
        k_linear1.scalar_type() == at::kFloat &&
        v_linear1.scalar_type() == at::kFloat &&
        out_weight.scalar_type() == at::kFloat &&
        out_bias.scalar_type() == at::kFloat;
    if (D == 4 && H == 2 && S_total > 0 && S_total <= 64 && no_mask &&
        same_shape && all_float) {
        auto qW = q_linear1.contiguous();
        auto kW = k_linear1.contiguous();
        auto vW = v_linear1.contiguous();
        auto ow = out_weight.contiguous();
        auto ob = out_bias.contiguous();
        auto output = at::empty({B, S_total, D}, options);
        hipStream_t stream = at::hip::getCurrentHIPStream();
        hipLaunchKernelGGL(fused_mha_d4_h2_kernel, dim3(B * S_total), dim3(128),
                           0, stream,
                           q0.data_ptr<float>(), k0.data_ptr<float>(),
                           v0.data_ptr<float>(), qW.data_ptr<float>(),
                           kW.data_ptr<float>(), vW.data_ptr<float>(),
                           ow.data_ptr<float>(), ob.data_ptr<float>(),
                           output.data_ptr<float>(), (int)S_total);
        return output;
    }

    // Views as [B,S,D]
    auto q_bsd = q0.view({B, S_total, D});
    auto k_bsd = k0.view({B, S_total, D});
    auto v_bsd = v0.view({B, S_total, D});

    // Projections via rocBLAS-backed addmm: [B*S, D] @ [D, D]
    auto q2d = q_bsd.reshape({B * S_total, D});
    auto k2d = k_bsd.reshape({B * S_total, D});
    auto v2d = v_bsd.reshape({B * S_total, D});

    auto qW = q_linear1.contiguous();
    auto kW = k_linear1.contiguous();
    auto vW = v_linear1.contiguous();

    // out = X @ W (no bias), shape [B*S, D]
    auto q_proj2d = at::matmul(q2d, qW);
    auto k_proj2d = at::matmul(k2d, kW);
    auto v_proj2d = at::matmul(v2d, vW);

    auto q_proj = q_proj2d.view({B, S_total, D});
    auto k_proj = k_proj2d.view({B, S_total, D});
    auto v_proj = v_proj2d.view({B, S_total, D});

    // Reshape to heads and move to [B,H,S,d_k]
    auto q_bhsd = q_proj.view({B, S_total, H, d_k}).permute({0, 2, 1, 3}).contiguous();
    auto k_bhsd = k_proj.view({B, S_total, H, d_k}).permute({0, 2, 1, 3}).contiguous();
    auto v_bhsd = v_proj.view({B, S_total, H, d_k}).permute({0, 2, 1, 3}).contiguous();

    // Prepare for batched GEMMs: merge B and H -> [BH, S, d_k]
    auto BH = B * H;
    auto q_bhsd_3d = q_bhsd.view({BH, (int64_t)S_total, d_k});
    auto k_bhsd_3d = k_bhsd.view({BH, (int64_t)S_total, d_k});
    auto v_bhsd_3d = v_bhsd.view({BH, (int64_t)S_total, d_k});

    // Scores = (Q) @ (K^T) / sqrt(d_k)  -> [BH, S, S]
    auto k_t = k_bhsd_3d.transpose(1, 2);
    auto scores = at::bmm(q_bhsd_3d, k_t);
    const double scale = 1.0 / std::sqrt((double)std::max(1, d_k));
    scores.mul_(scores.scalar_type() == at::kFloat ? (float)scale : (double)scale);

    // Apply mask broadcast over H if provided: mask [B,S,S] (from Python: mask.unsqueeze(1))
    if (mask_opt.has_value() && mask_opt.value().defined()) {
        auto mask = mask_opt.value().contiguous();
        TORCH_CHECK(mask.dim() == 3 && mask.size(0) == B && mask.size(1) == S_total && mask.size(2) == S_total,
                    "mask must be [B,S,S]");
        auto scores4d = scores.view({B, H, (int64_t)S_total, (int64_t)S_total}).contiguous();
        const long long total = (long long)B * H * S_total * S_total;
        const int threads = 256;
        dim3 grid(div_up((int)total, threads));
        hipStream_t stream = at::hip::getCurrentHIPStream();
        AT_DISPATCH_FLOATING_TYPES(scores4d.scalar_type(), "apply_mask_kernel", [&](){
            hipLaunchKernelGGL(HIP_KERNEL_NAME(apply_mask_kernel<scalar_t>), grid, threads, 0, stream,
                               scores4d.data_ptr<scalar_t>(), mask.data_ptr<uint8_t>(), (int)B, (int)H, (int)S_total);
        });
        scores = scores4d.view({BH, (int64_t)S_total, (int64_t)S_total});
    }

    // Softmax along last dim
    scores = at::softmax(scores, -1);

    // attn = scores @ V  -> [BH, S, d_k]
    auto attn_3d = at::bmm(scores, v_bhsd_3d);

    // Restore [B,H,S,d_k]
    auto attn_bhsd = attn_3d.view({B, H, (int64_t)S_total, d_k});

    // Concatenate heads: [B,S,H,d_k] -> [B,S,D]
    auto concat = attn_bhsd.permute({0, 2, 1, 3}).contiguous().view({B, (int64_t)S_total, D});

    // Output projection: F.linear(concat, out_weight, out_bias)
    auto ow = out_weight.contiguous();
    auto ob = out_bias.contiguous();
    auto out2d = concat.view({B * S_total, D});
    // out = out2d @ ow.T + ob
    auto output = at::addmm(ob, out2d, ow.t(), 1.0, 1.0).view({B, (int64_t)S_total, D});

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MultiHeadAttention HIP forward (ROCm, optimized BLAS path)");
}
