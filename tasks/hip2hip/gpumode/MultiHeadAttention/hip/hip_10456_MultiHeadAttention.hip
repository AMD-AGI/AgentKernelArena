// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
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
        AT_DISPATCH_FLOATING_TYPES(scores4d.scalar_type(), "apply_mask_kernel", [&](){
            hipLaunchKernelGGL(HIP_KERNEL_NAME(apply_mask_kernel<scalar_t>), grid, threads, 0, 0,
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

