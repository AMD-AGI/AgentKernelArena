"""Int4 group-quantized expert weights and router decisions of a Kimi-K2-style top-8 layer over 384 experts.

Weights. Each expert matrix is drawn normal with standard deviation 1/sqrt(K) of its reduction and quantized
symmetrically to int4 without a zero point (vLLM's uint4b8 type, as in vLLM's quantize_weights test helper and
GPTQ/compressed-tensors symmetric checkpoints), group size 32 along K: per group the scale is
max(max / 7, -min / 8), stored in bfloat16; the code is round(w / scale) clamped to [-8, 7] and stored with
offset 8 as an unsigned nibble, two nibbles per byte along K with the even K element in the low nibble. That is
vLLM's N-first uint8 layout for the Triton WNA16 kernel: qweight [E, N, K // 2] and scale [E, N, K // 32]. The
weight value is (nibble - 8) * scale. Every code from -8 to 7 occurs.

Routing. Every token selects 8 distinct experts uniformly (a stand-in for the bias-corrected top-8 selection of a
DeepSeek-V3-style router); its routing weights are sigmoid scores of the selected experts renormalized to sum to
one, as the noaux_tc router with norm_topk_prob does.
"""
import math

import torch

_CHUNK = 16


def _quantize(weight, group_size):
    experts, rows, k = weight.shape
    groups = weight.view(experts, rows, k // group_size, group_size)
    scale = torch.maximum(groups.amax(dim=-1) / 7.0, groups.amin(dim=-1) / -8.0)
    scale = scale.clamp_min(1.0e-8).to(torch.bfloat16)
    codes = torch.round(groups / scale.float().unsqueeze(-1)).clamp(-8, 7).to(torch.int16) + 8
    codes = codes.view(experts, rows, k // 2, 2).to(torch.uint8)
    return codes[..., 0] | (codes[..., 1] << 4), scale


def _quantized_bank(experts, rows, k, group_size, generator, device):
    qweight = torch.empty((experts, rows, k // 2), dtype=torch.uint8, device=device)
    scale = torch.empty((experts, rows, k // group_size), dtype=torch.bfloat16, device=device)
    for start in range(0, experts, _CHUNK):
        stop = min(start + _CHUNK, experts)
        weight = torch.randn((stop - start, rows, k), generator=generator, device=device, dtype=torch.float32)
        qweight[start:stop], scale[start:stop] = _quantize(weight.div_(math.sqrt(k)), group_size)
    return qweight, scale


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    num_tokens, top_k, num_experts = axes["num_tokens"], axes["top_k"], axes["num_experts"]
    hidden, inter, group_size = axes["hidden_size"], axes["intermediate_size"], axes["group_size"]
    w13_qweight, w13_scale = _quantized_bank(num_experts, 2 * inter, hidden, group_size, generator, device)
    w2_qweight, w2_scale = _quantized_bank(num_experts, hidden, inter, group_size, generator, device)
    keys = torch.rand((num_tokens, num_experts), generator=generator, device=device)
    ids = keys.topk(top_k, dim=-1).indices
    scores = torch.sigmoid(torch.randn((num_tokens, top_k), generator=generator, device=device, dtype=torch.float32))
    return {
        "w13_qweight": w13_qweight, "w13_scale": w13_scale, "w2_qweight": w2_qweight, "w2_scale": w2_scale,
        "topk_weights": scores / scores.sum(dim=-1, keepdim=True), "topk_ids": ids.to(torch.int32),
    }
