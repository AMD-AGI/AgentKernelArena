"""Routing and MXFP4 expert weights for the MXFP4 fused MoE task.

The router draws standard-normal logits over all experts and keeps the top-k softmax probabilities
(not renormalized), so every token picks top_k distinct experts. Expert weights are drawn in bfloat16
at 1/sqrt(fan-in) scale and quantized to MXFP4 with the operator's own rule (mxfp4_numerics.py), one
expert at a time: w1 holds [gate | up] along its rows, w2 is the down projection. Weights and scales
are stored in the plain row-major layout the kernels read (no preshuffle).
"""
import math

import torch

from mxfp4_numerics import quantize


def _quantized_experts(num_experts, rows, cols, generator, device):
    codes = torch.empty((num_experts, rows, cols // 2), dtype=torch.uint8, device=device)
    scales = torch.empty((num_experts, rows, cols // 32), dtype=torch.uint8, device=device)
    for expert in range(num_experts):
        weight = torch.randn((rows, cols), generator=generator, device=device, dtype=torch.float32)
        codes[expert], scales[expert] = quantize((weight / math.sqrt(cols)).to(torch.bfloat16))
    return codes, scales


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    num_tokens, num_experts, top_k = axes["num_tokens"], axes["num_experts"], axes["top_k"]
    hidden_size, intermediate_size = axes["hidden_size"], axes["intermediate_size"]
    logits = torch.randn((num_tokens, num_experts), generator=generator, device=device, dtype=torch.float32)
    topk_weights, topk_ids = logits.softmax(dim=-1).topk(top_k, dim=-1)
    w1, w1_scale = _quantized_experts(num_experts, 2 * intermediate_size, hidden_size, generator, device)
    w2, w2_scale = _quantized_experts(num_experts, hidden_size, intermediate_size, generator, device)
    return {"w1": w1, "w1_scale": w1_scale, "w2": w2, "w2_scale": w2_scale,
            "topk_weights": topk_weights.contiguous(), "topk_ids": topk_ids.to(torch.int32).contiguous()}
