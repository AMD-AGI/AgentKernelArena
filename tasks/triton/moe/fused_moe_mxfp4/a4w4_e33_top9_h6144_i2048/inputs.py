"""MXFP4 expert weights and a softmax top-k router for the fused MXFP4 MoE.

Expert weights are drawn normal with std 1 / sqrt(fan-in), rounded to bfloat16 and quantized to MXFP4
along the reduction axis as the old task's initializer did: the scale of a 1x32 group is the smallest
power of two 2^s with amax <= 6 * 2^s (AITER's default RoundUp scale), and each value divided by it
rounds to the nearest e2m1 value with ties to the even code. Two codes share a byte, the even index in
the low nibble; scales are e8m0 bytes (value 2^(byte - 127)). The router draws normal logits per token
over all experts and takes the top_k of their softmax: distinct expert ids and their (not renormalized)
probabilities.
"""
import math

import torch

_GROUP = 32
# Decision boundaries between neighbouring e2m1 magnitudes (0, .5, 1, 1.5, 2, 3, 4, 6); a value exactly
# on a boundary goes to the even code.
_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (False, True, False, True, False, True, False)
_ELEMENTS_PER_CHUNK = 1 << 25


def _encode_e2m1(scaled):
    magnitude = scaled.abs()
    code = torch.zeros(scaled.shape, dtype=torch.uint8, device=scaled.device)
    for bound, up in zip(_BOUNDS, _TIE_UP):
        code += ((magnitude >= bound) if up else (magnitude > bound)).to(torch.uint8)
    return code | ((scaled < 0).to(torch.uint8) << 3)


def _quantize_mxfp4(values):
    """values [..., k] float32 -> (codes [..., k // 2] uint8, e8m0 scales [..., k // 32] uint8)."""
    *lead, k = values.shape
    groups = values.reshape(*lead, k // _GROUP, _GROUP)
    amax = groups.abs().amax(dim=-1, keepdim=True)
    mantissa, exponent = torch.frexp(amax)
    exponent = exponent - 3 + (mantissa > 0.75).to(exponent.dtype)
    exponent = torch.where(amax > 0, exponent, torch.full_like(exponent, -127)).clamp(-127, 127)
    scaled = groups * torch.exp2(-exponent.to(torch.float32))
    codes = _encode_e2m1(scaled).reshape(*lead, k)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous(), (exponent + 127).squeeze(-1).to(torch.uint8)


def _quantized_experts(experts, rows, cols, generator, device):
    """Normal [experts, rows, cols] weights with std 1 / sqrt(cols), MXFP4-quantized along cols."""
    codes = torch.empty(experts, rows, cols // 2, dtype=torch.uint8, device=device)
    scales = torch.empty(experts, rows, cols // _GROUP, dtype=torch.uint8, device=device)
    step = max(1, _ELEMENTS_PER_CHUNK // (rows * cols))
    std = 1.0 / math.sqrt(cols)
    for start in range(0, experts, step):
        stop = min(start + step, experts)
        values = torch.randn(stop - start, rows, cols, generator=generator, device=device, dtype=torch.float32)
        values = (values * std).to(torch.bfloat16).to(torch.float32)
        codes[start:stop], scales[start:stop] = _quantize_mxfp4(values)
    return codes, scales


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    tokens, experts, top_k = axes["num_tokens"], axes["num_experts"], axes["top_k"]
    hidden, inter = axes["hidden_size"], axes["intermediate_size"]
    w1, w1_scale = _quantized_experts(experts, 2 * inter, hidden, generator, device)
    w2, w2_scale = _quantized_experts(experts, hidden, inter, generator, device)
    logits = torch.randn(tokens, experts, generator=generator, device=device, dtype=torch.float32)
    topk_weights, topk_ids = logits.softmax(dim=-1).topk(top_k, dim=-1)
    return {"w1": w1, "w1_scale": w1_scale, "w2": w2, "w2_scale": w2_scale,
            "topk_weights": topk_weights.contiguous(), "topk_ids": topk_ids.to(torch.int32).contiguous()}
