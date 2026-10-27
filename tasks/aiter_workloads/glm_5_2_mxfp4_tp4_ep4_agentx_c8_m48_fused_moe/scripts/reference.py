# SPDX-License-Identifier: MIT
"""Protected PyTorch MXFP4 MoE oracle; imports neither AITER nor candidate code.

The E2M1 codebook and input E8M0 scale rule follow ROCm/AITER fp4_utils.py
and quant.py at 2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0 (MIT). The fused
FP32 intermediate and scale calculation follow mixed_moe_gemm_2stage_common.py
at the same pin (Apache-2.0, Copyright 2025-2026 FlyDSL Project Contributors).
The EP global-to-local map follows aiter.fused_moe.torch_moe.
"""
from __future__ import annotations

import math

from cases import local_expert_map


def scale_to_float(scale):
    import torch

    codes = scale.contiguous().view(torch.uint8).to(torch.int32)
    if (codes == 255).any():
        raise ValueError("Nonfinite E8M0 scale")
    bits = codes << 23
    bits[codes == 0] = 0x00400000
    return bits.view(torch.float32)


def dequantize_weight(packed, scale):
    import torch

    raw = packed.contiguous().view(torch.uint8)
    nibbles = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=raw.device)
    values = levels[(nibbles & 7).long()] * torch.where(nibbles < 8, 1.0, -1.0)
    scales = scale_to_float(scale).reshape(*values.shape[:-1], values.shape[-1] // 32)
    return (values.reshape(*scales.shape, 32) * scales.unsqueeze(-1)).reshape(values.shape)


def quantize_dequantize(values, *, fused_intermediate=False):
    """Pinned input or fused-stage1 E8M0 rule, then E2M1 nearest-even."""
    import torch

    shape = values.shape
    groups = values.float().reshape(-1, 32)
    if not torch.isfinite(groups).all():
        raise ValueError("Nonfinite reference quantizer input")
    maximum = groups.abs().amax(-1)
    if fused_intermediate:
        # The selected *_fp4 FlyDSL epilogue quantizes FP32 SiLU*up directly.
        # Its scale differs from ceil_pow2(amax/6), notably at amax=6*2**n.
        bits = maximum.contiguous().view(torch.int32)
        rounded = (bits + 0x400000) & -0x800000
        exponent = ((rounded >> 23) - 2).clamp(min=0)
    else:
        bits = (maximum / 6.0).contiguous().view(torch.int32)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).to(torch.int32)
    if (exponent >= 255).any():
        raise ValueError("Reference quantizer scale overflow")
    scale_bits = exponent << 23
    scale_bits[exponent == 0] = 0x00400000
    scale = scale_bits.view(torch.float32)
    normalized = groups / scale[:, None]
    magnitude = normalized.abs().contiguous()
    thresholds = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], device=values.device)
    code = torch.bucketize(magnitude, thresholds)
    ties = (code < 7) & (magnitude == thresholds[code.clamp(max=6)])
    code = code + (ties & ((code & 1) == 1)).to(code.dtype)
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=values.device)
    quantized = levels[code] * torch.where(normalized < 0, -1.0, 1.0)
    return (quantized * scale[:, None]).reshape(shape)


def reference(payload):
    import torch

    hidden, _, _, weights, ids = payload["args"]
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("FP32 reference requires TF32 to be disabled")
    mask = payload["kwargs"]["expert_mask"]
    mapping = torch.tensor(local_expert_map(mask.cpu().tolist()), dtype=torch.int64, device=ids.device)
    if (ids < 0).any() or (ids >= mapping.numel()).any():
        raise ValueError("Invalid global routing IDs")
    local_ids = mapping[ids.long()]
    m, hidden_width = hidden.shape
    topk = ids.shape[1]
    canonical = payload["reference"]
    experts, _, packed_intermediate = canonical["w2"].shape
    intermediate = packed_intermediate * 2
    hidden_q = quantize_dequantize(hidden)
    routes = torch.zeros((m, topk, hidden_width), dtype=torch.float32, device=hidden.device)
    for expert in range(experts):
        token, slot = torch.where(local_ids == expert)
        if not token.numel():
            continue
        scale1 = canonical["w1_scale"].reshape(experts, 2 * intermediate, hidden_width // 32)[expert]
        w1 = dequantize_weight(canonical["w1"][expert], scale1)
        gate, up = (hidden_q[token] @ w1.t()).split(intermediate, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
        a2 = quantize_dequantize(activated, fused_intermediate=True)
        scale2 = canonical["w2_scale"].reshape(experts, hidden_width, intermediate // 32)[expert]
        w2 = dequantize_weight(canonical["w2"][expert], scale2)
        routes[token, slot] = (a2 @ w2.t()) * weights[token, slot, None]
    return routes.sum(dim=1).to(torch.bfloat16)


def compare_output(actual, expected):
    """Upstream checked-call rule with the fixed tighter warning-level D gate."""
    import torch

    if (not isinstance(expected, torch.Tensor) or expected.dtype != torch.bfloat16
            or expected.ndim != 2 or not expected.numel() or not torch.isfinite(expected).all()):
        raise ValueError("Reference output is invalid")
    if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
            or actual.dtype != expected.dtype or actual.device != expected.device
            or not torch.isfinite(actual).all()):
        raise AssertionError("Output geometry, dtype, device or finite values differ")
    close = torch.isclose(expected, actual, rtol=.01, atol=.01)
    lhs, rhs = expected.double(), actual.double()
    denominator = (lhs.square() + rhs.square()).sum()
    dot = (lhs * rhs).sum()
    if not math.isfinite(denominator.item()) or not math.isfinite(dot.item()):
        raise ValueError("Nonfinite comparison reduction")
    difference = 0.0 if denominator.item() == 0 else (1 - 2 * dot / denominator).item()
    return {"accepted": bool(close.all()) or difference <= .001,
            "max_absolute_error": (lhs - rhs).abs().max().item(),
            "mismatch_count": int((~close).sum()), "normalized_difference": difference,
            "compared_elements": expected.numel()}


def check_output(actual, expected):
    result = compare_output(actual, expected)
    if not result["accepted"]:
        raise AssertionError(f"Output exceeds frozen whole-call tolerance: {result}")
    return result


def reference_controls(payload):
    import torch

    zero = {**payload, "args": (torch.zeros_like(payload["args"][0]),) + payload["args"][1:]}
    expected_zero = torch.zeros_like(payload["args"][0])
    if not torch.equal(reference(zero), expected_zero):
        raise AssertionError("Zero-input MoE identity failed")
    mask = payload["kwargs"]["expert_mask"]
    remote = int(torch.where(mask == 0)[0][0])
    remote_args = payload["args"][:4] + (torch.full_like(payload["args"][4], remote),)
    if not torch.equal(reference({**payload, "args": remote_args}), expected_zero):
        raise AssertionError("Nonresident experts contributed to the local result")
    wrong = expected_zero.clone()
    wrong[0, 0] = 1
    if compare_output(wrong, expected_zero)["accepted"]:
        raise AssertionError("Comparator accepted a nonzero output against zero")
    return {"zero_input": "PASS", "remote_experts_zero": "PASS", "wrong_output_rejected": "PASS"}
