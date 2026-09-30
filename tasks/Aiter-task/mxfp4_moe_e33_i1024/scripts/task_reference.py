# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Correctness reference for this workload, copied from the schema bundle.

Everything below this docstring is the definition's ``reference`` callback, taken
without edit from schema v2 so that this task and the acceptance run that
verifies its result apply one implementation rather than two that agree today.
Do not edit it here: change it in the bundle and copy it down again, or the two
silently diverge -- which is exactly the failure this file exists to prevent.

``run`` is the entry point the bundle exports.
"""

from __future__ import annotations
from typing import Optional as Optional
import torch as torch

def _apply_gated_activation(x: torch.Tensor, activation: int) -> torch.Tensor:
    """SwiGLU-style gate/up split; w1 stores [gate | up] along its rows."""
    gate, up = x.chunk(2, dim=-1)
    activation = getattr(activation, "value", activation)
    if activation == 1:
        gate = torch.nn.functional.gelu(gate)
    elif activation in (None, 0):
        gate = torch.nn.functional.silu(gate)
    else:
        raise ValueError(f"unsupported MoE activation: {activation}")
    return gate * up


def _e8m0_to_f32(scale: torch.Tensor) -> torch.Tensor:
    """Biased exponent byte -> fp32 power of two."""
    return scale.view(torch.float8_e8m0fnu).to(torch.float32)


def _n_ones(n: int) -> int:
    return (1 << n) - 1


def _floatx_unpacked_to_f32(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """Decode sub-byte codes from the low bits of uint8 into FP32."""
    EBITS_F32, MBITS_F32 = 8, 23
    F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)

    assert x.dtype == torch.uint8
    assert 1 + ebits + mbits <= 8

    sign_mask = 1 << (ebits + mbits)
    exp_bias = _n_ones(ebits - 1)
    mantissa_mask = _n_ones(mbits)

    sign_lp = x & sign_mask

    x_pos = x ^ sign_lp

    zero_mask = x_pos == 0

    denormal_mask = torch.logical_and((x_pos > 0), ((x_pos >> mbits) == 0))

    # Rebuild the FP32 exponent and mantissa.
    exp_biased_lp = x_pos >> mbits
    exp_biased_f32 = exp_biased_lp - exp_bias + F32_EXP_BIAS
    exp_biased_f32 = exp_biased_f32.to(torch.int32) << MBITS_F32

    mantissa_lp_int32 = (x_pos & mantissa_mask).to(torch.int32)
    mantissa_f32 = mantissa_lp_int32 << (MBITS_F32 - mbits)
    result = exp_biased_f32 | mantissa_f32

    result[zero_mask] = 0

    denormal_exp_biased = 1 - exp_bias + F32_EXP_BIAS

    if mbits == 1:
        result[denormal_mask] = (denormal_exp_biased - mbits) << MBITS_F32

    else:
        # Normalize each subnormal mantissa before inserting its exponent.
        for i in range(mbits):
            for mantissa_cmp in range(1 << i, 1 << (i + 1)):
                left_shift = mbits - i
                mantissa_f32 = (mantissa_cmp - (1 << i)) << (
                    left_shift + MBITS_F32 - mbits
                )
                exp_biased_f32 = (denormal_exp_biased - left_shift) << MBITS_F32

                # Addition supports mixed SymInt/int operands in torch.compile.
                mantissa_lp_int32[mantissa_lp_int32 == mantissa_cmp] = (
                    exp_biased_f32 + mantissa_f32
                )

        result = torch.where(denormal_mask, mantissa_lp_int32, result)

    sign_f32 = sign_lp.to(torch.int32) << (MBITS_F32 - mbits + EBITS_F32 - ebits)
    result = result | sign_f32

    return result.view(torch.float)


def _mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    """Unpack two e2m1 values per byte, low nibble first."""
    x = x.view(torch.uint8).contiguous()
    shape = x.shape
    first_elements = (x & 0b1111).to(torch.uint8)
    second_elements = (x >> 4).to(torch.uint8)
    unpacked = torch.stack([first_elements, second_elements], dim=-1).view(
        *shape[:-1], shape[-1] * 2
    )
    return _floatx_unpacked_to_f32(unpacked, ebits=2, mbits=1)


def _unshuffle_scale(scale: torch.Tensor) -> torch.Tensor:
    """Invert the unpadded per-expert [N, K//32] shuffle_scale layout."""
    dtype = scale.dtype
    if dtype == getattr(torch, "float8_e8m0fnu", None):
        scale = scale.view(torch.uint8)
    num_experts, rows, cols = scale.shape
    if rows % 256 != 0 or cols % 8 != 0:
        raise ValueError(
            f"unexpected mxfp4 scale layout: N={rows}, K//32={cols}. "
            "shuffle_scale pads N to a multiple of 256 and K//32 to a "
            "multiple of 8; un-padding is not implemented."
        )
    unshuffled = scale.view(num_experts, rows // 32, cols // 8, 4, 16, 2, 2)
    unshuffled = unshuffled.permute(0, 1, 6, 4, 2, 5, 3).contiguous()
    return unshuffled.view(*scale.shape).view(dtype)


def _unshuffle_weight(weight: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
    """Invert the generic, non-guinterleaved ``shuffle_weight`` layout."""
    dtype = weight.dtype
    if dtype == getattr(torch, "float4_e2m1fn_x2", None):
        weight = weight.view(torch.uint8)
    block_n, block_k = layout
    block_k *= 2
    vector_size = 16 // weight.element_size()
    rows, cols = weight.shape[-2:]
    batch = weight.numel() // (rows * cols)
    unshuffled = weight.view(
        batch,
        rows // block_n,
        cols // block_k,
        block_k // vector_size,
        block_n,
        vector_size,
    )
    unshuffled = unshuffled.permute(0, 1, 4, 2, 3, 5).contiguous()
    return unshuffled.view(*weight.shape).view(dtype)


def _dequantize_weight(
    weight: torch.Tensor, scale: torch.Tensor | None, block: int = 32
) -> torch.Tensor:
    """Unshuffle weights/scales, then cast or MXFP4-dequantize to fp32."""
    weight = _unshuffle_weight(weight)
    if scale is None:
        return weight.to(torch.float32)
    weight = _mxfp4_to_f32(weight)
    scale = _e8m0_to_f32(_unshuffle_scale(scale))
    if scale.shape[-1] * block != weight.shape[-1]:
        raise ValueError(
            "unexpected mxfp4 scale layout: "
            f"weight K={weight.shape[-1]}, scale K={scale.shape[-1]}, block={block}."
        )
    return weight * scale.repeat_interleave(block, dim=-1)


def _f32_to_floatx_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """Convert FP32 to saturated sub-byte codes with round-to-nearest-even."""

    # FP4 conversion and packing adapted from torchao 0.17.0.
    EBITS_F32, MBITS_F32 = 8, 23
    F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)

    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))

    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (F32_EXP_BIAS - exp_bias) + (MBITS_F32 - mbits) + 1
    denorm_mask_int = denorm_exp << MBITS_F32

    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(
        torch.float32
    )

    # CPU bit shifts require int32 rather than uint32.
    x = x.view(torch.int32)
    sign = x & 0x80000000

    x = x ^ sign

    x = x.view(torch.float)

    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    # Adding the exponent offset rounds subnormals to nearest-even.
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    # Adjust the exponent and round the retained mantissa to nearest-even.
    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Discard sign extension from the signed right shift.
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


def _quantize_mxfp4(
    x: torch.Tensor, block: int = 32, min_amax: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack nearest-even E2M1 values with upward-rounded E8M0 block scales."""
    values = x.to(torch.float32).unflatten(-1, (-1, block))
    amax = values.abs().amax(dim=-1, keepdim=True)
    is_finite = torch.isfinite(amax)
    values = torch.where(is_finite, values, 0.0)
    amax = torch.where(is_finite, amax, 0.0).clamp_min(min_amax)
    # frexp preserves subnormal maxima; 6 = 0.75 * 2**3.
    mantissa, exponent = torch.frexp(amax)
    exponent = exponent - 3 + (mantissa > 0.75).to(torch.int32)
    exponent = torch.where(amax == 0, -127, exponent).clamp(-127, 127)
    biased_exponent = exponent + 127
    scaled = values * torch.exp2(exponent.neg().to(torch.float32))
    codes = _f32_to_floatx_unpacked(scaled, ebits=2, mbits=1).flatten(-2)
    shape = codes.shape
    assert shape[-1] % 2 == 0
    codes = codes.contiguous().view(-1)
    packed = (codes[::2] | codes[1::2] << 4).view(*shape[:-1], shape[-1] // 2)
    scales = torch.where(is_finite, biased_exponent, 0xFF).to(torch.uint8).squeeze(-1)
    return packed, scales


def _quantize_activation(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Round-trip MXFP4 with a 1e-10 amax floor and NaN propagation."""
    packed, scales = _quantize_mxfp4(x, block, min_amax=1e-10)
    return _mxfp4_to_f32(packed) * _e8m0_to_f32(scales).repeat_interleave(block, dim=-1)


def _fused_moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: int = 0,
    doweight_stage1: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute FP32 MoE with quantized activation inputs when scales exist."""
    if (w1_scale is None) != (w2_scale is None):
        raise ValueError("both MoE weight scales must be present or absent")
    is_quantized = w1_scale is not None
    w1 = _dequantize_weight(w1, w1_scale)
    w2 = _dequantize_weight(w2, w2_scale)
    num_tokens, model_dim = hidden_states.shape
    topk = topk_ids.shape[1]
    activations = (
        _quantize_activation(hidden_states)
        if is_quantized
        else hidden_states.to(torch.float32)
    )
    routing_weights = topk_weights.to(torch.float32)
    out = torch.zeros(
        (num_tokens, topk, model_dim), dtype=torch.float32, device=activations.device
    )
    for expert_id in range(w1.shape[0]):
        mask = topk_ids == expert_id
        if not mask.any():
            continue
        token_indices = mask.nonzero(as_tuple=True)[0]
        projected = activations[token_indices] @ w1[expert_id].transpose(0, 1)
        if doweight_stage1:
            # Stage-1 routing weights apply before the gated activation.
            projected = projected * routing_weights[mask].unsqueeze(-1)
        intermediate = _apply_gated_activation(projected, activation)
        if is_quantized:
            intermediate = _quantize_activation(intermediate.to(hidden_states.dtype))
        out[mask] = intermediate @ w2[expert_id].transpose(0, 1)
    if not doweight_stage1:
        out = out * routing_weights.view(num_tokens, topk, 1)
    return out.sum(dim=1).to(hidden_states.dtype)


_callable = _fused_moe_reference


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
