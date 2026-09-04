# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Correctness reference for the MXFP4 fused-MoE workload.

Transcribed from the workload schema's ``reference`` solution for this operator
(``solutions/reference/moe/*/fused_moe_torch_reference_*.json``, entry point
``main.py::run``): a dense per-expert MoE in fp32 that un-shuffles and
dequantizes the stored MXFP4 weights itself, and quantizes both activations
because ``per_1x32`` is the afp4_wfp4 path.

Three documented divergences from the schema source, so a future comparison
against the bundle knows what to expect:

1. ``from typing import Optional`` is added. The schema source annotates the
   scale arguments with it and never imports it; that survives there only
   because ``from __future__ import annotations`` keeps annotations
   unevaluated.
2. The schema source's unused ``math`` and ``torch.nn.functional`` imports are
   dropped.
3. The activation quantizer is ``aiter.get_torch_quant(QuantType.per_1x32)``
   applied to a bf16 cast, not ``aiter.ops.triton.quant.dynamic_mxfp4_quant``.
   See :func:`_quantize_activation` for the measurement that decided it.

Divergence 3 is not cosmetic. The correctness gate is derived from how far the
production implementation sits from this reference, so a reference that
disagrees with the operator's own quantization inflates the baseline's apparent
error and, with it, the gate that admits a candidate.
"""

from __future__ import annotations

from typing import Optional

import torch


# ----- reference dependencies -----
def _unshuffle_weight(w: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
    """Invert aiter's ``shuffle_weight`` (generic, non-guinterleave) layout.

    aiter preshuffles MoE weights once at load time for the CK/asm gemm
    kernels (``aiter/ops/shuffle.py::shuffle_weight``), permuting
    ``[..., N, K] -> [..., N//BN, K//BK, BK//K, BN, K]`` (BN, BK derived from
    ``layout``) then flattening back to ``[..., N, K]``. Shape is unchanged,
    only element order is, so this is the exact inverse permutation.
    """
    x_type = w.dtype
    w = w.view(torch.uint8) if x_type == getattr(torch, "float4_e2m1fn_x2", None) else w
    IN, IK = layout
    BK = IK * 2
    K = 16 // w.element_size()
    BN = IN
    batch = w.numel() // (w.shape[-2] * w.shape[-1])
    w_ = w.view(batch, w.shape[-2] // BN, w.shape[-1] // BK, BK // K, BN, K)
    w_ = w_.permute(0, 1, 4, 2, 3, 5).contiguous()
    return w_.view(*w.shape).view(x_type)


def _unshuffle_scale(scale: torch.Tensor) -> torch.Tensor:
    """Invert aiter's ``shuffle_scale`` (generic, non-guinterleave) layout.

    Same idea as :func:`_unshuffle_weight` but for the per-block e8m0 scale,
    applied per expert on a ``[N, K//32]`` slice (``aiter/ops/shuffle.py::
    shuffle_scale``). Assumes N is a multiple of 256 and K//32 a multiple of
    8, so no padding was introduced on the way in -- true for every shape
    this schema declares.
    """
    x_type = scale.dtype
    scale = scale.view(torch.uint8) if x_type == getattr(torch, "float8_e8m0fnu", None) else scale
    num_experts, sm, sn = scale.shape
    if sm % 256 != 0 or sn % 8 != 0:
        raise ValueError(
            f"unexpected mxfp4 scale layout: N={sm}, K//32={sn}. "
            "shuffle_scale pads N to a multiple of 256 and K//32 to a "
            "multiple of 8; un-padding is not implemented."
        )
    s_ = scale.view(num_experts, sm // 32, sn // 8, 4, 16, 2, 2)
    s_ = s_.permute(0, 1, 6, 4, 2, 5, 3).contiguous()
    return s_.view(*scale.shape).view(x_type)


def _act_and_mul(x: torch.Tensor, activation: int) -> torch.Tensor:
    """SwiGLU-style gate/up split; w1 stores [gate | up] along its rows."""
    gate, up = x.chunk(2, dim=-1)
    if activation == 1:
        gate = torch.nn.functional.gelu(gate)
    else:
        gate = torch.nn.functional.silu(gate)
    return gate * up


def _mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    """Unpack two e2m1 values per byte, low nibble first."""
    lut = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    x = x.view(torch.uint8).repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    table = torch.tensor(lut, dtype=torch.float32, device=x.device)
    return table[x.long()]


def _e8m0_to_f32(scale: torch.Tensor) -> torch.Tensor:
    """Biased exponent byte -> fp32 power of two."""
    scale = scale.view(torch.uint8)
    zero_case = scale == 0
    nan_case = scale == 0xFF
    bits = scale.to(torch.int32) << 23
    bits[zero_case] = 0x00400000
    bits[nan_case] = 0x7F800001
    return bits.view(torch.float32)


def _to_f32(w: torch.Tensor, scale: Optional[torch.Tensor], block: int = 32) -> torch.Tensor:
    """Weights as fp32: plain cast, or mxfp4 dequant when scales are given.

    ``w`` (and ``scale``, if given) are assumed preshuffled -- that's the
    layout aiter's real fused_moe kernel requires, so it's what every call
    captured from a live model actually carries. Un-shuffle first.
    """
    w = _unshuffle_weight(w)
    if scale is None:
        return w.to(torch.float32)
    w = _mxfp4_to_f32(w)
    scale = _e8m0_to_f32(_unshuffle_scale(scale))
    if scale.shape[-1] * block != w.shape[-1]:
        raise ValueError(
            "unexpected mxfp4 scale layout: "
            f"weight K={w.shape[-1]}, scale K={scale.shape[-1]}, block={block}."
        )
    return w * scale.repeat_interleave(block, dim=-1)


# ----- reference -----
_ACTIVATION_QUANT = None


def _quantize_activation(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Round activations through mxfp4, the way the kernel does internally.

    ``per_1x32`` is a4w4, not a16w4: aiter's ``fused_moe`` leaves the caller's
    activations in bf16 and dynamic-quantizes them to MXFP4 *inside* the kernel
    (``aiter/fused_moe.py``: for a Silu + mxfp4-weight call ``q_dtype_a`` is
    ``fp4x2``). Computing the matmuls in fp32 would model a different operator.

    The quantizer is ``aiter.get_torch_quant(QuantType.per_1x32)`` -- the one
    aiter validates ``fused_moe`` against -- applied to a bf16 cast, because
    that is what the operator receives. The schema source instead calls
    ``aiter.ops.triton.quant.dynamic_mxfp4_quant``, which rounds the e8m0 block
    scale differently. Measured on this definition, distance between the
    production implementation and this reference:

        num_tokens                            64        128
        dynamic_mxfp4_quant               0.1457     0.1459
        get_torch_quant, no bf16 cast     0.0346     0.0035
        get_torch_quant, bf16 cast        0.0016     0.0328

    Only the last row is a property of the kernels rather than of a mismatched
    quantizer: it tracks which tuned stage-1 variant the M bucket selects (the
    ``kw2_fp4`` variants fuse the output quantization into the epilogue and sit
    an order of magnitude further out). The other two rows are the reference
    disagreeing with the operator's own quantization contract.

    Grouping and rounding are part of what the definition *is* -- the checkpoint
    declares ``input_tensors`` as fp4, dynamic, per-group-32, e8m0 -- not part
    of the kernel under test. Everything that is being checked --
    dequantization, the per-expert matmuls, the activation, the routing -- stays
    in this file.
    """
    global _ACTIVATION_QUANT
    if _ACTIVATION_QUANT is None:
        import aiter

        _ACTIVATION_QUANT = aiter.get_torch_quant(aiter.QuantType.per_1x32)

    from aiter import dtypes

    packed, scale = _ACTIVATION_QUANT(x.to(torch.bfloat16), quant_dtype=dtypes.fp4x2)
    values = _mxfp4_to_f32(packed)
    return values * _e8m0_to_f32(scale).repeat_interleave(block, dim=-1)


def _fused_moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    activation: int = 0,
    doweight_stage1: bool = False,
) -> torch.Tensor:
    """Dense per-expert MoE. ``w1``/``w2``/scales are assumed preshuffled
    (aiter's real weight layout, see ``_to_f32``); mxfp4 weights are
    un-shuffled and dequantized first, and both matmuls take mxfp4 activations
    (see ``_quantize_activation``) and accumulate in fp32."""
    w1 = _to_f32(w1, w1_scale)
    w2 = _to_f32(w2, w2_scale)
    num_tokens, model_dim = hidden_states.shape
    topk = topk_ids.shape[1]
    x = _quantize_activation(hidden_states)
    weight = topk_weights.to(torch.float32)
    out = torch.zeros((num_tokens, topk, model_dim), dtype=torch.float32, device=x.device)
    for expert in range(w1.shape[0]):
        mask = topk_ids == expert
        if not mask.any():
            continue
        rows = mask.nonzero(as_tuple=True)[0]
        act_out = _act_and_mul(x[rows] @ w1[expert].transpose(0, 1), activation)
        if doweight_stage1:
            act_out = act_out * weight[mask].unsqueeze(-1)
        out[mask] = _quantize_activation(act_out) @ w2[expert].transpose(0, 1)
    if not doweight_stage1:
        out = out * weight.view(num_tokens, topk, 1)
    return out.sum(dim=1).to(hidden_states.dtype)


# ----- entry point -----
def run(*args, **kwargs):
    return _fused_moe_reference(*args, **kwargs)
