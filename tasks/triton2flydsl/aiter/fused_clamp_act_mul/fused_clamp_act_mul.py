# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone fused SwiGLU-clamp + act(gate)*up Triton kernel (GPT-OSS path).

Provenance: ported from aiter.ops.triton.fusions.fused_clamp_act_mul
(`fused_clamp_act_mul`) and its device kernel `_fused_clamp_silu_mul_kernel`
(aiter.ops.triton._triton_kernels.fusions.fused_clamp_act_mul). The optional FP8
group-quant / ue8m0 (MXFP8) output branches and their `_fp8_quant_op` import,
plus `AiterTritonLogger`, are dropped so the module depends only on `triton` +
`torch`; the `_apply_activation_from_str` activation helpers and the
constexpr-aware `make_kernel_repr` are inlined.

Op (non-quant clamped-SwiGLU, used by GPT-OSS / DeepSeek-V4-style FFNs):
    inp is [M, 2*N] with gate in the first N cols and up in the second N (same as
    chunk(2, dim=-1)). When swiglu_limit > 0: gate = min(gate, limit),
    up = clamp(up, -limit, limit). Then out = act(gate) * up, optionally scaled by
    per-row / per-element weights. Compute is fp32, output written in inp.dtype.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Inlined activation helpers (from _triton_kernels.activation)
# ---------------------------------------------------------------------------
@triton.jit
def _silu_exp2(x):
    return x / (1.0 + tl.exp2(-(x * 1.44269504089)))


@triton.jit
def _silu(x):
    return _silu_exp2(x)


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _gelu(x):
    M_SQRT1_2 = 0.70710678118654752440
    ALPHA = M_SQRT1_2
    return 0.5 * x * (1.0 + tl.erf(x * ALPHA))


@triton.jit
def _gelu_tanh(x):
    M_SQRT2 = 1.41421356237309504880
    M_2_SQRTPI = 1.12837916709551257390
    BETA = M_SQRT2 * M_2_SQRTPI * 0.5
    KAPPA = 0.044715
    x_cube = x * x * x
    inner = BETA * (x + KAPPA * x_cube)
    return 0.5 * x * (1.0 + _tanh(inner))


@triton.jit
def _relu(x):
    return tl.maximum(0.0, x)


@triton.jit
def _apply_activation_from_str(x, activation: tl.constexpr):
    if activation == "gelu":
        return _gelu(x)
    elif activation == "gelu_tanh":
        return _gelu_tanh(x)
    elif activation == "silu":
        return _silu(x)
    elif activation == "silu_exp2":
        return _silu_exp2(x)
    elif activation == "relu":
        return _relu(x)
    else:
        return x  # No activation if it is not recognized


# ---------------------------------------------------------------------------
# Inlined constexpr-aware kernel naming (from utils/_triton/kernel_repr.py)
# ---------------------------------------------------------------------------
def _sanitize_constexpr_value(value):
    if value is None:
        return "NONE"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return cleaned.upper() if cleaned else "NONE"


def make_kernel_repr(base_name, config_keys):
    def _repr(specialization):
        constants = specialization.constants
        parts = [
            f"{key}_{_sanitize_constexpr_value(constants.get(key, None))}"
            for key in config_keys
        ]
        return f"{base_name}_{'_'.join(parts)}" if parts else base_name

    return _repr


_fused_clamp_silu_mul_repr = make_kernel_repr(
    "_fused_clamp_silu_mul_kernel",
    [
        "BLOCK_SIZE_N",
        "HAVE_WEIGHTS",
        "WEIGHT_BROADCAST",
        "HAVE_SWIGLU_CLAMP",
    ],
)


@triton.jit(repr=_fused_clamp_silu_mul_repr)
def _fused_clamp_silu_mul_kernel(
    inp_ptr,
    out_ptr,
    weights_ptr,
    M,
    n_half,
    inp_stride_m,
    inp_stride_n,
    out_stride_m,
    out_stride_n,
    weights_stride_m,
    weights_stride_n,
    swiglu_limit,
    BLOCK_SIZE_N: tl.constexpr,
    HAVE_WEIGHTS: tl.constexpr,
    WEIGHT_BROADCAST: tl.constexpr,
    HAVE_SWIGLU_CLAMP: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    m_pid = tl.program_id(0)
    n_offs = tl.arange(0, BLOCK_SIZE_N)
    mask = n_offs < n_half

    gate = tl.load(
        inp_ptr + m_pid * inp_stride_m + n_offs * inp_stride_n,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    up = tl.load(
        inp_ptr + m_pid * inp_stride_m + (n_half + n_offs) * inp_stride_n,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)

    if HAVE_SWIGLU_CLAMP:
        up = tl.clamp(up, -swiglu_limit, swiglu_limit)
        gate = tl.minimum(gate, swiglu_limit)

    out = _apply_activation_from_str(gate, ACTIVATION) * up

    if HAVE_WEIGHTS:
        if WEIGHT_BROADCAST:
            w = tl.load(weights_ptr + m_pid * weights_stride_m).to(tl.float32)
            out = out * w
        else:
            w = tl.load(
                weights_ptr + m_pid * weights_stride_m + n_offs * weights_stride_n,
                mask=mask,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            out = out * w

    tl.store(
        out_ptr + m_pid * out_stride_m + n_offs * out_stride_n,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def fused_clamp_act_mul(
    inp: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    swiglu_limit: float = 0,
    activation: Literal["silu", "gelu", "gelu_tanh"] = "silu",
    weights: Optional[torch.Tensor] = None,
):
    """
    Fused clamp (SwiGLU-style) + act(gate) * up + optional weights (non-quant path).

    Args:
        inp: ``[M, D]`` with ``D = 2 * N``, contiguous; first ``N`` columns are gate,
            second ``N`` are up (same as ``chunk(2, dim=-1)`` on gate-up GEMM output).
        out: pre-allocated ``[M, N]`` output tensor. If ``None``, allocated internally
            with dtype = ``inp.dtype``.
        swiglu_limit: if ``> 0``, apply reference clamps; if ``<= 0``, skip clamping.
        activation: activation applied to the (clamped) gate before the up multiply.
        weights: optional ``[M, 1]`` (broadcast) or ``[M, N]`` row weights, multiplied
            into ``act(gate) * up`` (same as reference ``weights * x``).

    Constraints:
        ``N >= 128`` and ``N % 128 == 0``. The column block is the next power of
        two of ``N`` and masks the tail, so ``N`` itself need not be a power of two.
    """
    assert inp.dim() == 2
    M, D = inp.shape
    assert D % 2 == 0
    n_half = D // 2

    if out is None:
        out = torch.empty((M, n_half), dtype=inp.dtype, device=inp.device)
    else:
        assert out.shape == (M, n_half)

    assert n_half >= 128
    assert n_half % 128 == 0

    BLOCK_SIZE_N = triton.next_power_of_2(n_half)

    HAVE_WEIGHTS = weights is not None
    if HAVE_WEIGHTS:
        assert weights.is_cuda and weights.is_contiguous()
        assert weights.shape[0] == M
        if weights.shape[1] == 1:
            WEIGHT_BROADCAST = True
        else:
            assert weights.shape[1] == n_half
            WEIGHT_BROADCAST = False
    else:
        WEIGHT_BROADCAST = False

    if BLOCK_SIZE_N <= 512:
        num_warps = 1
    elif BLOCK_SIZE_N <= 2048:
        num_warps = 4
    else:
        num_warps = 8

    HAVE_SWIGLU_CLAMP = swiglu_limit > 0

    _fused_clamp_silu_mul_kernel[(M,)](
        inp,
        out,
        weights if HAVE_WEIGHTS else inp,
        M,
        n_half,
        inp.stride(0),
        inp.stride(1),
        out.stride(0),
        out.stride(1),
        weights.stride(0) if HAVE_WEIGHTS else 0,
        weights.stride(1) if HAVE_WEIGHTS else 0,
        swiglu_limit,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        HAVE_WEIGHTS=HAVE_WEIGHTS,
        WEIGHT_BROADCAST=WEIGHT_BROADCAST,
        HAVE_SWIGLU_CLAMP=HAVE_SWIGLU_CLAMP,
        ACTIVATION=activation,
        num_warps=num_warps,
    )

    return out
