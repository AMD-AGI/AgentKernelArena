# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Standalone Triton MXFP4 SageAttention v2 forward path (triton + torch only).
#
# Provenance: the @triton.jit attention + quant kernels (sage_fwd_mxfp4, the
# masking helpers, sage_quant_v_kernel, the rotation/delta kernels) are ported
# originally from aiter.ops.triton MXFP4 sage-attention; the host wrappers
# (sage_quant_mxfp4, fav3_sage_mxfp4_wrapper) and the mxfp4 downcast/upcast utils
# originate from the same module with helper dependencies inlined.
# Numerical repairs use nearest-even FP4 conversion, a two-component FP4
# expansion of Q/K, and a zero-safe V quantizer. The attention dot products
# remain E2M1/E8M0; the expansion increases internal work and storage.
#
# Dropped: the gluon / gfx1250 kernel path (one terse note). The block-sparse
# kernels are kept inlined.
from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple
import functools

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Inlined utils (arch detection, pid grid, fp8 dtype).
# ---------------------------------------------------------------------------
def _detect_arch_str() -> str:
    """arch string, e.g. 'gfx950' (inlined from arch_info.get_arch)."""
    try:
        return triton.runtime.driver.active.get_current_target().arch
    except Exception:  # noqa: BLE001 - import-only / no GPU
        return "unknown"


def get_arch() -> str:
    return _detect_arch_str()


# FP8 dtype selection: gfx950 / gfx12 use OCP e4m3 (e4m3fn); older CDNA uses e4m3fnuz.
if _detect_arch_str() in ("gfx950", "gfx1250", "gfx1200", "gfx1201"):
    fp8_dtype = torch.float8_e4m3fn
else:
    fp8_dtype = torch.float8_e4m3fnuz


def map_dims(shape, indices):
    return [shape[i] for i in indices]


@triton.jit
def pid_grid_3d(pid, num_pid_m, num_pid_n, num_pid_k):
    """Maps 1D pid to 3D grid coords (pid_m, pid_n, pid_k)."""
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n
    pid_k = pid // (num_pid_m * num_pid_n) % num_pid_k
    return pid_m, pid_n, pid_k


# ===========================================================================
# MXFP4 downcast / upcast (based on moe/quant_moe.py and its device kernels)
# ===========================================================================
@triton.jit
def _get_max_quant_val(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 6.0
    elif dtype == tl.float8e5:
        return 57344.0
    elif dtype == tl.float8e4nv:
        return 448.0
    else:
        tl.static_assert(False, f"Invalid {dtype=}")


@triton.jit
def _e2m1_rne_magnitude(magnitude):
    """E2M1 magnitude codes; midpoint ties select the even code."""
    return ((magnitude > 0.25).to(tl.uint32)
            + (magnitude >= 0.75).to(tl.uint32)
            + (magnitude > 1.25).to(tl.uint32)
            + (magnitude >= 1.75).to(tl.uint32)
            + (magnitude > 2.5).to(tl.uint32)
            + (magnitude >= 3.5).to(tl.uint32)
            + (magnitude > 5.0).to(tl.uint32))


@triton.jit
def _compute_mx_quant_and_scale(
    src_tensor,
    valid_src_mask,
    mx_tensor_dtype: tl.constexpr,
    DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr = 0,
):
    is_fp8: tl.constexpr = (
        mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    )
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // 32

    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(valid_src_mask, abs_tensor, -1.0)
    abs_tensor = tl.reshape(
        abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
    )
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
    if DEQUANT_SCALE_ROUNDING_MODE == 2:
        # E2M1 maximum power-of-two magnitude is 4. Zero blocks retain zero codes.
        maximum_exponent = ((max_val.to(tl.uint32, bitcast=True) >> 23) & 0xFF).to(tl.int32)
        scale_exponent = tl.maximum(0, maximum_exponent - 2)
        dequant_scale_exponent = scale_exponent.to(tl.uint32) << 23
        quant_scale = tl.exp2((127 - scale_exponent).to(tl.float32))
    elif DEQUANT_SCALE_ROUNDING_MODE == 0:
        dequant_scale_exponent = (
            dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF
        ) & 0x7F800000
    else:
        assert DEQUANT_SCALE_ROUNDING_MODE == 1
        dequant_scale_exponent = dequant_scale.to(tl.uint32, bitcast=True) & 0x7F800000
    if DEQUANT_SCALE_ROUNDING_MODE != 2:
        dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
        quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    f32_tensor = tl.reshape(
        f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
    )
    quant_tensor = f32_tensor * quant_scale

    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)
    dequant_scale_exponent = dequant_scale_exponent.reshape(
        [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE]
    )

    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)
    if is_fp8:
        out_tensor = quant_tensor.to(mx_tensor_dtype)
    else:
        signs = quant_tensor.to(tl.uint32, bitcast=True) & 0x80000000
        magnitude = tl.abs(quant_tensor)
        code = _e2m1_rne_magnitude(magnitude)
        e2m1_value = ((signs >> 28) | code).to(tl.uint8)

        e2m1_value = tl.reshape(
            e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2]
        )
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)

    return out_tensor, dequant_scale_exponent


@triton.jit
def _downcast_to_mxfp(
    mx_tensor_ptr,
    stride_mxt_outer,
    stride_mxt_quant: tl.constexpr,
    mx_scale_ptr,
    stride_mx_scale_outer,
    stride_mx_scale_quant,
    src_ptr,
    stride_src_outer,
    stride_src_quant,
    outer_dim,
    quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr,
):
    tl.static_assert(
        stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1."
    )
    tl.static_assert(
        BLOCK_SIZE_QUANT_DIM % 32 == 0,
        f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32",
    )

    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    tl.static_assert(
        mx_tensor_dtype == tl.uint8
        or (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5),
        f"Invalid {mx_tensor_dtype=}. Must be uint8 or float8.",
    )

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(
        mx_scale_ptr.dtype.element_ty == tl.uint8,
        f"{mx_scale_ptr.dtype.element_ty=} must be uint8",
    )
    tl.static_assert(
        (src_dtype == tl.bfloat16) or (src_dtype == tl.float16),
        f"{src_dtype=} must be bfloat16 or float16",
    )
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 32
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
    mx_scale_ptr += (
        start_mx_scale_quant * stride_mx_scale_quant + start_out * stride_mx_scale_outer
    )
    mx_tensor_ptr += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_scale_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

    mask_src_quant = start_src_quant + offs_src_quant < quant_dim
    mask_n = start_out + offs_outer < outer_dim
    full_mask_src = mask_src_quant & mask_n

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_mxt = mask_mxt_quant & mask_n

    scale_mask_k = start_mx_scale_quant + offs_scale_quant < tl.cdiv(quant_dim, 32)
    full_scale_mask = scale_mask_k & mask_n

    src_tensor_offsets = (
        offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
    )
    mx_scale_offsets = (
        offs_scale_quant * stride_mx_scale_quant + offs_outer * stride_mx_scale_outer
    )
    mx_tensor_offsets = offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer
    src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src)

    out_tensor, scale_tensor = _compute_mx_quant_and_scale(
        src_tensor, full_mask_src, mx_tensor_dtype, DEQUANT_SCALE_ROUNDING_MODE
    )

    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)


@triton.jit
def _upcast_from_mxfp(
    out_ptr,
    stride_o_outer,
    stride_o_quant: tl.constexpr,
    mx_scale_ptr,
    stride_scale_outer,
    stride_scale_quant,
    mx_tensor_ptr,
    stride_tensor_outer,
    stride_tensor_quant: tl.constexpr,
    outer_dim,
    quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
):
    tl.static_assert(
        stride_o_quant == 1, "the weight must be contiguous in the k dimension for mx"
    )
    tl.static_assert(
        BLOCK_SIZE_QUANT_DIM % 32 == 0, "BLOCK_SIZE_K must be a multiple of 32"
    )
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    dst_dtype: tl.constexpr = out_ptr.dtype.element_ty
    tl.static_assert(dst_dtype == tl.float16 or dst_dtype == tl.bfloat16)
    tl.static_assert(
        mx_tensor_dtype == tl.uint8
        or (
            (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5)
            or mx_tensor_dtype == dst_dtype
        ),
        "mx_tensor_ptr must be uint8 or float8 or dst_dtype",
    )
    tl.static_assert(
        mx_scale_ptr.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8"
    )

    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8
    is_fp8: tl.constexpr = (
        mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    )
    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 32
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    start_mxt_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    mx_tensor_ptr += (
        start_mxt_quant * stride_tensor_quant + start_out * stride_tensor_outer
    )
    mx_scale_ptr += (
        start_mx_scale_quant * stride_scale_quant + start_out * stride_scale_outer
    )
    out_ptr += start_out * stride_o_outer + start_out_quant * stride_o_quant

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_out_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)
    offs_scale = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)

    mask_outer = start_out + offs_outer < outer_dim
    mask_out_quant = start_out_quant + offs_out_quant < quant_dim
    full_mask_out = mask_out_quant & mask_outer

    mask_src_quant = start_mxt_quant + offs_src_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_src = mask_src_quant & mask_outer

    mask_scale = start_mx_scale_quant + offs_scale < tl.cdiv(quant_dim, 32)
    full_scale_mask = mask_scale & mask_outer

    tensor_offsets = (
        offs_src_quant * stride_tensor_quant + offs_outer * stride_tensor_outer
    )
    scale_offsets = offs_scale * stride_scale_quant + offs_outer * stride_scale_outer
    out_offsets = offs_out_quant * stride_o_quant + offs_outer * stride_o_outer

    tensor = tl.load(mx_tensor_ptr + tensor_offsets, mask=full_mask_src)
    scale = tl.load(mx_scale_ptr + scale_offsets, mask=full_scale_mask)

    if dst_dtype == tl.bfloat16:
        dst_scale = (scale.to(tl.uint16) << 7).to(dst_dtype, bitcast=True)
    else:
        tl.static_assert(dst_dtype == tl.float16)
        dst_scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        dst_scale = dst_scale.to(tl.float16)

    if is_fp8:
        dst_tensor = tensor.to(dst_dtype)
        if tensor.dtype == tl.float8e5:
            from_e_bits: tl.constexpr = 5
            from_m_bits: tl.constexpr = 2
            to_e_bits: tl.constexpr = 8 if dst_dtype == tl.bfloat16 else 5
            to_m_bits: tl.constexpr = 7 if dst_dtype == tl.bfloat16 else 10

            non_finite_mask_src: tl.constexpr = ((1 << from_e_bits) - 1) << from_m_bits
            non_finite_mask_dst: tl.constexpr = ((1 << to_e_bits) - 1) << to_m_bits
            dst_tensor = tl.where(
                (tensor.to(tl.uint8, bitcast=True) & non_finite_mask_src)
                == non_finite_mask_src,
                (dst_tensor.to(tl.uint16, bitcast=True) | non_finite_mask_dst).to(
                    dst_dtype, bitcast=True
                ),
                dst_tensor,
            )
    else:
        assert is_fp4
        dst_bias: tl.constexpr = 127 if dst_dtype == tl.bfloat16 else 15
        dst_0p5: tl.constexpr = 16128 if dst_dtype == tl.bfloat16 else 0x3800
        dst_m_bits: tl.constexpr = 7 if dst_dtype == tl.bfloat16 else 10
        em0 = tensor & 0x07
        em1 = tensor & 0x70
        x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | (
            (tensor & 0x08).to(tl.uint16) << 12
        )
        x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | (
            (tensor & 0x80).to(tl.uint16) << 8
        )
        x0 = tl.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
        x1 = tl.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)
        x0 = tl.where(em0 == 0x01, dst_0p5 | (x0 & 0x8000), x0)
        x1 = tl.where(em1 == 0x10, dst_0p5 | (x1 & 0x8000), x1)
        dst_tensor = tl.interleave(x0, x1).to(dst_dtype, bitcast=True)

    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32])
    dst_scale = dst_scale.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])
    scale = scale.reshape(dst_scale.shape)

    out_tensor = dst_tensor * dst_scale
    out_tensor = tl.where(scale == 0xFF, float("nan"), out_tensor)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    tl.store(out_ptr + out_offsets, out_tensor, mask=full_mask_out)


class DequantScaleRoundingMode(Enum):
    ROUND_UP = 0
    ROUND_DOWN = 1
    OCP = 2


def downcast_to_mxfp(
    src_tensor: torch.Tensor,
    out_quant_type: torch.dtype,
    axis: int,
    DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_UP,
):
    """Convert src weights to mx format, quantized along `axis`."""
    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    src_tensor = src_tensor.transpose(axis, src_tensor.ndim - 1)
    is_fp4 = out_quant_type == torch.uint8
    is_fp8 = out_quant_type in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
    )
    assert is_fp4 or is_fp8
    divisor = 2 if is_fp4 else 1
    L = src_tensor.shape[-1]
    if is_fp4:
        assert L % 2 == 0, f"axis dim must be divisible by 2 for e2m1. Got {L}"
    out_shape = src_tensor.shape[:-1] + (L // divisor,)
    out_scale_shape = src_tensor.shape[:-1] + (triton.cdiv(L, 32),)

    out_quant_tensor = src_tensor.new_empty(out_shape, dtype=out_quant_type)
    out_scale = src_tensor.new_empty(out_scale_shape, dtype=torch.uint8)

    kernel_src_tensor = src_tensor.reshape(-1, src_tensor.shape[-1])
    kernel_quant_tensor = out_quant_tensor.view(-1, out_quant_tensor.shape[-1])
    kernel_scale = out_scale.view(-1, out_scale.shape[-1])

    BLOCK_OUT_DIM = 128
    BLOCK_QUANT_DIM = 32
    grid_out = triton.cdiv(kernel_src_tensor.shape[0], BLOCK_OUT_DIM)
    grid_quant = triton.cdiv(kernel_src_tensor.shape[1], BLOCK_QUANT_DIM)

    _downcast_to_mxfp[(grid_out, grid_quant)](
        kernel_quant_tensor,
        *kernel_quant_tensor.stride(),
        kernel_scale,
        *kernel_scale.stride(),
        kernel_src_tensor,
        *kernel_src_tensor.stride(),
        *kernel_src_tensor.shape,
        BLOCK_OUT_DIM,
        BLOCK_QUANT_DIM,
        DEQUANT_SCALE_ROUNDING_MODE.value,
        num_warps=8,
    )

    out_quant_tensor = out_quant_tensor.transpose(axis, src_tensor.ndim - 1)
    out_scale = out_scale.transpose(axis, src_tensor.ndim - 1)
    return out_quant_tensor, out_scale


def upcast_from_mxfp(
    tensor: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, axis: int
):
    """Upcast an mxfp (packed) weight tensor back to float16/bfloat16."""
    ndim = tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    assert tensor.ndim == scale.ndim
    assert tensor.dtype in {
        torch.uint8,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    }, f"Invalid tensor dtype {tensor.dtype=}"
    assert scale.dtype == torch.uint8, f"Invalid scale dtype {scale.dtype=}"
    assert dtype in (torch.float16, torch.bfloat16), f"Invalid output dtype {dtype=}"
    logical_quant_dim = tensor.shape[axis] * (2 if tensor.dtype == torch.uint8 else 1)
    tensor = tensor.transpose(axis, tensor.ndim - 1).contiguous()
    scale = scale.transpose(axis, scale.ndim - 1).contiguous()
    out = torch.empty(
        (*tensor.shape[:-1], logical_quant_dim), dtype=dtype, device=tensor.device
    )
    reshaped_out = out.view(-1, out.shape[-1])
    reshaped_tensor = tensor.view(-1, tensor.shape[-1])
    reshaped_scale = scale.view(-1, scale.shape[-1])
    BLOCK_OUT_DIM = 128
    BLOCK_QUANT_DIM = 32
    blocks_out_dim = triton.cdiv(reshaped_out.shape[0], BLOCK_OUT_DIM)
    blocks_quant_dim = triton.cdiv(reshaped_out.shape[1], BLOCK_QUANT_DIM)
    _upcast_from_mxfp[(blocks_out_dim, blocks_quant_dim)](
        reshaped_out,
        *reshaped_out.stride(),
        reshaped_scale,
        *reshaped_scale.stride(),
        reshaped_tensor,
        *reshaped_tensor.stride(),
        *reshaped_out.shape,
        BLOCK_OUT_DIM,
        BLOCK_QUANT_DIM,
        num_warps=8,
    )
    out = out.transpose(axis, scale.ndim - 1).contiguous()
    return out


# ===========================================================================
# Sage V2 quantization kernels (verbatim from sage_attention_quant.py)
# ===========================================================================
@triton.jit
def sage_quant_v_kernel(
    V_Input,
    V_Output,
    V_Scale,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vsz,
    stride_vsh,
    BATCH,
    K_HEAD,
    K_NUM_BLKS,
    SEQLEN_K,
    D: tl.constexpr,
    BLK_K: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    offs_blk_k = tl.arange(0, BLK_K)
    offs_d = tl.arange(0, D)

    # V
    off_blk, off_h, off_b = pid_grid_3d(pid, K_NUM_BLKS, K_HEAD, BATCH)
    offs_kn = off_blk * BLK_K + offs_blk_k

    v_offs = (
        off_b * stride_kz
        + off_h * stride_kh
        + offs_kn[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )

    v_input_ptrs = V_Input + v_offs
    v_output_ptrs = V_Output + v_offs

    # just apply the per channel v_scales that have been computed outside
    v_scale_ptrs = V_Scale + off_b * stride_vsz + off_h * stride_vsh + offs_d[None, :]
    v = tl.load(v_input_ptrs, mask=offs_kn[:, None] < SEQLEN_K, other=0.0)
    v = v.to(tl.float32)
    v_scales = tl.load(v_scale_ptrs)
    v_quant = v / tl.where(v_scales == 0.0, 1.0, v_scales)
    v_quant = v_quant.to(v_output_ptrs.dtype.element_ty)
    tl.store(v_output_ptrs, v_quant, mask=offs_kn[:, None] < SEQLEN_K)


@triton.jit
def _rot_q_kernel(
    Q,
    Q_rot,
    Q_mean,
    R,  # Hadamard matrix
    sm_scale: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_qob,
    stride_qoh,
    stride_qom,
    stride_qod,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_md,
    stride_rm,
    stride_rd,
    n_heads,
    seq_len,
    d_model,
    q_smoothing: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,  # BLOCK_D is 32
):
    # Grid: (batch * n_heads, seq_len // BLOCK_M, d_model // BLOCK_D)
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1).to(tl.int64)
    pid_d = tl.program_id(2).to(tl.int64)

    pid_h = pid_bh % n_heads
    pid_b = pid_bh // n_heads

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # Load Q block and R (Hadamard)
    # Q block shape: [BLOCK_M, BLOCK_D]
    q_ptr = (
        Q
        + (pid_b * stride_qb)
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    r_ptr = (
        R
        + tl.arange(0, BLOCK_D)[:, None] * stride_rm
        + tl.arange(0, BLOCK_D)[None, :] * stride_rd
    )
    q_tile = tl.load(
        q_ptr, mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < d_model), other=0.0
    )
    r_mat = tl.load(r_ptr)  # 32x32

    # Rotate: Q_rot = Q @ R
    q_rot_tile = tl.dot(q_tile.to(r_mat.dtype), r_mat)
    if sm_scale is not None:
        q_rot_tile *= sm_scale

    # Store rotated Q
    rot_ptr = (
        Q_rot
        + (pid_b * stride_qob)
        + pid_h * stride_qoh
        + offs_m[:, None] * stride_qom
        + offs_d[None, :] * stride_qod
    )

    # Calculate mean for the block (reduction over d within the BLOCK_M)
    # q_mean shape: [B, H, Q_NUM_BLKS, D]
    if q_smoothing:
        m_row_mean = (
            tl.sum(q_rot_tile, axis=0) / BLOCK_M
        )  # Sum over BLOCK_M -> shape [BLOCK_D]

        q_rot_tile -= m_row_mean[None, :]
        # Store mean (Atomic add or structured store)
        # For simplicity in this layout, we store the block-sum
        # and divide by BLOCK_M in the host or final step
        mean_ptr = (
            Q_mean
            + (pid_b * stride_mb)
            + pid_h * stride_mh
            + pid_m * stride_mm
            + offs_d * stride_md
        )
        tl.store(mean_ptr, m_row_mean)

    tl.store(
        rot_ptr,
        q_rot_tile,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < d_model),
    )


@triton.jit
def _rot_k_only_kernel(
    K,
    K_rot,
    R,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_kob,
    stride_koh,
    stride_kon,
    stride_kod,
    stride_rm,
    stride_rd,
    n_heads,
    seq_k,
    d_model,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    pid_d = tl.program_id(2).to(tl.int64)

    pid_h = pid_bh % n_heads
    pid_b = pid_bh // n_heads

    offs_n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # Load K block and R
    k_ptr = (
        K
        + (pid_b * stride_kb)
        + (pid_h * stride_kh)
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )
    r_ptr = (
        R
        + tl.arange(0, BLOCK_D)[:, None] * stride_rm
        + tl.arange(0, BLOCK_D)[None, :] * stride_rd
    )

    k_tile = tl.load(
        k_ptr, mask=(offs_n[:, None] < seq_k) & (offs_d[None, :] < d_model), other=0.0
    )
    r_mat = tl.load(r_ptr)

    # Rotate K
    k_rot_tile = tl.dot(k_tile.to(r_mat.dtype), r_mat)

    # Store
    rot_ptr = (
        K_rot
        + (pid_b * stride_kob)
        + pid_h * stride_koh
        + offs_n[:, None] * stride_kon
        + offs_d[None, :] * stride_kod
    )
    tl.store(
        rot_ptr,
        k_rot_tile,
        mask=(offs_n[:, None] < seq_k) & (offs_d[None, :] < d_model),
    )


@triton.jit
def _compute_delta_s_kernel(
    Q_mean,
    K_rot,
    Delta_S,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_md,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_sb,
    stride_sh,
    stride_sm,
    stride_sn,
    n_heads_q,
    n_heads_k,
    seq_k,
    d_model,
    BLOCK_N: tl.constexpr,  # Number of K-tokens to process
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_m_q = tl.program_id(1).to(tl.int64)  # The Q-block index
    pid_n_k = tl.program_id(2).to(tl.int64)  # The K-block index

    pid_hq = pid_bh % n_heads_q
    pid_b = pid_bh // n_heads_q

    pid_hk = pid_hq // (n_heads_q // n_heads_k)

    offs_n = pid_n_k * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulate dot product across the whole d_model
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over d_model in steps of 32 (our block_size)
    for d_offset in range(0, d_model, 32):
        offs_d = d_offset + tl.arange(0, 32)

        # Load Q_mean segment: [32]
        qm_ptr = (
            Q_mean
            + pid_b * stride_mb
            + pid_hq * stride_mh
            + pid_m_q * stride_mm
            + offs_d * stride_md
        )
        qm_val = tl.load(qm_ptr)

        # Load K_rot segment: [BLOCK_N, 32]
        kn_ptr = (
            K_rot
            + pid_b * stride_kb
            + pid_hk * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        kn_val = tl.load(kn_ptr, mask=offs_n[:, None] < seq_k, other=0.0)

        # Compute dot product for this d-segment
        acc += tl.sum(qm_val[None, :] * kn_val, axis=1)

    # Store to Delta_S [B, H, Q_BLKS, seq_k]
    s_ptr = (
        Delta_S
        + pid_b * stride_sb
        + pid_hq * stride_sh
        + pid_m_q * stride_sm
        + offs_n * stride_sn
    )
    tl.store(s_ptr, acc, mask=offs_n < seq_k)


@functools.lru_cache(maxsize=16)
def create_hadamard_matrix(block_size, device="cuda", dtype=torch.bfloat16):
    """Returns an (unnormalized) Hadamard matrix of size block_size x block_size."""
    assert (block_size & (block_size - 1)) == 0, "block_size must be power of 2"
    assert block_size > 0, "block_size must be positive"

    if block_size == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)

    H_half = create_hadamard_matrix(block_size // 2, device=device, dtype=dtype)

    H = torch.zeros(block_size, block_size, device=device, dtype=dtype)
    half = block_size // 2
    H[:half, :half] = H_half
    H[:half, half:] = H_half
    H[half:, :half] = H_half
    H[half:, half:] = -H_half
    return H


def rotation_smooth_qk(
    q,
    k,
    BLOCK_SIZE_M,
    R=None,
    BLOCK_R=32,
    q_smoothing=False,
    sm_scale=None,
    layout="bhsd",
    smooth_k=True,
):
    if R is None:
        assert (
            BLOCK_R is not None
        ), "if not passing R (hadamard matrix), BLOCK_R must be provided."
        R = create_hadamard_matrix(BLOCK_R, device=q.device, dtype=q.dtype) / (
            BLOCK_R**0.5
        )
    else:
        BLOCK_R = R.shape[-1]

    bshd = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]

    b, s_q, h_q, d = map_dims(q.shape, bshd)
    _, s_k, h_k, _ = map_dims(k.shape, bshd)

    Q_rot = torch.empty_like(q)
    K_rot = torch.empty_like(k)

    Q_NUM_BLKS = (s_q + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    K_NUM_BLKS = (s_k + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    if q_smoothing:
        q_mean = torch.empty(
            (b, h_q, Q_NUM_BLKS, d), dtype=torch.float32, device=q.device
        )
        delta_s = torch.empty(
            (b, h_q, Q_NUM_BLKS, s_k), dtype=torch.float32, device=q.device
        )
    else:
        q_mean = None
        delta_s = None

    stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd)
    stride_qob, stride_qom, stride_qoh, stride_qod = map_dims(Q_rot.stride(), bshd)
    stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd)
    stride_kob, stride_kon, stride_koh, stride_kod = map_dims(K_rot.stride(), bshd)

    grid_q = (b * h_q, Q_NUM_BLKS, d // BLOCK_R)
    _rot_q_kernel[grid_q](
        q,
        Q_rot,
        q_mean,
        R,
        sm_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_qob,
        stride_qoh,
        stride_qom,
        stride_qod,
        q_mean.stride(0) if q_smoothing else None,
        q_mean.stride(1) if q_smoothing else None,
        q_mean.stride(2) if q_smoothing else None,
        q_mean.stride(3) if q_smoothing else None,
        R.stride(0),
        R.stride(1),
        h_q,
        s_q,
        d,
        q_smoothing=q_smoothing,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_D=BLOCK_R,
    )

    grid_k = (b * h_k, K_NUM_BLKS, d // BLOCK_R)
    _rot_k_only_kernel[grid_k](
        k,
        K_rot,
        R,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_kob,
        stride_koh,
        stride_kon,
        stride_kod,
        R.stride(0),
        R.stride(1),
        h_k,
        s_k,
        d,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_D=BLOCK_R,
    )

    if smooth_k:
        K_rot = K_rot - K_rot.mean(dim=1 if layout == "bshd" else 2, keepdim=True)

    if q_smoothing:
        grid_delta = (b * h_q, Q_NUM_BLKS, K_NUM_BLKS)
        _compute_delta_s_kernel[grid_delta](
            q_mean,
            K_rot,
            delta_s,
            q_mean.stride(0),
            q_mean.stride(1),
            q_mean.stride(2),
            q_mean.stride(3),
            stride_kb,
            stride_kh,
            stride_kn,
            stride_kd,
            delta_s.stride(0),
            delta_s.stride(1),
            delta_s.stride(2),
            delta_s.stride(3),
            h_q,
            h_k,
            s_k,
            d,
            BLOCK_N=BLOCK_SIZE_M,
        )

    return Q_rot, K_rot, delta_s



@triton.jit
def _fp4_residual_subtract(Source, Decoded, Residual, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    original = tl.load(Source + offsets, mask=offsets < N, other=0).to(tl.float32)
    decoded = tl.load(Decoded + offsets, mask=offsets < N, other=0).to(tl.float32)
    tl.store(Residual + offsets, original - decoded, mask=offsets < N)


def _fp4_residual(value, codes, scales):
    """Recover the residual on the GPU without changing the external operands."""
    original = value.contiguous()
    decoded = upcast_from_mxfp(codes, scales, dtype=value.dtype, axis=-1).contiguous()
    residual = torch.empty_like(original)
    size = original.numel()
    _fp4_residual_subtract[(triton.cdiv(size, 256),)](original, decoded, residual, size, 256)
    return residual


def sage_quant_mxfp4(
    q,
    k,
    v,
    FP8_TYPE,
    FP8_MAX,
    BLKQ,
    BLKK,
    sm_scale=None,
    q_smoothing=False,
    layout="bshd",
    USE_RNE=False,
    R=None,
    BLOCK_R=32,
    smooth_k=True,
    return_lse=False,
):
    v_fp8 = torch.empty_like(v, dtype=FP8_TYPE, device=v.device)

    if layout == "bhsd":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = v.shape

        stride_bz_v, stride_h_v, stride_seq_v, stride_d_v = (
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
        )
    elif layout == "bshd":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = v.shape

        stride_bz_v, stride_h_v, stride_seq_v, stride_d_v = (
            v.stride(0),
            v.stride(2),
            v.stride(1),
            v.stride(3),
        )
    else:
        raise ValueError(f"Unknown tensor layout: {layout}")
    K_NUM_BLKS = (kv_len + BLKK - 1) // BLKK

    v_scale = v.abs().amax(dim=1 if layout == "bshd" else 2).to(torch.float32) / FP8_MAX

    v_task_count = b * h_kv * K_NUM_BLKS
    grid = (v_task_count,)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    if return_lse and smooth_k:
        k_mean = k.mean(dim=1 if layout == "bshd" else 2, keepdim=True)
    else:
        k_mean = None

    q_orig = q
    q, k, delta_s = rotation_smooth_qk(
        q,
        k,
        BLKQ,
        R=R,
        BLOCK_R=BLOCK_R,
        q_smoothing=q_smoothing,
        layout=layout,
        sm_scale=(sm_scale * 1.4426950408889634),
        smooth_k=smooth_k,
    )

    sage_quant_v_kernel[grid](
        v,
        v_fp8,
        v_scale,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_d_v,
        v_scale.stride(0),
        v_scale.stride(1),
        b,
        h_kv,
        K_NUM_BLKS,
        kv_len,
        D=head_dim,
        BLK_K=BLKK,
        num_stages=3,
        num_warps=8,
    )

    q_fp4, q_scale = downcast_to_mxfp(q, torch.uint8, axis=-1,
        DEQUANT_SCALE_ROUNDING_MODE=DequantScaleRoundingMode.OCP)
    k_fp4, k_scale = downcast_to_mxfp(k, torch.uint8, axis=-1,
        DEQUANT_SCALE_ROUNDING_MODE=DequantScaleRoundingMode.OCP)

    # A two-component FP4 expansion retains both contraction operands in
    # E2M1/E8M0. Concatenating four feature products evaluates
    # (Q0 + Q1) @ (K0 + K1).T with the same dot_scaled attention kernel.
    q_residual = _fp4_residual(q, q_fp4, q_scale)
    k_residual = _fp4_residual(k, k_fp4, k_scale)
    q1, qs1 = downcast_to_mxfp(q_residual, torch.uint8, axis=-1,
        DEQUANT_SCALE_ROUNDING_MODE=DequantScaleRoundingMode.OCP)
    k1, ks1 = downcast_to_mxfp(k_residual, torch.uint8, axis=-1,
        DEQUANT_SCALE_ROUNDING_MODE=DequantScaleRoundingMode.OCP)
    q_fp4 = torch.cat((q_fp4, q1, q_fp4, q1), dim=-1)
    k_fp4 = torch.cat((k_fp4, k_fp4, k1, k1), dim=-1)
    q_scale = torch.cat((q_scale, qs1, q_scale, qs1), dim=-1)
    k_scale = torch.cat((k_scale, k_scale, ks1, ks1), dim=-1)

    if not return_lse:
        return q_fp4, q_scale, k_fp4, k_scale, v_fp8, v_scale, delta_s

    if k_mean is None:
        delta_lse = torch.zeros(
            (b, h_qo, qo_len), device=q_orig.device, dtype=torch.float32
        )
    else:
        if layout == "bhsd":
            q_bhsd = q_orig
            kmean_bhsd = k_mean
        else:
            q_bhsd = q_orig.transpose(1, 2)
            kmean_bhsd = k_mean.transpose(1, 2)
        if h_qo != h_kv:
            assert h_qo % h_kv == 0
            kmean_bhsd = kmean_bhsd.repeat_interleave(h_qo // h_kv, dim=1)
        delta_lse = (q_bhsd.to(torch.float32) * kmean_bhsd.to(torch.float32)).sum(
            dim=-1
        ) * sm_scale

    return q_fp4, q_scale, k_fp4, k_scale, v_fp8, v_scale, delta_s, delta_lse


# ===========================================================================
# MXFP4 SageAttention forward kernel
# (verbatim from fav3_sage_attention_mxfp4.py)
# ===========================================================================
@triton.jit
def compute_padding_info(seqlen_k, BLOCK_N: tl.constexpr):
    """Calculate padding information for the last K block."""
    # check if we will need to do masking due either BLOCK_N being bigger than seqlen_k or seqlen_k not being a factor of BLOCK_N
    # n_extra_tokens = 10 % 4 = 2
    # This means the last K block has 2 valid tokens and 2 padding positions
    # K blocks visualization:
    #         Block 0         Block 1         Block 2 (last)
    #         K0 K1 K2 K3    K4 K5 K6 K7     K8 K9 ?? ??
    #         ↑---------↑    ↑---------↑     ↑---↑ ↑---↑
    #         full block     full block      valid  pad
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    else:
        n_extra_tokens = 0
    return n_extra_tokens


@triton.jit
def compute_block_masking(
    seqlen_k,
    seqlen_q,
    start_m,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Classify K blocks for attention computation with sliding window support.

    Returns:
        - n_front_skip_blocks: Blocks completely before the window
        - n_front_masked_blocks: Blocks partially overlapping window front
        - n_full_blocks: Blocks completely inside the window
        - n_back_masked_blocks: Blocks partially overlapping window back
        - n_extra_tokens: Padding tokens in last K block
    """

    # common
    # q_start = start_m * BLOCK_M
    q_end = tl.minimum((start_m + 1) * BLOCK_M - 1, seqlen_q - 1)
    diag = seqlen_k - seqlen_q
    total_k_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    n_extra_tokens = compute_padding_info(seqlen_k, BLOCK_N)

    if IS_CAUSAL:
        # ========== CAUSAL MODE: Classify K Blocks ==========
        # Calculate causal boundary for this Q block
        #          [K0 K1 K2 K3] [K4 K5 K6 K7] [K8 K9 ?? ??]
        # Q0-Q3:   [ 1  0  0  0] [ 0  0  0  0] [ 0  0 -- --]  ← Q0
        #          [ 1  1  0  0] [ 0  0  0  0] [ 0  0 -- --]  ← Q1
        #          [ 1  1  1  0] [ 0  0  0  0] [ 0  0 -- --]  ← Q2
        #          [ 1  1  1  1] [ 1  1  0  0] [ 0  0 -- --]  ← Q3
        #                            ↑ can see up to K5
        #
        # Q4-Q7:   [ 1  1  1  1] [ 1  1  1  0] [ 0  0 -- --]  ← Q4
        #          [ 1  1  1  1] [ 1  1  1  1] [ 0  0 -- --]  ← Q5
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  0 -- --]  ← Q6
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -- --]  ← Q7

        # ------------------------------------------------------------
        # 1. figure out, in tokens, the right-most K position
        #    this Q-block may attend to
        # ------------------------------------------------------------
        k_max_token = q_end + diag  # last visible K index

        # this Q-block is entirely above the diagonal ⇒ nothing to do
        if k_max_token < 0:
            return 0, 0, 0, 0, n_extra_tokens

        k_max_token = tl.minimum(k_max_token, seqlen_k - 1)

        # ------------------------------------------------------------
        # 2. translate token indices into K-block indices
        # ------------------------------------------------------------
        last_visible_k_block = k_max_token // BLOCK_N
        n_visible_k_blocks = tl.minimum(last_visible_k_block + 1, total_k_blocks)

        # ------------------------------------------------------------
        # 3. classify those visible blocks
        #    – we *never* skip or mask blocks in front, because causal
        #      attention always starts at K0
        #    – the back side can require several masked blocks:
        #         • intersection of the causal diagonal with K-grid
        #           (at most  ⌈BLOCK_M / BLOCK_N⌉ blocks)
        #         • plus one extra block if this Q-block stops in the
        #           middle of a K-block or the last K-block is padded
        # ------------------------------------------------------------
        padded_last_k = n_extra_tokens != 0
        is_modulo_mn = (not padded_last_k) & (seqlen_q % BLOCK_M == 0)

        n_back_masked_blocks = BLOCK_M // BLOCK_N + tl.where(is_modulo_mn, 0, 1)
        n_back_masked_blocks = tl.minimum(n_back_masked_blocks, n_visible_k_blocks)

        n_front_skip_blocks = 0  # causal never skips the left side
        n_front_masked_blocks = 0  # ditto
        n_full_blocks = n_visible_k_blocks - n_back_masked_blocks
    else:
        # ========== NON-CAUSAL MODE ==========
        # Without causal mask, all positions can attend to all positions
        # Only need to handle the padding in the last block
        #          [K0 K1 K2 K3] [K4 K5 K6 K7] [K8 K9 ?? ??]
        # Q0-Q3:   [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #
        # Q4-Q7:   [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]

        n_front_skip_blocks = 0  # never skips the left side
        n_front_masked_blocks = 0  # ditto
        if n_extra_tokens != 0:
            n_back_masked_blocks = 1  # Last block needs padding mask
            n_full_blocks = total_k_blocks - 1
        else:
            n_back_masked_blocks = 0  # All blocks are aligned
            n_full_blocks = total_k_blocks

    return (
        n_front_skip_blocks,
        n_front_masked_blocks,
        n_full_blocks,
        n_back_masked_blocks,
        n_extra_tokens,
    )


@triton.jit
def _sage_fwd_no_mask_mxfp4(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
    bias_base_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    seqlen_k,
    seqlen_q,
    offs_m,
    offs_d_k,
    offs_d_v,
    block_min,
    block_max,
    q_descale,
    k_descale_base_ptrs,
    stride_ksn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
    ACCUMULATOR_TYPE: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    for start_n in range(block_min, block_max, BLOCK_N):
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk
        k_descale_ptrs = k_descale_base_ptrs + start_n * stride_ksn
        kv_offs_n = start_n + tl.arange(0, BLOCK_N)

        # Refactored K Load
        if PADDED_HEAD_QK:
            k_mask = offs_d_k[:, None] < ACTUAL_BLOCK_DMODEL_QK
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        else:
            k = tl.load(k_ptrs)

        k_descale = tl.load(k_descale_ptrs)

        if PRE_LOAD_V:
            # Refactored V Load
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)
        qk = tl.dot_scaled(
            q, q_descale, Q_DTYPE_STR, k, k_descale, K_DTYPE_STR, fast_math=True, acc=qk
        )

        if USE_BIAS:
            bias_mask = kv_offs_n < seqlen_k
            bias = tl.load(
                bias_base_ptrs + start_n * stride_bn, mask=bias_mask, other=0.0
            )
            qk += bias[None, :]

        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        if USE_BIAS:
            q_shifted = tl.where(
                m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None]
            )
        else:
            q_shifted = qk - m_ij[:, None]

        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)

        if USE_BIAS:
            m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        else:
            m_diff = m_i - m_ij

        alpha = tl.math.exp2(m_diff)
        acc = acc * alpha[:, None]

        if not PRE_LOAD_V:
            # Refactored V Load (Lazy)
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)

    return acc, l_i, m_i


@triton.jit
def _sage_fwd_mask_mxfp4(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
    bias_base_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    seqlen_k,
    seqlen_q,
    offs_m,
    offs_n,
    offs_d_k,
    offs_d_v,
    block_min,
    block_max,
    n_extra_tokens,
    q_descale,
    k_descale_base_ptrs,
    stride_ksn,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
    ACCUMULATOR_TYPE: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    seqlen_delta_qk = seqlen_k - seqlen_q
    for start_n in range(block_min, block_max, BLOCK_N):
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk
        k_descale_ptrs = k_descale_base_ptrs + start_n * stride_ksn
        kv_offs_n = start_n + tl.arange(0, BLOCK_N)

        # Refactored K Load with mandatory boundary check + optional padding check
        k_mask = kv_offs_n[None, :] < seqlen_k
        if PADDED_HEAD_QK:
            k_mask &= offs_d_k[:, None] < ACTUAL_BLOCK_DMODEL_QK

        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        k_descale = tl.load(
            k_descale_ptrs, mask=kv_offs_n[:, None] < seqlen_k, other=0.0
        )

        if PRE_LOAD_V:
            # Refactored V Load
            v_mask = kv_offs_n[:, None] < seqlen_k
            if PADDED_HEAD_V:
                v_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)

        if (n_extra_tokens != 0) and (start_n + BLOCK_N == block_max):
            mask = (start_n + offs_n[None, :]) < seqlen_k
            qk = tl.where(mask, qk, float("-inf"))

        qk = tl.dot_scaled(
            q, q_descale, Q_DTYPE_STR, k, k_descale, K_DTYPE_STR, fast_math=True, acc=qk
        )

        if IS_CAUSAL:
            qk = tl.where(
                offs_m[:, None] >= (start_n + offs_n - seqlen_delta_qk)[None, :],
                qk,
                float("-inf"),
            )

        if USE_BIAS:
            bias_mask = kv_offs_n < seqlen_k
            bias = tl.load(
                bias_base_ptrs + start_n * stride_bn, mask=bias_mask, other=0.0
            )
            qk += bias[None, :]

        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        q_shifted = tl.where(
            m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None]
        )

        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)

        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        alpha = tl.math.exp2(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            # Refactored V Load (Lazy)
            v_mask = kv_offs_n[:, None] < seqlen_k
            if PADDED_HEAD_V:
                v_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)

    return acc, l_i, m_i


@triton.jit
def _sage_fwd_blocksparse_nomask_mxfp4(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
    bias_base_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    seqlen_k,
    seqlen_q,
    offs_m,
    offs_d_k,
    offs_d_v,
    q_descale,
    k_descale_base_ptrs,
    stride_ksn,
    kv_block_indices,
    lut_start_val,
    n_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
    ACCUMULATOR_TYPE: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    for i in range(n_blocks):
        start_b = tl.load(kv_block_indices + lut_start_val + i)
        start_n = start_b * BLOCK_N
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk
        k_descale_ptrs = k_descale_base_ptrs + start_n * stride_ksn
        kv_offs_n = start_n + tl.arange(0, BLOCK_N)

        if PADDED_HEAD_QK:
            k_mask = offs_d_k[:, None] < ACTUAL_BLOCK_DMODEL_QK
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        else:
            k = tl.load(k_ptrs)

        k_descale = tl.load(k_descale_ptrs)

        if PRE_LOAD_V:
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)
        qk = tl.dot_scaled(
            q, q_descale, Q_DTYPE_STR, k, k_descale, K_DTYPE_STR, fast_math=True, acc=qk
        )

        if USE_BIAS:
            bias_mask = kv_offs_n < seqlen_k
            bias = tl.load(
                bias_base_ptrs + start_n * stride_bn, mask=bias_mask, other=0.0
            )
            qk += bias[None, :]

        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        if USE_BIAS:
            q_shifted = tl.where(
                m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None]
            )
        else:
            q_shifted = qk - m_ij[:, None]

        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)

        if USE_BIAS:
            m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        else:
            m_diff = m_i - m_ij

        alpha = tl.math.exp2(m_diff)
        acc = acc * alpha[:, None]

        if not PRE_LOAD_V:
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)

    return acc, l_i, m_i


@triton.jit
def _sage_fwd_blocksparse_mask_mxfp4(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
    bias_base_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    seqlen_k,
    seqlen_q,
    offs_m,
    offs_n,
    offs_d_k,
    offs_d_v,
    q_descale,
    k_descale_base_ptrs,
    stride_ksn,
    kv_block_indices,
    lut_start_val,
    n_blocks,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
    ACCUMULATOR_TYPE: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    seqlen_delta_qk = seqlen_k - seqlen_q
    for i in range(n_blocks):
        start_b = tl.load(kv_block_indices + lut_start_val + i)
        start_n = start_b * BLOCK_N
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk
        k_descale_ptrs = k_descale_base_ptrs + start_n * stride_ksn
        kv_offs_n = start_n + tl.arange(0, BLOCK_N)

        k_mask = kv_offs_n[None, :] < seqlen_k
        if PADDED_HEAD_QK:
            k_mask &= offs_d_k[:, None] < ACTUAL_BLOCK_DMODEL_QK

        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        k_descale = tl.load(
            k_descale_ptrs, mask=kv_offs_n[:, None] < seqlen_k, other=0.0
        )

        if PRE_LOAD_V:
            v_mask = kv_offs_n[:, None] < seqlen_k
            if PADDED_HEAD_V:
                v_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)

        # Padding mask: mask out positions beyond seqlen_k
        boundary_mask = kv_offs_n[None, :] < seqlen_k
        qk = tl.where(boundary_mask, qk, float("-inf"))

        qk = tl.dot_scaled(
            q, q_descale, Q_DTYPE_STR, k, k_descale, K_DTYPE_STR, fast_math=True, acc=qk
        )

        if IS_CAUSAL:
            qk = tl.where(
                offs_m[:, None] >= (start_n + offs_n - seqlen_delta_qk)[None, :],
                qk,
                float("-inf"),
            )

        if USE_BIAS:
            bias_mask = kv_offs_n < seqlen_k
            bias = tl.load(
                bias_base_ptrs + start_n * stride_bn, mask=bias_mask, other=0.0
            )
            qk += bias[None, :]

        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        q_shifted = tl.where(
            m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None]
        )

        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)

        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        alpha = tl.math.exp2(m_diff)
        acc = acc * alpha[:, None]

        if not PRE_LOAD_V:
            v_mask = kv_offs_n[:, None] < seqlen_k
            if PADDED_HEAD_V:
                v_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)

    return acc, l_i, m_i


@triton.jit
def sage_fwd_mxfp4(
    Q,
    K,
    V,
    bias,
    Q_Descale,
    K_Descale,
    V_Descale,
    stride_qsz,
    stride_qsh,
    stride_qsm,
    stride_ksz,
    stride_ksh,
    stride_ksn,
    stride_vsz,
    stride_vsh,
    Out,
    LSE,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_lse_z,
    stride_lse_h,
    stride_lse_m,
    cu_seqlens_q,
    cu_seqlens_k,
    kv_block_indices,
    lut_start,
    lut_count,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    USE_BIAS: tl.constexpr,
    USE_BLOCK_SPARSE: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    # Constants
    Q_HEAD_DIV: tl.constexpr = 2 if Q_DTYPE_STR == "e2m1" else 1
    K_HEAD_DIV: tl.constexpr = 2 if K_DTYPE_STR == "e2m1" else 1
    SCALE_GROUP: tl.constexpr = 32
    ACC_TYPE: tl.constexpr = tl.float32
    # b*h*s*d can grow to be larger than int32 max, so turn to int64
    start_m, off_h_q, off_z = (
        tl.program_id(0).to(tl.int64),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2).to(tl.int64),
    )
    off_h_k = off_h_q // (HQ // HK)

    PADDED_HEAD_QK: tl.constexpr = ACTUAL_BLOCK_DMODEL_QK != BLOCK_DMODEL_QK
    PADDED_HEAD_V: tl.constexpr = ACTUAL_BLOCK_DMODEL_V != BLOCK_DMODEL_V

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_q = tl.arange(0, BLOCK_DMODEL_QK // Q_HEAD_DIV)
    offs_d_k = tl.arange(0, BLOCK_DMODEL_QK // K_HEAD_DIV)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)
    offs_d_scale = tl.arange(0, BLOCK_DMODEL_QK // SCALE_GROUP)

    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + off_z)
        seqlen_q = tl.load(cu_seqlens_q + off_z + 1) - q_start
        k_start = tl.load(cu_seqlens_k + off_z)
        seqlen_k = tl.load(cu_seqlens_k + off_z + 1) - k_start
        if start_m * BLOCK_M >= seqlen_q:
            return
    else:
        q_start, k_start = 0, 0
        seqlen_q, seqlen_k = MAX_SEQLENS_Q, MAX_SEQLENS_K

    # Masking logic
    if USE_BLOCK_SPARSE:
        num_q_blocks = (seqlen_q + BLOCK_M - 1) // BLOCK_M
        n_extra = compute_padding_info(seqlen_k, BLOCK_N)
        lut_idx = off_z * (HQ * num_q_blocks) + off_h_q * num_q_blocks + start_m
        n_blocks = tl.load(lut_count + lut_idx)
        has_any_range = n_blocks > 0
    else:
        mask_info = compute_block_masking(
            seqlen_k, seqlen_q, start_m.to(tl.int32), IS_CAUSAL, BLOCK_M, BLOCK_N
        )  # need to turn start_m to int32 for consistent return values
        n_front_skip, n_front_masked, n_full, n_back_masked, n_extra = mask_info
        has_any_range = True

    # ============================================================
    #          PROGRAM EARLY EXIT (All K Blocks Skipped)
    # ============================================================
    if not USE_BLOCK_SPARSE:
        total_visible_blocks = n_front_masked + n_full + n_back_masked
    # Early exit: no K blocks to process
    if USE_BLOCK_SPARSE:
        _no_blocks = not has_any_range
    else:
        _no_blocks = total_visible_blocks == 0
    if _no_blocks:
        """
        No K blocks visible - write zeros and exit.
        """
        o_ptr = (
            Out
            + off_z * stride_oz
            + off_h_q * stride_oh
            + (q_start + offs_m[:, None]) * stride_om
            + offs_d_v[None, :]
        )
        o_mask = offs_m[:, None] < seqlen_q
        if PADDED_HEAD_V:
            o_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
        tl.store(
            o_ptr,
            tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=Out.type.element_ty),
            mask=o_mask,
        )

        if RETURN_LSE:
            l_offset = (
                LSE
                + off_z * stride_lse_z
                + off_h_q * stride_lse_h
                + q_start * stride_lse_m
            )
            l_ptrs = l_offset + offs_m * stride_lse_m
            tl.store(
                l_ptrs,
                tl.full([BLOCK_M], float("-inf"), dtype=tl.float32),
                mask=offs_m < seqlen_q,
            )

        return

    # Pointers
    q_ptrs = (
        Q
        + off_z * stride_qz
        + off_h_q * stride_qh
        + (q_start + offs_m[:, None]) * stride_qm
        + offs_d_q[None, :]
    )
    k_ptrs = (
        K
        + off_z * stride_kz
        + off_h_k * stride_kh
        + (k_start + offs_n[None, :]) * stride_kn
        + offs_d_k[:, None]
    )
    v_ptrs = (
        V
        + off_z * stride_vz
        + off_h_k * stride_vh
        + (k_start + offs_n[:, None]) * stride_vk
        + offs_d_v[None, :]
    )

    qd_ptrs = (
        Q_Descale
        + off_z * stride_qsz
        + off_h_q * stride_qsh
        + (q_start + offs_m[:, None]) * stride_qsm
        + offs_d_scale[None, :]
    )
    kd_ptrs = (
        K_Descale
        + off_z * stride_ksz
        + off_h_k * stride_ksh
        + (k_start + offs_n[:, None]) * stride_ksn
        + offs_d_scale[None, :]
    )
    vd_ptr = V_Descale + off_z * stride_vsz + off_h_k * stride_vsh + offs_d_v

    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q), other=0.0)
    q_descale = tl.load(qd_ptrs, mask=(offs_m[:, None] < seqlen_q), other=0.0)

    # Bias is delta s
    bias_ptrs = (
        (
            bias
            + off_z * stride_bz
            + off_h_q * stride_bh
            + start_m * stride_bm
            + tl.cast(offs_n, tl.int64) * stride_bn
        )
        if USE_BIAS
        else None
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=ACC_TYPE)
    l_i = tl.full([BLOCK_M], 1.0, dtype=ACC_TYPE)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=ACC_TYPE)

    if not USE_BLOCK_SPARSE:
        if n_full > 0:
            b_min = (n_front_skip + n_front_masked) * BLOCK_N
            b_max = b_min + n_full * BLOCK_N
            acc, l_i, m_i = _sage_fwd_no_mask_mxfp4(
                acc,
                l_i,
                m_i,
                q,
                k_ptrs,
                v_ptrs,
                bias_ptrs,
                stride_kn,
                stride_vk,
                stride_bn,
                seqlen_k,
                seqlen_q,
                offs_m,
                offs_d_k,
                offs_d_v,
                b_min,
                b_max,
                q_descale,
                kd_ptrs,
                stride_ksn,
                BLOCK_M,
                BLOCK_N,
                PRE_LOAD_V,
                PADDED_HEAD_QK,
                PADDED_HEAD_V,
                ACTUAL_BLOCK_DMODEL_QK,
                ACTUAL_BLOCK_DMODEL_V,
                Q_DTYPE_STR,
                K_DTYPE_STR,
                ACC_TYPE,
                USE_BIAS,
            )

        if n_back_masked > 0:
            b_min = (n_front_skip + n_front_masked + n_full) * BLOCK_N
            b_max = b_min + n_back_masked * BLOCK_N
            acc, l_i, m_i = _sage_fwd_mask_mxfp4(
                acc,
                l_i,
                m_i,
                q,
                k_ptrs,
                v_ptrs,
                bias_ptrs,
                stride_kn,
                stride_vk,
                stride_bn,
                seqlen_k,
                seqlen_q,
                offs_m,
                offs_n,
                offs_d_k,
                offs_d_v,
                b_min,
                b_max,
                n_extra,
                q_descale,
                kd_ptrs,
                stride_ksn,
                IS_CAUSAL,
                BLOCK_M,
                BLOCK_N,
                PRE_LOAD_V,
                PADDED_HEAD_QK,
                PADDED_HEAD_V,
                ACTUAL_BLOCK_DMODEL_QK,
                ACTUAL_BLOCK_DMODEL_V,
                Q_DTYPE_STR,
                K_DTYPE_STR,
                ACC_TYPE,
                USE_BIAS,
            )
    else:
        lut_start_val = tl.load(lut_start + lut_idx)
        acc, l_i, m_i = _sage_fwd_blocksparse_nomask_mxfp4(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            seqlen_k,
            seqlen_q,
            offs_m,
            offs_d_k,
            offs_d_v,
            q_descale,
            kd_ptrs,
            stride_ksn,
            kv_block_indices,
            lut_start_val,
            n_blocks - 1,
            BLOCK_M,
            BLOCK_N,
            PRE_LOAD_V,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            Q_DTYPE_STR,
            K_DTYPE_STR,
            ACC_TYPE,
            USE_BIAS,
        )
        invalid_q_rows = offs_m >= seqlen_q
        m_i = tl.where(invalid_q_rows, float("-inf"), m_i)
        l_i = tl.where(invalid_q_rows, 1.0, l_i)
        acc = tl.where(invalid_q_rows[:, None], 0.0, acc)
        acc, l_i, m_i = _sage_fwd_blocksparse_mask_mxfp4(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            seqlen_k,
            seqlen_q,
            offs_m,
            offs_n,
            offs_d_k,
            offs_d_v,
            q_descale,
            kd_ptrs,
            stride_ksn,
            kv_block_indices,
            lut_start_val + (n_blocks - 1),
            1,
            False,  # IS_CAUSAL is not supported for block sparse
            BLOCK_M,
            BLOCK_N,
            PRE_LOAD_V,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            Q_DTYPE_STR,
            K_DTYPE_STR,
            ACC_TYPE,
            USE_BIAS,
        )

    # Epilogue
    invalid_mask = m_i == float("-inf")
    l_i_safe = tl.where(invalid_mask, 1.0, l_i)
    l_i_safe = tl.maximum(l_i_safe, 1e-7)
    l_recip = 1 / l_i_safe[:, None]
    v_descale = tl.load(vd_ptr, mask=offs_d_v < ACTUAL_BLOCK_DMODEL_V, other=0.0)
    acc = acc * l_recip * v_descale
    acc = tl.where(invalid_mask[:, None], 0.0, acc)

    if RETURN_LSE:
        # m_i / l_i are in base-2 (sm_scale was pre-multiplied by 1/ln(2)).
        # Convert back to natural units to match the int8 sage convention.
        LN2: tl.constexpr = 0.6931471824645996
        log_l_i = tl.where(invalid_mask, 0.0, tl.math.log2(l_i_safe))
        softmax_lse = tl.where(invalid_mask, float("-inf"), (m_i + log_l_i) * LN2)
        l_offset = (
            LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + q_start * stride_lse_m
        )
        l_ptrs = l_offset + offs_m * stride_lse_m
        tl.store(l_ptrs, softmax_lse, mask=offs_m < seqlen_q)

    o_ptr = (
        Out
        + off_z * stride_oz
        + off_h_q * stride_oh
        + (q_start + offs_m[:, None]) * stride_om
        + offs_d_v[None, :]
    )
    o_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD_V:
        o_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
    tl.store(o_ptr, acc.to(Out.dtype.element_ty), mask=o_mask)


# ===========================================================================
# Host wrappers (ported from fav3_sage_attention_mxfp4_wrapper.py)
# ===========================================================================
def get_sage_fwd_configs_mxfp4():
    """Returns tuned config for MXFP4 on supported architectures."""
    arch = get_arch()
    if arch != "gfx950":
        raise RuntimeError(f"MXFP4 is not supported on {arch}")
    return {
        "BLOCK_M": 256,
        "BLOCK_N": 128,
        "waves_per_eu": 2,
        "PRE_LOAD_V": False,
        "num_stages": 3,
        "num_warps": 8,
    }


def fav3_sage_mxfp4_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    bias: torch.Tensor = None,
    causal: bool = False,
    layout: str = "bshd",
    config: Optional[dict] = None,
    kv_block_indices: Optional[torch.Tensor] = None,
    lut_start: Optional[torch.Tensor] = None,
    lut_count: Optional[torch.Tensor] = None,
    use_block_sparse: bool = False,
    return_lse: bool = False,
):
    """Direct MXFP4 kernel execution with unused parameters removed."""
    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    batch, seqlen_q, nheads_q, head_size_qk = map_dims(q.shape, bshd_map)

    head_size_qk *= 2
    _, seqlen_k, nheads_k, _ = map_dims(k.shape, bshd_map)
    _, _, _, head_size_v = map_dims(v.shape, bshd_map)

    assert q.dtype == torch.uint8 and k.dtype == torch.uint8, "MXFP4 Q/K must be uint8"
    assert nheads_q % nheads_k == 0, "GQA/MQA ratio mismatch"
    assert layout in ["bhsd", "bshd"], "Only bhsd and bshd supported for now."

    if config is None:
        config = get_sage_fwd_configs_mxfp4()

    out = torch.zeros(
        (q.shape[0], q.shape[1], q.shape[2], v.shape[-1]),
        dtype=torch.bfloat16,
        device=q.device,
    )
    softmax_lse = (
        torch.empty((batch, nheads_q, seqlen_q), device=q.device, dtype=torch.float32)
        if return_lse
        else None
    )

    stride_qb, stride_qm, stride_qh, _ = map_dims(q.stride(), bshd_map)
    stride_kb, stride_kn, stride_kh, _ = map_dims(k.stride(), bshd_map)
    stride_vb, stride_vn, stride_vh, _ = map_dims(v.stride(), bshd_map)
    stride_ob, stride_om, stride_oh, _ = map_dims(out.stride(), bshd_map)

    if bias is not None:
        USE_BIAS = True
        stride_bz, stride_bh, stride_bm, stride_bn = bias.stride()
    else:
        USE_BIAS = False
        stride_bz, stride_bm, stride_bh, stride_bn = 0, 0, 0, 0

    stride_qsz, stride_qsm, stride_qsh, _ = map_dims(q_descale.stride(), bshd_map)
    stride_ksz, stride_ksn, stride_ksh, _ = map_dims(k_descale.stride(), bshd_map)
    stride_vsz, stride_vsh, _ = v_descale.stride()

    stride_lse_z, stride_lse_h, stride_lse_m = (
        softmax_lse.stride() if return_lse else (0, 0, 0)
    )

    padded_d_qk = max(16, 1 << (head_size_qk - 1).bit_length())
    padded_d_v = max(16, 1 << (head_size_v - 1).bit_length())

    if use_block_sparse:
        if kv_block_indices is None or lut_start is None or lut_count is None:
            raise ValueError(
                "kv_block_indices, lut_start, and lut_count must be provided "
                "when use_block_sparse=True"
            )
        if causal:
            raise NotImplementedError(
                "The Triton block-sparse attention path selected by block_lut "
                "does not support causal masking. require causal=False."
            )
    else:
        kv_block_indices = torch.zeros(1, dtype=torch.int32, device=q.device)
        lut_start = torch.zeros(1, dtype=torch.int32, device=q.device)
        lut_count = torch.zeros(1, dtype=torch.int32, device=q.device)

    def grid(META):
        return (triton.cdiv(seqlen_q, META["BLOCK_M"]), nheads_q, batch)

    sage_fwd_mxfp4[grid](
        Q=q,
        K=k,
        V=v,
        bias=bias,
        Q_Descale=q_descale,
        K_Descale=k_descale,
        V_Descale=v_descale,
        stride_qsz=stride_qsz,
        stride_qsh=stride_qsh,
        stride_qsm=stride_qsm,
        stride_ksz=stride_ksz,
        stride_ksh=stride_ksh,
        stride_ksn=stride_ksn,
        stride_vsz=stride_vsz,
        stride_vsh=stride_vsh,
        Out=out,
        LSE=softmax_lse,
        stride_qz=stride_qb,
        stride_qh=stride_qh,
        stride_qm=stride_qm,
        stride_kz=stride_kb,
        stride_kh=stride_kh,
        stride_kn=stride_kn,
        stride_vz=stride_vb,
        stride_vh=stride_vh,
        stride_vk=stride_vn,
        stride_oz=stride_ob,
        stride_oh=stride_oh,
        stride_om=stride_om,
        stride_bz=stride_bz,
        stride_bh=stride_bh,
        stride_bm=stride_bm,
        stride_bn=stride_bn,
        stride_lse_z=stride_lse_z,
        stride_lse_h=stride_lse_h,
        stride_lse_m=stride_lse_m,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        kv_block_indices=kv_block_indices,
        lut_start=lut_start,
        lut_count=lut_count,
        Q_DTYPE_STR="e2m1",
        K_DTYPE_STR="e2m1",
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=seqlen_q,
        MAX_SEQLENS_K=seqlen_k,
        IS_VARLEN=False,
        IS_CAUSAL=causal,
        BLOCK_DMODEL_QK=padded_d_qk,
        BLOCK_DMODEL_V=padded_d_v,
        USE_BIAS=USE_BIAS,
        USE_BLOCK_SPARSE=use_block_sparse,
        RETURN_LSE=return_lse,
        **config,
    )

    if return_lse:
        return out, softmax_lse
    return out


class _FAv3SageMXFP4WrapperFunc(torch.autograd.Function):
    """Sage Attention v2 MXFP4 wrapper maintaining high-precision I/O."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        layout: str = "bshd",
        q_smooth: bool = False,
        hadamard_rotation: bool = True,
        config: Optional[dict] = None,
        R: torch.Tensor = None,
        BLOCK_R: int = 128,
        block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        return_lse: bool = False,
        smooth_k: bool = True,
    ):
        bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
        bhsd_map = [0, 2, 1, 3] if layout == "bshd" else [0, 1, 2, 3]
        batch, seqlen_q, num_q_heads, head_dim = map_dims(q.shape, bshd_map)
        _, seqlen_k, num_kv_heads, _ = map_dims(k.shape, bshd_map)

        if config is None:
            config = get_sage_fwd_configs_mxfp4()

        FP8_TYPE = fp8_dtype
        FP8_MAX = torch.finfo(FP8_TYPE).max

        assert hadamard_rotation, "hadamard_rotation=False not supported at the moment"
        sq_result = sage_quant_mxfp4(
            q,
            k,
            v,
            FP8_TYPE,
            FP8_MAX,
            BLKQ=config["BLOCK_M"],
            BLKK=64,
            layout=layout,
            R=R,
            BLOCK_R=BLOCK_R,
            q_smoothing=q_smooth,
            smooth_k=smooth_k,
            return_lse=return_lse,
        )
        if return_lse:
            (
                q_quantized,
                q_descale,
                k_quantized,
                k_descale,
                v_quantized,
                v_descale,
                delta_s,
                sage_lse_delta,
            ) = sq_result
        else:
            (
                q_quantized,
                q_descale,
                k_quantized,
                k_descale,
                v_quantized,
                v_descale,
                delta_s,
            ) = sq_result
            sage_lse_delta = None

        qd_mapped = map_dims(q_descale.shape, bhsd_map)
        kd_mapped = map_dims(k_descale.shape, bhsd_map)

        expected_q_ds = (batch, num_q_heads, seqlen_q, (2 * q_quantized.shape[-1]) // 32)
        expected_k_ds = (batch, num_kv_heads, seqlen_k, (2 * k_quantized.shape[-1]) // 32)

        assert tuple(qd_mapped) == expected_q_ds, "q_descale mismatch"
        assert tuple(kd_mapped) == expected_k_ds, "k_descale mismatch"

        if block_lut is not None:
            kv_block_indices, lut_start, lut_count = block_lut
            use_block_sparse = True
            if causal:
                raise NotImplementedError(
                    "The Triton block-sparse attention path selected by block_lut "
                    "does not support causal masking. require causal=False."
                )
        else:
            kv_block_indices = lut_start = lut_count = None
            use_block_sparse = False

        result = fav3_sage_mxfp4_func(
            q=q_quantized,
            k=k_quantized,
            v=v_quantized,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            bias=delta_s,
            causal=causal,
            layout=layout,
            config=config,
            kv_block_indices=kv_block_indices,
            lut_start=lut_start,
            lut_count=lut_count,
            use_block_sparse=use_block_sparse,
            return_lse=return_lse,
        )

        if return_lse:
            out, softmax_lse = result
            if sage_lse_delta is not None:
                softmax_lse = softmax_lse + sage_lse_delta.to(softmax_lse.dtype)
            return out, softmax_lse

        return result

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        assert False, "backward not implemented"
        return (None,) * 12


def fav3_sage_mxfp4_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    layout: str = "bshd",
    q_smooth: bool = False,
    hadamard_rotation: bool = False,
    config: Optional[dict] = None,
    R: torch.Tensor = None,
    BLOCK_R: int = 128,
    block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    return_lse: bool = False,
    smooth_k: bool = True,
):
    """High-precision entry point for MXFP4 SageAttention."""
    for tensor, name in zip([q, k, v], ["q", "k", "v"]):
        assert tensor.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], f"Expected high-precision for {name}, got {tensor.dtype}"

    return _FAv3SageMXFP4WrapperFunc.apply(
        q,
        k,
        v,
        causal,
        layout,
        q_smooth,
        hadamard_rotation,
        config,
        R,
        BLOCK_R,
        block_lut,
        return_lse,
        smooth_k,
    )
