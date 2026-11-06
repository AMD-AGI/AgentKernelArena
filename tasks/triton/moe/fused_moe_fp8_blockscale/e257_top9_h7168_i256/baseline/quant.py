# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER's Triton FP8 per-token group (1 x 128) quantization kernels.

Kernels: _fp8_quant_op, _fused_flatten_fp8_group_quant_kernel and
_fused_reduce_act_mul_fp8_group_quant in aiter/ops/triton/_triton_kernels/quant/fused_fp8_quant.py
(verbatim). Launchers: fused_flatten_fp8_group_quant and fused_reduce_act_mul_fp8_group_quant in
aiter/ops/triton/quant/fused_fp8_quant.py, the latter reduced to its two-dimensional (no split-K)
input path with the SiLU activation of aiter/ops/triton/_triton_kernels/activation.py.
"""
import torch
import triton
import triton.language as tl

from .utils import _silu

fp8_dtype = torch.float8_e4m3fn


@triton.jit
def _fp8_quant_op(
    x,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
):
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, QUANT_BLOCK_SIZE)
    m = tl.maximum(tl.max(tl.abs(x), axis=-1), 1e-10)
    scale_out = m.to(tl.float32) / DTYPE_MAX
    scale_recip = 1.0 / scale_out.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, 1)
    x = tl.clamp(x * scale_recip, DTYPE_MIN, DTYPE_MAX)

    return x, scale_out


@triton.jit
def _fused_flatten_fp8_group_quant_kernel(
    x_ptr,
    out_ptr,
    out_scales_ptr,
    x_stride_m,
    x_stride_n1,
    x_stride_n2,
    out_stride_m,
    out_stride_n,
    out_scales_stride_m,
    out_scales_stride_n,
    N2,
    BLOCK_SIZE_N2: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
):
    m = tl.program_id(0)
    n1 = tl.program_id(1)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N2 // QUANT_BLOCK_SIZE
    # In the flattened (M, N1 * N2) output, each n1 segment is exactly N2 wide
    # (not BLOCK_SIZE_N2), so stride between n1 segments must use N2 — otherwise
    # non-power-of-2 N2 (e.g. 7168) over-strides the output (and at the last n1
    # walks past the row boundary, causing OOB writes).
    n2_groups = tl.cdiv(N2, QUANT_BLOCK_SIZE)

    n2_offs = tl.arange(0, BLOCK_SIZE_N2)
    x_mask = n2_offs < N2
    x_offs = m * x_stride_m + n1 * x_stride_n1 + n2_offs * x_stride_n2
    x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

    out, out_block_scales = _fp8_quant_op(
        x, 1, BLOCK_SIZE_N2, QUANT_BLOCK_SIZE, DTYPE_MAX, DTYPE_MIN
    )
    out = tl.ravel(out)
    out_block_scales = tl.ravel(out_block_scales)

    tl.store(
        out_ptr + m * out_stride_m + (n1 * N2 + n2_offs) * out_stride_n,
        out.to(out_ptr.dtype.element_ty),
        mask=x_mask,
    )
    block_scale_offs = tl.arange(0, NUM_QUANT_BLOCKS)
    tl.store(
        out_scales_ptr
        + m * out_scales_stride_m
        + (n1 * n2_groups + block_scale_offs) * out_scales_stride_n,
        out_block_scales.to(out_scales_ptr.dtype.element_ty),
        mask=block_scale_offs < n2_groups,
    )


@triton.jit
def _fused_reduce_act_mul_fp8_group_quant(
    x_ptr,
    y_ptr,
    y_scale_ptr,
    x2_ptr,
    y2_ptr,
    M,
    N1,
    N2,
    stride_x_spk,
    stride_x_m,
    stride_x_n,
    stride_y_m,
    stride_y_n,
    stride_y_scale_m,
    stride_y_scale_n,
    stride_x2_spk,
    stride_x2_m,
    stride_x2_n,
    stride_y2_m,
    stride_y2_n,
    # Meta-parameters
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE_M2: tl.constexpr,
    BLOCK_SIZE_N1: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
    X_HAS_SPLITK: tl.constexpr,
    X_NUM_KSPLIT: tl.constexpr,
    X_NUM_KSPLIT_POW2: tl.constexpr,
    X_MASK: tl.constexpr,
):

    tl.assume(stride_x_spk > 0)
    tl.assume(stride_x_m > 0)
    tl.assume(stride_x_n > 0)
    tl.assume(stride_y_m > 0)
    tl.assume(stride_y_n > 0)
    tl.assume(stride_y_scale_m > 0)
    tl.assume(stride_y_scale_n > 0)
    tl.assume(stride_x2_spk > 0)
    tl.assume(stride_x2_m > 0)
    tl.assume(stride_x2_n > 0)
    tl.assume(stride_y2_m > 0)
    tl.assume(stride_y2_n > 0)

    m_pid = tl.program_id(axis=0)
    if X_HAS_SPLITK and m_pid >= M:
        pid2 = m_pid - M
        num_pid_n2 = tl.cdiv(N2, BLOCK_SIZE_N2)
        pid_m2 = pid2 // num_pid_n2
        pid_n2 = pid2 % num_pid_n2
        offs_m2 = (pid_m2 * BLOCK_SIZE_M2 + tl.arange(0, BLOCK_SIZE_M2)) % M
        offs_n2 = (pid_n2 * BLOCK_SIZE_N2 + tl.arange(0, BLOCK_SIZE_N2)) % N2
        offs_spk = tl.arange(0, X_NUM_KSPLIT_POW2)
        x2_ptrs = (
            x2_ptr
            + offs_spk[:, None, None] * stride_x2_spk
            + offs_m2[None, :, None] * stride_x2_m
            + offs_n2[None, None, :] * stride_x2_n
        )
        if X_NUM_KSPLIT_POW2 == X_NUM_KSPLIT:
            x2 = tl.load(x2_ptrs)
        else:
            x2 = tl.load(
                x2_ptrs, mask=offs_spk[:, None, None] < X_NUM_KSPLIT, other=0.0
            )
        x2 = tl.sum(x2, axis=0)

        x2 = x2.to(y2_ptr.type.element_ty)

        y2_out_ptrs = (
            y2_ptr + (offs_m2[:, None] * stride_y2_m) + (offs_n2[None, :] * stride_y2_n)
        )

        tl.store(y2_out_ptrs, x2)
        return

    n_offs = tl.arange(0, BLOCK_SIZE_N1)
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N1 // QUANT_BLOCK_SIZE

    mask = None
    other = None
    if X_HAS_SPLITK:
        offs_spk = tl.arange(0, X_NUM_KSPLIT_POW2)
        x_ptrs = (
            x_ptr
            + offs_spk[:, None] * stride_x_spk
            + m_pid * stride_x_m
            + n_offs[None, :] * stride_x_n
        )
        if X_MASK:
            mask = (offs_spk[:, None] < X_NUM_KSPLIT) & (n_offs[None, :] < N1)
            other = 0.0
        else:
            mask = offs_spk[:, None] < X_NUM_KSPLIT
            other = 0.0
    else:
        x_ptrs = x_ptr + m_pid * stride_x_m + n_offs * stride_x_n
        if X_MASK:
            mask = n_offs < N1
            other = 0.0

    x = tl.load(
        x_ptrs,
        mask=mask,
        other=other,
        cache_modifier=".cg",
    ).to(tl.float32)
    x_mul = tl.load(
        x_ptrs + N1 * stride_x_n,
        mask=mask,
        other=other,
        cache_modifier=".cg",
    ).to(tl.float32)

    if X_HAS_SPLITK:
        x = tl.sum(x, axis=0)
        x_mul = tl.sum(x_mul, axis=0)

    x = ACTIVATION(x) * x_mul

    y, y_scale = _fp8_quant_op(
        x, 1, BLOCK_SIZE_N1, QUANT_BLOCK_SIZE, DTYPE_MAX, DTYPE_MIN
    )
    y = tl.ravel(y)
    y_scale = tl.ravel(y_scale)

    if X_MASK:
        mask = n_offs < N1
    else:
        mask = n_offs < N1
    tl.store(
        y_ptr + m_pid * stride_y_m + n_offs * stride_y_n,
        y.to(y_ptr.dtype.element_ty),
        mask=mask,
    )
    g_offs = tl.arange(0, NUM_QUANT_BLOCKS)
    num_bs_cols = (N1 + QUANT_BLOCK_SIZE - 1) // QUANT_BLOCK_SIZE
    tl.store(
        y_scale_ptr + m_pid * stride_y_scale_m + g_offs * stride_y_scale_n,
        y_scale.to(y_scale_ptr.dtype.element_ty),
        mask=g_offs < num_bs_cols,
    )


def fused_flatten_fp8_group_quant(
    x: torch.Tensor,
    group_size,
    dtype_quant=fp8_dtype,
):
    """
    Flatten the last two dimension of x and perform FP8 per-token group quantization along the last dimension

    Key parameters:
    - x: Matrix X with shape (M, N1, N2).

    Returns:
    - out: The output matrix with shape (M, N1 * N2).
    - out_block_scales: The output matrix with shape (M, cdiv((N1 * N2), group_size)).
    """
    M, N1, N2 = x.shape

    BLOCK_SIZE_N2 = max(triton.next_power_of_2(N2), group_size)
    N = N1 * N2
    num_bs_cols = triton.cdiv(N, group_size)
    out = torch.empty((M, N), dtype=dtype_quant, device=x.device)
    out_block_scales = torch.empty(
        (M, num_bs_cols), dtype=torch.float32, device=x.device
    )

    DTYPE_MAX = (
        torch.finfo(out.dtype).max
        if torch.is_floating_point(out)
        else torch.iinfo(out.dtype).max
    )
    grid = (
        M,
        N1,
    )
    _fused_flatten_fp8_group_quant_kernel[grid](
        x,
        out,
        out_block_scales,
        *x.stride(),
        *out.stride(),
        *out_block_scales.stride(),
        N2,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2,
        QUANT_BLOCK_SIZE=group_size,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
    )

    return out, out_block_scales


def fused_act_mul_fp8_group_quant(
    x: torch.Tensor,
    group_size=128,
    dtype_quant=fp8_dtype,
):
    """
    fused_reduce_act_mul_fp8_group_quant(x, activation="silu", x2=None) for a two-dimensional x:
    y = silu(x[:, :N1]) * x[:, N1:], followed by per-token group quantization.

    Args:
        x: (M, 2*N1), dtype = fp16 or bf16.

    Returns:
        tuple: (y, y_scale)
            y: (M, N1), dtype = dtype_quant
            y_scale: (M, cdiv(N1, group_size)), dtype = fp32
    """
    assert x.dim() == 2, "The two-dimensional path takes a (M, 2*N1) input"
    x_num_splitk = 1
    N2 = 1
    M, N1 = x.shape

    assert (
        N1 % 2 == 0
    ), "The last dimension for x1 should be multiple of 2 for acitvation and multiplication"
    N1 = N1 // 2

    y = torch.empty((M, N1), dtype=dtype_quant, device=x.device)
    y_scale = torch.empty(
        (M, (N1 + group_size - 1) // group_size),
        dtype=torch.float32,
        device=x.device,
    )

    BLOCK_SIZE_N1 = max(triton.next_power_of_2(N1), group_size)
    BLOCK_SIZE_N2 = max(triton.next_power_of_2(N2), 32)
    BLOCK_SIZE_M2 = 1 if M <= 128 else 4
    X_MASK = N1 % BLOCK_SIZE_N1 != 0

    DTYPE_MAX = (
        torch.finfo(y.dtype).max
        if torch.is_floating_point(y)
        else torch.iinfo(y.dtype).max
    )
    num_pid = M
    grid = (num_pid,)
    # The split-K strides are unused without split-K; they are passed as 1 so that the kernel's
    # tl.assume(stride > 0) hints hold.
    _fused_reduce_act_mul_fp8_group_quant[grid](
        x,
        y,
        y_scale,
        None,
        None,
        M,
        N1,
        N2,
        1,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        y_scale.stride(0),
        y_scale.stride(1),
        1,
        1,
        1,
        1,
        1,
        ACTIVATION=_silu,
        BLOCK_SIZE_M2=BLOCK_SIZE_M2,
        BLOCK_SIZE_N1=BLOCK_SIZE_N1,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2,
        QUANT_BLOCK_SIZE=group_size,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        X_HAS_SPLITK=False,
        X_NUM_KSPLIT=x_num_splitk,
        X_NUM_KSPLIT_POW2=triton.next_power_of_2(x_num_splitk),
        X_MASK=X_MASK,
        num_warps=1 if max(BLOCK_SIZE_N1, BLOCK_SIZE_N2) <= 512 else 4,
    )

    return y, y_scale
