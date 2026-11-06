# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone RMSNorm (forward) Triton kernel (AITER Triton rms_norm).

Provenance: ported from aiter.ops.triton.normalization.rmsnorm (`rms_norm` ->
`_rmsnorm_forward` / `rmsnorm_forward_inference`) and its device kernels
`_rms_norm_kernel` / `_rmsnorm_kernel_large_m_small_n`
(aiter.ops.triton._triton_kernels.normalization.rmsnorm). The quant / fused-add /
backward paths and the `get_num_sms` helper are dropped/inlined so the module
depends only on `triton` + `torch`.

Op:
    y[i, :] = x[i, :] * rsqrt(mean(x[i, :]^2) + eps) * g
with the sum-of-squares + scale accumulated in fp32 and the result truncated to
the input dtype. A blocked persistent kernel handles wide rows; a tiled
`large_m_small_n` kernel handles the tall/narrow regime.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    # Pointers to matrices
    input_ptr,
    output_ptr,
    g_ptr,
    rsigma_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `input_row_stride` is
    # how much to increase `input_ptr` by to get the element one row down.
    input_row_stride,
    output_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    epsilon,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call rms_norm function
    below.

    Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    Key parameters:
    - Input: The input tensor to be normalized with shape (n_rows, n_cols).
    - Output: The output tensor with shape (n_rows, n_cols).
    - G: The learnable weights tensor with shape (n_cols, ).
    """
    # Map the program id to the first row of input and output it should compute.
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    if USE_BLOCKED:
        # Persistent loop for rows
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride

            # Accumulate sum of squares
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            sum_squares = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs).to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            sum_squares += tl.sum(x * x, axis=0)

            # Compute normalization factor
            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)

            # Store rsigma (norm_factor)
            tl.store(rsigma_ptr + row_idx, norm_factor)

            # Normalize and write output
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                rms_norm = x * norm_factor * g
                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            rms_norm = x * norm_factor * g
            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)

    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

            # Store rsigma (norm_factor)
            tl.store(rsigma_ptr + row_idx, norm_factor)

            rms_norm = row * norm_factor * g

            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16,))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def _rmsnorm_kernel_large_m_small_n(
    X,
    Y,
    W,
    RSIGMA,
    M,
    N,
    eps,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = tl.arange(0, BLOCK_N)

    mask_m = m_off < M
    mask_n = n_off < N
    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(
        X + m_off[:, None] * stride_xm + n_off[None, :] * stride_xn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    w = tl.load(W + n_off, mask=mask_n, other=0.0).to(tl.float32)

    x = tl.where(mask, x, 0.0)
    sum_sq = tl.sum(x * x, axis=1)
    var = sum_sq / N
    rsigma = tl.math.rsqrt(var + eps)

    y = x * rsigma[:, None] * w[None, :]
    tl.store(
        Y + m_off[:, None] * stride_ym + n_off[None, :] * stride_yn,
        y.to(Y.dtype.element_ty),
        mask=mask,
    )

    if RSIGMA is not None:
        tl.store(RSIGMA + m_off, rsigma, mask=mask_m)


def _get_num_sms():
    # Inlined from utils.device_info.get_num_sms (number of compute units).
    return torch.cuda.get_device_properties(0).multi_processor_count


def num_programs(x):
    return min(x.shape[0], _get_num_sms())


def block_size(x):
    return min(65536 // x.element_size(), triton.next_power_of_2(x.shape[1]))


def use_blocked(x):
    return x.shape[1] > block_size(x)


def _rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, epsilon: float):

    n_rows, n_cols = x.shape

    rsigma = torch.empty((n_rows,), dtype=torch.float32, device=x.device)

    blk_size = block_size(x)
    USE_BLOCKED = use_blocked(x)
    NUM_PRGMS = num_programs(x)

    grid = lambda meta: (NUM_PRGMS,)  # noqa: E731
    _rms_norm_kernel[grid](
        x,
        out,
        weight,
        rsigma,
        x.stride(0),
        out.stride(0),
        n_rows,
        n_cols,
        epsilon,
        blk_size,
        USE_BLOCKED,
        NUM_PRGMS,
    )


def _should_use_large_m_small_n(M: int, N: int) -> bool:

    if M > 8192 and N <= 2048:
        return True

    return False


def _rmsnorm_forward_large_m_small_n(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
):
    M, N = x.shape

    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_M = min(16384 // BLOCK_N, 32)
    BLOCK_M = max(BLOCK_M, 8)

    grid = (triton.cdiv(M, BLOCK_M),)
    _rmsnorm_kernel_large_m_small_n[grid](
        x,
        out,
        weight,
        None,
        M,
        N,
        eps,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )


def run(*, x, weight, epsilon, out) -> None:
    M, N = x.shape
    if _should_use_large_m_small_n(M, N):
        _rmsnorm_forward_large_m_small_n(x, weight, out, epsilon)
    else:
        _rmsnorm_forward(x, weight, out, epsilon)
