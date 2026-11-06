# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""RMSNorm backward (dx and per-program partial dgamma plus its reduction), from AITER's Triton kernels."""
import triton
import triton.language as tl
import torch


@triton.jit
def _rmsnorm_bwd_triton(
    grad_output_ptr,
    input_ptr,
    g_ptr,
    rsigma_ptr,
    dx_ptr,
    dg_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    if USE_BLOCKED:
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_grad_output_ptr = grad_output_ptr + row_idx * output_row_stride
            row_dx_ptr = dx_ptr + row_idx * input_row_stride
            row_dg_ptr = dg_ptr + row_idx * input_row_stride

            # Compute gradients sum of all colums for each row
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            # older version of triton doesn't accept below init
            # comment out for now to make it compatible with triton 3.1
            # grad_sum: tl.float32 = 0.0
            grad_sum = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                grad_output_ptrs = row_grad_output_ptr + cols

                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16,))

                x = tl.load(input_ptrs).to(tl.float32)
                grad_output = tl.load(grad_output_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                grad_sum += tl.sum(grad_output * x * g, axis=0)

            # remainder for grad_sum:
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_output_ptrs = row_grad_output_ptr + cols
            grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0).to(tl.float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_sum += tl.sum(grad_output * x * g, axis=0)

            # Load r_sigma
            norm_factor = tl.load(rsigma_ptr + row_idx).to(tl.float32)

            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                grad_output_ptrs = row_grad_output_ptr + cols

                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16,))

                x = tl.load(input_ptrs).to(tl.float32)
                grad_output = tl.load(grad_output_ptrs).to(tl.float32)

                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                grad_input = grad_output * norm_factor * g - (
                    norm_factor * norm_factor * norm_factor
                ) * x * (grad_sum / n_cols)

                dx_ptrs = row_dx_ptr + cols
                tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty))

                dg = grad_output * x * norm_factor
                dg_ptrs = row_dg_ptr + cols
                tl.store(dg_ptrs, dg.to(tl.float32))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols

            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_output_ptrs = row_grad_output_ptr + cols
            grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0).to(tl.float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_input = grad_output * norm_factor * g - (
                norm_factor * norm_factor * norm_factor
            ) * x * (grad_sum / n_cols)

            dx_ptrs = row_dx_ptr + cols
            tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty), mask=mask)

            dg = grad_output * x * norm_factor
            dg_ptrs = row_dg_ptr + cols
            tl.store(dg_ptrs, dg.to(tl.float32), mask=mask)

    else:
        mask = col_offsets < n_cols
        dg_col_redux = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            grad_output_ptrs = (
                grad_output_ptr + row_idx * output_row_stride + col_offsets
            )
            dx_ptrs = dx_ptr + row_idx * input_row_stride + col_offsets

            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16,))
            dx_ptrs = tl.multiple_of(dx_ptrs, (16,))

            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

            norm_factor = tl.load(rsigma_ptr + row_idx).to(tl.float32)
            grad_sum = tl.sum(grad_output * x * g, axis=0)

            grad_input = grad_output * norm_factor * g - (
                norm_factor * norm_factor * norm_factor
            ) * x * (grad_sum / n_cols)
            tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty), mask=mask)

            dg = grad_output * x * norm_factor
            dg_col_redux += dg.to(tl.float32)

        tl.store(
            dg_ptr + tl.program_id(0) * input_row_stride + col_offsets,
            dg_col_redux,
            mask=mask,
        )


@triton.jit
def _rmsnorm_bwd_dg_reduce_triton(
    dg_in_ptr,
    dg_out_ptr,
    dg_in_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # we want parallelism in N direction
    # if N is small, we will just use one CU,
    # otherwise, it can be split by N/BLOCK_SIZE
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, n_rows, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
        offs = rows[:, None] * n_cols + cols[None, :]
        acc += tl.load(dg_in_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg").to(
            tl.float32
        )

    sum_dg = tl.sum(acc, axis=0)
    tl.store(
        dg_out_ptr + cols, sum_dg.to(dg_out_ptr.type.element_ty), mask=cols < n_cols
    )



def run(*, dy, x, gamma, rsigma, dx, dgamma):
    M, N = x.shape
    blk_size = min(65536 // x.element_size(), triton.next_power_of_2(N))
    USE_BLOCKED = N > blk_size
    NUM_PRGMS = min(M, torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
    # One row of partial dgamma per program (the non-blocked path of the kernel; N = 4096 never blocks).
    dg_tmp = torch.empty(M if USE_BLOCKED else NUM_PRGMS, N, device=x.device, dtype=torch.float32)

    _rmsnorm_bwd_triton[(NUM_PRGMS,)](
        dy,
        x,
        gamma,
        rsigma,
        dx,
        dg_tmp,
        x.stride(0),
        dy.stride(0),
        M,
        N,
        blk_size,
        USE_BLOCKED,
        NUM_PRGMS,
        num_warps=8,
    )
    _rmsnorm_bwd_dg_reduce_triton[(triton.cdiv(N, 64),)](
        dg_tmp,
        dgamma,
        dg_tmp.stride(0),
        dg_tmp.shape[0],
        dg_tmp.shape[1],
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=64,
    )
