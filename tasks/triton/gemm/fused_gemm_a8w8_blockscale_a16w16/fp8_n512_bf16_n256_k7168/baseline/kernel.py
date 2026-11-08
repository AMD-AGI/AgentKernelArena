# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


"""Fused block-scaled fp8 GEMM and bf16 GEMM from ROCm/aiter (Triton): one launch computes the fp8 gate-up projection
(x_fp8 @ w_fp8^T with 1x128 activation and 128x128 weight scales) and the bf16 MoE router gate (x_bf16 @ w_bf16^T).

Provenance: the device kernels of aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_a8w8_blockscale_a16w16.py
(`_fused_gemm_a8w8_blockscale_a16w16_kernel` and `_fused_gemm_a8w8_blockscale_a16w16_reduce_kernel`), the launcher
`fused_gemm_a8w8_blockscale_a16w16` of aiter/ops/triton/gemm/fused/fused_gemm_a8w8_blockscale_a16w16.py, and `remap_xcd` and
`pid_grid` of aiter/ops/triton/utils/_triton/pid_preprocessing.py. The tuned table is the shape-specialized gfx950 file
configs/gemm/gfx950-FUSED-GEMM-A8W8_BLOCKSCALE-A16W16-N8=512-N16=256-K=7168.json, inlined with the get_gemm_config lookup
order. Split-K (NUM_KSPLIT up to 14 for M <= 256 in the table) writes float32 partial sums that the reduce kernel adds.
"""


import torch
import triton
import triton.language as tl


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    if tall_xcds == 0:
        tall_xcds = tl.cast(NUM_XCDS, tall_xcds.type)
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    return pid


@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
    """
    Maps 1D pid to 2D grid coords (pid_m, pid_n).

    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - GROUP_SIZE_M: tl.constexpr: default is 1
    """
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN_FP8": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N_fp8"], args["BLOCK_SIZE_N"]),
        "GRID_MN_BF16": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N_bf16"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _fused_gemm_a8w8_blockscale_a16w16_kernel(
    # Pointers to matrices
    a_fp8_ptr,
    b_fp8_ptr,
    bias_fp8_ptr,
    a_fp8_scale_ptr,
    b_fp8_scale_ptr,
    c_fp8_ptr,
    a_bf16_ptr,
    b_bf16_ptr,
    bias_bf16_ptr,
    c_bf16_ptr,
    # Matrix dimensions
    M,
    N_fp8,
    N_bf16,
    K,
    stride_a_fp8_m,
    stride_a_fp8_k,
    stride_b_fp8_k,
    stride_b_fp8_n,
    stride_a_fp8_scale_m,
    stride_a_fp8_scale_k,
    stride_b_fp8_scale_k,
    stride_b_fp8_scale_n,
    stride_c_fp8_k,
    stride_c_fp8_m,
    stride_c_fp8_n,
    stride_a_bf16_m,
    stride_a_bf16_k,
    stride_b_bf16_k,
    stride_b_bf16_n,
    stride_c_bf16_k,
    stride_c_bf16_m,
    stride_c_bf16_n,
    # Meta-parameters
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    ADD_BIAS_FP8: tl.constexpr,
    ADD_BIAS_BF16: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN_FP8: tl.constexpr,
    GRID_MN_BF16: tl.constexpr,
    SKIP_REDUCE: tl.constexpr,
    cache_modifier: tl.constexpr,
):

    tl.assume(stride_a_fp8_m > 0)
    tl.assume(stride_a_fp8_k > 0)
    tl.assume(stride_b_fp8_k > 0)
    tl.assume(stride_b_fp8_n > 0)
    tl.assume(stride_c_fp8_k > 0)
    tl.assume(stride_c_fp8_m > 0)
    tl.assume(stride_c_fp8_n > 0)
    tl.assume(stride_a_fp8_scale_m > 0)
    tl.assume(stride_a_fp8_scale_k > 0)
    tl.assume(stride_b_fp8_scale_k > 0)
    tl.assume(stride_b_fp8_scale_n > 0)

    tl.assume(stride_a_bf16_m > 0)
    tl.assume(stride_a_bf16_k > 0)
    tl.assume(stride_b_bf16_k > 0)
    tl.assume(stride_b_bf16_n > 0)
    tl.assume(stride_c_bf16_m > 0)
    tl.assume(stride_c_bf16_n > 0)

    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n_fp8 = tl.cdiv(N_fp8, BLOCK_SIZE_N)
    num_pid_n_bf16 = tl.cdiv(N_bf16, BLOCK_SIZE_N)
    num_pid_n = num_pid_n_fp8 + num_pid_n_bf16

    if NUM_KSPLIT == 1:
        GRID_MN: tl.constexpr = GRID_MN_FP8 + GRID_MN_BF16
        remap_xcd(pid, GRID_MN)

        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE) < K:

        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)
        acc_dtype = tl.float32 if c_fp8_ptr.type.element_ty != tl.int8 else tl.int32

        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_ks_step = BLOCK_SIZE_K // GROUP_K

        if pid_n < num_pid_n_fp8:
            offs_b_fp8_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_fp8
            a_fp8_ptrs = a_fp8_ptr + (
                offs_am[:, None] * stride_a_fp8_m
                + offs_k_split[None, :] * stride_a_fp8_k
            )
            b_fp8_ptrs = b_fp8_ptr + (
                offs_k_split[:, None] * stride_b_fp8_k
                + offs_b_fp8_n[None, :] * stride_b_fp8_n
            )

            offs_ks = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
            a_scale_ptrs = (
                a_fp8_scale_ptr
                + offs_am * stride_a_fp8_scale_m
                + offs_ks * stride_a_fp8_scale_k
            )
            offs_bsn = offs_b_fp8_n // GROUP_N
            b_scale_ptrs = (
                b_fp8_scale_ptr
                + offs_ks * stride_b_fp8_scale_k
                + offs_bsn * stride_b_fp8_scale_n
            )

            accumulator_fp8 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            if ADD_BIAS_FP8 and (NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0)):
                bias_fp8_vals = tl.load(bias_fp8_ptr + offs_b_fp8_n).to(dtype=acc_dtype)
                accumulator_fp8 += bias_fp8_vals[None, :]

            for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
                if EVEN_K:
                    a = tl.load(a_fp8_ptrs)
                    b = tl.load(b_fp8_ptrs, cache_modifier=cache_modifier)
                else:
                    a = tl.load(
                        a_fp8_ptrs,
                        mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                        other=0.0,
                    )
                    b = tl.load(
                        b_fp8_ptrs,
                        mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0,
                        cache_modifier=cache_modifier,
                    )

                a_scale = tl.load(a_scale_ptrs)
                b_scale = tl.load(b_scale_ptrs)

                accumulator_fp8 += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]

                a_fp8_ptrs += BLOCK_SIZE_K * stride_a_fp8_k
                b_fp8_ptrs += BLOCK_SIZE_K * stride_b_fp8_k
                a_scale_ptrs += offs_ks_step * stride_a_fp8_scale_k
                b_scale_ptrs += offs_ks_step * stride_b_fp8_scale_k

            c_fp8 = accumulator_fp8.to(c_fp8_ptr.type.element_ty)

            offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
                tl.int64
            )
            offs_c_fp8_n = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(
                0, BLOCK_SIZE_N
            ).to(tl.int64)
            c_fp8_ptrs = (
                c_fp8_ptr
                + stride_c_fp8_m * offs_cm[:, None]
                + stride_c_fp8_n * offs_c_fp8_n[None, :]
                + pid_k * stride_c_fp8_k
            )
            c_fp8_mask = (offs_cm[:, None] < M) & (offs_c_fp8_n[None, :] < N_fp8)
            tl.store(c_fp8_ptrs, c_fp8, mask=c_fp8_mask)
        else:
            pid_n -= num_pid_n_fp8

            offs_b_bf16_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_bf16
            a_ptrs = a_bf16_ptr + (
                offs_am[:, None] * stride_a_bf16_m
                + offs_k_split[None, :] * stride_a_bf16_k
            )
            b_ptrs = b_bf16_ptr + (
                offs_k_split[:, None] * stride_b_bf16_k
                + offs_b_bf16_n[None, :] * stride_b_bf16_n
            )

            accumulator_bf16 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
            if ADD_BIAS_BF16 and (NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0)):
                bias_bf16_vals = tl.load(bias_bf16_ptr + offs_b_bf16_n).to(
                    dtype=acc_dtype
                )
                accumulator_bf16 += bias_bf16_vals[None, :]

            for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
                if EVEN_K:
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs, cache_modifier=cache_modifier)
                else:
                    a = tl.load(
                        a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0,
                        cache_modifier=cache_modifier,
                    )

                accumulator_bf16 = tl.dot(a, b, acc=accumulator_bf16)

                a_ptrs += BLOCK_SIZE_K * stride_a_bf16_k
                b_ptrs += BLOCK_SIZE_K * stride_b_bf16_k

            c_bf16 = accumulator_bf16.to(c_bf16_ptr.type.element_ty)

            # Write back the block of the output matrix C with masks.
            offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
                tl.int64
            )
            offs_c_bf16_n = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(
                0, BLOCK_SIZE_N
            ).to(tl.int64)
            c_bf16_ptrs = (
                c_bf16_ptr
                + stride_c_bf16_m * offs_cm[:, None]
                + stride_c_bf16_n * offs_c_bf16_n[None, :]
                + pid_k * stride_c_bf16_k
            )
            c_bf16_mask = (offs_cm[:, None] < M) & (offs_c_bf16_n[None, :] < N_bf16)
            tl.store(c_bf16_ptrs, c_bf16, mask=c_bf16_mask)


@triton.jit
def _fused_gemm_a8w8_blockscale_a16w16_reduce_kernel(
    bias_fp8_ptr,
    c_fp8_in_ptr,
    c_fp8_out_ptr,
    bias_bf16_ptr,
    c_bf16_in_ptr,
    c_bf16_out_ptr,
    M,
    N_fp8,
    N_bf16,
    stride_c_fp8_in_k,
    stride_c_fp8_in_m,
    stride_c_fp8_in_n,
    stride_c_fp8_out_m,
    stride_c_fp8_out_n,
    stride_c_bf16_in_k,
    stride_c_bf16_in_m,
    stride_c_bf16_in_n,
    stride_c_bf16_out_m,
    stride_c_bf16_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
    ADD_BIAS_FP8: tl.constexpr,
    ADD_BIAS_BF16: tl.constexpr,
):

    tl.assume(stride_c_fp8_in_k > 0)
    tl.assume(stride_c_fp8_in_m > 0)
    tl.assume(stride_c_fp8_in_n > 0)
    tl.assume(stride_c_fp8_out_m > 0)
    tl.assume(stride_c_fp8_out_n > 0)

    tl.assume(stride_c_bf16_in_k > 0)
    tl.assume(stride_c_bf16_in_m > 0)
    tl.assume(stride_c_bf16_in_n > 0)
    tl.assume(stride_c_bf16_out_m > 0)
    tl.assume(stride_c_bf16_out_n > 0)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M

    num_pid_n_fp8 = tl.cdiv(N_fp8, BLOCK_SIZE_N)
    offs_k = tl.arange(0, MAX_KSPLIT)
    acc_dtype = tl.float32 if c_fp8_in_ptr.type.element_ty != tl.int8 else tl.int32

    if pid_n < num_pid_n_fp8:
        offs_fp8_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_fp8
        c_fp8_in_ptrs = (
            c_fp8_in_ptr
            + (offs_k[:, None, None] * stride_c_fp8_in_k)
            + (offs_m[None, :, None] * stride_c_fp8_in_m)
            + (offs_fp8_n[None, None, :] * stride_c_fp8_in_n)
        )

        if ACTUAL_KSPLIT == MAX_KSPLIT:
            c_fp8 = tl.load(c_fp8_in_ptrs)
        else:
            c_fp8 = tl.load(
                c_fp8_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT, other=0.0
            )
        c_fp8 = tl.sum(c_fp8, axis=0)
        if ADD_BIAS_FP8:
            bias_fp8 = tl.load(bias_fp8_ptr + offs_fp8_n).to(dtype=acc_dtype)
            bias_fp8 = tl.broadcast_to(bias_fp8[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
            c_fp8 += bias_fp8

        c_fp8 = c_fp8.to(c_fp8_out_ptr.type.element_ty)

        c_fp8_out_ptrs = (
            c_fp8_out_ptr
            + (offs_m[:, None] * stride_c_fp8_out_m)
            + (offs_fp8_n[None, :] * stride_c_fp8_out_n)
        )

        tl.store(c_fp8_out_ptrs, c_fp8)
    else:
        pid_n -= num_pid_n_fp8

        offs_bf16_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_bf16
        c_bf16_in_ptrs = (
            c_bf16_in_ptr
            + (offs_k[:, None, None] * stride_c_bf16_in_k)
            + (offs_m[None, :, None] * stride_c_bf16_in_m)
            + (offs_bf16_n[None, None, :] * stride_c_bf16_in_n)
        )

        if ACTUAL_KSPLIT == MAX_KSPLIT:
            c_bf16 = tl.load(c_bf16_in_ptrs)
        else:
            c_bf16 = tl.load(
                c_bf16_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT, other=0.0
            )
        c_bf16 = tl.sum(c_bf16, axis=0)
        if ADD_BIAS_BF16:
            bias_bf16 = tl.load(bias_bf16_ptr + offs_bf16_n).to(dtype=acc_dtype)
            bias_bf16 = tl.broadcast_to(
                bias_bf16[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
            )
            c_bf16 += bias_bf16

        c_bf16 = c_bf16.to(c_bf16_out_ptr.type.element_ty)

        c_bf16_out_ptrs = (
            c_bf16_out_ptr
            + (offs_m[:, None] * stride_c_bf16_out_m)
            + (offs_bf16_n[None, :] * stride_c_bf16_out_n)
        )
        c_bf16_mask = (offs_m[:, None] < M) & (offs_bf16_n[None, :] < N_bf16)
        tl.store(c_bf16_out_ptrs, c_bf16, mask=c_bf16_mask)


# configs/gemm/gfx950-FUSED-GEMM-A8W8_BLOCKSCALE-A16W16-N8=512-N16=256-K=7168.json
_CONFIGS = {
    "M_LEQ_8": dict(BLOCK_SIZE_M=8, BLOCK_SIZE_N=16, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=2, num_stages=2, waves_per_eu=8, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=14),
    "M_LEQ_16": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=2, num_stages=2, waves_per_eu=8, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=14),
    "M_LEQ_32": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=6, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=14),
    "M_LEQ_64": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=2, waves_per_eu=6, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=14),
    "M_LEQ_128": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=2, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=7),
    # Task fix: the upstream row has BLOCK_SIZE_K=256, but the kernel applies one scale pair per BLOCK_SIZE_K tile and
    # so needs BLOCK_SIZE_K == GROUP_K (128); with 256 it reads every second scale (about 16 dB SQNR). Only that field changed.
    "M_LEQ_256": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=4),
    "M_LEQ_1024": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_warps=4, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=32, cache_modifier=None, NUM_KSPLIT=1),
    "M_LEQ_2048": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=4, num_warps=4, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=32, cache_modifier=None, NUM_KSPLIT=1),
    "any": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_warps=2, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=32, cache_modifier=None, NUM_KSPLIT=1),
}


_STANDARD_M_BOUNDS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def _table_config(M):
    """Row of the tuned table for M, in the lookup order of AITER's get_gemm_config."""
    for bound in _STANDARD_M_BOUNDS:
        key = f"M_LEQ_{bound}"
        if M <= bound and key in _CONFIGS:
            return dict(_CONFIGS[key])
    for bound in reversed(_STANDARD_M_BOUNDS):
        key = f"M_GEQ_{bound}"
        if M >= bound and key in _CONFIGS:
            return dict(_CONFIGS[key])
    return dict(_CONFIGS["any"])


def _get_config(M, N_fp8, N_bf16, K):
    # Custom file naming upstream: N8={N_fp8}-N16={N_bf16}-K={K}; the table above is that file for (512, 256, 7168).
    return _table_config(M)


def run(*, x_fp8, x_fp8_scale, w_fp8, w_fp8_scale, x_bf16, w_bf16, y_a8w8, y_a16w16) -> None:
    """y_a8w8 = block-scaled fp8 x_fp8 @ w_fp8^T and y_a16w16 = x_bf16 @ w_bf16^T (the wrapper's y_fp8 and y_bf16)."""
    M, K = x_fp8.shape
    N_fp8, K = w_fp8.shape
    N_bf16, K = w_bf16.shape

    # Transpose w and w_scale
    w_fp8 = w_fp8.T
    w_bf16 = w_bf16.T
    w_fp8_scale = w_fp8_scale.T

    config = _get_config(M, N_fp8, N_bf16, K)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])
    if config["NUM_KSPLIT"] > 1:
        y_fp8_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N_fp8),
            dtype=torch.float32,
            device=x_fp8.device,
        )
        y_bf16_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N_bf16),
            dtype=torch.float32,
            device=x_bf16.device,
        )
    else:
        y_fp8_pp = None
        y_bf16_pp = None

    if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 4
    config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 16)

    # Scale block sizes
    config["GROUP_K"] = triton.next_power_of_2(triton.cdiv(K, w_fp8_scale.shape[0]))
    config["GROUP_N"] = triton.next_power_of_2(triton.cdiv(N_fp8, w_fp8_scale.shape[1]))

    # The kernel applies one scale pair per BLOCK_SIZE_K tile, so a tile must not span two scale blocks.
    assert config["GROUP_K"] == config["BLOCK_SIZE_K"], "GROUP_K must equal BLOCK_SIZE_K"

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * (
                triton.cdiv(N_fp8, META["BLOCK_SIZE_N"])
                + triton.cdiv(N_bf16, META["BLOCK_SIZE_N"])
            )
        ),
    )
    _fused_gemm_a8w8_blockscale_a16w16_kernel[grid](
        x_fp8,
        w_fp8,
        None,
        x_fp8_scale,
        w_fp8_scale,
        y_a8w8 if config["NUM_KSPLIT"] == 1 else y_fp8_pp,
        x_bf16,
        w_bf16,
        None,
        y_a16w16 if config["NUM_KSPLIT"] == 1 else y_bf16_pp,
        M,
        N_fp8,
        N_bf16,
        K,
        x_fp8.stride(0),
        x_fp8.stride(1),
        w_fp8.stride(0),
        w_fp8.stride(1),
        x_fp8_scale.stride(0),
        x_fp8_scale.stride(1),
        w_fp8_scale.stride(0),
        w_fp8_scale.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_fp8_pp.stride(0),
        y_a8w8.stride(0) if config["NUM_KSPLIT"] == 1 else y_fp8_pp.stride(1),
        y_a8w8.stride(1) if config["NUM_KSPLIT"] == 1 else y_fp8_pp.stride(2),
        x_bf16.stride(0),
        x_bf16.stride(1),
        w_bf16.stride(0),
        w_bf16.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_bf16_pp.stride(0),
        y_a16w16.stride(0) if config["NUM_KSPLIT"] == 1 else y_bf16_pp.stride(1),
        y_a16w16.stride(1) if config["NUM_KSPLIT"] == 1 else y_bf16_pp.stride(2),
        ADD_BIAS_FP8=False,
        ADD_BIAS_BF16=False,
        SKIP_REDUCE=False,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N_fp8, REDUCE_BLOCK_SIZE_N)
            + triton.cdiv(N_bf16, REDUCE_BLOCK_SIZE_N),
        )
        _fused_gemm_a8w8_blockscale_a16w16_reduce_kernel[grid_reduce](
            None,
            y_fp8_pp,
            y_a8w8,
            None,
            y_bf16_pp,
            y_a16w16,
            M,
            N_fp8,
            N_bf16,
            y_fp8_pp.stride(0),
            y_fp8_pp.stride(1),
            y_fp8_pp.stride(2),
            y_a8w8.stride(0),
            y_a8w8.stride(1),
            y_bf16_pp.stride(0),
            y_bf16_pp.stride(1),
            y_bf16_pp.stride(2),
            y_a16w16.stride(0),
            y_a16w16.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS_FP8=False,
            ADD_BIAS_BF16=False,
        )
