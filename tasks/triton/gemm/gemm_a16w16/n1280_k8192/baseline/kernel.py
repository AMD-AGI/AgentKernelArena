# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 GEMM out = a @ b.T in Triton, after AITER's gemm_a16w16 (single-pass, tile-grouped)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)

    # Grouped tile order for L2 reuse.
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K + offs_k
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile, tl.trans(w_tile))

        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def _get_config(M, N, K):
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    if M <= 32:
        config["BLOCK_SIZE_M"] = 32
    if N <= 32:
        config["BLOCK_SIZE_N"] = 32
    if K <= 32:
        config["BLOCK_SIZE_K"] = 16
    if M >= 2048 and N >= 2048:
        config["BLOCK_SIZE_M"] = 128
        config["BLOCK_SIZE_N"] = 128
        config["BLOCK_SIZE_K"] = 64
        config["GROUP_SIZE_M"] = 8
    return config


def run(*, x, w, out):
    M, K = x.shape
    N, _ = w.shape
    config = _get_config(M, N, K)
    grid = (triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    _gemm_kernel[grid](
        x,
        w,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        num_warps=4,
        num_stages=2,
    )
