# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused router GEMM + sigmoid + top-1 (+ shared expert column) in Triton.

Vendored from ROCm/aiter (aiter/ops/triton/_triton_kernels/moe/moe_routing_sigmoid_top1_fused.py and
aiter/ops/triton/moe/moe_routing_sigmoid_top1_fused.py): the kernel is unchanged except that the two output stores are
masked to the valid rows (upstream stores unmasked, which writes past the output when M is not a multiple of
BLOCK_M) and that X is rebased on each block's first row with a 64-bit offset (upstream forms offs_m * stride_xm in
32 bits, which wraps once M * K exceeds 2**31 elements); the gfx950 tuning table is inlined and the wrapper writes
into preallocated outputs.
"""

import functools

import triton
import triton.language as tl

# Tuning table of aiter/ops/triton/configs/moe/gfx950-MOE_ROUTING_SIGMOID_TOPK1.json (N128 entries).
_CONFIGS = {
    "small": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 8, "num_stages": 1, "waves_per_eu": 0, "kpack": 1},
    "medium": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 8, "num_stages": 1, "waves_per_eu": 0, "kpack": 1},
    "large": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "kpack": 1},
    "xlarge": {"BLOCK_M": 32, "BLOCK_K": 128, "num_warps": 8, "num_stages": 2, "waves_per_eu": 2, "kpack": 1},
}


@triton.jit
def _routing_sigmoid_top1_kernel(
    X_ptr,
    W_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_topk_ids_m,
    stride_topk_ids_n,
    stride_topk_weights_m,
    stride_topk_weights_n,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOPK: tl.constexpr,
    FUSED_SHARED_EXPERTS: tl.constexpr,
):
    # Program ID corresponds to the block index in M dimension
    pid_m = tl.program_id(axis=0)

    # Offsets for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    _TOPK: tl.constexpr = TOPK + 1 if FUSED_SHARED_EXPERTS else TOPK

    offs_topk = tl.arange(0, _TOPK)

    # Masks for bounds checking
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Rebase X on the block's first row with a 64-bit offset: M * K can exceed 2**31 elements, where the 32-bit
    # product offs_m * stride_xm wraps. Offsets inside the block stay 32-bit.
    X_rows_ptr = X_ptr + pid_m.to(tl.int64) * BLOCK_M * stride_xm
    offs_xm = tl.arange(0, BLOCK_M)

    # Initialize accumulator for matmul (will be in float32 due to default acc_type)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        # Compute pointers for A and B
        offs_k_iter = k + offs_k
        mask_k = offs_k_iter < K

        X_ptrs = X_rows_ptr + (
            # pyre-ignore
            offs_xm[:, None] * stride_xm
            + offs_k_iter[None, :] * stride_xk
        )
        W_ptrs = W_ptr + (
            offs_k_iter[:, None] * stride_wk + offs_n[None, :] * stride_wn
        )

        # Load A and B tiles
        # pyre-ignore
        x = tl.load(X_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        w = tl.load(W_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)

        # Compute partial matmul for the current block using FP16 inputs and FP32 accumulation
        acc = tl.dot(x, w, acc=acc)

    acc = tl.sigmoid(acc)
    # Get topk results
    topk_ids = tl.argmax(acc, axis=1, tie_break_left=True)  # Shape: (BLOCK_M,)
    topk_weights = tl.max(acc, axis=1)  # Shape: (BLOCK_M,)

    # Create buffers for results
    topk_ids_buffer = tl.zeros((BLOCK_M, _TOPK), dtype=tl.int32)
    topk_weights_buffer = tl.zeros((BLOCK_M, _TOPK), dtype=tl.float32)

    if FUSED_SHARED_EXPERTS:
        # Set the first column with broadcasting
        topk_ids_buffer = tl.where(
            (offs_topk[None, :] < _TOPK - 1), topk_ids[:, None], N
        )
        topk_weights_buffer = tl.where(
            (offs_topk[None, :] < _TOPK - 1), topk_weights[:, None], 1.0
        )
    else:
        topk_ids_buffer = topk_ids[:, None]
        topk_weights_buffer = topk_weights[:, None]

    topk_ids_ptrs = (
        topk_ids_ptr
        + offs_m[:, None] * stride_topk_ids_m
        + offs_topk[None, :] * stride_topk_ids_n
    )

    topk_weights_ptrs = (
        topk_weights_ptr
        + offs_m[:, None] * stride_topk_weights_m
        + offs_topk[None, :] * stride_topk_weights_n
    )

    tl.store(topk_ids_ptrs, topk_ids_buffer, mask=mask_m[:, None])
    tl.store(topk_weights_ptrs, topk_weights_buffer, mask=mask_m[:, None])


@functools.lru_cache(maxsize=1024)
def _get_config(M):
    m_key = "xlarge" if M >= 8192 else "large" if M >= 4096 else "medium" if M >= 2048 else "small"
    return _CONFIGS[m_key]


def run(*, x, w, topk_ids, topk_weights):
    M, K = x.shape
    Kb, N = w.shape
    assert K == Kb
    shared = topk_ids.shape[1] == 2
    config = _get_config(M)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]),)

    _routing_sigmoid_top1_kernel[grid](
        x,
        w,
        topk_ids,
        topk_weights,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        BLOCK_N=N,
        TOPK=1,
        FUSED_SHARED_EXPERTS=shared,
        **config,
    )
