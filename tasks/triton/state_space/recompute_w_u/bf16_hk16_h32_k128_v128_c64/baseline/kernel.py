# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project (MIT, see LICENSE;
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang).
# Adapted from vllm-project/vllm,
# vllm/third_party/flash_linear_attention/ops/wy_fast.py.
"""WY representation of the Gated DeltaNet chunk (recompute of w and u) in Triton.

Per 64-token chunk and value head h (key head h // (H // Hg)), with A the chunk's
(I + tril(beta * K K^T * decay))^{-1}:

    u = A @ (v * beta[:, None])
    w = A @ (k * beta[:, None] * exp(g)[:, None])

The beta-scaled (and gated) right operands are rounded to the input dtype (bfloat16) for the
MFMA dot, which accumulates in float32. Single sequence (no varlen); the warps and stages are
fixed instead of autotuned.
"""
import triton
import triton.language as tl

CHUNK_SIZE = 64
BLOCK_K = 64
BLOCK_V = 64
NUM_WARPS = 4
NUM_STAGES = 3


@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    p_beta = tl.make_block_ptr(beta + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g = tl.make_block_ptr(g + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(
        A + i_h * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + i_h * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_u = tl.make_block_ptr(
            u + i_h * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + i_h * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def run(*, k, v, beta, g, A, w, u):
    T, Hg, K = k.shape
    H, V = v.shape[1], v.shape[2]
    recompute_w_u_fwd_kernel[(triton.cdiv(T, CHUNK_SIZE), H)](
        k, v, beta, w, u, A, g, T,
        H=H, Hg=Hg, K=K, V=V, BT=CHUNK_SIZE, BK=BLOCK_K, BV=BLOCK_V,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
