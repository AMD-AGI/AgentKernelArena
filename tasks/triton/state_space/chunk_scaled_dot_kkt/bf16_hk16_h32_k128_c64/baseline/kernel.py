# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project (MIT, see LICENSE;
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang).
# Adapted from vllm-project/vllm,
# vllm/third_party/flash_linear_attention/ops/chunk_scaled_dot_kkt.py.
"""Gated, beta-scaled intra-chunk K K^T (the Gated DeltaNet prefill "A" matrix) in Triton.

Per 64-token chunk and value head h (key head h // (H // Hg)):

    A = tril_strict( (beta[:, None] * K) @ K^T * exp(g[:, None] - g[None, :]) )

The beta-scaled operand stays in k's dtype (bfloat16) for the MFMA dot, which accumulates in
float32. Single sequence (no varlen); the block size, warps and stages are fixed instead of
autotuned.
"""
import triton
import triton.language as tl

CHUNK_SIZE = 64
BLOCK_K = 64
NUM_WARPS = 8
NUM_STAGES = 3


@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,
    g,
    A,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_beta = tl.make_block_ptr(beta + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))

    p_g = tl.make_block_ptr(g + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_g_diff = b_g[:, None] - b_g[None, :]
    b_A = b_A * tl.exp(b_g_diff)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(
        A + i_h * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def run(*, k, beta, g, out):
    T, Hg, K = k.shape
    H = beta.shape[-1]
    chunk_scaled_dot_kkt_fwd_kernel[(triton.cdiv(T, CHUNK_SIZE), H)](
        k, beta, g, out, T,
        H=H, Hg=Hg, K=K, BT=CHUNK_SIZE, BK=BLOCK_K,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
