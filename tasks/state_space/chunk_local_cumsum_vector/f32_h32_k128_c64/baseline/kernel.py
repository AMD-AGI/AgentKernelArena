# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project (MIT, see LICENSE;
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang).
# Adapted from the flash-linear-attention chunk-local cumulative sum
# (vllm-project/vllm, vllm/third_party/flash_linear_attention/ops/cumsum.py).
"""Chunk-local cumulative sum of vector gates [T, H, S] as a masked matmul in Triton.

Each program loads a [BT, BS] tile of one chunk/head, multiplies it with a lower-triangular
ones matrix in float32 (no TF32) and stores the running sums. The block width and warp count
are fixed instead of autotuned.
"""
import torch
import triton
import triton.language as tl

CHUNK_SIZE = 64
BLOCK_S = 64
NUM_WARPS = 4


@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_vector_kernel(
    s, o,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    i_s, i_t, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)

    p_s = tl.make_block_ptr(
        s + i_h * S, (T, S), (H * S, 1),
        (i_t * BT, i_s * BS), (BT, BS), (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + i_h * S, (T, S), (H * S, 1),
        (i_t * BT, i_s * BS), (BT, BS), (1, 0),
    )
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def run(*, g, out):
    T, H, S = g.shape
    grid = (triton.cdiv(S, BLOCK_S), triton.cdiv(T, CHUNK_SIZE), H)
    chunk_local_cumsum_vector_kernel[grid](
        g, out, T, H=H, S=S, BT=CHUNK_SIZE, BS=BLOCK_S, num_warps=NUM_WARPS,
    )
