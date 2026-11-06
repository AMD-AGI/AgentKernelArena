# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code adapted from the flash-linear-attention project (MIT, see LICENSE;
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang), as carried in sgl-project/sglang
# (python/sglang/kernels/ops/attention/fla/chunk_o.py).
"""GDN chunked linear-attention output (chunk_fwd_o), scalar-gate path.

Per chunk of 64 tokens and value head h (key head hg = h // (H // Hg)):
    o = (q @ h_chunk^T * exp(g)) + tril(q @ k^T * exp(g_i - g_j)) @ v, both scaled by `scale`.
"""
import triton
import triton.language as tl


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * V * K

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, tl.trans(b_h))
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * tl.exp(b_g)[:, None]
        b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def run(*, q, k, v, h, g, scale, o) -> None:
    B, T, Hg, K = q.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = min(64, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q, k, v, h, g, o, None, None, scale,
        T=T, H=H, Hg=Hg, K=K, V=V,
        BT=BT, BK=128, BV=64,
        USE_G=True, IS_VARLEN=False,
        num_warps=4, num_stages=2,
    )
