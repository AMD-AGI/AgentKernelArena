# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project (MIT, see LICENSE;
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang).
# Adapted from vllm-project/vllm, vllm/third_party/flash_linear_attention/ops/kda.py
# (recompute_w_u_fwd_kernel and recompute_w_u_fwd).
"""Kimi Delta Attention: recompute the WY-representation tensors w, u (and the decayed keys kg).

Per 64-token chunk and head, with the inverse A of the unit-lower-triangular system (I + L), the
per-token write strength beta and the per-channel cumulative log2-decay gk:

    u  = A @ (beta * v)
    w  = A @ (beta * k * 2^gk)
    kg = k * 2^(gk[last] - gk)          (last = last token of the chunk)
"""
import triton
import triton.language as tl

BLOCK_K = 64
BLOCK_V = 64


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    p_b = tl.make_block_ptr(beta + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(
        A + i_h * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + i_h * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + i_h * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(
            w + i_h * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + i_h * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]

        p_gk = tl.make_block_ptr(
            gk + i_h * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kb *= tl.exp2(b_gk)

        last_idx = min(i_t * BT + BT, T) - 1

        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_gn = tl.load(gk + (last_idx * H + i_h) * K + o_k, mask=m_k, other=0.0)
        b_kg = b_k * tl.exp2(b_gn - b_gk)

        p_kg = tl.make_block_ptr(
            kg + i_h * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def run(*, k, v, beta, A, gk, w, u, kg):
    T, H, K = k.shape
    V = v.shape[-1]
    BT = A.shape[-1]
    NT = triton.cdiv(T, BT)
    recompute_w_u_fwd_kernel[(NT, H)](
        k=k, kg=kg, v=v, beta=beta, w=w, u=u, A=A, gk=gk,
        T=T, H=H, K=K, V=V, BT=BT, BK=BLOCK_K, BV=BLOCK_V,
        DOT_PRECISION="ieee",
    )
