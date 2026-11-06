# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project (MIT, see LICENSE;
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang).
# Adapted from vllm-project/vllm, vllm/third_party/flash_linear_attention/ops/kda.py
# (chunk_kda_scaled_dot_kkt_fwd and its two kernels).
"""Kimi Delta Attention: gated, beta-scaled intra-chunk K K^T and Q K^T in Triton.

For every 64-token chunk and head, with per-channel cumulative gates ``g`` in the log2 domain:

    A[i, j]   = beta[i] * sum_k k[i, k] * k[j, k] * 2^(g[i, k] - g[j, k])   for i > j
    Aqk[i, j] = scale   * sum_k q[i, k] * k[j, k] * 2^(g[i, k] - g[j, k])   for i >= j

Two kernels: ``sub_inter`` fills the 16x16 sub-blocks strictly below the diagonal with MFMA dots, and
``sub_intra`` fills the diagonal sub-blocks column by column. Production first clears the outputs
with ``torch.zeros``; here the ``sub_inter`` kernel stores the zero sub-blocks on and above the
diagonal itself, so the call needs no host-side fill and every output element is written.
"""
import triton
import triton.language as tl

CHUNK_SIZE = 64
SUB_CHUNK = 16


@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BC"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter(
    q,
    k,
    g,
    beta,
    A,
    Aqk,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
):
    i_t, i_c, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_i, i_j = i_c // NC, i_c % NC

    if i_t * BT + i_i * BC >= T:
        return

    q += i_h * K
    k += i_h * K
    g += i_h * K
    A += i_h * BT
    Aqk += i_h * BT

    p_A = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0)
    )
    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0)
    )
    if i_i <= i_j:
        # Task adaptation: write the zero sub-blocks on and above the diagonal (production
        # pre-clears the outputs with torch.zeros); the intra kernel overwrites the diagonal.
        b_zero = tl.zeros([BC, BC], dtype=tl.float32)
        tl.store(p_A, b_zero.to(A.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk, b_zero.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        return

    p_b = tl.make_block_ptr(
        beta + i_h, (T,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,)
    )
    b_b = tl.load(p_b, boundary_check=(0,))

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
        )
        p_g = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
        )
        b_kt = tl.make_block_ptr(
            k, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1)
        )
        p_gk = tl.make_block_ptr(
            g, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1)
        )

        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        # [BK,]
        b_gn = tl.load(g + (i_t * BT + i_i * BC) * H * K + o_k, mask=m_k, other=0)
        # [BC, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1)) * tl.exp2(b_g - b_gn[None, :])
        # [BK, BC]
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kt = tl.load(b_kt, boundary_check=(0, 1))
        # [BC, BC]
        b_ktg = b_kt * tl.exp2(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_k, b_ktg)

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_qg = b_q * tl.exp2(b_g - b_gn[None, :]) * scale
        b_Aqk += tl.dot(b_qg, b_ktg)

    b_A *= b_b[:, None]

    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["BK", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra(
    q,
    k,
    g,
    beta,
    A,
    Aqk,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    i_t, i_i, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + o_i) < T
    o_A = (i_t * BT + i_i * BC + o_i) * H * BT + i_h * BT + i_i * BC

    p_q = tl.make_block_ptr(
        q + i_h * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    p_k = tl.make_block_ptr(
        k + i_h * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    p_g = tl.make_block_ptr(
        g + i_h * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_b = beta + (i_t * BT + i_i * BC + o_i) * H + i_h
    b_k = b_k * tl.load(p_b, mask=m_A, other=0)[:, None]

    p_kt = k + (i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_gk = g + (i_t * BT + i_i * BC) * H * K + i_h * K + o_k

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_kt = tl.load(p_kt, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_ktg = b_kt[None, :] * tl.exp2(b_g - b_gk[None, :])
        b_A = tl.sum(b_k * b_ktg, 1)
        b_A = tl.where(o_i > j, b_A, 0.0)
        b_Aqk = tl.sum(b_q * b_ktg, 1)
        b_Aqk = tl.where(o_i >= j, b_Aqk * scale, 0.0)
        tl.store(A + o_A + j, b_A, mask=m_A)
        tl.store(Aqk + o_A + j, b_Aqk, mask=m_A)
        p_kt += H * K
        p_gk += H * K


def run(*, q, k, g, beta, scale, A, Aqk):
    T, H, K = k.shape
    BT = CHUNK_SIZE
    BC = SUB_CHUNK
    NC = BT // BC
    NT = triton.cdiv(T, BT)
    BK = max(triton.next_power_of_2(K), 16)
    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter[(NT, NC * NC, H)](
        q=q, k=k, g=g, beta=beta, A=A, Aqk=Aqk, scale=scale,
        T=T, H=H, K=K, BT=BT, BC=BC, NC=NC,
    )
    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra[(NT, NC, H)](
        q=q, k=k, g=g, beta=beta, A=A, Aqk=Aqk, scale=scale,
        T=T, H=H, K=K, BT=BT, BC=BC, BK=BK,
    )
