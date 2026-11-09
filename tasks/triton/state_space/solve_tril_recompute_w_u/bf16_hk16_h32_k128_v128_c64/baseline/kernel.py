# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Vendored from ROCm/aiter @ 652f2d4b768239aefd515c929e84817681ff598d
# (aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/fused_solve_tril_recompute.py), MIT, see LICENSE.
# The upstream header names flash-linear-attention (MIT, Copyright (c) 2023-2025, Songlin Yang, Yu Zhang).
#
# fused_solve_tril_recompute_w_u_kernel and recompute_w_u_head_major_kernel are unchanged. Their
# @triton.autotune decorators (one entry each, BK = BV = 64, 2 warps, 2 stages, unless the
# GATED_DELTA_RULE_TRITON_AUTOTUNE environment variable is set) are replaced by that entry passed at
# launch, and `exp` (tl.exp, utils/op.py) is defined here. run() replaces fused_solve_tril_recompute_w_u
# and _run_split_path for one sequence (no cu_seqlens, use_exp2 on), keeping the fused / split dispatch
# at 32 chunks (AITER_SOLVE_TRIL_RECOMPUTE_FUSE_NT_MAX default) and writing the preallocated w and u.
# The split path's triangular solve is in solve_tril.py.

import triton
import triton.language as tl

from .solve_tril import solve_tril

# solve_tril + recompute_w_u dispatch threshold in chunks (NT). At or below this the single fused kernel
# is used; above it the split path (solve_tril + recompute) is used (upstream default of
# AITER_SOLVE_TRIL_RECOMPUTE_FUSE_NT_MAX).
_SOLVE_TRIL_RECOMPUTE_FUSE_NT_MAX = 32

# Precision of the fused kernel's 16 x 16 block dots (upstream default of FLA_TRIL_PRECISION).
FLA_TRIL_PRECISION = "ieee"

exp = tl.exp


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def fused_solve_tril_recompute_w_u_kernel(
    A_raw,
    k,
    v,
    beta,
    g,
    w,
    u,
    cu_seqlens,
    sequence_ids,
    chunk_ids,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    USE_EXP2: tl.constexpr = False,
    LOWP_DTYPE_IS_BF16: tl.constexpr = False,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    T_flat = T
    # Preserve fp16 executions instead of unconditionally truncating to bf16.
    tl.static_assert(
        (k.dtype.element_ty == tl.float16 or k.dtype.element_ty == tl.bfloat16)
        and (v.dtype.element_ty == tl.float16 or v.dtype.element_ty == tl.bfloat16),
        "fused_solve_tril_recompute_w_u_kernel only supports fp16/bf16 k and v tensors.",
    )
    tl.static_assert(
        k.dtype.element_ty == v.dtype.element_ty,
        "fused_solve_tril_recompute_w_u_kernel expects k and v to have the same dtype.",
    )
    lowp_dtype = tl.bfloat16 if LOWP_DTYPE_IS_BF16 else tl.float16

    if IS_VARLEN:
        i_n, i_t = (
            tl.load(sequence_ids + i_t * INDEX_STRIDE).to(tl.int32),
            tl.load(chunk_ids + i_t * INDEX_STRIDE).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_b * T

    # ================================================================
    # Phase 1: compute (I + A)^{-1} in registers (triangular solve)
    # ================================================================
    o_i = tl.arange(0, 16)
    m_lo = o_i[:, None] > o_i[None, :]
    m_id = o_i[:, None] == o_i[None, :]
    A_base = A_raw + (bos * H + i_h) * BT

    p11 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
    )
    p22 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
    )
    p33 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)
    )
    p44 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)
    )
    b11 = -tl.where(m_lo, tl.load(p11, boundary_check=(0, 1)).to(tl.float32), 0)
    b22 = -tl.where(m_lo, tl.load(p22, boundary_check=(0, 1)).to(tl.float32), 0)
    b33 = -tl.where(m_lo, tl.load(p33, boundary_check=(0, 1)).to(tl.float32), 0)
    b44 = -tl.where(m_lo, tl.load(p44, boundary_check=(0, 1)).to(tl.float32), 0)

    for i in range(2, min(16, T - i_t * BT)):
        r = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i)
        r = r + tl.sum(r[:, None] * b11, 0)
        b11 = tl.where((o_i == i)[:, None], r, b11)
    for i in range(18, min(32, T - i_t * BT)):
        r = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i + 16)
        r = r + tl.sum(r[:, None] * b22, 0)
        b22 = tl.where((o_i == i - 16)[:, None], r, b22)
    for i in range(34, min(48, T - i_t * BT)):
        r = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i + 32)
        r = r + tl.sum(r[:, None] * b33, 0)
        b33 = tl.where((o_i == i - 32)[:, None], r, b33)
    for i in range(50, min(64, T - i_t * BT)):
        r = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i + 48)
        r = r + tl.sum(r[:, None] * b44, 0)
        b44 = tl.where((o_i == i - 48)[:, None], r, b44)
    b11 += m_id
    b22 += m_id
    b33 += m_id
    b44 += m_id

    rA21 = tl.load(
        tl.make_block_ptr(
            A_base, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    rA31 = tl.load(
        tl.make_block_ptr(
            A_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    rA32 = tl.load(
        tl.make_block_ptr(
            A_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    rA41 = tl.load(
        tl.make_block_ptr(
            A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    rA42 = tl.load(
        tl.make_block_ptr(
            A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    rA43 = tl.load(
        tl.make_block_ptr(
            A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)

    b21 = -tl.dot(
        tl.dot(b22, rA21, input_precision=DOT_PRECISION),
        b11,
        input_precision=DOT_PRECISION,
    )
    b32 = -tl.dot(
        tl.dot(b33, rA32, input_precision=DOT_PRECISION),
        b22,
        input_precision=DOT_PRECISION,
    )
    b43 = -tl.dot(
        tl.dot(b44, rA43, input_precision=DOT_PRECISION),
        b33,
        input_precision=DOT_PRECISION,
    )
    b31 = -tl.dot(
        b33,
        tl.dot(rA31, b11, input_precision=DOT_PRECISION)
        + tl.dot(rA32, b21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b42 = -tl.dot(
        b44,
        tl.dot(rA42, b22, input_precision=DOT_PRECISION)
        + tl.dot(rA43, b32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b41 = -tl.dot(
        b44,
        tl.dot(rA41, b11, input_precision=DOT_PRECISION)
        + tl.dot(rA42, b21, input_precision=DOT_PRECISION)
        + tl.dot(rA43, b31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    h11 = b11.to(lowp_dtype)
    h22 = b22.to(lowp_dtype)
    h33 = b33.to(lowp_dtype)
    h44 = b44.to(lowp_dtype)
    h21 = b21.to(lowp_dtype)
    h31 = b31.to(lowp_dtype)
    h32 = b32.to(lowp_dtype)
    h41 = b41.to(lowp_dtype)
    h42 = b42.to(lowp_dtype)
    h43 = b43.to(lowp_dtype)

    # ================================================================
    # Phase 2: u = Ai @ (v * beta), w = Ai @ (k * beta * exp(g))
    # ================================================================
    beta_base = beta + bos * H + i_h
    if IS_VARLEN:
        g_base = g + i_h * T_flat + bos
    else:
        g_base = g + (i_b * H + i_h) * T_flat

    p_b0 = tl.make_block_ptr(beta_base, (T,), (H,), (i_t * BT,), (16,), (0,))
    p_b1 = tl.make_block_ptr(beta_base, (T,), (H,), (i_t * BT + 16,), (16,), (0,))
    p_b2 = tl.make_block_ptr(beta_base, (T,), (H,), (i_t * BT + 32,), (16,), (0,))
    p_b3 = tl.make_block_ptr(beta_base, (T,), (H,), (i_t * BT + 48,), (16,), (0,))
    bb0 = tl.load(p_b0, boundary_check=(0,))
    bb1 = tl.load(p_b1, boundary_check=(0,))
    bb2 = tl.load(p_b2, boundary_check=(0,))
    bb3 = tl.load(p_b3, boundary_check=(0,))

    p_g0 = tl.make_block_ptr(g_base, (T,), (1,), (i_t * BT,), (16,), (0,))
    p_g1 = tl.make_block_ptr(g_base, (T,), (1,), (i_t * BT + 16,), (16,), (0,))
    p_g2 = tl.make_block_ptr(g_base, (T,), (1,), (i_t * BT + 32,), (16,), (0,))
    p_g3 = tl.make_block_ptr(g_base, (T,), (1,), (i_t * BT + 48,), (16,), (0,))
    if USE_EXP2:
        eg0 = tl.math.exp2(tl.load(p_g0, boundary_check=(0,)))
        eg1 = tl.math.exp2(tl.load(p_g1, boundary_check=(0,)))
        eg2 = tl.math.exp2(tl.load(p_g2, boundary_check=(0,)))
        eg3 = tl.math.exp2(tl.load(p_g3, boundary_check=(0,)))
    else:
        eg0 = exp(tl.load(p_g0, boundary_check=(0,)))
        eg1 = exp(tl.load(p_g1, boundary_check=(0,)))
        eg2 = exp(tl.load(p_g2, boundary_check=(0,)))
        eg3 = exp(tl.load(p_g3, boundary_check=(0,)))

    v_base = v + (bos * H + i_h) * V
    if IS_VARLEN:
        u_base = u + (i_h * T_flat + bos) * V
    else:
        u_base = u + (((i_b * H + i_h) * T_flat) * V)

    for i_v in range(tl.cdiv(V, BV)):
        pv0 = tl.make_block_ptr(
            v_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (16, BV), (1, 0)
        )
        pv1 = tl.make_block_ptr(
            v_base, (T, V), (H * V, 1), (i_t * BT + 16, i_v * BV), (16, BV), (1, 0)
        )
        pv2 = tl.make_block_ptr(
            v_base, (T, V), (H * V, 1), (i_t * BT + 32, i_v * BV), (16, BV), (1, 0)
        )
        pv3 = tl.make_block_ptr(
            v_base, (T, V), (H * V, 1), (i_t * BT + 48, i_v * BV), (16, BV), (1, 0)
        )
        vb0 = (tl.load(pv0, boundary_check=(0, 1)) * bb0[:, None]).to(lowp_dtype)
        vb1 = (tl.load(pv1, boundary_check=(0, 1)) * bb1[:, None]).to(lowp_dtype)
        vb2 = (tl.load(pv2, boundary_check=(0, 1)) * bb2[:, None]).to(lowp_dtype)
        vb3 = (tl.load(pv3, boundary_check=(0, 1)) * bb3[:, None]).to(lowp_dtype)

        u0 = tl.dot(h11, vb0, allow_tf32=False)
        u1 = tl.dot(h21, vb0, allow_tf32=False) + tl.dot(h22, vb1, allow_tf32=False)
        u2 = (
            tl.dot(h31, vb0, allow_tf32=False)
            + tl.dot(h32, vb1, allow_tf32=False)
            + tl.dot(h33, vb2, allow_tf32=False)
        )
        u3 = (
            tl.dot(h41, vb0, allow_tf32=False)
            + tl.dot(h42, vb1, allow_tf32=False)
            + tl.dot(h43, vb2, allow_tf32=False)
            + tl.dot(h44, vb3, allow_tf32=False)
        )

        pu0 = tl.make_block_ptr(
            u_base, (T, V), (V, 1), (i_t * BT, i_v * BV), (16, BV), (1, 0)
        )
        pu1 = tl.make_block_ptr(
            u_base, (T, V), (V, 1), (i_t * BT + 16, i_v * BV), (16, BV), (1, 0)
        )
        pu2 = tl.make_block_ptr(
            u_base, (T, V), (V, 1), (i_t * BT + 32, i_v * BV), (16, BV), (1, 0)
        )
        pu3 = tl.make_block_ptr(
            u_base, (T, V), (V, 1), (i_t * BT + 48, i_v * BV), (16, BV), (1, 0)
        )
        tl.store(pu0, u0.to(pu0.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pu1, u1.to(pu1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pu2, u2.to(pu2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pu3, u3.to(pu3.dtype.element_ty), boundary_check=(0, 1))

    k_base = k + (bos * Hg + i_h // (H // Hg)) * K
    if IS_VARLEN:
        w_base = w + (i_h * T_flat + bos) * K
    else:
        w_base = w + (((i_b * H + i_h) * T_flat) * K)

    for i_k in range(tl.cdiv(K, BK)):
        pk0 = tl.make_block_ptr(
            k_base, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (16, BK), (1, 0)
        )
        pk1 = tl.make_block_ptr(
            k_base, (T, K), (Hg * K, 1), (i_t * BT + 16, i_k * BK), (16, BK), (1, 0)
        )
        pk2 = tl.make_block_ptr(
            k_base, (T, K), (Hg * K, 1), (i_t * BT + 32, i_k * BK), (16, BK), (1, 0)
        )
        pk3 = tl.make_block_ptr(
            k_base, (T, K), (Hg * K, 1), (i_t * BT + 48, i_k * BK), (16, BK), (1, 0)
        )
        kb0 = (tl.load(pk0, boundary_check=(0, 1)) * bb0[:, None] * eg0[:, None]).to(
            lowp_dtype
        )
        kb1 = (tl.load(pk1, boundary_check=(0, 1)) * bb1[:, None] * eg1[:, None]).to(
            lowp_dtype
        )
        kb2 = (tl.load(pk2, boundary_check=(0, 1)) * bb2[:, None] * eg2[:, None]).to(
            lowp_dtype
        )
        kb3 = (tl.load(pk3, boundary_check=(0, 1)) * bb3[:, None] * eg3[:, None]).to(
            lowp_dtype
        )

        w0 = tl.dot(h11, kb0)
        w1 = tl.dot(h21, kb0) + tl.dot(h22, kb1)
        w2 = tl.dot(h31, kb0) + tl.dot(h32, kb1) + tl.dot(h33, kb2)
        w3 = tl.dot(h41, kb0) + tl.dot(h42, kb1) + tl.dot(h43, kb2) + tl.dot(h44, kb3)

        pw0 = tl.make_block_ptr(
            w_base, (T, K), (K, 1), (i_t * BT, i_k * BK), (16, BK), (1, 0)
        )
        pw1 = tl.make_block_ptr(
            w_base, (T, K), (K, 1), (i_t * BT + 16, i_k * BK), (16, BK), (1, 0)
        )
        pw2 = tl.make_block_ptr(
            w_base, (T, K), (K, 1), (i_t * BT + 32, i_k * BK), (16, BK), (1, 0)
        )
        pw3 = tl.make_block_ptr(
            w_base, (T, K), (K, 1), (i_t * BT + 48, i_k * BK), (16, BK), (1, 0)
        )
        tl.store(pw0, w0.to(pw0.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pw1, w1.to(pw1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pw2, w2.to(pw2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pw3, w3.to(pw3.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_head_major_kernel(
    k,
    v,
    beta,
    Ai,
    g,
    w,
    u,
    cu_seqlens,
    sequence_ids,
    chunk_ids,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    USE_EXP2: tl.constexpr = False,
):
    """
    Compute u = Ai @ (v * beta) and w = Ai @ (k * beta * exp(g)),
    where Ai = (I + A_strict_lower)^{-1} is precomputed by `solve_tril`.

    Output layout: [B, H, T, K/V] head-major contiguous (matches fused path).
    Hg <= H supports GQA (k shared across H/Hg heads).
    `g` is consumed in head-major layout [B, H, T], matching the fused kernel;
    USE_EXP2 selects whether the cumulative gate is interpreted in log2 space.
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    T_flat = T

    if IS_VARLEN:
        i_n, i_t = (
            tl.load(sequence_ids + i_t * INDEX_STRIDE).to(tl.int32),
            tl.load(chunk_ids + i_t * INDEX_STRIDE).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_b * T

    # Load beta, Ai (the inverted A), and the per-chunk gate eagerly so the
    # compiler can hoist them out of the V/K loops and overlap memory latency.
    if IS_VARLEN:
        g_base = g + i_h * T_flat + bos
    else:
        g_base = g + (i_b * H + i_h) * T_flat
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(g_base, (T,), (1,), (i_t * BT,), (BT,), (0,))
    p_Ai = tl.make_block_ptr(
        Ai + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_Ai = tl.load(p_Ai, boundary_check=(0, 1))
    b_g_raw = tl.load(p_g, boundary_check=(0,))
    b_g = tl.math.exp2(b_g_raw) if USE_EXP2 else exp(b_g_raw)

    # ---- u = Ai @ (v * beta) -> head-major store ----
    v_base = v + (bos * H + i_h) * V
    if IS_VARLEN:
        u_base = u + (i_h * T_flat + bos) * V
    else:
        u_base = u + ((i_b * H + i_h) * T_flat) * V

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_u = tl.make_block_ptr(
            u_base, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_Ai, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    # ---- w = Ai @ (k * beta * exp(g)) -> head-major store ----
    k_base = k + (bos * Hg + i_h // (H // Hg)) * K
    if IS_VARLEN:
        w_base = w + (i_h * T_flat + bos) * K
    else:
        w_base = w + ((i_b * H + i_h) * T_flat) * K

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k_base, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_w = tl.make_block_ptr(
            w_base, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # Single fused multiply-cast before the dot.
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_Ai, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def run(*, A_raw, k, v, beta, g_cumsum, w, u) -> None:
    """Fused triangular solve + recompute of w and u for one sequence of T tokens.

    A_raw: [T, H, 64] float32, strictly lower triangular; k: [T, Hg, K] and v: [T, H, V] bfloat16;
    beta: [T, H] float32; g_cumsum: [H, T] float32, chunk-local cumulative gate in log2 space;
    w: [H, T, K] and u: [H, T, V] bfloat16, head-major.
    """
    T, Hg, K = k.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = A_raw.shape[-1]
    NT = triton.cdiv(T, BT)

    if NT > _SOLVE_TRIL_RECOMPUTE_FUSE_NT_MAX:
        # Long-sequence path: precompute Ai via solve_tril (stored in k.dtype), then a plain matmul.
        Ai = solve_tril(A_raw, output_dtype=k.dtype)
        recompute_w_u_head_major_kernel[(NT, H)](
            k,
            v,
            beta,
            Ai,
            g_cumsum,
            w,
            u,
            None,
            None,
            None,
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            BK=64,
            BV=64,
            INDEX_STRIDE=1,
            USE_EXP2=True,
            num_warps=2,
            num_stages=2,
        )
        return

    fused_solve_tril_recompute_w_u_kernel[(NT, H)](
        A_raw,
        k,
        v,
        beta,
        g_cumsum,
        w,
        u,
        None,
        None,
        None,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=64,
        BV=64,
        INDEX_STRIDE=1,
        DOT_PRECISION=FLA_TRIL_PRECISION,
        USE_EXP2=True,
        LOWP_DTYPE_IS_BF16=True,
        num_warps=2,
        num_stages=2,
    )
