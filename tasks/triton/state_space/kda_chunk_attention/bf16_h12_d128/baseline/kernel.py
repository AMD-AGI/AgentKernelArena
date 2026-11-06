# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# ruff: noqa: E501
#
# Adapted from vllm-project/vllm, vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py
# (chunk_kda_with_fused_gate, the ROCm Kimi Delta Attention prefill). The three kernels below are
# unchanged except that their autotune decorators are replaced by fixed launch configurations;
# run() is the production call sequence for one fixed-length batch, writing the output and the
# final state into the preallocated buffers instead of overwriting v.
"""Kimi Delta Attention (KDA) chunked prefill with the fused safe gate, in Triton.

run() chains the production kernels:
  1. l2norm_fwd(q), l2norm_fwd(k)                    (l2norm.py)
  2. kda_gate_chunk_cumsum_vector_kernel              g = chunk-local cumsum of
     lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias)) / ln 2, beta = sigmoid(raw_beta)
  3. chunk_kda_fwd_intra                              (chunk_intra.py) Aqk and the WY inverse
  4. recompute_w_u_fwd_kernel                         w, u, kg
  5. chunk_gated_delta_rule_fwd_h                     (chunk_delta_h.py) chunk states, v_new,
                                                      final state
  6. chunk_gla_fwd_kernel_o                           output
"""
import torch
import triton
import triton.language as tl

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_intra import chunk_kda_fwd_intra
from .l2norm import l2norm_fwd

exp2 = tl.exp2
log = tl.log

FLA_CHUNK_SIZE = 64
RCP_LN2 = 1.4426950216  # vllm.utils.math_utils.RCP_LN2

# Fixed entries of the production autotune spaces.
WU_NUM_WARPS = 4  # num_warps in {2, 4, 8}
WU_NUM_STAGES = 2  # num_stages in {2, 3, 4}
O_BLOCK_K = 64  # BK in {32, 64}
O_BLOCK_V = 64  # BV in {64, 128}
O_NUM_WARPS = 4  # num_warps in {2, 4, 8}
O_NUM_STAGES = 2  # num_stages in {2, 3, 4}
GATE_BLOCK_S = 64  # BS in {32, 64}
GATE_NUM_WARPS = 4  # num_warps in {2, 4, 8}
GATE_NUM_STAGES = 3  # the autotune configs leave num_stages at Triton's default, 3


@triton.heuristics(
    {
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)

    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
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
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]

        p_gk = tl.make_block_ptr(
            gk + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kb *= exp2(b_gk)
        if STORE_QG:
            p_q = tl.make_block_ptr(
                q + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_qg = tl.make_block_ptr(
                qg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))
        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1

            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(
                gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.0
            )
            b_kg = b_k * exp2(b_gn - b_gk)

            p_kg = tl.make_block_ptr(
                kg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    BK = 64
    BV = 64
    NT = triton.cdiv(T, BT)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        q=None,
        k=k,
        qg=None,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        DOT_PRECISION="ieee",
        num_warps=WU_NUM_WARPS,
        num_stages=WU_NUM_STAGES,
    )
    return w, u, kg


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_h = tl.make_block_ptr(
            h + (i_tg * H + i_h) * K * V,
            (V, K),
            (K, 1),
            (i_v * BV, i_k * BK),
            (BV, BK),
            (1, 0),
        )

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK]
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))



def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    chunk_size: int = FLA_CHUNK_SIZE,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT)

    grid = (triton.cdiv(V, O_BLOCK_V), NT, B * H)
    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=None,
        chunk_indices=None,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=O_BLOCK_K,
        BV=O_BLOCK_V,
        num_warps=O_NUM_WARPS,
        num_stages=O_NUM_STAGES,
    )
    return o


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["g_bias"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def kda_gate_chunk_cumsum_vector_kernel(
    s,
    raw_beta,
    A_log,
    g_bias,
    o,
    beta_out,
    cu_seqlens,
    chunk_indices,
    cumsum_scale,
    lower_bound,
    beta,
    threshold,
    T,
    stride_beta_batch,
    stride_beta_token,
    stride_beta_head,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_b * T

    if i_s == 0:
        o_beta_t = tl.arange(0, BT)
        m_beta = i_t * BT + o_beta_t < T
        if IS_VARLEN:
            p_beta = (
                raw_beta
                + (bos + i_t * BT + o_beta_t) * stride_beta_token
                + i_h * stride_beta_head
            )
        else:
            p_beta = (
                raw_beta
                + i_b * stride_beta_batch
                + (i_t * BT + o_beta_t) * stride_beta_token
                + i_h * stride_beta_head
            )
        b_beta = tl.load(p_beta, mask=m_beta, other=0.0).to(tl.float32)
        p_beta_out = beta_out + (bos + i_t * BT + o_beta_t) * H + i_h
        tl.store(p_beta_out, tl.sigmoid(b_beta), mask=m_beta)
        return

    i_s -= 1

    p_s = tl.make_block_ptr(
        s + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )

    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        p_bias = tl.make_block_ptr(
            g_bias + i_h * S,
            (S,),
            (1,),
            (i_s * BS,),
            (BS,),
            (0,),
        )
        b_bias = tl.load(p_bias, boundary_check=(0,)).to(tl.float32)
        b_s += b_bias[None, :]

    b_a = tl.exp(tl.load(A_log + i_h).to(tl.float32))
    if USE_LOWER_BOUND:
        b_gate = lower_bound * tl.sigmoid(b_a * b_s)
    else:
        b_g_scaled = b_s * beta
        b_softplus = tl.where(
            b_g_scaled > threshold,
            b_s,
            (1.0 / beta) * log(1.0 + tl.exp(b_g_scaled)),
        )
        b_gate = -b_a * b_softplus

    # Boundary loads return zero, but bias and gate activation can make padded
    # rows nonzero. Padding trails valid rows, so it only affects masked stores.
    b_o = tl.cumsum(b_gate, axis=0) * cumsum_scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))



def fused_kda_gate_chunk_cumsum(
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    lower_bound: float | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    output_dtype: torch.dtype | None = torch.float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, D = raw_g.shape
    if raw_beta.shape != (B, T, H):
        raise ValueError(f"Expected raw_beta shape {(B, T, H)}, got {raw_beta.shape}")
    NT = triton.cdiv(T, chunk_size)

    A_log = A_log.reshape(-1)
    if g_bias is not None:
        g_bias = g_bias.reshape(-1)
    y = torch.empty_like(raw_g, dtype=output_dtype or raw_g.dtype)
    beta_out = torch.empty(raw_beta.shape, device=raw_beta.device, dtype=torch.float32)

    # For each (chunk, head), program 0 computes beta without extending a
    # gate tile's critical path. The remaining programs cover the gate dim.
    grid = (triton.cdiv(D, GATE_BLOCK_S) + 1, NT, B * H)
    kda_gate_chunk_cumsum_vector_kernel[grid](
        s=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        g_bias=g_bias,
        o=y,
        beta_out=beta_out,
        cu_seqlens=None,
        chunk_indices=None,
        # RCP_LN2 folds in the natural-log -> log2 conversion so downstream
        # exp2-based kernels reproduce exp(g).
        cumsum_scale=RCP_LN2,
        lower_bound=lower_bound or 0.0,
        beta=beta,
        threshold=threshold,
        T=T,
        stride_beta_batch=raw_beta.stride(0),
        stride_beta_token=raw_beta.stride(1),
        stride_beta_head=raw_beta.stride(2),
        H=H,
        S=D,
        BT=chunk_size,
        USE_LOWER_BOUND=lower_bound is not None,
        BS=GATE_BLOCK_S,
        num_warps=GATE_NUM_WARPS,
        num_stages=GATE_NUM_STAGES,
    )
    return y, beta_out


def run(*, q, k, v, raw_g, raw_beta, A_log, dt_bias, initial_state, lower_bound, o, final_state):
    """chunk_kda_with_fused_gate(..., output_final_state=True, use_qk_l2norm_in_kernel=True).

    q, k, v, raw_g: [N, L, H, D] bfloat16; raw_beta: [N, L, H] bfloat16 (pre-sigmoid);
    A_log: [H] and dt_bias: [H * D] float32; initial_state, final_state: [N, H, D(v), D(k)]
    float32; o: [N, L, H, D] bfloat16.
    """
    scale = k.shape[-1] ** -0.5
    chunk_size = FLA_CHUNK_SIZE
    q = l2norm_fwd(q)
    k = l2norm_fwd(k)
    g, beta = fused_kda_gate_chunk_cumsum(
        raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        g_bias=dt_bias,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
    )
    Aqk, A = chunk_kda_fwd_intra(
        q=q,
        k=k,
        gk=g,
        beta=beta,
        scale=scale,
        chunk_size=chunk_size,
    )
    w, u, kg = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, gk=g)
    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        final_state=final_state,
        gk=g,
        initial_state=initial_state,
        chunk_size=chunk_size,
        use_exp2=True,
    )
    chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        o=o,
        scale=scale,
        chunk_size=chunk_size,
    )
