# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# The kernel is adapted from LightLLM's context_attention_fwd (via vLLM's prefix_prefill.py):
# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py
"""Paged prefix-prefill attention in Triton, from vLLM prefix_prefill.py (_fwd_kernel).

Queries attend to the cached context through a block table (5D K cache, 4D V cache) and to the
new tokens of their own sequence causally. Every sequence of a call has the same query length, so
the launch grid derives max_input_len from the packed token count.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    K_cache,
    V_cache,
    B_Loc,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    x: tl.constexpr,
    Out,
    stride_b_loc_b,
    stride_b_loc_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl: tl.constexpr,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    num_queries_per_kv: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_PADDED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PHYSICAL_BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    num_unroll_cache: tl.constexpr,
    num_unroll_request: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // num_queries_per_kv

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

    # start position inside of the query
    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_bs_n = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )

    dim_mask = tl.where(tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(
        tl.int1
    )

    q = tl.load(
        Q + off_q,
        mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len),
        other=0.0,
    )

    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)

    # compute query against context (no causal mask here)
    for start_n in tl.range(
        0, cur_batch_ctx_len, BLOCK_SIZE, loop_unroll_factor=num_unroll_cache
    ):
        token_indices = start_n + offs_bs_n
        bn_logical_indices = token_indices // PHYSICAL_BLOCK_SIZE

        bn = tl.load(
            B_Loc + cur_batch * stride_b_loc_b + bn_logical_indices * stride_b_loc_s,
            mask=token_indices < cur_batch_ctx_len,
            other=0,
        ).to(tl.int64)

        internal_offsets = token_indices % PHYSICAL_BLOCK_SIZE

        # Addressing of K (5D)
        off_k = (
            bn[None, :] * stride_k_cache_bs
            + cur_kv_head * stride_k_cache_h
            + (offs_d[:, None] // x) * stride_k_cache_d
            + internal_offsets[None, :] * stride_k_cache_bl
            + (offs_d[:, None] % x) * stride_k_cache_x
        )

        # Addressing of V (4D)
        off_v = (
            bn[:, None] * stride_v_cache_bs
            + cur_kv_head * stride_v_cache_h
            + offs_d[None, :] * stride_v_cache_d
            + internal_offsets[:, None] * stride_v_cache_bl
        )

        if (
            start_n + BLOCK_SIZE > cur_batch_ctx_len
            or BLOCK_DMODEL != BLOCK_DMODEL_PADDED
        ):
            k = tl.load(
                K_cache + off_k,
                mask=dim_mask[:, None]
                & ((start_n + offs_bs_n[None, :]) < cur_batch_ctx_len),
                other=0.0,
            )
        else:
            k = tl.load(K_cache + off_k)

        qk = sm_scale * tl.dot(q, k)
        qk = tl.where(
            (start_n + offs_bs_n[None, :]) < cur_batch_ctx_len, qk, float("-inf")
        )
        if SLIDING_WINDOW > 0:
            qk = tl.where(
                (cur_batch_ctx_len + offs_m[:, None]) - (start_n + offs_bs_n[None, :])
                < SLIDING_WINDOW,
                qk,
                float("-inf"),
            )

        # compute running maximum
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        acc = acc * alpha[:, None]

        if (
            start_n + BLOCK_SIZE > cur_batch_ctx_len
            or BLOCK_DMODEL != BLOCK_DMODEL_PADDED
        ):
            v = tl.load(
                V_cache + off_v,
                mask=dim_mask[None, :]
                & ((start_n + offs_bs_n[:, None]) < cur_batch_ctx_len),
                other=0.0,
            )
        else:
            v = tl.load(V_cache + off_v)

        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc=acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    off_k = (
        offs_n[None, :] * stride_kbs
        + cur_kv_head * stride_kh
        + offs_d[:, None] * stride_kd
    )
    off_v = (
        offs_n[:, None] * stride_vbs
        + cur_kv_head * stride_vh
        + offs_d[None, :] * stride_vd
    )
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)

    # compute query against itself (with causal mask)
    for start_n in tl.range(
        0,
        block_mask * (start_m + 1) * BLOCK_M,
        BLOCK_N,
        loop_unroll_factor=num_unroll_request,
    ):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=dim_mask[:, None]
            & ((start_n + offs_n[None, :]) < cur_batch_query_len),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        if SLIDING_WINDOW > 0:
            qk = tl.where(
                offs_m[:, None] - (start_n + offs_n[None, :]) < SLIDING_WINDOW,
                qk,
                float("-inf"),
            )

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        acc = acc * alpha[:, None]

        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=dim_mask[None, :]
            & ((start_n + offs_n[:, None]) < cur_batch_query_len),
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc=acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / (l_i[:, None] + 1e-10)

    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs, acc, mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len)
    )
    return


def run(*, q, k, v, k_cache, v_cache, b_loc, b_start_loc, b_seq_len, sm_scale, out):
    head_dim = q.shape[-1]
    head_dim_padded = triton.next_power_of_2(head_dim)
    batch, n_heads = b_seq_len.shape[0], q.shape[1]
    num_queries_per_kv = n_heads // k.shape[1]
    max_input_len = q.shape[0] // batch  # uniform query length per sequence

    real_block_size = v_cache.shape[3]
    is_pow2 = real_block_size > 0 and (real_block_size & (real_block_size - 1) == 0)
    if is_pow2:
        BLOCK_M = 128
        BLOCK_N = 64
    else:
        BLOCK_M = 32
        BLOCK_N = 32

    TRITON_BLOCK_SIZE = 32

    grid = (batch, n_heads, triton.cdiv(max_input_len, BLOCK_M))
    _fwd_kernel[grid](
        q,
        k,
        v,
        k_cache,
        v_cache,
        b_loc,
        sm_scale,
        b_start_loc,
        b_seq_len,
        k_cache.shape[4],  # x
        out,
        b_loc.stride(0),
        b_loc.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        stride_k_cache_bs=k_cache.stride(0),
        stride_k_cache_h=k_cache.stride(1),
        stride_k_cache_d=k_cache.stride(2),
        stride_k_cache_bl=k_cache.stride(3),
        stride_k_cache_x=k_cache.stride(4),
        stride_v_cache_bs=v_cache.stride(0),
        stride_v_cache_h=v_cache.stride(1),
        stride_v_cache_d=v_cache.stride(2),
        stride_v_cache_bl=v_cache.stride(3),
        BLOCK_SIZE=TRITON_BLOCK_SIZE,
        PHYSICAL_BLOCK_SIZE=real_block_size,
        num_queries_per_kv=num_queries_per_kv,
        BLOCK_DMODEL=head_dim,
        BLOCK_DMODEL_PADDED=head_dim_padded,
        SLIDING_WINDOW=0,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_unroll_cache=4,
        num_unroll_request=1,
        num_warps=4,
        num_stages=1,
    )
