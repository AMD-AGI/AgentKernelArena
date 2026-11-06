# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# The kernel is adapted from LightLLM's context_attention_fwd (via vLLM's prefix_prefill.py):
# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py
"""Paged prefix-prefill attention with ALiBi in Triton, from vLLM prefix_prefill.py (_fwd_kernel_alibi).

Queries attend to the cached context through a block table (5D K cache, 4D V cache) and to the
new tokens of their own sequence causally; the score of every key is biased by
alibi_slope * (key_position - query_position). Every sequence of a call has the same query length,
so the launch grid derives max_input_len from the packed token count.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_alibi(
    Q,
    K,
    V,
    K_cache,
    V_cache,
    B_Loc,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Alibi_slopes,
    block_size,
    x,
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
    stride_k_cache_bl,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    num_queries_per_kv: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_PADDED: tl.constexpr,
    BLOCK_N: tl.constexpr,
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

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
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
        mask=dim_mask[None, :]
        & (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len),
        other=0.0,
    )

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)

    alibi_slope = tl.load(Alibi_slopes + cur_head)
    alibi_start_q = tl.arange(0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
    alibi_start_k = 0

    # Phase 1: attend to context (no causal mask, with ALiBi)
    for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        bn = tl.load(
            B_Loc
            + cur_batch * stride_b_loc_b
            + ((start_n + offs_n) // block_size) * stride_b_loc_s,
            mask=(start_n + offs_n) < cur_batch_ctx_len,
            other=0,
        ).to(tl.int64)
        off_k = (
            bn[None, :] * stride_k_cache_bs
            + cur_kv_head * stride_k_cache_h
            + (offs_d[:, None] // x) * stride_k_cache_d
            + ((start_n + offs_n[None, :]) % block_size) * stride_k_cache_bl
            + (offs_d[:, None] % x) * stride_k_cache_x
        )
        off_v = (
            bn[:, None] * stride_v_cache_bs
            + cur_kv_head * stride_v_cache_h
            + offs_d[None, :] * stride_v_cache_d
            + (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl
        )
        k = tl.load(
            K_cache + off_k,
            mask=dim_mask[:, None] & ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk)
        qk = tl.where(
            (start_n + offs_n[None, :]) < cur_batch_ctx_len, qk, float("-inf")
        )
        qk *= sm_scale

        # ALiBi bias
        alibi = (
            tl.arange(0, BLOCK_N)[None, :] + alibi_start_k - alibi_start_q[:, None]
        ) * alibi_slope
        alibi = tl.where(
            (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
            alibi,
            float("-inf"),
        )
        qk += alibi
        alibi_start_k += BLOCK_N

        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        acc_scale = alpha
        acc = acc * acc_scale[:, None]

        v = tl.load(
            V_cache + off_v,
            mask=dim_mask[None, :] & ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc=acc)
        l_i = l_i_new
        m_i = m_i_new

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

    block_mask = tl.where(block_start_loc < cur_batch_seq_len - cur_batch_ctx_len, 1, 0)

    # Phase 2: attend to query tokens (causal + ALiBi)
    alibi_slope = tl.load(Alibi_slopes + cur_head)
    alibi_start_q = tl.arange(0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
    alibi_start_k = cur_batch_ctx_len

    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=dim_mask[:, None]
            & ((start_n + offs_n[None, :]) < cur_batch_seq_len - cur_batch_ctx_len),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        # ALiBi bias
        alibi = (
            tl.arange(0, BLOCK_N)[None, :] + alibi_start_k - alibi_start_q[:, None]
        ) * alibi_slope
        alibi = tl.where(
            (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
            alibi,
            float("-inf"),
        )
        qk += alibi
        alibi_start_k += BLOCK_N

        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        acc_scale = alpha
        acc = acc * acc_scale[:, None]

        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=dim_mask[None, :]
            & ((start_n + offs_n[:, None]) < cur_batch_seq_len - cur_batch_ctx_len),
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc=acc)
        l_i = l_i_new
        m_i = m_i_new

    acc = acc / l_i[:, None]

    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs,
        acc,
        mask=dim_mask[None, :]
        & (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len),
    )
    return


def run(*, q, k, v, k_cache, v_cache, b_loc, b_start_loc, b_seq_len, alibi_slopes, sm_scale, out):
    head_dim = q.shape[-1]
    head_dim_padded = triton.next_power_of_2(head_dim)
    batch, n_heads = b_seq_len.shape[0], q.shape[1]
    num_queries_per_kv = n_heads // k.shape[1]
    max_input_len = q.shape[0] // batch  # uniform query length per sequence

    q_dtype_is_f32 = q.dtype is torch.float32
    BLOCK = 64 if q_dtype_is_f32 else 128

    grid = (batch, n_heads, triton.cdiv(max_input_len, BLOCK))
    _fwd_kernel_alibi[grid](
        q,
        k,
        v,
        k_cache,
        v_cache,
        b_loc,
        sm_scale,
        b_start_loc,
        b_seq_len,
        alibi_slopes,
        v_cache.shape[3],  # block_size
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
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        k_cache.stride(4),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        num_queries_per_kv=num_queries_per_kv,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=head_dim,
        BLOCK_DMODEL_PADDED=head_dim_padded,
        BLOCK_N=BLOCK,
        num_warps=4,
        num_stages=1,
    )
