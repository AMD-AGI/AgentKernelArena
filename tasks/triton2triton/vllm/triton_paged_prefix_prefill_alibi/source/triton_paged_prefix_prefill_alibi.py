"""
Triton paged attention kernel for prefix prefill with ALiBi positional bias.

Adapted from vLLM's prefix_prefill.py (originally from LightLLM).
Implements prefill attention with paged KV cache and ALiBi (Attention with
Linear Biases) positional encoding.

Two-phase attention:
1. Query-vs-Context: queries attend to cached context tokens with ALiBi bias
2. Query-vs-Query: queries attend to new query tokens (causal) with ALiBi bias

Features:
- Paged KV cache with block table indirection
- K cache in 5D layout: [num_blocks, num_kv_heads, head_size/x, block_size, x]
- V cache in 4D layout: [num_blocks, num_kv_heads, head_size, block_size]
- ALiBi per-head slopes for positional encoding
- Grouped-query attention (GQA/MQA)
- Causal masking for query-vs-query portion
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


def context_attention_fwd_alibi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    b_loc: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    alibi_slopes: torch.Tensor,
    sm_scale: float | None = None,
):
    """
    Paged attention for prefix prefill with ALiBi positional bias.

    Args:
        q:        [total_tokens, num_heads, head_dim]  packed query
        k, v:     [total_tokens, num_kv_heads, head_dim]  packed new K/V
        o:        [total_tokens, num_heads, head_dim]  output
        k_cache:  [num_blocks, num_kv_heads, head_dim // x, block_size, x]  paged K
        v_cache:  [num_blocks, num_kv_heads, head_dim, block_size]  paged V
        b_loc:    [batch, max_num_blocks]  block table
        b_start_loc: [batch + 1]  cumulative start positions
        b_seq_len:   [batch]  total sequence length (context + query)
        max_input_len: max query length in this batch
        alibi_slopes: [num_heads]  per-head ALiBi slopes
        sm_scale: softmax scale (default 1/sqrt(head_dim))
    """
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    Lk_padded = triton.next_power_of_2(Lk)

    if sm_scale is None:
        sm_scale = 1.0 / (Lq ** 0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    num_queries_per_kv = q.shape[1] // k.shape[1]

    q_dtype_is_f32 = q.dtype is torch.float32
    BLOCK = 64 if q_dtype_is_f32 else 128

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4

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
        v_cache.shape[3],   # block_size
        k_cache.shape[4],   # x
        o,
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
        o.stride(0),
        o.stride(1),
        o.stride(2),
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
        BLOCK_DMODEL=Lk,
        BLOCK_DMODEL_PADDED=Lk_padded,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
