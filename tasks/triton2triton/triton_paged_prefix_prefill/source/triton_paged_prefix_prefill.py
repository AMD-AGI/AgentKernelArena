"""
Triton paged attention kernel for prefix prefill.

Adapted from vLLM's prefix_prefill.py (originally from LightLLM).
Implements prefill attention with paged KV cache support — the kernel processes
both cached context tokens (read from paged KV cache) and new query tokens
(read from dense K/V tensors) in two separate loops.

Features:
- Paged KV cache with block table indirection
- K cache in 5D layout: [num_blocks, num_kv_heads, head_size/x, block_size, x]
- V cache in 4D layout: [num_blocks, num_kv_heads, head_size, block_size]
- Grouped-query attention (GQA/MQA)
- Causal masking for the query-vs-query portion
- Sliding window attention

Input layout: [total_tokens, num_heads, head_dim] (variable-length packed batch)
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
            B_Loc + cur_batch * stride_b_loc_b + bn_logical_indices * stride_b_loc_s
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


def context_attention_fwd(
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
    sm_scale: float | None = None,
    sliding_window: int | None = None,
):
    """
    Paged attention for prefix prefill.

    The kernel processes two phases:
    1. Query tokens attend to cached context tokens (from paged KV cache)
    2. Query tokens attend to other query tokens (causal, from dense K/V)

    Args:
        q:        [total_tokens, num_heads, head_dim]  packed query
        k, v:     [total_tokens, num_kv_heads, head_dim]  packed new K/V
        o:        [total_tokens, num_heads, head_dim]  output
        k_cache:  [num_blocks, num_kv_heads, head_dim // x, block_size, x]  paged K
        v_cache:  [num_blocks, num_kv_heads, head_dim, block_size]  paged V
        b_loc:    [batch, max_num_blocks]  block table
        b_start_loc: [batch + 1]  cumulative start positions (note: length = batch+1)
        b_seq_len:   [batch]  total sequence length (context + query)
        max_input_len: max query length in this batch
        sm_scale: softmax scale (default 1/sqrt(head_dim))
        sliding_window: sliding window size (None or 0 = disabled)
    """
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    Lk_padded = triton.next_power_of_2(Lk)

    if sm_scale is None:
        sm_scale = 1.0 / (Lq ** 0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    num_queries_per_kv = q.shape[1] // k.shape[1]

    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    real_block_size = v_cache.shape[3]
    is_pow2 = real_block_size > 0 and (real_block_size & (real_block_size - 1) == 0)
    if is_pow2:
        BLOCK_M = 128
        BLOCK_N = 64
    else:
        BLOCK_M = 32
        BLOCK_N = 32

    TRITON_BLOCK_SIZE = 32

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK_M))
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
        BLOCK_DMODEL=Lk,
        BLOCK_DMODEL_PADDED=Lk_padded,
        SLIDING_WINDOW=sliding_window,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_unroll_cache=4,
        num_unroll_request=1,
        num_warps=4,
        num_stages=1,
    )
