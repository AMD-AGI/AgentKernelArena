# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved. (MIT, see LICENSE)
# Taken from AITER aiter/ops/triton/_triton_kernels/attention/unified_attention.py
# (kernel_unified_attention_2d) and aiter/ops/triton/attention/unified_attention.py
# (select_2d_config, launch) as a Triton stand-in for AITER's Composable Kernel
# mha_batch_prefill: the block table is replaced by the kv_indptr / kv_page_indices /
# kv_last_page_lens page list of the CK batch-prefill interface.
"""Paged-KV batch prefill attention with logits soft-capping (bottom-right causal, GQA)."""
import triton
import triton.language as tl


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.math.exp2(Sdiv)
    p2 = tl.math.exp2(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def kernel_batch_prefill_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_pages, page_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_pages, page_size, num_kv_heads, head_size]
    kv_indptr_ptr,  # [num_seqs + 1]
    kv_page_indices_ptr,  # [>= kv_indptr[num_seqs]]
    kv_last_page_lens_ptr,  # [num_seqs]
    scale: tl.constexpr,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int, page size
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_SOFTCAP: tl.constexpr,  # bool
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
):
    kv_head_idx = tl.program_id(0)
    q_block_global_idx = tl.program_id(1)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    if BLOCK_M >= num_query_heads:
        Q_cache_modifier: tl.constexpr = ".cg"
    else:
        Q_cache_modifier: tl.constexpr = ""
    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
        cache_modifier=Q_cache_modifier,
    )

    # This sequence's pages: kv_page_indices[kv_indptr[seq] : kv_indptr[seq + 1]],
    # all full except the last, which holds kv_last_page_lens[seq] tokens.
    page_list_start = tl.load(kv_indptr_ptr + seq_idx)
    num_seq_pages = tl.load(kv_indptr_ptr + seq_idx + 1) - page_list_start

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = (num_seq_pages - 1) * BLOCK_SIZE + tl.load(kv_last_page_lens_ptr + seq_idx)

    # context length for this particular sequences (bottom-right causal alignment)
    context_len = seq_len - cur_batch_query_len

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # iterate through tiles
    for j in range(0, num_tiles):
        seq_offset = j * TILE_SIZE + offs_t
        # to reduce the masking effect when not needed
        if TILE_SIZE == BLOCK_SIZE:
            tile_mask = tl.full((1,), 1, dtype=tl.int1)
        else:
            tile_mask = seq_offset < max_seq_prefix_len

        # The page-list load is masked: past the sequence's last page the list
        # belongs to the next sequence or ends.
        physical_block_idx = tl.load(
            kv_page_indices_ptr + page_list_start + seq_offset // BLOCK_SIZE,
            mask=seq_offset < max_seq_prefix_len,
            other=0,
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        v_mask = dim_mask[None, :] & tile_mask[:, None]

        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )
        k_mask = dim_mask[:, None] & tile_mask[None, :]

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(key_cache_ptr + k_offset, mask=k_mask, other=0.0)
        K = K_load.to(Q.dtype)

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(value_cache_ptr + v_offset, mask=v_mask, other=0.0)
        V = V_load.to(Q.dtype)

        # S : (BLOCK_M, TILE_SIZE)
        # qk_scale = scale * RCP_LN2 (log_2 e) so that we can use exp2 later
        S = qk_scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            # softcap here uses exp2 and consumes RCP_LN2 conversion.
            # multiply by RCP_LN2 again to be used in later exp2
            S = apply_softcap(S, softcap) * RCP_LN2
        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # A fully masked (padding) row keeps -inf; use 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE)
        P = tl.math.exp2(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.math.exp2(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = tl.dot(P.to(V.dtype), V, acc=acc)

    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    one_over_L = 1.0 / L[:, None]
    acc = acc * one_over_L

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


def select_2d_config(head_size, max_seqlen_q, num_queries_per_kv):
    """AITER's select_2d_config for a CDNA GPU, prefill (not all-decode), unshuffled bf16 KV."""
    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )

    TILE_SIZE = 64
    waves_per_eu = 2

    max_num_stages_2d = 2 if head_size > 128 else 4

    # base prefill, for short cases
    if head_size >= 512:
        num_warps, num_stages_2d = 4, 2
        TILE_SIZE = 16
    elif head_size >= 256:
        num_warps, num_stages_2d = 2, 2
        TILE_SIZE = 32
    else:
        # large prefill config
        if max_seqlen_q >= 256:
            BLOCK_M = 128
            num_stages_2d, num_warps = 1, 4
        else:
            num_stages_2d, num_warps = 1, 2

    BLOCK_Q = BLOCK_M // num_queries_per_kv
    num_stages_2d = min(max_num_stages_2d, num_stages_2d)

    return {
        "BLOCK_M": BLOCK_M,
        "BLOCK_Q": BLOCK_Q,
        "TILE_SIZE": TILE_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages_2d,
        "waves_per_eu": waves_per_eu,
    }


def run(*, q, k_cache, v_cache, cu_seqlens_q, kv_indptr, kv_page_indices, kv_last_page_lens,
        softmax_scale, logits_soft_cap, out):
    num_tokens, num_query_heads, head_size = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    num_seqs = kv_last_page_lens.shape[0]
    num_queries_per_kv = num_query_heads // num_kv_heads

    # Production passes max_seqlen_q; the mean query length selects the same
    # (short or long prefill) configuration without a host round trip.
    config = select_2d_config(head_size, num_tokens // num_seqs, num_queries_per_kv)
    total_num_q_blocks = num_tokens // config["BLOCK_Q"] + num_seqs

    kernel_batch_prefill_2d[(num_kv_heads, total_num_q_blocks)](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k_cache,
        value_cache_ptr=v_cache,
        kv_indptr_ptr=kv_indptr,
        kv_page_indices_ptr=kv_page_indices,
        kv_last_page_lens_ptr=kv_last_page_lens,
        scale=softmax_scale,
        softcap=logits_soft_cap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_SOFTCAP=(logits_soft_cap > 0),
        stride_k_cache_0=k_cache.stride(0),
        stride_k_cache_1=k_cache.stride(1),
        stride_k_cache_2=k_cache.stride(2),
        stride_k_cache_3=k_cache.stride(3),
        stride_v_cache_0=v_cache.stride(0),
        stride_v_cache_1=v_cache.stride(1),
        stride_v_cache_2=v_cache.stride(2),
        stride_v_cache_3=v_cache.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        num_seqs=num_seqs,
        **config,
    )
