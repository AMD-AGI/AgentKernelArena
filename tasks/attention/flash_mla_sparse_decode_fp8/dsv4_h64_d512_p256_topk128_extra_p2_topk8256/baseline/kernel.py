# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 sparse FP8 MLA decode: vLLM's ROCm split-K Triton kernels.

Vendored from vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
(_sparse_attn_decode_partial_kernel, _sparse_attn_decode_reduce_kernel,
_decode_cu_count, _decode_partial_iters, _decode_num_splits and the gfx942/gfx950
branch of _rocm_sparse_attn_decode_ragged_triton). Task modifications, marked
"task:" below:

- The partial kernel reads the dense [batch, width] index tables and their
  per-row lengths directly. Upstream's dense entry point first packs them into
  ragged (indices, indptr) form with torch.clamp/torch.cumsum and a pack kernel;
  the selected entries and their order are the same. The split heuristic sees
  the table widths as the average segment lengths, exactly as upstream's dense
  path does (its packed buffer holds batch * width entries).
- The reduce kernel also writes the log-sum-exp of the logits (the sink is
  excluded; +inf for a row without KV entries), which the flash_mla_with_kvcache
  interface returns.
- Both pools use OCP float8_e4m3fn codes and arrive as float8_e4m3fn storage,
  viewed as bytes; the fallback kernel for un-tuned architectures and the vLLM
  platform/env plumbing are removed.
"""
import functools
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_attn_decode_partial_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_lens_ptr,  # task: per-row prefix lengths of the dense main table
    extra_cache_ptr,
    extra_indices_ptr,
    extra_lens_ptr,  # task: per-row prefix lengths of the dense extra table
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    q_stride0,
    q_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    main_indices_stride0,  # task: dense table row stride
    extra_indices_stride0,  # task: dense table row stride
    pm_stride0,
    pm_stride_s,
    pa_stride0,
    pa_stride_s,
    pa_stride_h,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    main_width,  # task: dense table width
    extra_width,  # task: dense table width
    scale,
    num_heads,
    HAS_EXTRA: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    # `main_cache` is the SWA K-cache (written by the C++ encoder, FNUZ on
    # gfx942 / OCP on gfx950). `extra_cache` is the compressed K-cache
    # (Triton encoder, OCP on every platform). Reading both with the same
    # `IS_FNUZ` would decode one of them with the wrong FNUZ/OCP scale ratio.
    IS_FNUZ_MAIN: tl.constexpr,
    IS_FNUZ_EXTRA: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    query_idx = tl.program_id(0)
    split_id = tl.program_id(1)
    pid_h = tl.program_id(2)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row_ptr = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_nope = tl.load(
        q_row_ptr + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row_ptr + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    zero_nope = tl.zeros((BLOCK_K, NOPE_BLOCK), dtype=tl.bfloat16)
    zero_rope = tl.zeros((BLOCK_K, ROPE_DIM), dtype=tl.bfloat16)

    # Each split processes a contiguous slice of this query's main (SWA) and
    # extra (topk) segments. Slices are handled independently so a block never
    # straddles the main/extra boundary.
    # task: the segment is the first `len` entries of the query's dense row.
    main_start = query_idx.to(tl.int64) * main_indices_stride0
    main_len = tl.load(main_lens_ptr + query_idx)
    main_len = tl.minimum(tl.maximum(main_len, 0), main_width)
    main_chunk = (main_len + NUM_SPLITS - 1) // NUM_SPLITS
    main_lo = split_id * main_chunk
    main_hi = tl.minimum(main_lo + main_chunk, main_len)

    for k_start in tl.range(main_lo, main_hi, BLOCK_K, num_stages=NUM_STAGES):
        k_pos = k_start + k_offsets
        in_range = k_pos < main_hi
        slot = tl.load(main_indices_ptr + main_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < main_num_rows)
        safe_slot = tl.where(valid, slot, 0)

        block_idx = safe_slot // main_block_size
        pos_in_block = safe_slot % main_block_size
        cache_block_ptr = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data_ptr = cache_block_ptr + pos_in_block * 576
        token_scale_ptr = cache_block_ptr + main_block_size * 576 + pos_in_block * 8

        x_uint8 = tl.load(
            token_data_ptr[:, None] + nope_offsets[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        if IS_FNUZ_MAIN:
            x_fp8 = x_uint8.to(tl.float8e4b8, bitcast=True)
        else:
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
        encoded_scales = tl.load(
            token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
            mask=valid[:, None] & nope_mask[None, :],
            other=127,
        )
        scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
        k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)
        k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
        k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

        rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(
            rope_ptr[:, None] + rope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        k_rope = tl.where(valid[:, None], k_rope, zero_rope)
        k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

        scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))
        scores *= scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
        m_i = m_new
        l_i = l_new

    if HAS_EXTRA:
        # task: dense extra row, first `len` entries.
        extra_start = query_idx.to(tl.int64) * extra_indices_stride0
        extra_len = tl.load(extra_lens_ptr + query_idx)
        extra_len = tl.minimum(tl.maximum(extra_len, 0), extra_width)
        extra_chunk = (extra_len + NUM_SPLITS - 1) // NUM_SPLITS
        extra_lo = split_id * extra_chunk
        extra_hi = tl.minimum(extra_lo + extra_chunk, extra_len)

        for k_start in tl.range(extra_lo, extra_hi, BLOCK_K, num_stages=NUM_STAGES):
            k_pos = k_start + k_offsets
            in_range = k_pos < extra_hi
            slot = tl.load(
                extra_indices_ptr + extra_start + k_pos, mask=in_range, other=-1
            )
            valid = in_range & (slot >= 0) & (slot < extra_num_rows)
            safe_slot = tl.where(valid, slot, 0)

            block_idx = safe_slot // extra_block_size
            pos_in_block = safe_slot % extra_block_size
            cache_block_ptr = (
                extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
            )
            token_data_ptr = cache_block_ptr + pos_in_block * 576
            token_scale_ptr = (
                cache_block_ptr + extra_block_size * 576 + pos_in_block * 8
            )

            x_uint8 = tl.load(
                token_data_ptr[:, None] + nope_offsets[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0,
            )
            if IS_FNUZ_EXTRA:
                x_fp8 = x_uint8.to(tl.float8e4b8, bitcast=True)
            else:
                x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            encoded_scales = tl.load(
                token_scale_ptr[:, None] + nope_offsets[None, :] // 64,
                mask=valid[:, None] & nope_mask[None, :],
                other=127,
            )
            scales = tl.exp2(encoded_scales.to(tl.float32) - 127.0)
            k_nope = x_fp8.to(tl.bfloat16) * scales.to(tl.bfloat16)
            k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_nope)
            k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)

            rope_ptr = (token_data_ptr + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            k_rope = tl.load(
                rope_ptr[:, None] + rope_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            k_rope = tl.where(valid[:, None], k_rope, zero_rope)
            k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)

            scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(
                q_rope,
                tl.trans(k_rope),
            )
            scores *= scale
            scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            acc_nope = acc_nope * alpha[:, None] + tl.dot(p.to(k_nope.dtype), k_nope)
            acc_rope = acc_rope * alpha[:, None] + tl.dot(p.to(k_rope.dtype), k_rope)
            m_i = m_new
            l_i = l_new

    # Store raw (un-normalized) partial state for this split. Softmax sink and
    # final normalization happen in the reduce kernel.
    pm_base = query_idx * pm_stride0 + split_id * pm_stride_s + head_offsets
    tl.store(part_m_ptr + pm_base, m_i, mask=head_mask)
    tl.store(part_l_ptr + pm_base, l_i, mask=head_mask)
    acc_base = (
        part_acc_ptr
        + query_idx * pa_stride0
        + split_id * pa_stride_s
        + head_offsets[:, None] * pa_stride_h
    )
    tl.store(
        acc_base + nope_offsets[None, :],
        acc_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        acc_base + NOPE_DIM + rope_offsets[None, :],
        acc_rope,
        mask=head_mask[:, None],
    )


@triton.jit
def _sparse_attn_decode_reduce_kernel(
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    attn_sink_ptr,
    out_ptr,
    lse_ptr,  # task: log-sum-exp output
    out_stride0,
    out_stride1,
    lse_stride0,  # task: log-sum-exp row stride
    pm_stride0,
    pm_stride_s,
    pa_stride0,
    pa_stride_s,
    pa_stride_h,
    num_heads,
    HAS_ATTN_SINK: tl.constexpr,
    COMB_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLITS_PAD: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    comb_offsets = tl.arange(0, COMB_DIM)
    # SPLITS_PAD is NUM_SPLITS rounded up to a power of two so the parallel
    # split-axis load is a legal arange for any split count; padding lanes are
    # masked off.
    split_offsets = tl.arange(0, SPLITS_PAD)
    split_mask = split_offsets < NUM_SPLITS

    neg_large = -3.4028234663852886e38

    # Phase 1: load every split's running max/sum at once and reduce the max
    # in parallel (tl.max over the split axis) instead of walking the splits
    # serially. This breaks the long online-softmax dependency chain that made
    # the reduce latency-bound.
    load_mask = split_mask[:, None] & head_mask[None, :]
    pm_split = (
        part_m_ptr
        + query_idx * pm_stride0
        + split_offsets[:, None] * pm_stride_s
        + head_offsets[None, :]
    )
    m_all = tl.load(pm_split, mask=load_mask, other=neg_large)  # [S, H]
    l_all = tl.load(
        part_l_ptr
        + query_idx * pm_stride0
        + split_offsets[:, None] * pm_stride_s
        + head_offsets[None, :],
        mask=load_mask,
        other=0.0,
    )

    m_comb = tl.max(m_all, axis=0)  # [H]

    # task: log-sum-exp of the logits alone (the sink is not a logit), +inf
    # for a row without any selected KV entry.
    l_comb = tl.sum(
        tl.where(load_mask, tl.exp(m_all - m_comb[None, :]), 0.0) * l_all, axis=0
    )
    lse = tl.where(l_comb > 0.0, m_comb + tl.log(l_comb), float("inf"))
    tl.store(lse_ptr + query_idx * lse_stride0 + head_offsets, lse, mask=head_mask)

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_comb, sink)
    else:
        m_final = m_comb

    w_all = tl.exp(m_all - m_final[None, :])  # [S, H]
    w_all = tl.where(load_mask, w_all, 0.0)
    l_final = tl.sum(w_all * l_all, axis=0)  # [H]
    if HAS_ATTN_SINK:
        l_final = l_final + tl.exp(sink - m_final)
    denom = tl.maximum(l_final, 1.0e-30)

    # Phase 2: weighted sum of the per-split accumulators. The combine weight
    # for each split only depends on the (already known) global max, so the
    # acc loads carry no cross-split dependency and the compiler can pipeline
    # them; only the cheap FMA into `acc` is loop-carried.
    acc = tl.zeros((BLOCK_H, COMB_DIM), dtype=tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        m_s = tl.load(
            part_m_ptr + query_idx * pm_stride0 + s * pm_stride_s + head_offsets,
            mask=head_mask,
            other=neg_large,
        )
        w_s = tl.exp(m_s - m_final)
        acc_base = (
            part_acc_ptr
            + query_idx * pa_stride0
            + s * pa_stride_s
            + head_offsets[:, None] * pa_stride_h
        )
        acc_s = tl.load(
            acc_base + comb_offsets[None, :],
            mask=head_mask[:, None],
            other=0.0,
        )
        acc += w_s[:, None] * acc_s

    out = tl.where(l_final[:, None] > 0.0, acc / denom[:, None], 0.0)

    out_row_ptr = (
        out_ptr + query_idx * out_stride0 + head_offsets[:, None] * out_stride1
    )
    tl.store(
        out_row_ptr + comb_offsets[None, :],
        out,
        mask=head_mask[:, None],
    )


@functools.lru_cache
def _decode_cu_count() -> int:
    try:
        return torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        return 256  # For gfx950 arch, gated behind a fallback path for other archs.


def _decode_partial_iters(
    avg_main_len: float, avg_extra_len: float, splits: int, block_k: int
) -> int:
    """BLOCK_K iterations one partial workgroup walks for ``splits`` splits.

    Each split processes ``ceil(seg_len / splits)`` tokens of a segment, walked
    ``BLOCK_K`` at a time, and the main/extra segments are handled separately.
    """
    main_iters = (
        math.ceil(math.ceil(avg_main_len / splits) / block_k) if avg_main_len > 0 else 0
    )
    extra_iters = (
        math.ceil(math.ceil(avg_extra_len / splits) / block_k)
        if avg_extra_len > 0
        else 0
    )
    return main_iters + extra_iters


def _decode_num_splits(
    num_queries: int,
    heads_blocks: int,
    avg_main_len: float = 0.0,
    avg_extra_len: float = 0.0,
    block_k: int = 32,
) -> int:
    """Pick a flash-decode split count to keep the GPU busy across batch sizes.

    Decode launches only ``num_queries * heads_blocks`` workgroups otherwise,
    which severely under-fills the device for the low-concurrency regime that
    dominates latency. Splitting the KV sequence adds parallelism.

    We model the relative partial-kernel latency for a given split count ``s``
    as ``waves * (1/s + mu)`` where ``waves = ceil(base * s / CU)`` and ``mu``
    is a small per-wave overhead penalty:

      - ``waves / s`` captures the partial compute: each wave walks roughly
        ``total_tokens / s`` tokens and there are ``waves`` of them, so dividing
        by ``s`` makes more splits cheaper *until* they spill into extra waves.
      - ``mu * waves`` charges per-wave launch/tail overhead so we do not
        over-split into many mostly-idle waves (e.g. batch 224 on 256 CUs is
        best left at 1 split rather than 8 splits across 7 waves).

    The minimiser naturally prefers split counts that pack the device into full
    waves (``base * s`` near a multiple of ``CU``) and falls back to 1 split
    once the batch already fills the device. Ties favour the smaller split
    count (less reduce work).

    Finally we "snap down" the chosen split count to the smallest value that
    yields the same wave count *and* the same per-workgroup BLOCK_K iteration
    count. Because latency tracks iteration count (not raw token count), extra
    splits that do not lower the iteration count add only reduce/HBM overhead
    for no parallelism gain (e.g. batch 24: s8 and s10 both walk 4 extra iters
    in one wave, so s8 is strictly better). Snapping needs the average segment
    lengths, which the caller derives sync-free from the ragged index sizes.
    """
    base = max(1, num_queries * heads_blocks)
    # Target ~1 workgroup per CU: enough to fill the device while keeping the
    # reduce cost (which grows with split count) small. Tuned on gfx950.
    cu = max(1, _decode_cu_count())
    # Per-wave overhead penalty: higher values discourage split counts that
    # spill into extra GPU waves. Tuned on gfx950.
    mu = 0.04
    best_splits = 1
    best_cost = None
    # Search up to 16 splits; beyond that the reduce/HBM overhead dominates.
    for splits in range(1, 17):
        waves = (base * splits + cu - 1) // cu
        cost = waves * (1.0 / splits + mu)
        if best_cost is None or cost < best_cost - 1e-9:
            best_splits = splits
            best_cost = cost

    if best_splits > 1 and (avg_main_len > 0 or avg_extra_len > 0):
        target_waves = (base * best_splits + cu - 1) // cu
        target_iters = _decode_partial_iters(
            avg_main_len, avg_extra_len, best_splits, block_k
        )
        for splits in range(1, best_splits):
            waves = (base * splits + cu - 1) // cu
            iters = _decode_partial_iters(avg_main_len, avg_extra_len, splits, block_k)
            if waves == target_waves and iters == target_iters:
                best_splits = splits
                break
    return best_splits


_NOPE_HEAD_DIM = 448
_ROPE_HEAD_DIM = 64


def run(
    *,
    q,
    kv_cache,
    sparse_indices,
    sparse_lens,
    extra_kv_cache,
    extra_sparse_indices,
    extra_sparse_lens,
    sm_scale,
    sinks,
    output,
    lse,
):
    """Sparse decode over the main and extra packed pools into output and lse."""
    num_queries, _, num_heads, head_dim = q.shape
    # One query token per row: [batch, 1, heads, dim] is [batch, heads, dim].
    q3 = q.view(num_queries, num_heads, head_dim)
    out = output.view(num_queries, num_heads, head_dim)
    lse2 = lse.view(num_queries, num_heads)
    main_indices = sparse_indices.view(num_queries, -1)
    extra_indices = extra_sparse_indices.view(num_queries, -1)
    attn_sink = sinks.contiguous()
    # task: the pools are float8_e4m3fn storage of packed bytes; the kernels read bytes.
    kv_cache = kv_cache.view(torch.uint8)
    extra_kv_cache = extra_kv_cache.view(torch.uint8)
    nope_head_dim, rope_head_dim = _NOPE_HEAD_DIM, _ROPE_HEAD_DIM

    block_h = 16
    heads_blocks = triton.cdiv(num_heads, block_h)
    nope_block = triton.next_power_of_2(nope_head_dim)
    comb_dim = nope_head_dim + rope_head_dim

    block_k = 32  # KV tokens walked per split-K iteration. Tuned on gfx950.
    # task: upstream's dense path packs batch * width entries, so its average
    # segment lengths are the table widths.
    avg_main_len = float(main_indices.shape[1])
    avg_extra_len = float(extra_indices.shape[1])
    num_splits = _decode_num_splits(
        num_queries, heads_blocks, avg_main_len, avg_extra_len, block_k
    )

    part_m = torch.empty(
        (num_queries, num_splits, num_heads), dtype=torch.float32, device=q.device
    )
    part_l = torch.empty_like(part_m)
    part_acc = torch.empty(
        (num_queries, num_splits, num_heads, comb_dim),
        dtype=torch.float32,
        device=q.device,
    )

    _sparse_attn_decode_partial_kernel[(num_queries, num_splits, heads_blocks)](
        q3,
        kv_cache,
        main_indices,
        sparse_lens,
        extra_kv_cache,
        extra_indices,
        extra_sparse_lens,
        part_m,
        part_l,
        part_acc,
        q3.stride(0),
        q3.stride(1),
        kv_cache.stride(0),
        extra_kv_cache.stride(0),
        main_indices.stride(0),
        extra_indices.stride(0),
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        kv_cache.shape[0] * kv_cache.shape[1],
        extra_kv_cache.shape[0] * extra_kv_cache.shape[1],
        kv_cache.shape[1],
        extra_kv_cache.shape[1],
        main_indices.shape[1],
        extra_indices.shape[1],
        sm_scale,
        num_heads,
        HAS_EXTRA=True,
        NOPE_DIM=nope_head_dim,
        NOPE_BLOCK=nope_block,
        ROPE_DIM=rope_head_dim,
        IS_FNUZ_MAIN=False,
        IS_FNUZ_EXTRA=False,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        NUM_SPLITS=num_splits,
        NUM_STAGES=1,
        num_warps=4,
    )

    _sparse_attn_decode_reduce_kernel[(num_queries, num_heads)](
        part_m,
        part_l,
        part_acc,
        attn_sink,
        out,
        lse2,
        out.stride(0),
        out.stride(1),
        lse2.stride(0),
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        num_heads,
        HAS_ATTN_SINK=True,
        COMB_DIM=comb_dim,
        BLOCK_H=1,
        NUM_SPLITS=num_splits,
        SPLITS_PAD=triton.next_power_of_2(num_splits),
        num_warps=4,
    )
