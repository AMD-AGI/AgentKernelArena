# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>
"""vLLM's Triton unified (paged) attention for bfloat16, wrapped in the run() interface.

Kernels and the 2D / 3D dispatch: vllm/v1/attention/ops/triton_unified_attention.py (vllm-project/vllm
edbc4969a76b25c531f7b9bb16b79984e65ef023); the shared device helpers are in helpers.py. The kernels are
unchanged; the host side keeps the dispatch decisions (tile sizes, 2D versus 3D plus reduce_segments)
and the TritonAttentionImpl calling convention for a bfloat16 causal layer.
"""

import torch
import triton
import triton.language as tl

from .helpers import (
    apply_alibi_to_score,
    apply_softcap,
    cdiv_fn,
    compute_kv_seq_mask,
    compute_tile_loop_bounds,
    find_seq_idx,
    init_softmax_M,
    load_qq_bias_tile,
    resolve_seq_and_query_len,
    softmax_step,
    store_segm_reduce_scalars,
)

# vllm.platforms.current_platform.fp8_dtype() on gfx950.
float8_info = torch.finfo(torch.float8_e4m3fn)


@triton.jit
def _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE: tl.constexpr):
    """Cast a loaded KV tile to Q's dtype, dequantizing if needed.

    Modes handled inside the core kernel:

    - ``KV_QUANT_MODE == 0`` (NONE) and ``2`` (INT8 per-token-head) and
      ``3`` (FP8 per-token-head): plain cast.  Per-token-head modes apply
      their scales separately on S/P inside the loop.
    - ``KV_QUANT_MODE == 1`` (FP8 per-tensor): dequantize using the
      tensor-wide scale, unless Q is also FP8 and the caller folds the scales
      into the attention score and output accumulator.
    """
    if KV_QUANT_MODE == 1:
        if Q.dtype.is_fp8():
            return data.to(Q.dtype)
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)
    return data.to(Q.dtype)


# ---------------------------------------------------------------------------
# Tensor-descriptor (TD) helpers
#
# Used when the caller enables ``USE_TD`` (Intel Xe2/Xe3 HW 2D block reads).
# When ``USE_TD`` is False the helpers are dead-code-eliminated at Triton
# compile time, leaving the pointer-arithmetic load/store path untouched.
# ---------------------------------------------------------------------------


@triton.jit
def _load_q_td(
    query_ptr,
    q_block_local_len,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    cur_batch_in_all_start_index,
    q_block_local_idx,
    kv_head_idx,
    num_queries_per_kv: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    """Load Q via a 2D tensor descriptor.

    Caller guarantees (via the wrapper's ``use_td_qo`` gate):
      * ``HEAD_SIZE == HEAD_SIZE_PADDED`` (head_size is a power of 2),
      * ``num_queries_per_kv`` is a power of 2,
      * the ``num_queries_per_kv`` heads of the current KV group are
        contiguous in memory (``query_stride_1 == HEAD_SIZE``, which is
        the default vLLM query layout).

    Under those preconditions the inner two axes are flattened into one
    row of size ``num_queries_per_kv * HEAD_SIZE`` with stride 1, which
    avoids the non-power-of-2 ``block_shape`` error from the Triton
    tensor-descriptor validator.  Returns (BLOCK_M, HEAD_SIZE_PADDED).
    """
    q_base = (
        query_ptr
        + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q) * query_stride_0
        + (kv_head_idx * num_queries_per_kv) * query_stride_1
    )
    q_desc = tl.make_tensor_descriptor(
        base=q_base,
        shape=(q_block_local_len, num_queries_per_kv * HEAD_SIZE),
        strides=(query_stride_0, 1),
        block_shape=(BLOCK_Q, num_queries_per_kv * HEAD_SIZE_PADDED),
    )
    return q_desc.load([0, 0]).reshape(BLOCK_M, HEAD_SIZE_PADDED)


@triton.jit
def _load_kv_tile_td(
    cache_ptr,
    physical_block_idx_scalar,
    kv_head_idx,
    offset_in_block,
    stride_cache_0: tl.int64,
    stride_cache_1: tl.int64,
    stride_cache_2: tl.int64,
    stride_cache_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    """Load a KV cache tile via tensor descriptor.

    Returns shape (TILE_SIZE, HEAD_SIZE_PADDED). Caller transposes for K.
    Tensor descriptors zero-pad reads beyond the shape boundary, so
    ``HEAD_SIZE_PADDED > HEAD_SIZE`` is handled correctly.
    """
    base = (
        cache_ptr
        + physical_block_idx_scalar * stride_cache_0
        + kv_head_idx * stride_cache_2
    )
    desc = tl.make_tensor_descriptor(
        base=base,
        shape=(BLOCK_SIZE, HEAD_SIZE),
        strides=(stride_cache_1, stride_cache_3),
        block_shape=(TILE_SIZE, HEAD_SIZE_PADDED),
    )
    return desc.load([offset_in_block, 0])


@triton.jit
def _store_output_td(
    base_ptr,
    acc,
    q_block_local_len,
    stride_token: tl.int64,
    stride_head: tl.int64,
    num_queries_per_kv: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    """Store an output tile via a tensor descriptor.

    The 2D and 3D epilogues differ only in ``base_ptr`` and the
    ``(stride_token, stride_head)`` pair: 2D writes directly to the
    flat output buffer, 3D writes to a single per-segment slice of
    ``segm_output_ptr``.  Descriptor shape / block_shape / reshape
    are the same in both modes, so share one helper.
    """
    acc = acc.to(base_ptr.dtype.element_ty)
    output_desc = tl.make_tensor_descriptor(
        base=base_ptr,
        shape=(q_block_local_len, num_queries_per_kv, HEAD_SIZE),
        strides=(stride_token, stride_head, 1),
        block_shape=(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED),
    )
    output_desc.store(
        [0, 0, 0],
        acc.reshape(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED),
    )


@triton.jit
def kernel_unified_attention(
    # Output destination for the 2D path.  In 3D mode per-segment partials
    # go to the ``segm_*`` tensors (see bottom of signature) and
    # ``output_ptr`` is unused (callers may pass any non-null pointer).
    output_ptr,
    # Inputs
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    # Scalars
    scale,
    q_scale,
    k_scale,
    v_scale,
    out_scale,
    softcap,
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    USE_CAUSAL: tl.constexpr,  # bool
    USE_PER_SEQ_CAUSAL: tl.constexpr,  # bool
    per_seq_causal_ptr,  # [num_seqs] bool, or None
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,
    rswa_prefix_lens_ptr,
    R_SWA_WINDOW: tl.constexpr,  # int
    USE_R_SWA: tl.constexpr,  # bool
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
    # Toggles 2D vs 3D layout.  The 2D path runs the full sequence in one
    # tile loop and writes to ``output_ptr``.  The 3D path scopes the loop
    # to ``[segm_idx, segm_idx+1) × tiles_per_segment`` and writes
    # per-segment partials, finalized by ``reduce_segments``.
    IS_3D: tl.constexpr,
    # Parameters below default to None so Triton can skip materialising them
    # on call sites where the corresponding constexpr branch is dead.
    # Credit: @quinnlp identified this as a perf regression source in
    # intel/intel-xpu-backend-for-triton#6758 (review comment r3204641104).
    # Per-segment outputs: used in 3D mode; unused in 2D (IS_3D=False).
    segm_output_ptr=None,
    segm_max_ptr=None,
    segm_expsum_ptr=None,
    # Per-(token, head) scale caches: used iff KV_QUANT_MODE in {2, 3}.
    k_scale_cache_ptr=None,
    v_scale_cache_ptr=None,
    # ``tl.int64`` cannot be combined with a ``None`` default — Triton's JIT
    # rejects ``Optional[tl.int64]`` / ``tl.int64 | None`` at trace time, and
    # plain ``tl.int64 = None`` raises ``TypeError: 'NoneType' object cannot
    # be interpreted as an integer`` when callers omit these arguments.
    # ``int | None`` is the only annotation that lets the wrapper pass
    # ``None`` here so Triton can skip materialising the strides when the
    # ``USE_PER_TOKEN_HEAD_SCALES`` branch is dead.
    stride_ks_blk: int | None = None,
    stride_ks_slot: int | None = None,
    stride_ks_head: int | None = None,
    stride_vs_blk: int | None = None,
    stride_vs_slot: int | None = None,
    stride_vs_head: int | None = None,
    # KV cache quantization mode handled inside this kernel via constexpr
    # branches: NONE (0), FP8_PER_TENSOR (1), INT8_PER_TOKEN_HEAD (2),
    # FP8_PER_TOKEN_HEAD (3). Sub-byte INT4 (4) uses its own
    # int4_per_token_head kernel, not this one.
    KV_QUANT_MODE: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    # Chunked / block-local attention.  ``CHUNK_LOOKBACK >= 0`` enables
    # chunked masking (used by Gemma3 block-local layers); takes precedence
    # over ``SLIDING_WINDOW`` inside the helpers.  ``-1`` disables.
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    # Tensor-descriptor load/store for HW 2D block reads on Intel Xe2/Xe3.
    # ``USE_TD`` gates KV tile loads; ``USE_TD_QO`` separately gates Q/output
    # (see ``unified_attention`` wrapper for the gating rules).
    USE_TD: tl.constexpr = False,
    USE_TD_QO: tl.constexpr = False,
    Q_IS_FP8: tl.constexpr = False,
    # Gemma4: clamp mm_prefix bidirectional ranges by the sliding window
    # instead of letting them override it. Default False preserves the
    # original (causal AND SW) OR mm_prefix behavior for all other models.
    MM_PREFIX_CLAMP_SW: tl.constexpr = False,
):
    # Per-(token, head) scale caches: used iff KV_QUANT_MODE in {2, 3}.
    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = (KV_QUANT_MODE >= 2) and (
        KV_QUANT_MODE <= 3
    )
    USE_FP8_Q_DESCALE: tl.constexpr = KV_QUANT_MODE == 1 and Q_IS_FP8

    if USE_TD:
        tl.static_assert(
            BLOCK_SIZE % TILE_SIZE == 0,
            "USE_TD requires BLOCK_SIZE to be a multiple of TILE_SIZE",
        )

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    # Number of valid query rows in this block (used by TD descriptor
    # shapes, but always computed so the variable stays in scope).
    q_block_local_len = tl.minimum(
        BLOCK_Q, cur_batch_query_len - q_block_local_idx * BLOCK_Q
    )

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

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    if USE_TD_QO:
        Q = _load_q_td(
            query_ptr,
            q_block_local_len,
            query_stride_0,
            query_stride_1,
            cur_batch_in_all_start_index,
            q_block_local_idx,
            kv_head_idx,
            num_queries_per_kv,
            BLOCK_Q,
            BLOCK_M,
            HEAD_SIZE,
            HEAD_SIZE_PADDED,
        )
    else:
        Q = tl.load(
            query_ptr + query_offset,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            other=0.0,
        )

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    # acc : (BLOCK_M, HEAD_SIZE_PADDED)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    score_scale = scale
    value_scale = 1.0
    if USE_FP8_Q_DESCALE:
        score_scale = scale * tl.load(q_scale) * tl.load(k_scale)
        value_scale = tl.load(v_scale)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX or USE_R_SWA,
        IS_3D,
        USE_CAUSAL,
        USE_PER_SEQ_CAUSAL,
        CHUNK_LOOKBACK,
        CHUNK_SIZE,
    )

    # iterate through tiles (now limited to the sliding window range)
    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        if USE_TD:
            # All TILE_SIZE slots within a single KV tile map to one
            # physical block (guaranteed by ``BLOCK_SIZE % TILE_SIZE == 0``
            # from the static_assert above), so load the block index as
            # a scalar instead of a broadcast reduction.
            offset_in_block = (j * TILE_SIZE) % BLOCK_SIZE
            physical_block_scalar = tl.load(
                block_tables_ptr + block_table_offset + (j * TILE_SIZE) // BLOCK_SIZE
            ).to(tl.int64)
            # K : (HEAD_SIZE, TILE_SIZE)
            K_load = _load_kv_tile_td(
                key_cache_ptr,
                physical_block_scalar,
                kv_head_idx,
                offset_in_block,
                stride_k_cache_0,
                stride_k_cache_1,
                stride_k_cache_2,
                stride_k_cache_3,
                BLOCK_SIZE,
                TILE_SIZE,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            ).T
            # V : (TILE_SIZE, HEAD_SIZE)
            V_load = _load_kv_tile_td(
                value_cache_ptr,
                physical_block_scalar,
                kv_head_idx,
                offset_in_block,
                stride_v_cache_0,
                stride_v_cache_1,
                stride_v_cache_2,
                stride_v_cache_3,
                BLOCK_SIZE,
                TILE_SIZE,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            # K : (HEAD_SIZE, TILE_SIZE)
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
            )
            # V : (TILE_SIZE, HEAD_SIZE)
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
            )
        K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)
        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)

        # Per-(token, head) scales for INT8 / FP8 per-token-head modes.
        if USE_PER_TOKEN_HEAD_SCALES:
            scale_idx = (
                physical_block_idx * stride_ks_blk
                + (seq_offset % BLOCK_SIZE) * stride_ks_slot
                + kv_head_idx * stride_ks_head
            )
            k_token_head_scales = tl.load(
                k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
            )
            v_scale_idx = (
                physical_block_idx * stride_vs_blk
                + (seq_offset % BLOCK_SIZE) * stride_vs_slot
                + kv_head_idx * stride_vs_head
            )
            v_token_head_scales = tl.load(
                v_scale_cache_ptr + v_scale_idx, mask=tile_mask, other=1.0
            )

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            seq_len,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
            USE_CAUSAL,
            USE_PER_SEQ_CAUSAL,
            per_seq_causal_ptr,
            rswa_prefix_lens_ptr,
            R_SWA_WINDOW,
            USE_R_SWA,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
            MM_PREFIX_CLAMP_SW,
        )

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: fuse softmax_scale with per-head k_scale
            # to avoid a separate BLOCK_M × TILE_SIZE multiply on S.
            S += tl.dot(Q, K) * (score_scale * k_token_head_scales[None, :])
        else:
            S += score_scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            dist = context_len + qpos_lo - seq_offset[:, None]
            if USE_PER_SEQ_CAUSAL:
                is_causal_seq = tl.load(per_seq_causal_ptr + seq_idx)
                sw_mask_v = tl.where(
                    is_causal_seq,
                    dist < SLIDING_WINDOW,
                    (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW),
                )
            elif USE_CAUSAL:
                sw_mask_v = dist < SLIDING_WINDOW
            else:
                sw_mask_v = (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW)
            V = tl.where(sw_mask_v, V, 0.0)
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: apply v_scale to P instead of V.
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # ---- Epilogue ---------------------------------------------------------
    if IS_3D:
        if USE_FP8_Q_DESCALE:
            acc *= value_scale
        # Store per-segment partials; finalized by ``reduce_segments``.
        if USE_TD_QO:
            # 3D target: segm_output[token, head, segm_idx, :].  Advance
            # the base to the correct (token-start, head-start, segm)
            # slice; strides step between tokens / heads of the flattened
            # (T, H, SEGS, PAD) layout.
            segm_base = (
                segm_output_ptr
                + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q).to(
                    tl.int64
                )
                * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + (kv_head_idx * num_queries_per_kv)
                * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + segm_idx * HEAD_SIZE_PADDED
            )
            _store_output_td(
                segm_base,
                acc,
                q_block_local_len,
                num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED,
                NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED,
                num_queries_per_kv,
                BLOCK_Q,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            segm_output_offset = (
                query_offset_0[:, None].to(tl.int64)
                * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + segm_idx * HEAD_SIZE_PADDED
                + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
            )
            tl.store(
                segm_output_ptr + segm_output_offset,
                acc,
                mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc = acc / L[:, None]
        if USE_FP8_Q_DESCALE:
            acc *= value_scale
        if USE_FP8:
            acc = acc * tl.load(out_scale)
            acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
        if USE_TD_QO:
            # 2D target: flat output[token, head, :].  Strides come
            # straight from the caller (``output_stride_0`` per token,
            # ``output_stride_1`` per head).
            output_base = (
                output_ptr
                + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q)
                * output_stride_0
                + (kv_head_idx * num_queries_per_kv) * output_stride_1
            )
            _store_output_td(
                output_base,
                acc,
                q_block_local_len,
                output_stride_0,
                output_stride_1,
                num_queries_per_kv,
                BLOCK_Q,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
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


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    """Detect Gemma3 models via unique (head_size, sliding_window) signature.

    Gemma3 models are the only ones using sliding_window=1024 with
    head_size 128 (27B) or 256 (1B, 4B, 12B). Other SWA models use
    different window sizes (Mistral=4096, Phi-3=2047).
    """
    return sliding_window == 1024 and head_size in (128, 256)


def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization."""
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    return 16 if element_size >= 2 else 32


# TritonAttentionImpl: the 2D kernel needs at least this many programs to fill the GPU; below it
# (fewer sequences than seq_threshold_3D, decode only) the segmented 3D kernel is used.
MIN_LAUNCH_GRID_SIZE_2D = 128
NUM_PAR_SOFTMAX_SEGMENTS = 16


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    softmax_scale,
    window_size,
    block_table,
    seq_threshold_3D,
    num_par_softmax_segments,
    softmax_segm_output,
    softmax_segm_max,
    softmax_segm_expsum,
):
    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    TILE_SIZE_PREFILL = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=True
    )
    TILE_SIZE_DECODE = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=False
    )

    head_size_padded = triton.next_power_of_2(head_size)

    # Launch the 2D kernel if
    # 1. The batch includes at least one prefill request, or
    # 2. The number of sequences exceeds the configured threshold.
    use_3d = not (max_seqlen_q > 1 or num_seqs > seq_threshold_3D)

    # 3D needs real segm tensors; 2D never touches them.  Pass ``None`` in
    # 2D mode so Triton can skip materialising these pointer arguments.
    segm_output_ptr = softmax_segm_output if use_3d else None
    segm_max_ptr = softmax_segm_max if use_3d else None
    segm_expsum_ptr = softmax_segm_expsum if use_3d else None
    num_segments = num_par_softmax_segments if use_3d else 1

    if not use_3d:
        grid = (total_num_q_blocks, num_kv_heads)
        tile_size = TILE_SIZE_PREFILL
    else:
        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        tile_size = TILE_SIZE_DECODE

    kernel_unified_attention[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=None,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=None,
        qq_bias_ptr=None,
        k_scale_cache_ptr=None,
        v_scale_cache_ptr=None,
        scale=softmax_scale,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        out_scale=1.0,
        softcap=0.0,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        USE_ALIBI_SLOPES=False,
        USE_ALIBI_SQRT=False,
        USE_QQ_BIAS=False,
        USE_SOFTCAP=False,
        USE_SINKS=False,
        SLIDING_WINDOW=(1 + window_size[0]),
        USE_CAUSAL=True,
        USE_PER_SEQ_CAUSAL=False,
        per_seq_causal_ptr=None,
        USE_MM_PREFIX=False,
        MAX_MM_RANGES=0,
        mm_prefix_range_ptr=None,
        rswa_prefix_lens_ptr=seqused_k,
        R_SWA_WINDOW=0,
        USE_R_SWA=False,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        stride_ks_blk=None,
        stride_ks_slot=None,
        stride_ks_head=None,
        stride_vs_blk=None,
        stride_vs_slot=None,
        stride_vs_head=None,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        USE_FP8=False,
        IS_3D=use_3d,
        KV_QUANT_MODE=0,
        Q_IS_FP8=False,
        CHUNK_LOOKBACK=-1,
        CHUNK_SIZE=-1,
        USE_TD=False,
        USE_TD_QO=False,
        MM_PREFIX_CLAMP_SW=False,
    )

    if use_3d:
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=False,
        )


def run(*, q, kv_cache, cu_seqlens_q, seqused_k, block_table, out) -> None:
    """Causal paged attention with a 1024-token sliding window, scale head_size ** -0.5; every sequence has the same query length.

    TritonAttentionImpl.forward's calling convention: the packed cache is viewed as separate K and V
    (a transpose-free split of the last dimension), the per-step segment buffers of the 3D path are
    sized for seq_threshold_3D sequences, and max_seqlen_q is the number of query tokens per sequence.
    """
    head_size = q.shape[2]
    key_cache, value_cache = kv_cache.split(head_size, dim=-1)
    num_kv_heads = key_cache.shape[2]
    seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads
    segm_shape = (seq_threshold_3D, q.shape[1], NUM_PAR_SOFTMAX_SEGMENTS)
    unified_attention(
        q,
        key_cache,
        value_cache,
        out,
        cu_seqlens_q,
        q.shape[0] // seqused_k.shape[0],
        seqused_k,
        head_size**-0.5,
        (1023, 0),
        block_table,
        seq_threshold_3D,
        NUM_PAR_SOFTMAX_SEGMENTS,
        torch.empty(
            segm_shape + (triton.next_power_of_2(head_size),),
            dtype=torch.float32,
            device=q.device,
        ),
        torch.empty(segm_shape, dtype=torch.float32, device=q.device),
        torch.empty(segm_shape, dtype=torch.float32, device=q.device),
    )
