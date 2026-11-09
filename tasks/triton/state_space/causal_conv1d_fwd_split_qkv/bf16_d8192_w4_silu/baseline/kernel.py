# Vendored from ROCm/aiter @ 652f2d4b768239aefd515c929e84817681ff598d
# (aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/causal_conv1d_fwd_split_qkv.py).
# The upstream file carries no SPDX header (the ROCm/aiter LICENSE, MIT, is copied to LICENSE); its kernel
# derives from vLLM's causal_conv1d kernel, see README.md for the unresolved provenance.
#
# Only the 2D-tiled kernel is kept, unchanged; the 1D per-token kernel is removed. The launch-map
# kernel and run() at the end of this file replace aiter's host-side wrapper
# (aiter/ops/triton/gated_delta_net/causal_conv1d_prefill.py, causal_conv1d_split_qkv_triton_tile_fn).

"""Prefill causal conv1d (fused split q/k/v) — Triton ``@triton.jit`` kernels.

Two interchangeable kernels: a 1D per-token kernel (serial token loop, any conv
width) and a 2D-tiled kernel (vectorized over feature/token, conv width in
{2, 3, 4}). Host-side launch wrappers live in the public namespace
``aiter.ops.triton.gated_delta_net.causal_conv1d_prefill``.
"""

import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1



# 2D-tiled variant: loads a [feature, token] tile and vectorizes the conv over
# both axes. grid.x decodes a flattened (sequence, chunk) schedule through
# batch_ptr / token_chunk_offset_ptr; grid.y is the feature block.


@triton.jit()
def _causal_conv1d_fwd_split_qkv_tile_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    initial_states_ptr,
    cache_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    block_idx_first_scheduled_token,
    block_idx_last_scheduled_token,
    initial_state_idx,
    num_computed_tokens,
    query_ptr,
    key_ptr,
    value_ptr,
    dim,
    k_dim_size,
    v_dim_size,
    cu_seqlen,
    num_cache_lines,
    stride_x_dim,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_istate_seq,
    stride_istate_dim,
    stride_istate_token,
    stride_cache_indices,
    query_dim_stride,
    query_token_stride: tl.constexpr,
    key_dim_stride,
    key_token_stride: tl.constexpr,
    value_dim_stride,
    value_token_stride: tl.constexpr,
    stride_block_m: tl.constexpr,
    pad_slot_id,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """2D vectorized causal conv1d with conv state update and APC support."""
    k_start_dim = k_dim_size
    v_start_dim = k_dim_size * 2
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1

    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))

    if idx_seq == pad_slot_id:
        return

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index

    B_size: tl.constexpr = stride_block_m * BLOCK_M

    if IS_APC_ENABLED:
        current_first_index = tl.load(block_idx_first_scheduled_token + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
        sequence_completed_index = tl.load(num_computed_tokens + idx_seq)

        sequence_completed_offset_token = sequence_completed_index % B_size
        seq_completed_offset = B_size - sequence_completed_offset_token
        seq_end_offset = (seqlen - seq_completed_offset) % B_size
        last_full_block_token_index = sequence_end_index - seq_end_offset
        if seq_end_offset == 0:
            last_full_block_token_index = last_full_block_token_index - B_size

        n_block_to_fill = current_last_index - current_first_index
        conv_state_init_index = tl.load(initial_state_idx + idx_seq)
    else:
        n_block_to_fill = 0
        current_last_index = 0
        conv_state_init_index = 0
        current_first_index = 0
        last_full_block_token_index = 0

    token_offset = BLOCK_M * chunk_offset
    segment_len = tl.minimum(BLOCK_M, seqlen - token_offset)

    if segment_len <= 0:
        return

    valid_feat = idx_feats < dim

    x_feat_base = (
        x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
    )

    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_cache_indices + conv_state_init_index
    ).to(tl.int64)

    if USE_PAD_SLOT and conv_states_input_coord == pad_slot_id:
        return

    conv_states_base = (
        conv_states_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )

    # Pre-define initial state columns as zeros (SSA: both paths define cols)
    if KERNEL_WIDTH >= 2:
        col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
    if KERNEL_WIDTH >= 3:
        col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
    if KERNEL_WIDTH >= 4:
        col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
    has_prior_state = False

    if chunk_offset == 0:
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            has_prior_state = True
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = valid_feat
            if KERNEL_WIDTH == 4:
                col2 = tl.load(prior_tokens, mask_w, 0.0)
                col1 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
                col0 = tl.load(prior_tokens - 2 * stride_conv_state_tok, mask_w, 0.0)
            elif KERNEL_WIDTH == 3:
                col1 = tl.load(prior_tokens, mask_w, 0.0)
                col0 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
            elif KERNEL_WIDTH == 2:
                col0 = tl.load(prior_tokens, mask_w, 0.0)

        # Update conv_state cache
        if state_len <= seqlen:
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)

            conv_states_output_coord = tl.load(
                conv_state_indices_ptr
                + idx_seq * stride_cache_indices
                + current_last_index
            ).to(tl.int64)

            conv_states_ptrs_target = (
                conv_states_ptr
                + (conv_states_output_coord * stride_conv_state_seq)
                + (idx_feats * stride_conv_state_dim)
            )[None, :] + (idx_tokens_conv * stride_conv_state_tok)[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, loaded_x, mask)

        else:
            if load_init_state:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_states_input_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )
                mask = (
                    (conv_states_input_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen
                x_ptrs = (
                    x_feat_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                tl.debug_barrier()
                new_conv_state = tl.where(mask, conv_state, loaded_x)
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                VAL = state_len - seqlen
                x_ptrs = (
                    x_feat_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # APC: store intermediate conv states at chunk boundaries
        if (chunk_offset - 1) < n_block_to_fill:
            idx_tokens_last = (
                last_full_block_token_index
                - (n_block_to_fill - chunk_offset) * B_size
                - state_len
            ) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (
                x_ptr
                + (idx_tokens_last * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x = (idx_tokens_last >= 0)[:, None] & (idx_feats < dim)[None, :]
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)

            conv_states_output_coord = tl.load(
                conv_state_indices_ptr
                + idx_seq * stride_cache_indices
                + current_first_index
                + (chunk_offset - 1)
            ).to(tl.int64)

            conv_states_ptrs_target = (
                conv_states_ptr
                + (conv_states_output_coord * stride_conv_state_seq)
                + (idx_feats * stride_conv_state_dim)
            )[None, :] + (idx_tokens_conv * stride_conv_state_tok)[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, loaded_x, mask)

    idx_tokens = tl.arange(0, BLOCK_M)
    token_global = token_offset + idx_tokens
    valid_token = idx_tokens < segment_len
    valid_2d = valid_feat[:, None] & valid_token[None, :]

    x_t = tl.load(
        x_feat_base[:, None] + token_global[None, :] * stride_x_token,
        mask=valid_2d,
        other=0.0,
    ).to(tl.float32)

    if KERNEL_WIDTH >= 2:
        x_s1 = tl.load(
            x_feat_base[:, None] + (token_global[None, :] - 1) * stride_x_token,
            mask=valid_2d & (token_global[None, :] >= 1),
            other=0.0,
        ).to(tl.float32)
    if KERNEL_WIDTH >= 3:
        x_s2 = tl.load(
            x_feat_base[:, None] + (token_global[None, :] - 2) * stride_x_token,
            mask=valid_2d & (token_global[None, :] >= 2),
            other=0.0,
        ).to(tl.float32)
    if KERNEL_WIDTH >= 4:
        x_s3 = tl.load(
            x_feat_base[:, None] + (token_global[None, :] - 3) * stride_x_token,
            mask=valid_2d & (token_global[None, :] >= 3),
            other=0.0,
        ).to(tl.float32)

    # Blend initial state into boundary tokens
    if has_prior_state:
        if KERNEL_WIDTH == 2:
            x_s1 = tl.where(
                (token_global[None, :] == 0) & valid_feat[:, None],
                col0.to(tl.float32)[:, None],
                x_s1,
            )
        if KERNEL_WIDTH == 3:
            x_s1 = tl.where(
                (token_global[None, :] == 0) & valid_feat[:, None],
                col1.to(tl.float32)[:, None],
                x_s1,
            )
            x_s2 = tl.where(
                (token_global[None, :] == 0) & valid_feat[:, None],
                col0.to(tl.float32)[:, None],
                x_s2,
            )
            x_s2 = tl.where(
                (token_global[None, :] == 1) & valid_feat[:, None],
                col1.to(tl.float32)[:, None],
                x_s2,
            )
        if KERNEL_WIDTH == 4:
            x_s1 = tl.where(
                (token_global[None, :] == 0) & valid_feat[:, None],
                col2.to(tl.float32)[:, None],
                x_s1,
            )
            x_s2 = tl.where(
                (token_global[None, :] == 0) & valid_feat[:, None],
                col1.to(tl.float32)[:, None],
                x_s2,
            )
            x_s2 = tl.where(
                (token_global[None, :] == 1) & valid_feat[:, None],
                col2.to(tl.float32)[:, None],
                x_s2,
            )
            x_s3 = tl.where(
                (token_global[None, :] == 0) & valid_feat[:, None],
                col0.to(tl.float32)[:, None],
                x_s3,
            )
            x_s3 = tl.where(
                (token_global[None, :] == 1) & valid_feat[:, None],
                col1.to(tl.float32)[:, None],
                x_s3,
            )
            x_s3 = tl.where(
                (token_global[None, :] == 2) & valid_feat[:, None],
                col2.to(tl.float32)[:, None],
                x_s3,
            )

    w_base = w_ptr + idx_feats * stride_w_dim
    mask_w = valid_feat
    if KERNEL_WIDTH == 4:
        w0 = tl.load(w_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
        w1 = tl.load(w_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
        w2 = tl.load(w_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
        w3 = tl.load(w_base + 3 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    elif KERNEL_WIDTH == 3:
        w0 = tl.load(w_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
        w1 = tl.load(w_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
        w2 = tl.load(w_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    elif KERNEL_WIDTH == 2:
        w0 = tl.load(w_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
        w1 = tl.load(w_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)

    if HAS_BIAS:
        acc = tl.load(bias_ptr + idx_feats, mask=valid_feat, other=0.0).to(tl.float32)
        acc = acc[:, None] + tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    if KERNEL_WIDTH == 4:
        acc += (
            w0[:, None] * x_s3
            + w1[:, None] * x_s2
            + w2[:, None] * x_s1
            + w3[:, None] * x_t
        )
    elif KERNEL_WIDTH == 3:
        acc += w0[:, None] * x_s2 + w1[:, None] * x_s1 + w2[:, None] * x_t
    elif KERNEL_WIDTH == 2:
        acc += w0[:, None] * x_s1 + w1[:, None] * x_t

    if SILU_ACTIVATION:
        exp_neg = tl.math.exp2(acc * (-1.4426950408889634))
        rcp = tl.inline_asm_elementwise(
            "v_rcp_f32 $0, $1",
            "=v,v",
            args=[1.0 + exp_neg],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        acc = acc * rcp

    token_pos = sequence_start_index + token_global
    valid_token_2d = valid_token[None, :]

    q_feat_idx = idx_feats
    is_q_block = idx_feats < k_start_dim
    q_ptrs = (
        query_ptr
        + token_pos[None, :] * query_token_stride
        + q_feat_idx[:, None] * query_dim_stride
    )
    tl.store(q_ptrs, acc, mask=valid_token_2d & is_q_block[:, None])

    k_feat_idx = idx_feats - k_start_dim
    is_k_block = (idx_feats >= k_start_dim) & (idx_feats < v_start_dim)
    k_ptrs = (
        key_ptr
        + token_pos[None, :] * key_token_stride
        + k_feat_idx[:, None] * key_dim_stride
    )
    tl.store(k_ptrs, acc, mask=valid_token_2d & is_k_block[:, None])

    v_feat_idx = idx_feats - v_start_dim
    is_v_block = (idx_feats >= v_start_dim) & (idx_feats < dim)
    v_ptrs = (
        value_ptr
        + token_pos[None, :] * value_token_stride
        + v_feat_idx[:, None] * value_dim_stride
    )
    tl.store(v_ptrs, acc, mask=valid_token_2d & is_v_block[:, None])


BLOCK_M = 64
BLOCK_N = 32


@triton.jit
def _build_launch_map_kernel(
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    n_seqs,
    max_programs,
    BLOCK_M: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_FILL: tl.constexpr,
):
    # aiter builds this (sequence, chunk) map on the host from query_start_loc (a CPU round trip, or
    # prebuilt metadata shared by the layers of a batch). Here every program n < n_seqs writes the
    # chunks of sequence n; the last program pads the rest of the fixed-size map with -1 so the main
    # kernel's surplus programs exit through the pad id.
    pid = tl.program_id(0)
    seqs = tl.arange(0, BLOCK_SEQ)
    valid = seqs < n_seqs
    starts = tl.load(query_start_loc_ptr + seqs, mask=valid, other=0)
    ends = tl.load(query_start_loc_ptr + seqs + 1, mask=valid, other=0)
    chunks = (ends - starts + BLOCK_M - 1) // BLOCK_M
    if pid < n_seqs:
        first = tl.sum(tl.where(seqs < pid, chunks, 0), axis=0)
        count = tl.sum(tl.where(seqs == pid, chunks, 0), axis=0)
        for c0 in range(0, count, BLOCK_FILL):
            c = c0 + tl.arange(0, BLOCK_FILL)
            m = c < count
            tl.store(batch_ptr + first + c, tl.zeros((BLOCK_FILL,), dtype=tl.int32) + pid, mask=m)
            tl.store(token_chunk_offset_ptr + first + c, c, mask=m)
    else:
        total = tl.sum(chunks, axis=0)
        for p0 in range(total, max_programs, BLOCK_FILL):
            p = p0 + tl.arange(0, BLOCK_FILL)
            m = p < max_programs
            pad = tl.zeros((BLOCK_FILL,), dtype=tl.int32) - 1
            tl.store(batch_ptr + p, pad, mask=m)
            tl.store(token_chunk_offset_ptr + p, pad, mask=m)


def run(*, x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state, q, k, v) -> None:
    """Varlen causal depthwise conv1d + bias + SiLU with conv-state update and q | k | v split.

    x: (n_tokens, dim) with channels packed [Q | K | V]; weight: (dim, width); conv_states:
    (cache_lines, width - 1, dim), updated in place; q, k: (n_tokens, key_dim); v: (n_tokens, value_dim).
    The kernel addresses channel-last (dim, n_tokens) views.
    """
    n_tokens, dim = x.shape
    key_dim = q.shape[1]
    value_dim = v.shape[1]
    n_seqs = cache_indices.shape[0]
    width = weight.shape[1]
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    x_t = x.t()
    states = conv_states.transpose(1, 2)  # (cache_lines, dim, width - 1), dim-contiguous

    # A sequence of L tokens needs ceil(L / BLOCK_M) chunk programs, so
    # sum(ceil(L_i / BLOCK_M)) <= ceil(n_tokens / BLOCK_M) + n_seqs.
    max_programs = triton.cdiv(n_tokens, BLOCK_M) + n_seqs
    batch_ptr = torch.empty(max_programs, dtype=torch.int32, device=x.device)
    token_chunk_offset_ptr = torch.empty(max_programs, dtype=torch.int32, device=x.device)
    _build_launch_map_kernel[(n_seqs + 1,)](
        query_start_loc, batch_ptr, token_chunk_offset_ptr, n_seqs, max_programs,
        BLOCK_M=BLOCK_M, BLOCK_SEQ=triton.next_power_of_2(n_seqs), BLOCK_FILL=128,
    )

    _causal_conv1d_fwd_split_qkv_tile_kernel[(max_programs, triton.cdiv(dim, BLOCK_N))](
        x_t,
        weight,
        bias,
        states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        batch_ptr,
        token_chunk_offset_ptr,
        None,  # block_idx_first_scheduled_token (APC disabled)
        None,  # block_idx_last_scheduled_token  (APC disabled)
        None,  # initial_state_idx               (APC disabled)
        None,  # num_computed_tokens             (APC disabled)
        q,
        k,
        v,
        dim,
        key_dim,
        value_dim,
        n_tokens,
        states.shape[0],
        x_t.stride(0),
        x_t.stride(1),
        weight.stride(0),
        weight.stride(1),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        cache_indices.stride(0),
        q.stride(1),
        q.stride(0),
        k.stride(1),
        k.stride(0),
        v.stride(1),
        v.stride(0),
        0,  # stride_block_m (APC only)
        PAD_SLOT_ID,
        HAS_BIAS=True,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=True,
        IS_APC_ENABLED=False,
        USE_PAD_SLOT=True,
        NP2_STATELEN=np2_statelen,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=1,
    )
