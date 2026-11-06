# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
#
# Vendored from vllm/model_executor/layers/mamba/ops/causal_conv1d.py (vLLM, revision
# edbc4969a76b25c531f7b9bb16b79984e65ef023). The kernel body is unchanged except that the
# CUDA-only programmatic-dependent-launch hooks (launch_pdl) are removed. The launch-map
# kernel and run() at the end of this file replace vLLM's host-side wrapper.

import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1
NULL_BLOCK_ID = 0
BLOCK_M = 8
BLOCK_N = 256




@triton.jit(do_not_specialize_on_alignment=["num_cache_lines"])
def _causal_conv1d_fwd_kernel(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    cache_indices_ptr,  # (batch, n_blocks + padding) The second dimension contains
    # the block indices relevant for each sequence
    # plus potential 0-padding at the beginning and at the end
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    block_idx_first_scheduled_token,  # (batch,)
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    num_computed_tokens,  # (batch,)
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
    # Matrix dimensions
    dim: tl.constexpr,
    num_cache_lines,  # added to support vLLM larger cache lines
    # Strides
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.int64,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_cache_indices: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.int64,
    stride_block_m: tl.constexpr,  # Stride block to align divided by BLOCK_M
    # others
    pad_slot_id: tl.constexpr,
    null_block_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    HAS_NULL_BLOCK: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = (
        KERNEL_WIDTH - 1
    )  # can be passed via argument if it's not the same as this value


    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    # single-sequence id
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    B_size: tl.constexpr = stride_block_m * BLOCK_M

    if IS_APC_ENABLED:
        # Handle the case if prefix caching is enabled.
        # In particular, if prefix caching is enabled, the program write additional cache states to "cache_indices_ptr"

        # Get the length of the completed sequence so far and compute the offset.
        current_first_index = tl.load(block_idx_first_scheduled_token + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
        sequence_completed_index = tl.load(num_computed_tokens + idx_seq)

        # Compute the offset where the first stride_block_m-aligned first full block is
        # Value in "token-space"
        sequence_completed_offset_token = sequence_completed_index % B_size
        seq_completed_offset = B_size - sequence_completed_offset_token
        seq_end_offset = (seqlen - seq_completed_offset) % B_size
        last_full_block_token_index = sequence_end_index - seq_end_offset
        # If the sequence without the sequence_offset_index is stride_cache_chunk-aligned, then the last full chunk is the second-to-last one
        if seq_end_offset == 0:
            last_full_block_token_index = last_full_block_token_index - B_size

        # Get the number of blocks to be filled for the current sequence
        # If n_block_to_fill = 0, then only the state at the sequence end is stored
        n_block_to_fill = current_last_index - current_first_index

        # Get the index of the init block
        conv_state_init_index = tl.load(initial_state_idx + idx_seq)
    else:
        n_block_to_fill = 0
        current_last_index = 0
        conv_state_init_index = 0
        current_first_index = 0
        last_full_block_token_index = 0

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    # base of the sequence
    x_base = (
        x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
    )  # [BLOCK_N,]

    # cache_idx
    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_cache_indices + conv_state_init_index
    ).to(tl.int64)

    if HAS_NULL_BLOCK:  # noqa
        if conv_states_input_coord == null_block_id:
            # not processing as this is a null block (padding)
            return
    conv_states_base = (
        conv_states_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            # load from conv_states
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            # prior-tokens are zeros
            if KERNEL_WIDTH >= 2:  # STRATEGY1
                # first chunk and does not have prior-token, so just set to 0
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:  # STRATEGY1
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:  # STRATEGY1
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:  # STRATEGY1
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # STEP 2:
        # here prepare data for updating conv_state
        if (
            state_len <= seqlen
        ):  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
            # just read from 'x'
            # copy 'x' data to conv_state
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
            idx_tokens_last = (seqlen - state_len) + tl.arange(
                0, NP2_STATELEN
            )  # [BLOCK_M]
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )  # [BLOCK_M,BLOCK_N,]
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

            # Compute the offset where the last block should be written in the conv_states
            conv_states_output_coord = tl.load(
                conv_state_indices_ptr
                + idx_seq * stride_cache_indices
                + current_last_index
            ).to(tl.int64)

            conv_states_ptrs_target = (
                conv_states_ptr
                + (conv_states_output_coord * stride_conv_state_seq)  # Offset from seq
                + (idx_feats * stride_conv_state_dim)
            )[None, :] + (  # [BLOCK_N,]
                idx_tokens_conv * stride_conv_state_tok
            )[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, loaded_x, mask)

        else:
            if load_init_state:
                # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_states_input_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (
                    (conv_states_input_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                tl.debug_barrier()  # need this due to the bug in tl.where not enforcing this when data is the result of another tl.load
                new_conv_state = tl.where(
                    mask, conv_state, loaded_x
                )  # BUG in 'tl.where'  which requires a barrier before this
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:  # load_init_state == False
                # update conv_state by shifting left, BUT
                # set cols prior to 'x' as zeros + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # read prior-token data from `x`
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 5:
            # ruff: noqa: F841
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

        # Store intermediate states aligned with stride_block_m
        # The additional states are cached starting from the last stride_block_m.
        # For example:
        # If n_block_to_fill = 0, then only the state at the sequence end is cached and the process below is not involved.
        # If n_block_to_fill > 0, then the states at the sequence end and at the n_block_to_fill-last
        # stride_block_m are cached.
        # For example chunk_offset = n_block_to_fill stores the state at last_full_block
        if (chunk_offset - 1) < n_block_to_fill:
            # Store the states at the chunk boundaries from the start of the sequence
            idx_tokens_last = (
                last_full_block_token_index
                - (n_block_to_fill - chunk_offset) * B_size
                - state_len
            ) + tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            x_ptrs = (
                x_ptr
                + (idx_tokens_last * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )  # [BLOCK_M,BLOCK_N,]

            mask_x = (idx_tokens_last >= 0)[:, None] & (idx_feats < dim)[
                None, :
            ]  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

            # cache_idx
            conv_states_output_coord = tl.load(
                conv_state_indices_ptr
                + idx_seq * stride_cache_indices
                + current_first_index
                + (chunk_offset - 1)
            ).to(tl.int64)

            conv_states_ptrs_target = (
                conv_states_ptr
                + (conv_states_output_coord * stride_conv_state_seq)  # Offset from seq
                + (idx_feats * stride_conv_state_dim)
            )[None, :] + (  # [BLOCK_N,]
                idx_tokens_conv * stride_conv_state_tok
            )[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, loaded_x, mask)

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim


    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (sequence_start_index + token_offset + idx_token) * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)


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
    # vLLM builds this (sequence, chunk) map on the host from query_start_loc. Here every
    # program n < n_seqs writes the chunks of sequence n; the last program pads the rest
    # of the fixed-size map with PAD_SLOT_ID so the main kernel's surplus programs exit.
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


def run(*, x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state, out) -> None:
    """Varlen causal depthwise conv1d + bias + SiLU with conv-state update.

    x, out: (n_tokens, dim); weight: (dim, width); conv_states: (cache_lines, width - 1, dim),
    updated in place. The kernel addresses channel-last (dim, n_tokens) views.
    """
    n_tokens, dim = x.shape
    n_seqs = cache_indices.shape[0]
    width = weight.shape[1]
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    x_t = x.t()
    out_t = out.t()
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

    _causal_conv1d_fwd_kernel[(max_programs, triton.cdiv(dim, BLOCK_N))](
        x_t,
        weight,
        bias,
        states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        batch_ptr,
        token_chunk_offset_ptr,
        None,
        None,
        None,
        None,
        out_t,
        dim,
        states.shape[0],
        x_t.stride(0),
        x_t.stride(1),
        weight.stride(0),
        weight.stride(1),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        cache_indices.stride(0),
        out_t.stride(0),
        out_t.stride(1),
        1,
        PAD_SLOT_ID,
        NULL_BLOCK_ID,
        HAS_BIAS=True,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=True,
        IS_APC_ENABLED=False,
        HAS_NULL_BLOCK=False,
        NP2_STATELEN=np2_statelen,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=2,
    )
