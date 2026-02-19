# Triton causal conv1d update kernel
# Adapted from vLLM mamba ops

import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1


@triton.jit()
def _causal_conv1d_update_kernel(
    x_ptr, w_ptr, bias_ptr, conv_state_ptr, conv_state_indices_ptr,
    num_accepted_tokens_ptr, query_start_loc_ptr,
    block_idx_last_scheduled_token, initial_state_idx,
    o_ptr,
    batch: int, dim: tl.constexpr, seqlen: tl.constexpr,
    state_len: tl.constexpr, num_cache_lines: tl.constexpr,
    stride_x_seq: tl.constexpr, stride_x_dim: tl.constexpr, stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr, stride_conv_state_dim: tl.constexpr, stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr, stride_o_dim: tl.constexpr, stride_o_token: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr, KERNEL_WIDTH: tl.constexpr, SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr, IS_APC_ENABLED: tl.constexpr, IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr, USE_PAD_SLOT: tl.constexpr, BLOCK_N: tl.constexpr,
):
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_APC_ENABLED:
        conv_state_init = tl.load(initial_state_idx + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
    else:
        conv_state_init = 0
        current_last_index = 0

    conv_states_input_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init).to(tl.int64)

    if USE_PAD_SLOT:
        if conv_states_input_coord == pad_slot_id:
            return

    if IS_VARLEN:
        query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
        query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64)
        state_len = state_len - (seqlen - (query_end_index - query_start_index))
        seqlen = query_end_index - query_start_index
        x_offset = query_start_index * stride_x_token
        o_offset = query_start_index * stride_o_token
    else:
        query_start_index = idx_seq * seqlen
        query_end_index = query_start_index + seqlen
        x_offset = idx_seq * stride_x_seq
        o_offset = idx_seq * stride_o_seq

    if query_start_index == query_end_index:
        return

    if IS_SPEC_DECODING:
        conv_state_token_offset = tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1
    else:
        conv_state_token_offset = 0

    conv_states_base = conv_state_ptr + (conv_states_input_coord * stride_conv_state_seq) + (idx_feats * stride_conv_state_dim)
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        col0 = tl.load(prior_tokens, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        col1 = tl.load(prior_tokens + 1 * stride_conv_state_tok, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        col2 = tl.load(prior_tokens + 2 * stride_conv_state_tok, mask_w, 0.0)
    if KERNEL_WIDTH >= 5:
        col3 = tl.load(prior_tokens + 3 * stride_conv_state_tok, mask_w, 0.0)
    if KERNEL_WIDTH >= 6:
        col4 = tl.load(prior_tokens + 4 * stride_conv_state_tok, mask_w, 0.0)

    # Update conv state
    idx_tokens = tl.arange(0, NP2_STATELEN)
    conv_state_ptrs_source = (
        conv_state_ptr + (conv_states_input_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[:, None]
    )
    mask = (conv_states_input_coord < num_cache_lines) & ((idx_tokens + seqlen) < state_len)[:, None] & (idx_feats < dim)[None, :]
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)
    x_ptrs = x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    mask_x = (idx_tokens - VAL >= 0)[:, None] & (idx_tokens - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :]
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()
    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_states_offset = tl.load(conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index).to(tl.int64)
    conv_state_ptrs_target = (
        conv_state_ptr + (conv_states_offset * stride_conv_state_seq) + (idx_feats * stride_conv_state_dim)
    )[None, :] + (idx_tokens * stride_conv_state_tok)[:, None]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    w_base = w_ptr + (idx_feats * stride_w_dim)
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_col0 = tl.load(w_base + (0 * stride_w_width), mask_w, other=0.0)
        w_col1 = tl.load(w_base + (1 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + (2 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + (3 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 5:
        w_col4 = tl.load(w_base + (4 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 6:
        w_col5 = tl.load(w_base + (5 * stride_w_width), mask_w, other=0.0)

    x_base_1d = x_base
    mask_x_1d = idx_feats < dim

    for idx_token in tl.range(seqlen):
        acc = acc_preload
        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
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
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 5:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 6:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = col4
                elif j == 5:
                    matrix_w = w_col5
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            acc += matrix_x * matrix_w

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        elif KERNEL_WIDTH == 5:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = matrix_x
        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (idx_feats < dim)
        o_ptrs = o_ptr + o_offset + idx_token * stride_o_token + (idx_feats * stride_o_dim)
        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None,
                         conv_state_indices=None, pad_slot_id=PAD_SLOT_ID):
    """
    x: (batch, dim, seqlen)
    conv_state: (num_cache_lines, dim, state_len)
    weight: (dim, width)
    """
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    batch, dim, seqlen = x.shape
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()

    out = x  # in-place
    stride_w_dim, stride_w_width = weight.stride()
    stride_x_seq, stride_x_dim, stride_x_token = x.stride()
    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0) if conv_state_indices is not None else 0
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (batch, triton.cdiv(dim, META["BLOCK_N"]))

    _causal_conv1d_update_kernel[grid](
        x, weight, bias, conv_state, conv_state_indices,
        None, None, None, None, out,
        batch, dim, seqlen, state_len, num_cache_lines,
        stride_x_seq, stride_x_dim, stride_x_token,
        stride_w_dim, stride_w_width,
        stride_istate_seq, stride_istate_dim, stride_istate_token,
        stride_state_indices,
        stride_o_seq, stride_o_dim, stride_o_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_VARLEN=False,
        IS_APC_ENABLED=False,
        IS_SPEC_DECODING=False,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=256,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_x_dtype)
