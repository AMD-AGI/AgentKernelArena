# Triton causal conv1d forward kernel
# Adapted from vLLM mamba ops

import numpy as np
import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1


@triton.jit()
def _causal_conv1d_fwd_kernel(
    x_ptr, w_ptr, bias_ptr, initial_states_ptr, cache_indices_ptr,
    has_initial_states_ptr, query_start_loc_ptr, batch_ptr, token_chunk_offset_ptr,
    block_idx_first_scheduled_token, block_idx_last_scheduled_token,
    initial_state_idx, num_computed_tokens, o_ptr,
    dim: tl.constexpr, seqlen: tl.int32, num_cache_lines: tl.constexpr,
    stride_x_dim: tl.constexpr, stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr,
    stride_istate_seq: tl.constexpr, stride_istate_dim: tl.constexpr, stride_istate_token: tl.constexpr,
    stride_cache_indices: tl.constexpr,
    stride_o_dim: tl.constexpr, stride_o_token: tl.constexpr,
    stride_block_m: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr, KERNEL_WIDTH: tl.constexpr, SILU_ACTIVATION: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr, USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1

    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

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
    segment_len = min(BLOCK_M, seqlen - token_offset)

    x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim

    conv_states_input_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices + conv_state_init_index).to(tl.int64)

    if USE_PAD_SLOT:
        if conv_states_input_coord == pad_slot_id:
            return
    conv_states_base = conv_states_ptr + (conv_states_input_coord * stride_conv_state_seq) + (idx_feats * stride_conv_state_dim)

    w_base = w_ptr + (idx_feats * stride_w_dim)

    if chunk_offset == 0:
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                col0 = tl.load(prior_tokens, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                col1 = tl.load(prior_tokens, mask_w, 0.0)
                col0 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                col2 = tl.load(prior_tokens, mask_w, 0.0)
                col1 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
                col0 = tl.load(prior_tokens - 2 * stride_conv_state_tok, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                col3 = tl.load(prior_tokens, mask_w, 0.0)
                col2 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
                col1 = tl.load(prior_tokens - 2 * stride_conv_state_tok, mask_w, 0.0)
                col0 = tl.load(prior_tokens - 3 * stride_conv_state_tok, mask_w, 0.0)
        else:
            if KERNEL_WIDTH >= 2:
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        if state_len <= seqlen:
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = x_ptr + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None] + (idx_feats * stride_x_dim)[None, :]
            mask_x = (idx_tokens_last >= 0)[:, None] & (idx_tokens_last < seqlen)[:, None] & (idx_feats < dim)[None, :]
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            conv_states_output_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices + current_last_index).to(tl.int64)
            conv_states_ptrs_target = (conv_states_ptr + (conv_states_output_coord * stride_conv_state_seq) + (idx_feats * stride_conv_state_dim))[None, :] + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, loaded_x, mask)
        else:
            if load_init_state:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                conv_states_ptrs_source = conv_states_ptr + (conv_states_input_coord * stride_conv_state_seq) + (idx_feats * stride_conv_state_dim)[None, :] + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                mask = (conv_states_input_coord < num_cache_lines) & ((idx_tokens_conv + seqlen) < state_len)[:, None] & (idx_feats < dim)[None, :]
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)
                VAL = state_len - seqlen
                x_ptrs = x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                mask_x = (idx_tokens_conv - VAL >= 0)[:, None] & (idx_tokens_conv - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :]
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                tl.debug_barrier()
                new_conv_state = tl.where(mask, conv_state, loaded_x)
                conv_states_ptrs_target = conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                VAL = state_len - seqlen
                x_ptrs = x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                mask_x = (idx_tokens_conv - VAL >= 0)[:, None] & (idx_tokens_conv - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :]
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
                conv_states_ptrs_target = conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
    else:
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            col0 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            col1 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
            col0 = tl.load(prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            col2 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
            col1 = tl.load(prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")
            col0 = tl.load(prior_tokens - 2 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 5:
            col3 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
            col2 = tl.load(prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")
            col1 = tl.load(prior_tokens - 2 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")
            col0 = tl.load(prior_tokens - 3 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")

        if IS_APC_ENABLED:
            if (chunk_offset - 1) < n_block_to_fill:
                idx_tokens_last = (last_full_block_token_index - (n_block_to_fill - chunk_offset) * B_size - state_len) + tl.arange(0, NP2_STATELEN)
                x_ptrs = x_ptr + (idx_tokens_last * stride_x_token)[:, None] + (idx_feats * stride_x_dim)[None, :]
                mask_x = (idx_tokens_last >= 0)[:, None] & (idx_feats < dim)[None, :]
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                conv_states_output_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices + current_first_index + (chunk_offset - 1)).to(tl.int64)
                conv_states_ptrs_target = (conv_states_ptr + (conv_states_output_coord * stride_conv_state_seq) + (idx_feats * stride_conv_state_dim))[None, :] + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.debug_barrier()
                tl.store(conv_states_ptrs_target, loaded_x, mask)

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token

    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_col0 = tl.load(w_base + (0 * stride_w_width), mask_w, other=0.0)
        w_col1 = tl.load(w_base + (1 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + (2 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + (3 * stride_w_width), mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
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

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (idx_feats < dim)
        o_ptrs = o_ptr + (sequence_start_index + token_offset + idx_token) * stride_o_token + (idx_feats * stride_o_dim)
        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_fwd(x, weight, bias, conv_states, query_start_loc, cache_indices,
                      has_initial_state, activation="silu", pad_slot_id=PAD_SLOT_ID,
                      block_size_to_align=0):
    """
    x: (dim, cu_seq_len) channel-last
    weight: (dim, width)
    conv_states: (num_cache_lines, dim, width-1)
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    original_x_dtype = x.dtype
    x = x.to(conv_states.dtype)
    out = torch.empty_like(x)

    seqlens = query_start_loc.diff().to("cpu")
    MAX_NUM_PROGRAMS = 1024
    batch_ptr = torch.full((MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=x.device)
    token_chunk_offset_ptr = torch.full((MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=x.device)

    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    BLOCK_M = 8
    num_cache_lines = conv_states.size(0)
    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = conv_states.stride(0)
    stride_istate_dim = conv_states.stride(1)
    stride_istate_token = conv_states.stride(2)
    stride_o_dim = out.stride(0)
    stride_o_token = out.stride(1)
    stride_cache_indices = cache_indices.stride(0) if cache_indices is not None else 0

    def num_program(META, seqlens):
        nums = -(-seqlens // META["BLOCK_M"])
        tot = nums.sum().item()
        mlist = np.repeat(np.arange(len(nums)), nums)
        offsetlist = []
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num))
        if META["batch_ptr"].nelement() < len(mlist):
            newlen = len(mlist) + 1
            META["batch_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)
            META["token_chunk_offset_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)
        if META["batch_ptr"].nelement() >= len(mlist):
            META["batch_ptr"][0:len(mlist)].copy_(torch.from_numpy(np.array(mlist)))
            META["token_chunk_offset_ptr"][0:len(mlist)].copy_(torch.from_numpy(np.array(offsetlist)))
        META["batch_ptr"] = META["batch_ptr"].to(META["x_ptr"].device)
        META["token_chunk_offset_ptr"] = META["token_chunk_offset_ptr"].to(META["x_ptr"].device)
        return tot

    def grid(META):
        return (num_program(META, seqlens), triton.cdiv(dim, META["BLOCK_N"]))

    if batch_ptr.device != x.device:
        batch_ptr = batch_ptr.to(x.device)
        token_chunk_offset_ptr = token_chunk_offset_ptr.to(x.device)

    _causal_conv1d_fwd_kernel[grid](
        x, weight, bias, conv_states, cache_indices,
        has_initial_state, query_start_loc, batch_ptr, token_chunk_offset_ptr,
        None, None, None, None, out,
        dim, cu_seqlen, num_cache_lines,
        stride_x_dim, stride_x_token, stride_w_dim, stride_w_width,
        stride_istate_seq, stride_istate_dim, stride_istate_token,
        stride_cache_indices, stride_o_dim, stride_o_token,
        (block_size_to_align // BLOCK_M) if block_size_to_align > 0 else 1,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_APC_ENABLED=False,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        BLOCK_M=BLOCK_M,
        BLOCK_N=256,
        num_stages=2,
    )
    return out.to(original_x_dtype)
