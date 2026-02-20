"""Copy and expand EAGLE inputs kernel from vLLM speculative decoding.

Copies target model inputs to drafting buffers, handling bonus tokens, parallel
drafting slots, and rejected token regions for EAGLE speculative decoding.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def copy_and_expand_eagle_inputs_kernel(
    target_token_ids_ptr,
    target_positions_ptr,
    next_token_ids_ptr,
    out_input_ids_ptr,
    out_positions_ptr,
    out_is_rejected_token_mask_ptr,
    out_is_masked_token_mask_ptr,
    out_new_token_indices_ptr,
    out_hidden_state_mapping_ptr,
    query_start_loc_ptr,
    query_end_loc_ptr,
    padding_token_id,
    parallel_drafting_token_id,
    total_input_tokens,
    num_padding_slots_per_request,
    shift_input_ids,
    BLOCK_SIZE_TOKENS: tl.constexpr,
):
    request_idx = tl.program_id(axis=0)
    token_batch_idx = tl.program_id(axis=1)

    query_start_loc = tl.load(query_start_loc_ptr + request_idx)
    next_query_start_loc = tl.load(query_start_loc_ptr + request_idx + 1)
    query_end_loc = tl.load(query_end_loc_ptr + request_idx)

    if shift_input_ids:
        num_valid_tokens = query_end_loc - query_start_loc
        input_offset = 1
        output_start = query_start_loc + request_idx * (
            num_padding_slots_per_request - 1
        )
    else:
        num_valid_tokens = query_end_loc - query_start_loc + 1
        input_offset = 0
        output_start = query_start_loc + request_idx * num_padding_slots_per_request

    num_rejected = next_query_start_loc - query_end_loc - 1
    total_output_tokens = (
        num_valid_tokens + num_padding_slots_per_request + num_rejected
    )

    j = token_batch_idx * BLOCK_SIZE_TOKENS + tl.arange(0, BLOCK_SIZE_TOKENS)

    in_bounds = j < total_output_tokens
    is_valid_region = j < num_valid_tokens
    is_bonus_region = j == num_valid_tokens
    is_parallel_draft_region = (j > num_valid_tokens) & (
        j < num_valid_tokens + num_padding_slots_per_request
    )
    is_rejected_region = j >= num_valid_tokens + num_padding_slots_per_request

    out_idx = output_start + j

    in_idx = query_start_loc + input_offset + j
    in_idx_clamped = tl.minimum(in_idx, total_input_tokens - 1)

    token_ids = tl.load(
        target_token_ids_ptr + in_idx_clamped, mask=is_valid_region & in_bounds, other=0
    )

    start_pos = tl.load(target_positions_ptr + query_start_loc)
    bonus_token = tl.load(next_token_ids_ptr + request_idx)

    token_ids = tl.where(is_bonus_region, bonus_token, token_ids)
    token_ids = tl.where(
        is_parallel_draft_region, parallel_drafting_token_id, token_ids
    )
    token_ids = tl.where(is_rejected_region, padding_token_id, token_ids)

    positions = start_pos + j
    positions = tl.where(is_rejected_region, 0, positions)

    is_rejected_out = is_rejected_region & in_bounds
    is_masked_out = is_parallel_draft_region & in_bounds

    is_new_token_region = (j >= num_valid_tokens) & (
        j < num_valid_tokens + num_padding_slots_per_request
    )
    new_token_local_idx = j - num_valid_tokens
    new_token_out_idx = (
        request_idx * num_padding_slots_per_request + new_token_local_idx
    )

    if shift_input_ids:
        num_input_tokens_this_request = next_query_start_loc - query_start_loc
        is_input_region = j < num_input_tokens_this_request
        src_idx = query_start_loc + j
        tl.store(out_hidden_state_mapping_ptr + src_idx, out_idx, mask=is_input_region)

    tl.store(out_input_ids_ptr + out_idx, token_ids, mask=in_bounds)
    tl.store(out_positions_ptr + out_idx, positions, mask=in_bounds)
    tl.store(out_is_rejected_token_mask_ptr + out_idx, is_rejected_out, mask=in_bounds)
    tl.store(out_is_masked_token_mask_ptr + out_idx, is_masked_out, mask=in_bounds)
    tl.store(
        out_new_token_indices_ptr + new_token_out_idx,
        out_idx,
        mask=is_new_token_region & in_bounds,
    )


def copy_and_expand_eagle_inputs(
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    next_token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_end_loc: torch.Tensor,
    padding_token_id: int,
    parallel_drafting_token_id: int,
    num_padding_slots_per_request: int,
    shift_input_ids: bool,
    max_output_tokens_per_req: int,
) -> tuple:
    """
    Args:
        target_token_ids: [total_tokens_in_batch]
        target_positions: [total_tokens_in_batch]
        next_token_ids: [num_reqs]
        query_start_loc: [num_reqs + 1]
        query_end_loc: [num_reqs]
        padding_token_id: int
        parallel_drafting_token_id: int
        num_padding_slots_per_request: int
        shift_input_ids: bool
        max_output_tokens_per_req: int (upper bound for grid sizing)
    Returns:
        out_input_ids, out_positions, out_is_rejected, out_is_masked,
        out_new_token_indices, out_hidden_state_mapping
    """
    num_reqs = next_token_ids.shape[0]
    total_input = target_token_ids.shape[0]
    # Overallocate output buffers
    total_out = total_input + num_reqs * (num_padding_slots_per_request + 10)

    device = target_token_ids.device
    out_input_ids = torch.zeros(total_out, dtype=torch.int32, device=device)
    out_positions = torch.zeros(total_out, dtype=torch.int32, device=device)
    out_is_rejected = torch.zeros(total_out, dtype=torch.bool, device=device)
    out_is_masked = torch.zeros(total_out, dtype=torch.bool, device=device)
    out_new_token_indices = torch.zeros(
        num_padding_slots_per_request * num_reqs, dtype=torch.int32, device=device
    )
    out_hidden_state_mapping = torch.zeros(total_input, dtype=torch.int32, device=device)

    BLOCK_SIZE_TOKENS = 128
    num_token_blocks = (max_output_tokens_per_req + BLOCK_SIZE_TOKENS - 1) // BLOCK_SIZE_TOKENS
    grid = (num_reqs, num_token_blocks)

    copy_and_expand_eagle_inputs_kernel[grid](
        target_token_ids,
        target_positions,
        next_token_ids,
        out_input_ids,
        out_positions,
        out_is_rejected,
        out_is_masked,
        out_new_token_indices,
        out_hidden_state_mapping,
        query_start_loc,
        query_end_loc,
        padding_token_id,
        parallel_drafting_token_id,
        total_input,
        num_padding_slots_per_request,
        shift_input_ids,
        BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
    )
    return (out_input_ids, out_positions, out_is_rejected, out_is_masked,
            out_new_token_indices, out_hidden_state_mapping)
