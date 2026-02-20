"""Prepare EAGLE inputs kernel from vLLM eagle worker.

Shifts target input_ids by one, copies positions, and computes last token
indices for EAGLE speculative decoding.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_eagle_inputs_kernel(
    last_token_indices_ptr,
    eagle_input_ids_ptr,
    eagle_positions_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    num_rejected = tl.load(num_rejected_ptr + batch_idx)
    query_len -= num_rejected

    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    if num_sampled > 0:
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        next_token = tl.load(next_prefill_tokens_ptr + req_state_idx)

    # Shift target_input_ids by one
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(eagle_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)

    last_token_index = query_start + query_len - 1
    tl.store(last_token_indices_ptr + batch_idx, last_token_index)
    tl.store(eagle_input_ids_ptr + last_token_index, next_token)

    # Copy positions
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(eagle_positions_ptr + query_start + block, target_pos, mask=mask)


def prepare_eagle_inputs(
    target_input_ids: torch.Tensor,
    target_positions: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> tuple:
    """
    Args:
        target_input_ids: [total_tokens]
        target_positions: [total_tokens]
        idx_mapping: [num_reqs]
        last_sampled: [max_num_reqs] int64
        next_prefill_tokens: [max_num_reqs] int32
        num_sampled: [num_reqs]
        num_rejected: [num_reqs]
        query_start_loc: [num_reqs + 1]
    Returns:
        last_token_indices: [num_reqs]
        eagle_input_ids: [total_tokens]
        eagle_positions: [total_tokens]
    """
    num_reqs = idx_mapping.shape[0]
    total_tokens = target_input_ids.shape[0]
    device = target_input_ids.device

    last_token_indices = torch.empty(num_reqs, dtype=torch.int64, device=device)
    eagle_input_ids = torch.zeros_like(target_input_ids)
    eagle_positions = torch.zeros_like(target_positions)

    _prepare_eagle_inputs_kernel[(num_reqs,)](
        last_token_indices,
        eagle_input_ids,
        eagle_positions,
        target_input_ids,
        target_positions,
        idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        BLOCK_SIZE=1024,
    )
    return last_token_indices, eagle_input_ids, eagle_positions
