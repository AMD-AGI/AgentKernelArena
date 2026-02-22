"""EAGLE prepare_next_token_padded kernel from vLLM speculative decoding.

Computes number of valid (1+accepted) tokens and next token ID for each request
in EAGLE speculative decoding.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def eagle_prepare_next_token_padded_kernel(
    sampled_token_ids_ptr,            # [num_reqs, num_sampled_tokens_per_req]
    discard_request_mask_ptr,         # [num_reqs]
    backup_next_token_ids_ptr,        # [num_reqs]
    next_token_ids_ptr,               # [num_reqs] (output)
    valid_sampled_tokens_count_ptr,   # [num_reqs] (output)
    vocab_size,                       # tl.int32
    num_sampled_tokens_per_req,       # tl.int32
    num_reqs,                         # tl.int32
    stride_sampled_token_ids,         # tl.int32
    BLOCK_SIZE_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    is_discarded = tl.load(discard_request_mask_ptr + req_idx)

    if is_discarded:
        backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
        valid_count = tl.full((), 0, dtype=tl.uint32)
        tl.store(next_token_ids_ptr + req_idx, backup_token)
        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)
    else:
        token_offs = tl.arange(0, BLOCK_SIZE_TOKENS)
        token_mask = token_offs < num_sampled_tokens_per_req

        row_ptr = sampled_token_ids_ptr + req_idx * stride_sampled_token_ids
        token_ids = tl.load(row_ptr + token_offs, mask=token_mask, other=-1)

        is_valid_mask = (token_ids != -1) & (token_ids < vocab_size) & token_mask
        valid_count = tl.sum(is_valid_mask)

        if valid_count > 0:
            last_valid_index = tl.max(tl.where(is_valid_mask, token_offs, -1))
            last_valid_token = tl.sum(
                tl.where(token_offs == last_valid_index, token_ids, 0)
            )
            tl.store(next_token_ids_ptr + req_idx, last_valid_token)
        else:
            backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
            tl.store(next_token_ids_ptr + req_idx, backup_token)

        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)


def eagle_prepare_next_token_padded(
    sampled_token_ids: torch.Tensor,
    discard_request_mask: torch.Tensor,
    backup_next_token_ids: torch.Tensor,
    vocab_size: int,
) -> tuple:
    """
    Args:
        sampled_token_ids: [num_reqs, num_sampled_tokens_per_req]
        discard_request_mask: [num_reqs] bool
        backup_next_token_ids: [num_reqs]
        vocab_size: int
    Returns:
        next_token_ids: [num_reqs]
        valid_sampled_tokens_count: [num_reqs]
    """
    num_reqs, num_sampled = sampled_token_ids.shape
    next_token_ids = torch.empty(num_reqs, dtype=torch.int32, device=sampled_token_ids.device)
    valid_count = torch.empty(num_reqs, dtype=torch.int32, device=sampled_token_ids.device)

    BLOCK_SIZE_TOKENS = triton.next_power_of_2(num_sampled)

    grid = (num_reqs,)
    eagle_prepare_next_token_padded_kernel[grid](
        sampled_token_ids,
        discard_request_mask,
        backup_next_token_ids,
        next_token_ids,
        valid_count,
        vocab_size,
        num_sampled,
        num_reqs,
        sampled_token_ids.stride(0),
        BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
    )
    return next_token_ids, valid_count
