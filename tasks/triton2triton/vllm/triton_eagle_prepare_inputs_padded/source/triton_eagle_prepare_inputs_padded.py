"""EAGLE prepare_inputs_padded kernel from vLLM speculative decoding.

Computes per-request token index to sample and number of rejected tokens,
given cumulative draft token counts, valid sampled token counts, and
query start locations.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def eagle_prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,          # [num_reqs]
    valid_sampled_tokens_count_ptr,   # [num_reqs]
    query_start_loc_gpu_ptr,          # [num_reqs + 1]
    token_indices_to_sample_ptr,      # [num_reqs] (output)
    num_rejected_tokens_gpu_ptr,      # [num_reqs] (output)
    num_reqs,                         # tl.int32
):
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + req_idx)

    num_draft_tokens = 0
    if req_idx == 0:
        num_draft_tokens = cu_draft_curr
    else:
        cu_draft_prev = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        num_draft_tokens = cu_draft_curr - cu_draft_prev

    valid_count = tl.load(valid_sampled_tokens_count_ptr + req_idx)
    num_rejected_tokens = num_draft_tokens + 1 - valid_count
    num_rejected_tokens = tl.where(num_draft_tokens > 0, num_rejected_tokens, 0)

    q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + req_idx + 1) - 1

    index_to_sample = q_last_tok_idx - num_rejected_tokens
    tl.store(token_indices_to_sample_ptr + req_idx, index_to_sample)
    tl.store(num_rejected_tokens_gpu_ptr + req_idx, num_rejected_tokens)


def eagle_prepare_inputs_padded(
    cu_num_draft_tokens: torch.Tensor,
    valid_sampled_tokens_count: torch.Tensor,
    query_start_loc_gpu: torch.Tensor,
) -> tuple:
    """
    Args:
        cu_num_draft_tokens: [num_reqs] inclusive cumsum of draft tokens
        valid_sampled_tokens_count: [num_reqs] number of valid (1+accepted) tokens
        query_start_loc_gpu: [num_reqs + 1] query start locations
    Returns:
        token_indices_to_sample: [num_reqs]
        num_rejected_tokens_gpu: [num_reqs]
    """
    num_reqs = cu_num_draft_tokens.shape[0]
    token_indices_to_sample = torch.empty(num_reqs, dtype=torch.int32, device=cu_num_draft_tokens.device)
    num_rejected_tokens_gpu = torch.empty(num_reqs, dtype=torch.int32, device=cu_num_draft_tokens.device)

    grid = (num_reqs,)
    eagle_prepare_inputs_padded_kernel[grid](
        cu_num_draft_tokens,
        valid_sampled_tokens_count,
        query_start_loc_gpu,
        token_indices_to_sample,
        num_rejected_tokens_gpu,
        num_reqs,
    )
    return token_indices_to_sample, num_rejected_tokens_gpu
