"""Prepare prefill inputs kernel from vLLM v1 worker/gpu/input_batch.py."""
import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_prefill_inputs_kernel(
    input_ids_ptr,
    next_prefill_tokens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    if num_computed >= prefill_len:
        # Not prefill.
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    request_ptr = all_token_ids_ptr + req_state_idx * all_token_ids_stride
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        tokens = tl.load(request_ptr + num_computed + block, mask=mask)
        tl.store(input_ids_ptr + query_start + block, tokens, mask=mask)

    next_pos = num_computed + query_len
    if next_pos < prefill_len:
        next_token = tl.load(request_ptr + next_pos)
        tl.store(next_prefill_tokens_ptr + req_state_idx, next_token)


def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    _prepare_prefill_inputs_kernel[(num_reqs,)](
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        prefill_len,
        num_computed_tokens,
        BLOCK_SIZE=1024,
    )
