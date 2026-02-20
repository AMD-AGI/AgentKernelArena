"""Rejection greedy sample kernel, adapted from vLLM rejection_sampler.py."""
import torch
import triton
import triton.language as tl

PLACEHOLDER_TOKEN_ID = -1


@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    target_argmax_ptr,
    bonus_token_ids_ptr,
    is_greedy_ptr,
    max_spec_len,
):
    req_idx = tl.program_id(0)
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        return

    start_idx = 0
    if req_idx > 0:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1).to(tl.int32)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx).to(tl.int32)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                target_argmax_id,
            )
            if draft_token_id != target_argmax_id:
                rejected = True

    if not rejected:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


def rejection_greedy_sample(
    output_token_ids, cu_num_draft_tokens, draft_token_ids,
    target_argmax, bonus_token_ids, is_greedy, max_spec_len,
):
    """Run rejection greedy sampling."""
    batch_size = cu_num_draft_tokens.shape[0]
    rejection_greedy_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
        max_spec_len,
    )
    return output_token_ids
