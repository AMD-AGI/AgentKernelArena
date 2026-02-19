"""Rejection random sample kernel, adapted from vLLM rejection_sampler.py."""
import torch
import triton
import triton.language as tl

PLACEHOLDER_TOKEN_ID = -1


@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    bonus_token_ids_ptr,
    recovered_token_ids_ptr,
    uniform_probs_ptr,
    is_greedy_ptr,
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
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
            if NO_DRAFT_PROBS:
                draft_prob = 1
            else:
                draft_prob = tl.load(
                    draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
            target_prob = tl.load(
                target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
            )
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if not rejected:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


def rejection_random_sample(
    output_token_ids, cu_num_draft_tokens, draft_token_ids,
    draft_probs, target_probs, bonus_token_ids, recovered_token_ids,
    uniform_probs, is_greedy, max_spec_len, vocab_size,
):
    """Run rejection random sampling."""
    batch_size = cu_num_draft_tokens.shape[0]
    rejection_random_sample_kernel[(batch_size,)](
        output_token_ids, cu_num_draft_tokens, draft_token_ids,
        draft_probs, target_probs, bonus_token_ids, recovered_token_ids,
        uniform_probs, is_greedy, max_spec_len, vocab_size,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return output_token_ids
