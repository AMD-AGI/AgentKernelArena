"""Sample recovered tokens kernel, adapted from vLLM rejection_sampler.py."""
import torch
import triton
import triton.language as tl


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    q_ptr,
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0
    if req_idx > 0:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1).to(tl.int32)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx).to(tl.int32)
    num_draft_tokens = end_idx - start_idx

    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    vocab_offset = tl.arange(0, PADDED_VOCAB_SIZE)
    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        prob = tl.load(
            target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=((vocab_offset < vocab_size) & (vocab_offset != draft_token_id)),
            other=0,
        )
    else:
        draft_prob = tl.load(
            draft_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0,
        )
        target_prob = tl.load(
            target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0,
        )
        prob = tl.maximum(target_prob - draft_prob, 0)

    q = tl.load(
        q_ptr + req_idx * vocab_size + vocab_offset,
        mask=vocab_offset < vocab_size,
        other=float("-inf"),
    )
    recovered_id = tl.argmax(prob / q, axis=-1)
    tl.store(output_token_ids_ptr + start_idx + pos, recovered_id)


def sample_recovered_tokens(
    cu_num_draft_tokens, draft_token_ids, draft_probs,
    target_probs, q, max_spec_len, vocab_size,
):
    """Sample recovered tokens for rejection sampling."""
    batch_size = cu_num_draft_tokens.shape[0]
    recovered_token_ids = torch.empty_like(draft_token_ids)
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids, cu_num_draft_tokens, draft_token_ids,
        draft_probs, target_probs, q, vocab_size,
        triton.next_power_of_2(vocab_size),
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return recovered_token_ids
