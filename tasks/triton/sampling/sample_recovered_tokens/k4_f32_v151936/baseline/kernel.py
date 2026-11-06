# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import triton
import triton.language as tl


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    inv_q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    USE_FP64_GUMBEL: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    token_idx = start_idx + pos

    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

    if USE_FP64_GUMBEL:
        max_val = tl.full((), float("-inf"), tl.float64)
    else:
        max_val = tl.full((), float("-inf"), tl.float32)
    recovered_id = 0
    for v in range(0, vocab_size, BLOCK_SIZE):
        vocab_offset = v + tl.arange(0, BLOCK_SIZE)
        vocab_mask = vocab_offset < vocab_size

        if NO_DRAFT_PROBS:
            prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=(vocab_mask & (vocab_offset != draft_token_id)),
                other=0.0,
            )
        else:
            draft_prob = tl.load(
                draft_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            target_prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            prob = tl.maximum(target_prob - draft_prob, 0.0)
            # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
            # `tl.argmax` will select the maximum value.

        inv_q = tl.load(
            inv_q_ptr + req_idx * vocab_size + vocab_offset,
            mask=vocab_mask,
            other=0.0,
        )

        # Local tile reduction.
        # Mask out-of-vocabulary entries to -inf so they can never win
        # the argmax — prevents producing recovered_id >= vocab_size
        # when all valid entries in the last tile have zero probability.
        score = prob * inv_q
        score = tl.where(vocab_mask, score, float("-inf"))
        local_max, local_id = tl.max(score, axis=0, return_indices=True)

        if local_max > max_val:
            max_val = local_max
            recovered_id = v + local_id

    recovered_id = tl.minimum(recovered_id, vocab_size - 1)
    tl.store(output_token_ids_ptr + token_idx, recovered_id)


def sample_recovered_tokens(
    max_spec_len,
    cu_num_draft_tokens,
    draft_token_ids,
    draft_probs,
    target_probs,
    inv_q,
    recovered_token_ids,
):
    batch_size = cu_num_draft_tokens.shape[0]
    vocab_size = target_probs.shape[-1]
    BLOCK_SIZE = 8192
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
        vocab_size,
        BLOCK_SIZE,
        NO_DRAFT_PROBS=False,
        USE_FP64_GUMBEL=False,
    )


def run(*, cu_num_draft_tokens, draft_token_ids, draft_probs, target_probs, inv_q, recovered_token_ids):
    batch_size = cu_num_draft_tokens.shape[0]
    max_spec_len = draft_token_ids.shape[0] // batch_size
    sample_recovered_tokens(
        max_spec_len, cu_num_draft_tokens, draft_token_ids, draft_probs, target_probs, inv_q,
        recovered_token_ids,
    )
