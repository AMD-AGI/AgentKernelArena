# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import triton
import triton.language as tl


@triton.jit
def _penalties_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    token_ids_ptr,
    expanded_local_pos_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
    freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
    pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)

    use_rep_penalty = rep_penalty != 1.0
    use_freq_penalty = freq_penalty != 0.0
    use_pres_penalty = pres_penalty != 0.0
    use_penalty = use_rep_penalty or use_freq_penalty or use_pres_penalty
    if not use_penalty:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)

    base_output_counts = tl.load(
        output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
        mask=mask,
        other=0,
    )

    # Accumulate draft token counts from previous positions directly into
    # output_bin_counts (preserves its native tensor layout, avoiding an
    # expensive shared-memory layout conversion after the loop).
    pos = tl.load(expanded_local_pos_ptr + token_idx)
    start_idx = token_idx - pos
    output_bin_counts = base_output_counts
    for prev_pos in tl.range(pos):
        prev_token = tl.load(token_ids_ptr + start_idx + prev_pos + 1)
        token_match = block == prev_token
        output_bin_counts = output_bin_counts + token_match.to(tl.int32)
    output_bin_mask = output_bin_counts > 0

    # Apply repetition penalties.
    if use_rep_penalty:
        packed_block = block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
        packed_mask = tl.load(
            prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + packed_block,
            mask=packed_block < tl.cdiv(vocab_size, 32),
            other=0,
        )
        prompt_bin_mask = (packed_mask[:, None] >> (tl.arange(0, 32)[None, :])) & 1
        prompt_bin_mask = prompt_bin_mask.to(tl.int1)
        prompt_bin_mask = prompt_bin_mask.reshape(BLOCK_SIZE)

        # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
        scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
        # If logits are positive, divide by penalty, otherwise multiply by penalty.
        logits *= tl.where(logits > 0, 1.0 / scale, scale)

    # Apply frequency penalties.
    logits -= freq_penalty * output_bin_counts
    # Apply presence penalties.
    logits -= pres_penalty * output_bin_mask
    # Store back to logits.
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_penalties(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _penalties_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def run(*, logits, expanded_idx_mapping, token_ids, expanded_local_pos, repetition_penalty,
        frequency_penalty, presence_penalty, prompt_bin_mask, output_bin_counts):
    apply_penalties(logits, expanded_idx_mapping, token_ids, expanded_local_pos, repetition_penalty,
                    frequency_penalty, presence_penalty, prompt_bin_mask, output_bin_counts)
