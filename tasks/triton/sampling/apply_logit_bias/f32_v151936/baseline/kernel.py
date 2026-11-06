# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allowed-token, logit-bias and minimum-length masking in Triton, vendored from vLLM's v1 GPU model runner."""
import triton
import triton.language as tl


@triton.jit
def _bias_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    expanded_idx_mapping_ptr,
    # Allowed token IDs.
    num_allowed_token_ids_ptr,
    allowed_token_ids_ptr,
    allowed_token_ids_stride,
    # Logit bias.
    num_logit_bias_ptr,
    bias_token_ids_ptr,
    bias_token_ids_stride,
    bias_ptr,
    bias_stride,
    # Min tokens.
    pos_ptr,
    min_lens_ptr,
    num_stop_token_ids_ptr,
    stop_token_ids_ptr,
    stop_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
    LOGITS_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    block = tl.arange(0, BLOCK_SIZE)

    # Allowed token IDs.
    num_allowed_token_ids = tl.load(num_allowed_token_ids_ptr + req_state_idx)
    if num_allowed_token_ids > 0:
        block = tl.arange(0, BLOCK_SIZE)
        mask = block < num_allowed_token_ids

        # Save logits for allowed token IDs.
        allowed_token_ids = tl.load(
            allowed_token_ids_ptr + req_state_idx * allowed_token_ids_stride + block,
            mask=mask,
        )
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + allowed_token_ids, mask=mask
        )

        tl.debug_barrier()  # save must read original logits before the -inf overwrite

        # Set logits to -inf for all tokens.
        for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
            offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
            tl.store(
                logits_ptr + token_idx * logits_stride + offset,
                -float("inf"),
                mask=offset < vocab_size,
            )

        tl.debug_barrier()  # -inf overwrite must finish before restoring saved logits

        # Restore logits for allowed token IDs.
        tl.store(
            logits_ptr + token_idx * logits_stride + allowed_token_ids,
            logits,
            mask=mask,
        )

        # Task modification: the restore must be visible to the bias loads below.
        tl.debug_barrier()

    # Logit bias.
    num_logit_bias = tl.load(num_logit_bias_ptr + req_state_idx)
    if num_logit_bias > 0:
        mask = block < num_logit_bias
        token_ids = tl.load(
            bias_token_ids_ptr + req_state_idx * bias_token_ids_stride + block,
            mask=mask,
        )
        bias = tl.load(bias_ptr + req_state_idx * bias_stride + block, mask=mask)
        logits = tl.load(logits_ptr + token_idx * logits_stride + token_ids, mask=mask)
        logits += bias
        tl.store(logits_ptr + token_idx * logits_stride + token_ids, logits, mask=mask)

        # Task modification: the biased values must be visible to the stop-token stores below.
        tl.debug_barrier()

    # Apply min tokens.
    num_stop_token_ids = tl.load(num_stop_token_ids_ptr + req_state_idx)
    pos = tl.load(pos_ptr + token_idx)
    min_len = tl.load(min_lens_ptr + req_state_idx)
    if num_stop_token_ids > 0 and pos + 1 < min_len:
        mask = block < num_stop_token_ids
        stop_token_ids = tl.load(
            stop_token_ids_ptr + req_state_idx * stop_token_ids_stride + block,
            mask=mask,
        )
        tl.store(
            logits_ptr + token_idx * logits_stride + stop_token_ids,
            -float("inf"),
            mask=mask,
        )


def run(
    *,
    logits,
    expanded_idx_mapping,
    pos,
    num_allowed_token_ids,
    allowed_token_ids,
    num_logit_bias,
    logit_bias_token_ids,
    logit_bias,
    min_lens,
    num_stop_token_ids,
    stop_token_ids,
):
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(
        max(
            allowed_token_ids.shape[-1],
            logit_bias_token_ids.shape[-1],
            stop_token_ids.shape[-1],
        )
    )
    LOGITS_BLOCK_SIZE = 8192
    _bias_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        vocab_size,
        expanded_idx_mapping,
        num_allowed_token_ids,
        allowed_token_ids,
        allowed_token_ids.stride(0),
        num_logit_bias,
        logit_bias_token_ids,
        logit_bias_token_ids.stride(0),
        logit_bias,
        logit_bias.stride(0),
        pos,
        min_lens,
        num_stop_token_ids,
        stop_token_ids,
        stop_token_ids.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE,
    )
