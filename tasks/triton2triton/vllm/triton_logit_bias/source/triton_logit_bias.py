"""Logit bias kernel, adapted from vLLM logit_bias.py."""
import torch
import triton
import triton.language as tl


@triton.jit
def _bias_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    idx_mapping_ptr,
    num_allowed_token_ids_ptr,
    allowed_token_ids_ptr,
    allowed_token_ids_stride,
    num_logit_bias_ptr,
    bias_token_ids_ptr,
    bias_token_ids_stride,
    bias_ptr,
    bias_stride,
    pos_ptr,
    min_lens_ptr,
    num_stop_token_ids_ptr,
    stop_token_ids_ptr,
    stop_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
    LOGITS_BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    block = tl.arange(0, BLOCK_SIZE)

    # Allowed token IDs.
    num_allowed_token_ids = tl.load(num_allowed_token_ids_ptr + req_state_idx)
    if num_allowed_token_ids > 0:
        block = tl.arange(0, BLOCK_SIZE)
        mask = block < num_allowed_token_ids

        allowed_token_ids = tl.load(
            allowed_token_ids_ptr + req_state_idx * allowed_token_ids_stride + block,
            mask=mask,
        )
        logits = tl.load(
            logits_ptr + batch_idx * logits_stride + allowed_token_ids, mask=mask
        )

        for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
            offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
            tl.store(
                logits_ptr + batch_idx * logits_stride + offset,
                -float("inf"),
                mask=offset < vocab_size,
            )

        tl.store(
            logits_ptr + batch_idx * logits_stride + allowed_token_ids,
            logits,
            mask=mask,
        )

    # Logit bias.
    num_logit_bias = tl.load(num_logit_bias_ptr + req_state_idx)
    if num_logit_bias > 0:
        mask = block < num_logit_bias
        token_ids = tl.load(
            bias_token_ids_ptr + req_state_idx * bias_token_ids_stride + block,
            mask=mask,
        )
        bias = tl.load(bias_ptr + req_state_idx * bias_stride + block, mask=mask)
        logits = tl.load(logits_ptr + batch_idx * logits_stride + token_ids, mask=mask)
        logits += bias
        tl.store(logits_ptr + batch_idx * logits_stride + token_ids, logits, mask=mask)

    # Apply min tokens.
    num_stop_token_ids = tl.load(num_stop_token_ids_ptr + req_state_idx)
    pos = tl.load(pos_ptr + batch_idx)
    min_len = tl.load(min_lens_ptr + req_state_idx)
    if num_stop_token_ids > 0 and pos < min_len:
        mask = block < num_stop_token_ids
        stop_token_ids = tl.load(
            stop_token_ids_ptr + req_state_idx * stop_token_ids_stride + block,
            mask=mask,
        )
        tl.store(
            logits_ptr + batch_idx * logits_stride + stop_token_ids,
            -float("inf"),
            mask=mask,
        )


def apply_logit_bias(
    logits, idx_mapping, pos,
    num_allowed_token_ids, allowed_token_ids,
    num_logit_bias, logit_bias_token_ids, logit_bias,
    min_lens, num_stop_token_ids, stop_token_ids,
):
    """Apply logit bias, allowed token filtering, and min-token stop masking."""
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(
        max(allowed_token_ids.shape[-1], logit_bias_token_ids.shape[-1], stop_token_ids.shape[-1])
    )
    LOGITS_BLOCK_SIZE = 8192
    _bias_kernel[(num_reqs,)](
        logits, logits.stride(0), vocab_size, idx_mapping,
        num_allowed_token_ids, allowed_token_ids, allowed_token_ids.stride(0),
        num_logit_bias, logit_bias_token_ids, logit_bias_token_ids.stride(0),
        logit_bias, logit_bias.stride(0),
        pos, min_lens, num_stop_token_ids, stop_token_ids, stop_token_ids.stride(0),
        BLOCK_SIZE=BLOCK_SIZE, LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE,
    )
