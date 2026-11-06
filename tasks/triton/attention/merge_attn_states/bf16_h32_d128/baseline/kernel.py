# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Merge two partial attention states with log-sum-exp rescaling (section 2.2 of
https://www.arxiv.org/pdf/2501.01005), from vLLM's vllm/v1/attention/ops/triton_merge_attn_states.py,
wrapped in the run() interface. The optional FP8 output quantization is removed."""

import torch
import triton
import triton.language as tl


@triton.jit
def merge_attn_states_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_head_stride,
    output_head_stride,
    prefix_lse_head_stride,
    prefix_lse_token_stride,
    suffix_lse_head_stride,
    suffix_lse_token_stride,
    output_lse_head_stride,
    output_lse_token_stride,
    prefill_tokens_with_context,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    prefix_mask = token_idx < prefill_tokens_with_context

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE

    # For tokens without context (token_idx >= prefill_tokens_with_context),
    # directly copy from suffix_output
    if not prefix_mask:
        s_lse = tl.load(
            suffix_lse
            + head_idx * suffix_lse_head_stride
            + token_idx * suffix_lse_token_stride
        )
        if OUTPUT_LSE:
            tl.store(
                output_lse
                + head_idx * output_lse_head_stride
                + token_idx * output_lse_token_stride,
                s_lse,
            )

        s_out = tl.load(
            suffix_output
            + token_idx * num_heads * prefix_head_stride
            + head_idx * prefix_head_stride
            + head_arange,
            mask=head_mask,
        )

        tl.store(
            output
            + token_idx * num_heads * output_head_stride
            + head_idx * output_head_stride
            + head_arange,
            s_out,
            mask=head_mask,
        )
        return

    # For tokens with context (token_idx < prefill_tokens_with_context),
    # perform normal merge operation
    p_lse = tl.load(
        prefix_lse
        + head_idx * prefix_lse_head_stride
        + token_idx * prefix_lse_token_stride
    )
    s_lse = tl.load(
        suffix_lse
        + head_idx * suffix_lse_head_stride
        + token_idx * suffix_lse_token_stride
    )

    # FA2 and FA3 have different behavior for when the sum-exp is 0, this namely
    # arises with 0 len seqlens. FA3 returns -inf here while FA2 returns inf.
    # If we see an inf assume FA2 and convert inf to -inf for consistency
    # and correctness. Inf generally doesn't make sense in this context outside
    # of undefined-behavior/FA2-case, so I think this a safe assumption.
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    # Will reuse precomputed Exp values for scale factor computation.
    p_se = tl.exp(p_lse)
    s_se = tl.exp(s_lse)
    out_se = p_se + s_se

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        # Both sides empty (max_lse == -inf) => undefined merge; keep -inf so
        # downstream merges continue to treat the token as empty.
        out_lse = tl.where(max_lse == float("-inf"), float("-inf"), out_lse)
        tl.store(
            output_lse
            + head_idx * output_lse_head_stride
            + token_idx * output_lse_token_stride,
            out_lse,
        )

    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )

    # NOTE(woosuk): Be careful with the numerical stability.
    # We should compute the scale first, and then multiply it with the output.
    # Do not multiply the output with tl.exp(p_lse) or tl.exp(s_lse) directly.
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    out = p_out * p_scale + s_out * s_scale
    # If both sides are empty (max_lse == -inf) the scales are 0/0 = NaN; emit
    # zeros rather than NaN. Callers with empty chunks (see mask_empty_context)
    # zero those inputs, so this only guards the fully-undefined corner.
    out = tl.where(max_lse == float("-inf"), 0.0, out)

    tl.store(
        output
        + token_idx * num_heads * output_head_stride
        + head_idx * output_head_stride
        + head_arange,
        out,
        mask=head_mask,
    )



def run(*, prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse):
    num_tokens = output.shape[0]
    num_query_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)
    # The output stride on num_head is not always the same as the prefix/suffix one: they may be
    # padded by the attention backend.
    prefix_head_stride = prefix_output.stride(1)
    output_head_stride = output.stride(1)

    merge_attn_states_kernel[(num_tokens, num_query_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        prefix_head_stride,
        output_head_stride,
        prefix_lse.stride(0),
        prefix_lse.stride(1),
        suffix_lse.stride(0),
        suffix_lse.stride(1),
        output_lse.stride(0),
        output_lse.stride(1),
        num_tokens,  # every token has prefix context
        head_size,
        padded_head_size,
        True,
    )
