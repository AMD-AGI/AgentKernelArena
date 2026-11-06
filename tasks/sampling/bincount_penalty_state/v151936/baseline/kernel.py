# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Penalty-state bincount in Triton, adapted from vLLM v1 worker/gpu/sample/penalties.py."""
import torch
import triton
import triton.language as tl


@triton.jit
def _reset_bin_state_kernel(
    idx_mapping_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    mask_words,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Zeroes the mapped state rows (vLLM does this with `prompt_bin_mask[idx_mapping] = 0`
    # and `output_bin_counts[idx_mapping] = 0` before launching the bincount kernel).
    batch_idx = tl.program_id(0)
    block = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx).to(tl.int64)
    tl.store(
        output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
        tl.zeros((BLOCK_SIZE,), tl.int32),
        mask=block < vocab_size,
    )
    tl.store(
        prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + block,
        tl.zeros((BLOCK_SIZE,), tl.int32),
        mask=block < mask_words,
    )


@triton.jit
def _bincount_kernel(
    idx_mapping_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    prefill_len_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if block_idx * BLOCK_SIZE >= prefill_len:
        return

    prompt_len = tl.load(prompt_len_ptr + req_state_idx)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if block_idx * BLOCK_SIZE < prompt_len:
        mask = block < prompt_len
        prompt_tokens = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + block, mask=mask
        )
        idx = prompt_tokens // 32
        bit_idx = prompt_tokens % 32
        bit = tl.full((BLOCK_SIZE,), 1, tl.int32) << bit_idx
        tl.atomic_or(
            prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + idx,
            bit, mask=mask,
        )

    if (block_idx + 1) * BLOCK_SIZE >= prompt_len:
        mask = block < prefill_len
        mask &= block >= prompt_len
        output_tokens = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + block, mask=mask
        )
        tl.atomic_add(
            output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + output_tokens,
            1, mask=mask,
        )


def run(*, idx_mapping, all_token_ids, prompt_len, prefill_len, prompt_bin_mask, output_bin_counts):
    num_reqs = idx_mapping.shape[0]
    max_prefill_len = all_token_ids.shape[1]
    vocab_size = output_bin_counts.shape[1]
    mask_words = prompt_bin_mask.shape[1]
    RESET_BLOCK = 2048
    _reset_bin_state_kernel[(num_reqs, triton.cdiv(vocab_size, RESET_BLOCK))](
        idx_mapping,
        prompt_bin_mask, prompt_bin_mask.stride(0), mask_words,
        output_bin_counts, output_bin_counts.stride(0), vocab_size,
        BLOCK_SIZE=RESET_BLOCK,
    )
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(max_prefill_len, BLOCK_SIZE)
    _bincount_kernel[(num_reqs, num_blocks)](
        idx_mapping, all_token_ids, all_token_ids.stride(0),
        prompt_len, prefill_len,
        prompt_bin_mask, prompt_bin_mask.stride(0),
        output_bin_counts, output_bin_counts.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
