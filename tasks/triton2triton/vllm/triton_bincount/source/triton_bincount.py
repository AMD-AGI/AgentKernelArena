"""Bincount kernel, adapted from vLLM penalties.py."""
import torch
import triton
import triton.language as tl


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


def bincount(
    idx_mapping, all_token_ids, prompt_len, prefill_len,
    prompt_bin_mask, output_bin_counts, max_prefill_len,
):
    """Compute prompt bitmask and output bin counts for penalty computation."""
    prompt_bin_mask[idx_mapping] = 0
    output_bin_counts[idx_mapping] = 0
    num_reqs = idx_mapping.shape[0]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(max_prefill_len, BLOCK_SIZE)
    _bincount_kernel[(num_reqs, num_blocks)](
        idx_mapping, all_token_ids, all_token_ids.stride(0),
        prompt_len, prefill_len,
        prompt_bin_mask, prompt_bin_mask.stride(0),
        output_bin_counts, output_bin_counts.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
