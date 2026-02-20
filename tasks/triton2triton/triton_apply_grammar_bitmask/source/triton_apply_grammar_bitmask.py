"""Apply grammar bitmask kernel from vLLM v1 worker/gpu/structured_outputs.py.
Uses packed bitmask (int32 with 32 bits each) to mask logits."""
import torch
import triton
import triton.language as tl


@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    bitmask_idx = tl.program_id(0)
    logits_idx = tl.load(logits_indices_ptr + bitmask_idx)

    # Load the bitmask.
    block_id = tl.program_id(1)
    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(
        bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
        mask=bitmask_offset < bitmask_stride,
    )
    # Unpack the bitmask.
    bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
    bitmask = bitmask.reshape(BLOCK_SIZE)

    # Apply the bitmask to the logits.
    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        logits_ptr + logits_idx * logits_stride + block_offset,
        -float("inf"),
        mask=bitmask & (block_offset < vocab_size),
    )


def apply_grammar_bitmask(
    logits: torch.Tensor,
    logits_indices: torch.Tensor,
    bitmask: torch.Tensor,
    vocab_size: int,
) -> None:
    """Apply grammar bitmask to logits in-place.

    For each bitmask row, the corresponding logits row (given by logits_indices)
    is masked: positions where the bit is 0 are set to -inf.

    Args:
        logits: [num_total_logits, vocab_size] float tensor
        logits_indices: [num_masks] int32 mapping bitmask row -> logits row
        bitmask: [num_masks, ceil(vocab_size/32)] int32 packed bitmask
        vocab_size: vocabulary size
    """
    num_masks = bitmask.shape[0]
    BLOCK_SIZE = 8192
    grid = (num_masks, triton.cdiv(vocab_size, BLOCK_SIZE))
    _apply_grammar_bitmask_kernel[grid](
        logits,
        logits.stride(0),
        logits_indices,
        bitmask,
        bitmask.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
