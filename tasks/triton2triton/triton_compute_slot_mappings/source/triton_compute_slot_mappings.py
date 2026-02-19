"""Compute slot mappings kernel from vLLM v1 worker/gpu/block_table.py.
Simplified: TOTAL_CP_WORLD_SIZE=1, TOTAL_CP_RANK=0 (no context parallelism).
Uses a single contiguous block table instead of pointer-of-pointers."""
import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1


@triton.jit
def _compute_slot_mappings_kernel(
    num_tokens,
    max_num_tokens,
    idx_mapping,       # [num_reqs]
    query_start_loc,   # [num_reqs + 1]
    pos,               # [num_tokens]
    block_table_ptr,   # [max_num_reqs, max_num_blocks]
    block_table_stride,
    block_size,
    slot_mappings_ptr, # [max_num_tokens]
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    if batch_idx == tl.num_programs(0) - 1:
        # Pad remaining slots to PAD_ID for CUDA graphs.
        for i in range(num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(slot_mappings_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)

    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)
        block_indices = positions // block_size
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices
        )
        block_offsets = positions % block_size
        slot_ids = block_numbers * block_size + block_offsets
        tl.store(slot_mappings_ptr + offset, slot_ids, mask=offset < end_idx)


def compute_slot_mappings(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    max_num_tokens: int,
) -> torch.Tensor:
    """Compute slot mappings for paged KV cache.

    Args:
        idx_mapping: [num_reqs] batch_idx -> req_state_idx
        query_start_loc: [num_reqs + 1] cumulative token counts
        positions: [num_tokens] position of each token
        block_table: [max_num_reqs, max_num_blocks] block IDs
        block_size: KV cache block size
        max_num_tokens: max tokens for padding
    Returns:
        slot_mappings: [max_num_tokens] slot IDs
    """
    num_reqs = idx_mapping.shape[0]
    num_tokens = positions.shape[0]
    slot_mappings = torch.empty(max_num_tokens, dtype=torch.int64, device=positions.device)
    _compute_slot_mappings_kernel[(num_reqs + 1,)](
        num_tokens,
        max_num_tokens,
        idx_mapping,
        query_start_loc,
        positions,
        block_table,
        block_table.stride(0),
        block_size,
        slot_mappings,
        PAD_ID=PAD_SLOT_ID,
        TRITON_BLOCK_SIZE=1024,
    )
    return slot_mappings[:num_tokens]
