"""Gather block tables kernel from vLLM v1 worker/gpu/block_table.py.
Simplified: works with a single contiguous block table (no pointer-of-pointers).
Instead of _load_ptr, we pass the block table directly."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather_block_tables_kernel(
    batch_idx_to_req_idx,  # [batch_size]
    src_block_table,       # [max_num_reqs, max_num_blocks]
    dst_block_table,       # [max_num_reqs, max_num_blocks]
    block_table_stride,    # stride of dim 0
    num_blocks_ptr,        # [max_num_reqs]
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)

    num_blocks = tl.load(num_blocks_ptr + req_idx)

    src_row_ptr = src_block_table + req_idx * block_table_stride
    dst_row_ptr = dst_block_table + batch_idx * block_table_stride

    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)


def gather_block_tables(
    idx_mapping: torch.Tensor,
    src_block_table: torch.Tensor,
    dst_block_table: torch.Tensor,
    num_blocks: torch.Tensor,
) -> torch.Tensor:
    """Gather block table rows from src to dst based on idx_mapping.

    Args:
        idx_mapping: [num_reqs] mapping from batch_idx to req_idx
        src_block_table: [max_num_reqs, max_num_blocks] source block table
        dst_block_table: [max_num_reqs, max_num_blocks] destination block table
        num_blocks: [max_num_reqs] number of valid blocks per request
    Returns:
        dst_block_table[:num_reqs]
    """
    num_reqs = idx_mapping.shape[0]
    _gather_block_tables_kernel[(num_reqs,)](
        idx_mapping,
        src_block_table,
        dst_block_table,
        src_block_table.stride(0),
        num_blocks,
        BLOCK_SIZE=1024,
    )
    return dst_block_table[:num_reqs]
