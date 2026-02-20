"""
Triton kernel to convert request-local token indices to global cache indices.

Extracted from vLLM v1 attention backends (mla/sparse_utils.py).
Simple index translation kernel:
  out[token_id, indice_id] =
      block_table[req_id[token_id],
          token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
      + token_indices[token_id, indice_id] % BLOCK_SIZE

Tokens with index == -1 propagate as -1. Out-of-bounds block indices also yield -1.
Optionally counts valid (non -1) indices per row via atomic add.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    valid_count_ptr,  # int32 [num_tokens] - output valid count per row
    # shapes (compile-time where possible)
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    COUNT_VALID: tl.constexpr,  # whether to count valid indices
    # strides (in elements)
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token
    req = tl.load(req_id_ptr + token_id)

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    is_invalid_tok = tok < 0

    # Compute block id and in-block offset
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    # Guard block_table access
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    is_invalid_tok |= ~valid_block
    base = tl.load(bt_ptr, mask=valid_block, other=0)
    out_val = base * BLOCK_SIZE + inblock_off

    out_val = tl.where(is_invalid_tok, -1, out_val)

    # Store results
    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)

    # Count valid indices in this tile and atomically add to row total
    if COUNT_VALID:
        tile_valid_count = tl.sum((~is_invalid_tok).to(tl.int32))
        tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)


def convert_req_to_global_index(
    req_id: torch.Tensor,        # int32 [num_tokens]
    block_table: torch.Tensor,   # int32 [num_requests, max_num_blocks_per_req]
    token_indices: torch.Tensor, # int32 [num_tokens, NUM_TOPK_TOKENS]
    BLOCK_SIZE: int = 64,
    BLOCK_N: int = 128,
    return_valid_counts: bool = False,
) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
    """
    Translate request-local token indices to global cache slot indices.

    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Entries with token_indices == -1 or out-of-bounds block_id yield -1.
    """
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32

    NUM_TOPK_TOKENS = token_indices.shape[1]
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by BLOCK_N ({BLOCK_N})"
    )

    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    valid_counts = None
    if return_valid_counts:
        valid_counts = torch.zeros(
            num_tokens, dtype=torch.int32, device=token_indices.device
        )

    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        valid_counts,
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        return_valid_counts,
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )

    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out
