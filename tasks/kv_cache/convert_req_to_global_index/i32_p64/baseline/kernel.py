# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-local to global cache-slot index translation (sparse attention metadata) in Triton.

Extracted from vLLM v1 attention backends (mla/sparse_utils.py).
  out[token_id, indice_id] =
      block_table[req_id[token_id], token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
      + token_indices[token_id, indice_id] % BLOCK_SIZE
Entries with index < 0 and out-of-range block indices yield -1.
"""
import triton
import triton.language as tl

BLOCK_SIZE = 64
BLOCK_N = 128


@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    valid_count_ptr,  # int32 [num_tokens] - output valid count per row
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    COUNT_VALID: tl.constexpr,  # whether to count valid indices
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    # program_id(0) -> token_id (row); program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    req = tl.load(req_id_ptr + token_id)

    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)

    # Only token == -1 (any negative) should propagate as -1
    is_invalid_tok = tok < 0

    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    # Guard block_table access
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    is_invalid_tok |= ~valid_block
    base = tl.load(bt_ptr, mask=valid_block, other=0)
    out_val = base * BLOCK_SIZE + inblock_off

    out_val = tl.where(is_invalid_tok, -1, out_val)

    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)

    # Count valid indices in this tile and atomically add to the row total
    if COUNT_VALID:
        tile_valid_count = tl.sum((~is_invalid_tok).to(tl.int32))
        tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)


def run(*, req_id, block_table, token_indices, out):
    num_tokens, num_topk_tokens = token_indices.shape
    tiles_per_row = num_topk_tokens // BLOCK_N
    _convert_req_index_to_global_index_kernel[(num_tokens, tiles_per_row)](
        req_id,
        block_table,
        token_indices,
        out,
        None,
        block_table.shape[1],
        BLOCK_SIZE,
        BLOCK_N,
        False,
        block_table.stride(0),
        block_table.stride(1),
        token_indices.stride(0),
        token_indices.stride(1),
        out.stride(0),
        out.stride(1),
    )
