# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scatter K/V tokens into a paged KV cache, adapted from vLLM triton_reshape_and_cache_flash.py."""
import triton
import triton.language as tl


@triton.jit
def reshape_and_cache_kernel_flash(
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    key_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    value_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping_ptr,  # [num_tokens]
    # strides
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    head_stride: tl.int64,
    page_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    # tune parameters
    TILE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        # Padding token that should be ignored.
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, TILE_SIZE)
    tile_pos = tile_i * TILE_SIZE + tile_offs
    src_key_idx = token_idx * key_stride
    src_value_idx = token_idx * value_stride

    cur_head = tile_pos // head_size
    cur_dim = tile_pos % head_size
    tgt_idx_k = (
        block_idx * block_stride
        + block_offset * page_stride
        + cur_head * head_stride
        + cur_dim
    )
    tgt_idx_v = tgt_idx_k

    # [TILE_SIZE]
    key_tile = tl.load(
        key_ptr + src_key_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
    )
    # [TILE_SIZE]
    value_tile = tl.load(
        value_ptr + src_value_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
    )

    tl.store(
        key_cache_ptr + tgt_idx_k,
        key_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    tl.store(
        value_cache_ptr + tgt_idx_v,
        value_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    return


def run(*, key, value, key_cache, value_cache, slot_mapping):
    num_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache.shape[1]

    n = num_heads * head_size
    key_stride = key.stride()[0]
    value_stride = value.stride()[0]
    block_stride = key_cache.stride()[0]
    page_stride = key_cache.stride()[1]
    head_stride = key_cache.stride()[2]

    # heuristics instead of autotuning (vLLM's ROCm values)
    TILE_SIZE = min(2048, triton.next_power_of_2(n))
    num_stages = 4
    num_warps = 8

    grid = lambda meta: (
        slot_mapping.shape[0],
        triton.cdiv(n, meta["TILE_SIZE"]),
    )

    reshape_and_cache_kernel_flash[grid](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        key_stride=key_stride,
        value_stride=value_stride,
        block_stride=block_stride,
        head_stride=head_stride,
        page_stride=page_stride,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
