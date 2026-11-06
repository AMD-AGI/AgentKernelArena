# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scatter K/V tokens with different head sizes into one paged cache, adapted from vLLM
triton_reshape_and_cache_flash.py (reshape_and_cache_flash_diffkv)."""
import triton
import triton.language as tl


@triton.jit
def reshape_and_cache_kernel_flash_diffkv(
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size_v]
    kv_cache_ptr,  # [num_blocks, block_size, num_heads, head_size + head_size_v]
    slot_mapping_ptr,  # [num_tokens]
    # strides
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    page_stride: tl.int64,
    head_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size_k: tl.constexpr,
    head_size_v: tl.constexpr,
    block_size: tl.constexpr,
    # tune parameters
    TILE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        # Padding token that should be ignored.
        return

    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, TILE_SIZE)

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    src_key_idx = token_idx * key_stride + tile_i * head_size_k
    src_value_idx = token_idx * value_stride + tile_i * head_size_v

    tgt_idx = (
        block_idx * block_stride + block_offset * page_stride + tile_i * head_stride
    )

    # [TILE_SIZE]
    key_tile = tl.load(key_ptr + src_key_idx + tile_offs, mask=tile_offs < head_size_k)
    # [TILE_SIZE]
    value_tile = tl.load(
        value_ptr + src_value_idx + tile_offs, mask=tile_offs < head_size_v
    )

    tl.store(
        kv_cache_ptr + tgt_idx + tile_offs,
        key_tile,
        mask=tile_offs < head_size_k,
    )
    tl.store(
        kv_cache_ptr + tgt_idx + head_size_k + tile_offs,
        value_tile,
        mask=tile_offs < head_size_v,
    )
    return


def run(*, key, value, kv_cache, slot_mapping):
    num_heads = key.shape[1]
    head_size_k = key.shape[2]
    head_size_v = value.shape[2]
    block_size = kv_cache.shape[1]

    k_stride = key.stride()[0]
    v_stride = value.stride()[0]
    block_stride = kv_cache.stride()[0]
    page_stride = kv_cache.stride()[1]
    head_stride = kv_cache.stride()[2]

    # heuristics instead of autotuning (vLLM's ROCm values)
    TILE_SIZE = max(head_size_k, head_size_v)
    TILE_SIZE = triton.next_power_of_2(TILE_SIZE)
    num_stages = 4
    num_warps = 8

    grid = lambda meta: (
        slot_mapping.shape[0],
        num_heads,
    )

    reshape_and_cache_kernel_flash_diffkv[grid](
        key_ptr=key,
        value_ptr=value,
        kv_cache_ptr=kv_cache,
        slot_mapping_ptr=slot_mapping,
        key_stride=k_stride,
        value_stride=v_stride,
        block_stride=block_stride,
        page_stride=page_stride,
        head_stride=head_stride,
        num_heads=num_heads,
        head_size_k=head_size_k,
        head_size_v=head_size_v,
        block_size=block_size,
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
