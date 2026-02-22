import torch
import triton
import triton.language as tl


@triton.jit
def reshape_and_cache_kernel_flash_diffkv(
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size_v]
    kv_cache_ptr,  # [num_blocks, block_size, num_heads, head_size + head_size_v]
    slot_mapping_ptr,  # [num_tokens]
    k_scale,  # float32
    v_scale,  # float32
    # strides
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    page_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size_k: tl.constexpr,
    head_size_v: tl.constexpr,
    block_size: tl.constexpr,
    # FP8 flags
    FP8_KV_CACHE: tl.constexpr,
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
        block_idx * block_stride
        + block_offset * page_stride
        + tile_i * (head_size_k + head_size_v)
    )

    # [TILE_SIZE]
    key_load = tl.load(key_ptr + src_key_idx + tile_offs, mask=tile_offs < head_size_k)
    if FP8_KV_CACHE:
        key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
    else:
        key_tile = key_load

    # [TILE_SIZE]
    value_load = tl.load(
        value_ptr + src_value_idx + tile_offs, mask=tile_offs < head_size_v
    )
    if FP8_KV_CACHE:
        if value_load.dtype.is_fp8():
            value_tile = value_load
        else:
            value_tile = value_load / tl.load(v_scale)
    else:
        value_tile = value_load

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


def reshape_and_cache_flash_diffkv(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size_k]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size_v]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, num_heads, head_size_k + head_size_v]
    slot_mapping: torch.Tensor,  # [num_tokens]
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor = None,
    v_scale: torch.Tensor = None,
):
    """Scatter K/V tokens into paged cache slots with different K/V head dimensions.

    Args:
        key: [num_tokens, num_heads, head_size_k]
        value: [num_tokens, num_heads, head_size_v]
        kv_cache: [num_blocks, block_size, num_heads, head_size_k + head_size_v]
        slot_mapping: [num_tokens] - maps each token to a cache slot
        kv_cache_dtype: "auto" (no conversion) or "fp8"
        k_scale: scaling factor for FP8
        v_scale: scaling factor for FP8
    """
    num_heads = key.shape[1]
    head_size_k = key.shape[2]
    head_size_v = value.shape[2]
    block_size = kv_cache.shape[1]

    k_stride = key.stride()[0]
    v_stride = value.stride()[0]
    block_stride = kv_cache.stride()[0]
    page_stride = kv_cache.stride()[1]

    if k_scale is None:
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=key.device)
    if v_scale is None:
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=key.device)

    FP8_KV_CACHE = kv_cache_dtype.startswith("fp8") if kv_cache_dtype != "auto" else False

    TILE_SIZE = max(head_size_k, head_size_v)
    TILE_SIZE = triton.next_power_of_2(TILE_SIZE)
    num_stages = 10
    num_warps = 16

    grid = lambda meta: (
        slot_mapping.shape[0],
        num_heads,
    )

    reshape_and_cache_kernel_flash_diffkv[grid](
        key_ptr=key,
        value_ptr=value,
        kv_cache_ptr=kv_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
        key_stride=k_stride,
        value_stride=v_stride,
        block_stride=block_stride,
        page_stride=page_stride,
        num_heads=num_heads,
        head_size_k=head_size_k,
        head_size_v=head_size_v,
        block_size=block_size,
        FP8_KV_CACHE=FP8_KV_CACHE,
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
