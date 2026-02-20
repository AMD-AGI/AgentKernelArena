import torch
import triton
import triton.language as tl


@triton.jit
def reshape_and_cache_kernel_flash(
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    key_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    value_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping_ptr,  # [num_tokens]
    k_scale,  # float32
    v_scale,  # float32
    # strides
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    head_stride: tl.int64,
    dim_stride_k: tl.int64,
    dim_stride_v: tl.int64,
    page_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    x: tl.constexpr,
    USE_HEAD_MAJOR_LAYOUT: tl.constexpr,
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

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, TILE_SIZE)
    tile_pos = tile_i * TILE_SIZE + tile_offs
    src_key_idx = token_idx * key_stride
    src_value_idx = token_idx * value_stride

    if USE_HEAD_MAJOR_LAYOUT:
        # Decompose the tile index back into head and dim coordinates.
        cur_head = tile_pos // head_size
        cur_dim = tile_pos % head_size
        # Value addressing (4D): [Block, Head, Dim, Slot]
        tgt_idx_v = (
            block_idx * block_stride
            + cur_head * head_stride
            + cur_dim * dim_stride_v
            + block_offset * 1
        )
        # Key addressing (5D): [Block, Head, Dim//8, Slot, 8]
        tgt_idx_k = (
            block_idx * block_stride
            + cur_head * head_stride
            + (cur_dim // x) * dim_stride_k
            + block_offset * x
            + (cur_dim % x)
        )
    else:
        tgt_base = block_idx * block_stride + block_offset * page_stride
        tgt_idx_k = tgt_base + tile_pos
        tgt_idx_v = tgt_base + tile_pos

    # [TILE_SIZE]
    key_load = tl.load(
        key_ptr + src_key_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
    )
    if FP8_KV_CACHE:
        key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
    else:
        key_tile = key_load

    # [TILE_SIZE]
    value_load = tl.load(
        value_ptr + src_value_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
    )
    if FP8_KV_CACHE:
        if value_load.dtype.is_fp8():
            value_tile = value_load
        else:
            value_tile = value_load / tl.load(v_scale)
    else:
        value_tile = value_load

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


def reshape_and_cache_flash(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_heads, head_size]
    value_cache: torch.Tensor,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping: torch.Tensor,  # [num_tokens]
    kv_cache_dtype: str = "auto",  # "auto" or "fp8"
    k_scale: torch.Tensor = None,  # float32
    v_scale: torch.Tensor = None,  # float32
):
    """Scatter K/V tokens into paged cache slots.

    Args:
        key: [num_tokens, num_heads, head_size]
        value: [num_tokens, num_heads, head_size]
        key_cache: [num_blocks, block_size, num_heads, head_size]
        value_cache: [num_blocks, block_size, num_heads, head_size]
        slot_mapping: [num_tokens] - maps each token to a cache slot
        kv_cache_dtype: "auto" (no conversion) or "fp8" (scale and cast)
        k_scale: scaling factor for FP8 key cache
        v_scale: scaling factor for FP8 value cache
    """
    num_heads = key.shape[1]
    head_size = key.shape[2]

    use_head_major_layout = key_cache.ndim == 5
    if use_head_major_layout:
        block_size = key_cache.shape[3]
        x = key_cache.shape[4]
        head_stride = key_cache.stride(1)
        dim_stride_k = key_cache.stride(2)
        dim_stride_v = value_cache.stride(2)
    else:
        block_size = key_cache.shape[1]
        x = 1
        dim_stride_k = 0
        dim_stride_v = 0
        head_stride = key_cache.stride()[2]

    n = num_heads * head_size
    key_stride = key.stride()[0]
    value_stride = value.stride()[0]
    block_stride = key_cache.stride()[0]
    page_stride = key_cache.stride()[1]

    if k_scale is None:
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=key.device)
    if v_scale is None:
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=key.device)

    FP8_KV_CACHE = kv_cache_dtype.startswith("fp8") if kv_cache_dtype != "auto" else False

    # heuristics
    TILE_SIZE = min(2048, triton.next_power_of_2(n))
    num_stages = 10
    num_warps = 16

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
        k_scale=k_scale,
        v_scale=v_scale,
        key_stride=key_stride,
        value_stride=value_stride,
        block_stride=block_stride,
        head_stride=head_stride,
        dim_stride_k=dim_stride_k,
        dim_stride_v=dim_stride_v,
        page_stride=page_stride,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        x=x,
        USE_HEAD_MAJOR_LAYOUT=use_head_major_layout,
        FP8_KV_CACHE=FP8_KV_CACHE,
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
