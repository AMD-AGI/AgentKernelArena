"""Lightning Attention - Linear attention decode kernel."""
import torch
import triton
import triton.language as tl


@triton.jit
def _linear_attn_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kv_cache_ptr,
    slope_rate,
    slot_idx,
    output_ptr,
    D: tl.constexpr,
    D_v: tl.constexpr,
    qkv_b_stride,
    qkv_h_stride,
    v_b_stride,
    v_h_stride,
    out_b_stride,
    out_h_stride,
    cache_b_stride,
    cache_h_stride,
    cache_d0_stride,
    cache_d1_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for linear attention decoding with KV cache.

    This kernel computes attention for a single token using the KV cache.
    """
    pid_b = tl.program_id(0)  # batch index
    pid_h = tl.program_id(1)  # head index
    pid_d = tl.program_id(2)  # dimension block index

    # Load slot index for the current batch
    slot_id = tl.load(slot_idx + pid_b).to(tl.int64)

    # Skip if slot_id is -1 (padding)
    if slot_id == -1:
        return

    batch_id = pid_b
    head_id = pid_h

    # Load decay rate for the current head
    ratio = tl.load(slope_rate + pid_h)

    # Calculate offsets for dimensions
    qk_d_offsets = tl.arange(0, D)
    v_d_offsets = tl.arange(0, BLOCK_SIZE) + pid_d * BLOCK_SIZE
    cache_d_offsets = (
        qk_d_offsets[:, None] * cache_d0_stride + v_d_offsets[None, :] * cache_d1_stride
    )

    # Calculate offsets for the current batch and head
    q_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    k_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride

    cache_offset = slot_id * cache_b_stride + head_id * cache_h_stride

    # Create masks for loading tensors
    qk_mask = qk_d_offsets < D
    v_mask = v_d_offsets < D_v

    # Load query, key, and value tensors
    q = tl.load(q_ptr + q_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    k = tl.load(k_ptr + k_offset + qk_d_offsets, mask=qk_mask, other=0.0)

    v_offset = batch_id * v_b_stride + head_id * v_h_stride
    v = tl.load(v_ptr + v_offset + v_d_offsets, mask=v_mask, other=0.0)

    # Compute key-value outer product
    kv_outer = k[:, None] * v[None, :]
    kv_mask = qk_mask[:, None] & v_mask[None, :]

    # Apply decay to previous KV cache
    ratio = tl.exp(-ratio)
    kv_ptr = kv_cache_ptr + cache_offset + cache_d_offsets
    kv_cache_old = tl.load(kv_ptr, mask=kv_mask, other=0.0)
    kv_outer = kv_outer + ratio * kv_cache_old

    # Compute attention output
    output = q[:, None].to(tl.float32) * kv_outer
    output = tl.sum(output, axis=0)

    # Update KV cache and store output
    tl.store(kv_ptr, kv_outer, mask=kv_mask)
    out_offset = batch_id * out_b_stride + head_id * out_h_stride
    tl.store(output_ptr + out_offset + v_d_offsets, output, mask=v_mask)


def linear_attn_decode_forward(q, k, v, kv_caches, slope_rate, slot_idx, BLOCK_SIZE=32):
    """
    Standalone wrapper for the linear attention decode kernel.

    Args:
        q: Query tensor [B, H, 1, D]
        k: Key tensor [B, H, 1, D]
        v: Value tensor [B, H, 1, D_v]
        kv_caches: KV cache tensor [num_slots, H, D, D_v] (modified in-place)
        slope_rate: Decay rate tensor [H]
        slot_idx: Slot indices [B] (int32/int64)
        BLOCK_SIZE: Block size for value dimension (default 32)

    Returns:
        output: Attention output [B, H, 1, D_v]
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    slope_rate = slope_rate.contiguous()
    slot_idx = slot_idx.contiguous()

    B, H, _, D = q.shape
    D_v = v.shape[-1]

    output = torch.empty(B, H, 1, D_v, device=q.device, dtype=q.dtype)

    grid = (B, H, D_v // BLOCK_SIZE)

    qkv_b_stride = q.stride(0)
    qkv_h_stride = q.stride(1)

    cache_b_stride = kv_caches.stride(0)
    cache_h_stride = kv_caches.stride(1)
    cache_d0_stride = kv_caches.stride(2)
    cache_d1_stride = kv_caches.stride(3)

    v_b_stride = v.stride(0)
    v_h_stride = v.stride(1)
    out_b_stride = output.stride(0)
    out_h_stride = output.stride(1)

    _linear_attn_decode_kernel[grid](
        q, k, v,
        kv_caches,
        slope_rate,
        slot_idx,
        output,
        D,
        D_v,
        qkv_b_stride,
        qkv_h_stride,
        v_b_stride,
        v_h_stride,
        out_b_stride,
        out_h_stride,
        cache_b_stride,
        cache_h_stride,
        cache_d0_stride,
        cache_d1_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
