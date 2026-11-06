# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm-project/vllm, vllm/model_executor/layers/lightning_attn.py
# (_linear_attn_decode_kernel, linear_decode_forward_triton).
"""MiniMax lightning-attention decode step in Triton.

Single-token linear attention with a per-head exponentially decayed KV state. For every batch
entry b with state slot slot_idx[b] and head h with decay slope s[h]:

    state = k (x) v + exp(-s) * state        # [D, D] outer product, float32 state, in place
    out   = q . state                        # contraction over the key dimension

Each program updates a [D, BLOCK_SIZE] tile of one (entry, head) state.
"""
import triton
import triton.language as tl

BLOCK_SIZE = 32
PAD_SLOT_ID = -1


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
    qkv_b_stride,
    qkv_h_stride,
    out_b_stride,
    out_h_stride,
    cache_b_stride,
    cache_h_stride,
    cache_d0_stride,
    cache_d1_stride,
    pad_slot_id: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index
    pid_h = tl.program_id(1)  # head index
    pid_d = tl.program_id(2)  # value-dimension block index

    # Load slot index for the current batch entry
    slot_id = tl.load(slot_idx + pid_b).to(tl.int64)

    # Skip if slot_id is the padding slot
    if slot_id == pad_slot_id:
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
    v_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride

    cache_offset = slot_id * cache_b_stride + head_id * cache_h_stride

    # Create masks for loading tensors
    qk_mask = qk_d_offsets < D
    v_mask = v_d_offsets < D

    # Load query, key, and value tensors
    q = tl.load(q_ptr + q_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    k = tl.load(k_ptr + k_offset + qk_d_offsets, mask=qk_mask, other=0.0)
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


def run(*, q, k, v, kv_cache, slope_rate, slot_idx, out):
    B, H, D = q.shape
    out3 = out.view(B, H, D)
    _linear_attn_decode_kernel[(B, H, D // BLOCK_SIZE)](
        q, k, v,
        kv_cache,
        slope_rate,
        slot_idx,
        out3,
        D,
        q.stride(0),
        q.stride(1),
        out3.stride(0),
        out3.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        pad_slot_id=PAD_SLOT_ID,
        BLOCK_SIZE=BLOCK_SIZE,
    )
