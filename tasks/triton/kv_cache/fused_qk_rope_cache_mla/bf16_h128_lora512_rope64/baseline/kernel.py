# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused RoPE + concat + MLA KV-cache write, vendored from ROCm/aiter.

Kernels: aiter/ops/triton/_triton_kernels/fusions/fused_kv_cache.py
(_fused_qk_rope_cat_and_cache_mla_kernel, _unit_rope), _triton_kernels/kv_cache.py
(_store_mla_kv_cache, FP4 branches removed) and _triton_kernels/rope/rope.py
(rotation helpers). Launch code follows aiter/ops/triton/fusions/fused_kv_cache.py
(fused_qk_rope_cat_and_cache_mla) for a bfloat16, non-shuffled cache.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _get_neox_rotated_x_1D(
    x,
    x_rotated_mask,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    x_rotated = tl.where(x_rotated_mask, x, -x)
    x_rotated = tl.reshape(x_rotated, (2, BLOCK_D_HALF))
    x_rotated = tl.flip(x_rotated, 1)
    x_rotated = tl.reshape(x_rotated, (BLOCK_D,))
    x_rotated = tl.flip(x_rotated, 0)
    return x_rotated


@triton.jit
def _get_gptj_rotated_x_1D(
    x,
    x_rotated_mask,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    x_rotated = tl.where(x_rotated_mask, x, -x)
    x_rotated = tl.reshape(x_rotated, (BLOCK_D_HALF, 2))
    x_rotated = tl.flip(x_rotated, 1)
    x_rotated = tl.reshape(x_rotated, (BLOCK_D,))
    return x_rotated


@triton.jit
def _store_mla_kv_cache(
    kv_cache_ptr,
    pid_t_slot,
    pid_hk,
    pid_blk,
    d_nope_offs,
    d_pe_offs,
    kv_cache_stride_b,
    kv_cache_stride_h,
    kv_cache_stride_d,
    k_nope,
    k_pe,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SHUFFLED_KV_CACHE: tl.constexpr,
    SCALE_K_WIDTH_NOPE: tl.constexpr,
    SCALE_K_WIDTH_ROPE: tl.constexpr,
):
    if SHUFFLED_KV_CACHE:
        if kv_cache_ptr.dtype.element_ty == tl.bfloat16:
            # BF16
            K_WIDTH: tl.constexpr = 8
        else:
            # FP8 E4M3 or packed FP4 E2M1
            K_WIDTH: tl.constexpr = 16

        BLOCK_D_nope_STORE: tl.constexpr = BLOCK_D_nope
        BLOCK_D_pe_STORE: tl.constexpr = BLOCK_D_pe

        d_nope_offs_shfl = tl.arange(0, BLOCK_D_nope_STORE // K_WIDTH).to(tl.int64)
        d_pe_offs_shfl = tl.arange(0, BLOCK_D_pe_STORE // K_WIDTH).to(tl.int64)
        k_width_shfl = tl.arange(0, K_WIDTH).to(tl.int64)
        k_nope = k_nope.reshape((BLOCK_D_nope_STORE // K_WIDTH, K_WIDTH))
        k_pe = k_pe.reshape((BLOCK_D_pe_STORE // K_WIDTH, K_WIDTH))

        kv_cache_ptrs = (
            kv_cache_ptr + pid_t_slot * kv_cache_stride_b + pid_hk * kv_cache_stride_h
        )
        kv_cache_nope_offs = (
            (pid_blk // 16) * BLOCK_D_nope_STORE * 16
            + (pid_blk % 16) * K_WIDTH
            + d_nope_offs_shfl[:, None] * K_WIDTH * 16
            + k_width_shfl[None, :]
        ) * kv_cache_stride_d

        nope_scale_offset: tl.constexpr = 0
        kv_cache_pe_offs = (
            BLOCK_SIZE * (BLOCK_D_nope_STORE + nope_scale_offset)
            + (pid_blk // 16) * BLOCK_D_pe_STORE * 16
            + (pid_blk % 16) * K_WIDTH
            + d_pe_offs_shfl[:, None] * K_WIDTH * 16
            + k_width_shfl[None, :]
        ) * kv_cache_stride_d

        tl.store(
            kv_cache_ptrs + kv_cache_nope_offs, k_nope.to(kv_cache_ptr.dtype.element_ty)
        )
        tl.store(
            kv_cache_ptrs + kv_cache_pe_offs, k_pe.to(kv_cache_ptr.dtype.element_ty)
        )
    else:
        # non-shuffled KV cache
        kv_cache_ptrs = (
            kv_cache_ptr + pid_t_slot * kv_cache_stride_b + pid_hk * kv_cache_stride_h
        )
        kv_cache_nope_offs = d_nope_offs * kv_cache_stride_d
        kv_cache_pe_offs = (d_pe_offs + BLOCK_D_nope) * kv_cache_stride_d
        tl.store(
            kv_cache_ptrs + kv_cache_nope_offs, k_nope.to(kv_cache_ptr.dtype.element_ty)
        )
        tl.store(
            kv_cache_ptrs + kv_cache_pe_offs, k_pe.to(kv_cache_ptr.dtype.element_ty)
        )


@triton.jit
def _unit_rope(
    x_ptrs,
    cos,
    sin,
    d_pe_offs,
    IS_NEOX: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
):
    x_pe = tl.load(x_ptrs)

    if IS_NEOX:
        x_rotated_mask = d_pe_offs < BLOCK_D_HALF_pe
        x_pe_rotated = _get_neox_rotated_x_1D(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )
    else:
        x_rotated_mask = d_pe_offs % 2 == 0
        x_pe_rotated = _get_gptj_rotated_x_1D(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )

    x_pe = x_pe * cos + x_pe_rotated * sin

    return x_pe


@triton.jit
def _fused_qk_rope_cat_and_cache_mla_kernel(
    q_nope_ptr,
    q_pe_ptr,
    k_nope_ptr,
    k_pe_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    decode_q_pe_out_ptr,
    k_pe_out_ptr,
    q_nope_zeros_out_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    B,
    B_slot,
    num_decode_toks_for_zeros,
    q_nope_stride_b,
    q_nope_stride_h,
    q_nope_stride_d,
    q_pe_stride_b,
    q_pe_stride_h,
    q_pe_stride_d,
    k_nope_stride_b,
    k_nope_stride_h,
    k_nope_stride_d,
    k_pe_stride_b,
    k_pe_stride_h,
    k_pe_stride_d,
    pos_stride_b,
    cos_stride_b,
    cos_stride_d,
    q_out_stride_b,
    q_out_stride_h,
    q_out_stride_d,
    decode_q_pe_out_stride_b,
    decode_q_pe_out_stride_h,
    decode_q_pe_out_stride_d,
    k_pe_out_stride_b,
    k_pe_out_stride_h,
    k_pe_out_stride_d,
    q_nope_zeros_out_stride_b,
    q_nope_zeros_out_stride_h,
    q_nope_zeros_out_stride_d,
    kv_cache_stride_b,
    kv_cache_stride_h,
    kv_cache_stride_d,
    k_scale_ptr,
    QH_PER_KH: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 1,
    SHUFFLED_KV_CACHE: tl.constexpr = False,
    SCALE_K_WIDTH_NOPE: tl.constexpr = 4,
    SCALE_K_WIDTH_ROPE: tl.constexpr = 4,
    OUTPUT_Q_NOPE_ZEROS_AND_Q_PE: tl.constexpr = False,
    HAVE_K_SCALE: tl.constexpr = False,
    UPCAST_OPERAND: tl.constexpr = False,
):
    pid = tl.program_id(0)

    d_nope_offs = tl.arange(0, BLOCK_D_nope).to(tl.int64)
    d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

    if pid < B * QH:
        # pid_b = pid // QH
        # pid_hq = pid % QH
        # This is a new optimization that prioritized heavy workload WGs first
        pid_hq = pid // B
        pid_b = pid % B

        if REUSE_FREQS_FRONT_PART:
            if IS_NEOX:
                d_cos_offs = d_pe_offs
                d_cos_offs = tl.where(
                    (d_cos_offs >= BLOCK_D_HALF_pe) & (d_cos_offs < BLOCK_D_pe),
                    d_cos_offs - BLOCK_D_HALF_pe,
                    d_cos_offs,
                ).to(d_cos_offs.dtype)
            else:
                d_cos_offs = d_pe_offs // 2
        else:
            d_cos_offs = d_pe_offs

        pos = tl.load(pos_ptr + pid_b * pos_stride_b)
        cos_offs = pos * cos_stride_b + d_cos_offs * cos_stride_d
        cos = tl.load(cos_ptr + cos_offs)
        sin = tl.load(sin_ptr + cos_offs)
        if UPCAST_OPERAND:
            cos = cos.to(tl.float32)
            sin = sin.to(tl.float32)

        q_nope_ptrs = (
            q_nope_ptr
            + pid_b * q_nope_stride_b
            + pid_hq * q_nope_stride_h
            + d_nope_offs * q_nope_stride_d
        )
        q_pe_ptrs = (
            q_pe_ptr
            + pid_b * q_pe_stride_b
            + pid_hq * q_pe_stride_h
            + d_pe_offs * q_pe_stride_d
        )
        q_out_ptrs = q_out_ptr + pid_b * q_out_stride_b + pid_hq * q_out_stride_h
        q_nope = tl.load(q_nope_ptrs)
        q_pe = _unit_rope(
            q_pe_ptrs,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            BLOCK_D_pe,
            BLOCK_D_HALF_pe,
        )
        tl.store(
            q_out_ptrs + d_nope_offs * q_out_stride_d,
            q_nope.to(q_out_ptr.dtype.element_ty),
        )
        tl.store(
            q_out_ptrs + (d_pe_offs + BLOCK_D_nope) * q_out_stride_d,
            q_pe.to(q_out_ptr.dtype.element_ty),
        )

        if OUTPUT_Q_NOPE_ZEROS_AND_Q_PE and pid < num_decode_toks_for_zeros * QH:
            decode_q_pe_out_ptrs = (
                decode_q_pe_out_ptr
                + pid_b * decode_q_pe_out_stride_b
                + pid_hq * decode_q_pe_out_stride_h
            )
            tl.store(
                decode_q_pe_out_ptrs + d_pe_offs * decode_q_pe_out_stride_d,
                q_pe.to(decode_q_pe_out_ptr.dtype.element_ty),
            )
            z = tl.zeros((BLOCK_D_nope,), dtype=q_nope_zeros_out_ptr.dtype.element_ty)
            tl.store(
                q_nope_zeros_out_ptr
                + pid_b * q_nope_zeros_out_stride_b
                + pid_hq * q_nope_zeros_out_stride_h
                + d_nope_offs * q_nope_zeros_out_stride_d,
                z,
            )

        # pid_hk = pid_hq // QH_PER_KH
        # is_kv = pid_hq % QH_PER_KH == 0
        # This is a new optimization that prioritized heavy workload WGs first
        pid_hk = pid_hq
        is_kv = pid_hk < KH
        if is_kv:
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
            if pid_slot >= 0:
                if BLOCK_SIZE > 1:
                    pid_t_slot = pid_slot // BLOCK_SIZE
                    pid_blk = pid_slot % BLOCK_SIZE
                else:
                    pid_t_slot = pid_slot
                    pid_blk = 0
                if BLOCK_SIZE > 1:
                    pid_t_slot = pid_slot // BLOCK_SIZE
                    pid_blk = pid_slot % BLOCK_SIZE
                else:
                    pid_t_slot = pid_slot
                    pid_blk = 0
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1

                k_nope_ptrs = (
                    k_nope_ptr
                    + pid_b * k_nope_stride_b
                    + pid_hk * k_nope_stride_h
                    + d_nope_offs * k_nope_stride_d
                )
                k_pe_ptrs = (
                    k_pe_ptr
                    + pid_b * k_pe_stride_b
                    + pid_hk * k_pe_stride_h
                    + d_pe_offs * k_pe_stride_d
                )
                k_pe_out_ptrs = (
                    k_pe_out_ptr
                    + pid_b * k_pe_out_stride_b
                    + pid_hk * k_pe_out_stride_h
                    + d_pe_offs * k_pe_out_stride_d
                )
                k_nope = tl.load(k_nope_ptrs)
                k_pe = _unit_rope(
                    k_pe_ptrs,
                    cos,
                    sin,
                    d_pe_offs,
                    IS_NEOX,
                    BLOCK_D_pe,
                    BLOCK_D_HALF_pe,
                )
                tl.store(k_pe_out_ptrs, k_pe.to(k_pe_out_ptr.dtype.element_ty))
                k_scale_rcprl = (1 / k_scale).to(tl.float32)
                k_nope = k_nope.to(tl.float32) * k_scale_rcprl
                k_pe = k_pe.to(tl.float32) * k_scale_rcprl

                _store_mla_kv_cache(
                    kv_cache_ptr,
                    pid_t_slot,
                    pid_hk,
                    pid_blk,
                    d_nope_offs,
                    d_pe_offs,
                    kv_cache_stride_b,
                    kv_cache_stride_h,
                    kv_cache_stride_d,
                    k_nope,
                    k_pe,
                    BLOCK_D_nope,
                    BLOCK_D_pe,
                    BLOCK_SIZE,
                    SHUFFLED_KV_CACHE,
                    SCALE_K_WIDTH_NOPE,
                    SCALE_K_WIDTH_ROPE,
                )
    else:
        pid = pid - B * QH + B * KH
        if pid < B_slot * KH:
            pid_b = pid // KH
            pid_hk = pid % KH
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
            if pid_slot >= 0:
                if BLOCK_SIZE > 1:
                    pid_t_slot = pid_slot // BLOCK_SIZE
                    pid_blk = pid_slot % BLOCK_SIZE
                else:
                    pid_t_slot = pid_slot
                    pid_blk = 0
                if BLOCK_SIZE > 1:
                    pid_t_slot = pid_slot // BLOCK_SIZE
                    pid_blk = pid_slot % BLOCK_SIZE
                else:
                    pid_t_slot = pid_slot
                    pid_blk = 0
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1

                k_nope_ptrs = (
                    k_nope_ptr
                    + pid_b * k_nope_stride_b
                    + pid_hk * k_nope_stride_h
                    + d_nope_offs * k_nope_stride_d
                )
                k_pe_ptrs = (
                    k_pe_ptr
                    + pid_b * k_pe_stride_b
                    + pid_hk * k_pe_stride_h
                    + d_pe_offs * k_pe_stride_d
                )
                k_pe_out_ptrs = (
                    k_pe_out_ptr
                    + pid_b * k_pe_out_stride_b
                    + pid_hk * k_pe_out_stride_h
                    + d_pe_offs * k_pe_out_stride_d
                )
                k_nope = tl.load(k_nope_ptrs)
                k_pe = tl.load(k_pe_ptrs)
                tl.store(k_pe_out_ptrs, k_pe.to(k_pe_out_ptr.dtype.element_ty))
                k_scale_rcprl = (1 / k_scale).to(tl.float32)
                k_nope = k_nope.to(tl.float32) * k_scale_rcprl
                k_pe = k_pe.to(tl.float32) * k_scale_rcprl

                _store_mla_kv_cache(
                    kv_cache_ptr,
                    pid_t_slot,
                    pid_hk,
                    pid_blk,
                    d_nope_offs,
                    d_pe_offs,
                    kv_cache_stride_b,
                    kv_cache_stride_h,
                    kv_cache_stride_d,
                    k_nope,
                    k_pe,
                    BLOCK_D_nope,
                    BLOCK_D_pe,
                    BLOCK_SIZE,
                    SHUFFLED_KV_CACHE,
                    SCALE_K_WIDTH_NOPE,
                    SCALE_K_WIDTH_ROPE,
                )


def run(*, q_nope, q_pe, k_nope, k_pe, positions, slot_mapping, cos, sin, kv_cache, q_out, k_pe_out):
    b, qh, d_nope = q_nope.shape
    d_pe = q_pe.shape[-1]
    _, kh, _ = k_nope.shape
    reuse_freqs_front_part = cos.shape[-1] == d_pe // 2
    # Decode-copy outputs, the zeros output and the scale are disabled (constexpr flags below);
    # the pointers only need a valid tensor of the right element type.
    grid = (b * qh, 1, 1)
    _fused_qk_rope_cat_and_cache_mla_kernel[grid](
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        positions,
        cos,
        sin,
        q_out,
        q_out,
        k_pe_out,
        q_out,
        kv_cache,
        slot_mapping,
        b,
        b,
        0,
        *q_nope.stride(),
        *q_pe.stride(),
        *k_nope.stride(),
        *k_pe.stride(),
        positions.stride(0),
        cos.stride(0),
        cos.stride(-1),
        *q_out.stride(),
        *q_out.stride(),
        *k_pe_out.stride(),
        *q_out.stride(),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        k_scale_ptr=cos,
        QH_PER_KH=qh // kh,
        QH=qh,
        KH=kh,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=False,
        BLOCK_D_nope=d_nope,
        BLOCK_D_pe=d_pe,
        BLOCK_D_HALF_pe=d_pe // 2,
        BLOCK_SIZE=1,
        SHUFFLED_KV_CACHE=False,
        SCALE_K_WIDTH_NOPE=4,
        SCALE_K_WIDTH_ROPE=4,
        OUTPUT_Q_NOPE_ZEROS_AND_Q_PE=False,
        HAVE_K_SCALE=False,
        UPCAST_OPERAND=False,
        num_warps=1,
    )
