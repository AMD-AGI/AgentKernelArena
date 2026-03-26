#!/usr/bin/env python3
"""
Fused QK RoPE + KV Cache Kernel for MLA

Fused QK RoPE concatenation + KV cache write for MLA. Combines Q nope/pe RoPE,
K nope/pe RoPE, and cache store into a single kernel launch.

Primary benchmark target: fused_qk_rope_cosine_cache_llama
Also tests: fused_qk_rope_cat_and_cache_mla, fused_qk_rope_reshape_and_cache
"""

import torch
import math
import statistics
import logging

import triton
import triton.language as tl

_LOGGER = logging.getLogger("AITER_TRITON")


# ============================================================================
# INLINED TRITON KERNELS (from aiter.ops.triton)
# ============================================================================


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
def _fused_qk_rope_cosine_cache_llama_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    offs_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    q_out_ptr,
    T,
    T_slot,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_t,
    k_stride_h,
    k_stride_d,
    v_stride_t,
    v_stride_h,
    v_stride_d,
    cos_stride_t,
    cos_stride_d,
    q_out_stride_t,
    q_out_stride_h,
    q_out_stride_d,
    key_cache_stride_t,
    key_cache_stride_h,
    key_cache_stride_d,
    key_cache_stride_b,
    key_cache_stride_x,
    value_cache_stride_t,
    value_cache_stride_h,
    value_cache_stride_d,
    value_cache_stride_b,
    k_scale_ptr,
    v_scale_ptr,
    QH_PER_KH: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    X_SIZE: tl.constexpr,
    FLASH_LAYOUT: tl.constexpr,
    HAVE_POS: tl.constexpr = False,
    HAVE_K_SCALE: tl.constexpr = False,
    HAVE_V_SCALE: tl.constexpr = False,
):
    pid = tl.program_id(0)

    d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

    if pid < T * QH:
        pid_t = pid // QH
        pid_hq = pid % QH
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
                d_cos_mask = d_cos_offs < BLOCK_D_HALF_pe

        else:
            d_cos_offs = d_pe_offs

        pos = tl.load(pos_ptr + pid_t)
        if HAVE_POS:
            offset = tl.load(offs_ptr + pid_t)
            pos = pos + offset
        cos_offs = pos * cos_stride_t + d_cos_offs * cos_stride_d
        cos = tl.load(cos_ptr + cos_offs).to(tl.float64)
        sin = tl.load(sin_ptr + cos_offs).to(tl.float64)

        q_ptrs = (
            q_ptr + pid_t * q_stride_t + pid_hq * q_stride_h + d_pe_offs * q_stride_d
        )
        q_pe = _unit_rope(
            q_ptrs,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            BLOCK_D_pe,
            BLOCK_D_HALF_pe,
        )
        q_out_ptrs = (
            q_out_ptr
            + pid_t * q_out_stride_t
            + pid_hq * q_out_stride_h
            + d_pe_offs * q_out_stride_d
        )
        tl.store(q_out_ptrs, q_pe.to(q_out_ptr.dtype.element_ty))

        if pid_hq % QH_PER_KH == 0:
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)
            if pid_slot >= 0:
                pid_t_slot = pid_t
                pid_b = pid_slot
                pid_hk = pid_hq // QH_PER_KH
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1
                k_ptrs = (
                    k_ptr
                    + pid_t * k_stride_t
                    + pid_hk * k_stride_h
                    + d_pe_offs * k_stride_d
                )
                k_pe = _unit_rope(
                    k_ptrs,
                    cos,
                    sin,
                    d_pe_offs,
                    IS_NEOX,
                    BLOCK_D_pe,
                    BLOCK_D_HALF_pe,
                )

                k_scale_rcprl = 1 / k_scale
                k_pe = k_pe * k_scale_rcprl

                if FLASH_LAYOUT:
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_b * key_cache_stride_b
                        + pid_hk * key_cache_stride_h
                        + d_pe_offs * key_cache_stride_d
                    )
                else:
                    k_pe = tl.reshape(k_pe, (BLOCK_D_pe // X_SIZE, X_SIZE))
                    dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)
                    x_offs = tl.arange(0, X_SIZE).to(tl.int64)
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_hk * key_cache_stride_h
                        + dx_offs[:, None] * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + x_offs[None, :] * key_cache_stride_x
                    )

                tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))

                v_ptrs = (
                    v_ptr
                    + pid_t * v_stride_t
                    + pid_hk * v_stride_h
                    + d_pe_offs * v_stride_d
                )
                if HAVE_V_SCALE:
                    v_scale = tl.load(v_scale_ptr)
                else:
                    v_scale = 1
                v_scale_rcprl = 1 / v_scale
                v = tl.load(v_ptrs) * v_scale_rcprl
                v_out_ptrs = (
                    value_cache_ptr
                    + pid_t_slot * value_cache_stride_t
                    + pid_hk * value_cache_stride_h
                    + d_pe_offs * value_cache_stride_d
                    + pid_b * value_cache_stride_b
                )
                tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))
    else:
        pid = pid - T * QH + T * KH
        if pid < T_slot * KH:
            pid_t = pid // KH
            pid_hk = pid % KH
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)
            if pid_slot >= 0:
                pid_t_slot = pid_t
                pid_b = pid_slot
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1
                k_ptrs = (
                    k_ptr
                    + pid_t * k_stride_t
                    + pid_hk * k_stride_h
                    + d_pe_offs * k_stride_d
                )

                k_pe = tl.load(k_ptrs)

                k_scale_rcprl = 1 / k_scale
                k_pe = k_pe * k_scale_rcprl

                if FLASH_LAYOUT:
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + d_pe_offs * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + pid_hk * key_cache_stride_h
                    )
                else:
                    k_pe = tl.reshape(k_pe, (BLOCK_D_pe // X_SIZE, X_SIZE))
                    dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)
                    x_offs = tl.arange(0, X_SIZE).to(tl.int64)
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_hk * key_cache_stride_h
                        + dx_offs[:, None] * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + x_offs[None, :] * key_cache_stride_x
                    )
                tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))

                v_ptrs = (
                    v_ptr
                    + pid_t * v_stride_t
                    + pid_hk * v_stride_h
                    + d_pe_offs * v_stride_d
                )
                if HAVE_V_SCALE:
                    v_scale = tl.load(v_scale_ptr)
                else:
                    v_scale = 1
                v_scale_rcprl = 1 / v_scale
                v = tl.load(v_ptrs) * v_scale_rcprl
                v_out_ptrs = (
                    value_cache_ptr
                    + pid_t_slot * value_cache_stride_t
                    + pid_hk * value_cache_stride_h
                    + d_pe_offs * value_cache_stride_d
                    + pid_b * value_cache_stride_b
                )
                tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))


def fused_qk_rope_cosine_cache_llama(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pos: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    is_neox: bool,
    flash_layout: bool,
    apply_scale: bool = True,
    offs: torch.Tensor = None,
    q_out: torch.Tensor = None,
):
    _LOGGER.info(
        f"FUSED_QK_ROPE_COSINE_CACHE_LLAMA: q={tuple(q.shape)} k={tuple(k.shape)} "
        + f"pos={tuple(pos.shape)} cos={tuple(cos.shape)} sin={tuple(sin.shape)} key_cache={tuple(key_cache.shape)} value_cache={tuple(value_cache.shape)} slot_mapping={tuple(slot_mapping.shape)}"
    )

    t, qh, d = q.shape
    tk, kh, dk = k.shape
    tv, vh, dv = v.shape
    if flash_layout:
        t_cache, block_size, kh_cache, dk_cache = key_cache.shape
        t_cache_v, block_size_v, vh_cache, dv_cache = value_cache.shape
    else:
        t_cache, kh_cache, dkx_cache, block_size, x_cache = key_cache.shape
        t_cache_v, vh_cache, dv_cache, block_size_v = value_cache.shape
    (t_slot,) = slot_mapping.shape

    assert (
        t == tk == tv and t_slot <= tk
    ), f"Number of tokens should be identical for q, kand v. The number of tokens of slot_mapping should no more than that of q, k and v, {t=} {tk=} {tv=} {t_slot=}"
    assert (
        block_size == block_size_v
    ), f"block size should be identical for key_cache, and value_cache {block_size} {block_size_v}"
    assert (
        kh == vh == kh_cache == vh_cache
    ), "KV head should be identical for k, v, key_cache, and value_cache"
    assert (
        t_cache == t_cache_v
    ), "Number of tokens should be identical for key_cache, and value_cache"
    if flash_layout:
        assert (
            d == dk == dv == dk_cache == dv_cache
        ), "D dimension should be identical for q, k, and v"
    else:
        assert (
            d == dk == dv == dkx_cache * x_cache == dv_cache
        ), "D dimension should be identical for q, k, and v"
        assert x_cache == triton.next_power_of_2(x_cache), "x_size should be power of 2"

    assert d == triton.next_power_of_2(d), "D dimension should be power of 2"
    assert qh % kh == 0, "Q heads must be multiple of H heads"
    d_freq = cos.shape[-1]
    assert (d_freq == d // 2) or (
        d_freq == d
    ), "cos/sin last dim should be the same or half of the qk last dim"
    reuse_freqs_front_part = d_freq == d // 2

    if q_out is None:
        q_out = torch.empty((t, qh, d), dtype=q.dtype, device=q.device)

    n_pid = t * qh + (t_slot - t) * kh
    grid = (n_pid, 1, 1)
    _fused_qk_rope_cosine_cache_llama_kernel[grid](
        q,
        k,
        v,
        pos,
        cos,
        sin,
        offs,
        key_cache,
        value_cache,
        slot_mapping,
        q_out,
        t,
        t_slot,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        cos.stride(0),
        cos.stride(-1),
        *q_out.stride(),
        key_cache.stride(0) if not flash_layout else key_cache.stride(0),
        key_cache.stride(1) if not flash_layout else key_cache.stride(2),
        key_cache.stride(2) if not flash_layout else key_cache.stride(3),
        key_cache.stride(3) if not flash_layout else key_cache.stride(1),
        key_cache.stride(4) if not flash_layout else 0,
        value_cache.stride(0) if not flash_layout else value_cache.stride(0),
        value_cache.stride(1) if not flash_layout else value_cache.stride(2),
        value_cache.stride(2) if not flash_layout else value_cache.stride(3),
        value_cache.stride(3) if not flash_layout else value_cache.stride(1),
        k_scale_ptr=k_scale,
        v_scale_ptr=v_scale,
        QH_PER_KH=qh // kh,
        QH=qh,
        KH=kh,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        BLOCK_D_pe=d,
        BLOCK_D_HALF_pe=d // 2,
        BLOCK_SIZE=block_size,
        X_SIZE=x_cache if not flash_layout else 0,
        FLASH_LAYOUT=flash_layout,
        HAVE_POS=(offs is not None),
        HAVE_K_SCALE=(k_scale is not None and apply_scale),
        HAVE_V_SCALE=(v_scale is not None and apply_scale),
        num_warps=1,
    )
    return q_out, key_cache, value_cache

# ============================================================================
# INPUT GENERATION
# ============================================================================

BLOCK_SIZE = 16
DTYPE = torch.bfloat16


def _generate_rope_freqs(T, D, device="cuda"):
    """Generate RoPE frequency tensors for testing."""
    freqs = torch.randn(T, 1, 1, D // 2, dtype=torch.float32, device=device)
    cos = torch.cos(freqs).squeeze(1).squeeze(1)
    sin = torch.sin(freqs).squeeze(1).squeeze(1)
    return cos, sin


def _generate_llama_inputs(T, QH_per_KH, KH, D, seed=42, device="cuda"):
    """Generate inputs for fused_qk_rope_cosine_cache_llama."""
    torch.manual_seed(seed)
    QH = QH_per_KH * KH
    num_kv_cache_tokens = max(T, 128)

    q = torch.randn(T, QH, D, dtype=DTYPE, device=device)
    k = torch.randn(T, KH, D, dtype=DTYPE, device=device)
    v = torch.randn(T, KH, D, dtype=DTYPE, device=device)

    cos, sin = _generate_rope_freqs(num_kv_cache_tokens, D, device)
    positions = torch.arange(T, device=device, dtype=torch.int64)

    key_cache = torch.zeros(T, num_kv_cache_tokens, KH, D, dtype=DTYPE, device=device)
    value_cache = torch.zeros(T, num_kv_cache_tokens, KH, D, dtype=DTYPE, device=device)

    k_scale = torch.ones(1, dtype=torch.float32, device=device)[0]
    v_scale = torch.ones(1, dtype=torch.float32, device=device)[0]
    slot_mapping = torch.randperm(T, device=device)

    return (q, k, v, key_cache, value_cache, slot_mapping, positions,
            cos, sin, k_scale, v_scale)


# ============================================================================
# REFERENCE IMPLEMENTATION
# ============================================================================


def _rotate_half_gptj(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _ref_rope_fwd(x, cos, sin):
    """Simple RoPE forward for GPTJ style with reuse_freqs_front_part."""
    x_f32 = x.float()
    D = x.shape[-1]
    cos_expanded = cos[:, :D // 2].repeat_interleave(2, dim=-1)
    sin_expanded = sin[:, :D // 2].repeat_interleave(2, dim=-1)

    if x_f32.dim() == 3:
        cos_expanded = cos_expanded.unsqueeze(1)
        sin_expanded = sin_expanded.unsqueeze(1)

    return (x_f32 * cos_expanded + _rotate_half_gptj(x_f32) * sin_expanded).to(x.dtype)


# ============================================================================
# ENTRY POINTS
# ============================================================================


def triton_op(q, k, v, key_cache, value_cache, slot_mapping, positions,
              cos, sin, k_scale, v_scale):
    kc = key_cache.clone()
    vc = value_cache.clone()
    q_out, kc_out, vc_out = fused_qk_rope_cosine_cache_llama(
        q, k, v, kc, vc, slot_mapping, positions, cos, sin,
        k_scale, v_scale, False,
        flash_layout=True, apply_scale=False, offs=None, q_out=q.clone(),
    )
    return q_out


def torch_op(q, k, v, key_cache, value_cache, slot_mapping, positions,
             cos, sin, k_scale, v_scale):
    pos_cos = cos[positions]
    pos_sin = sin[positions]
    return _ref_rope_fwd(q, pos_cos, pos_sin)


# ============================================================================
# TEST CONFIGURATIONS (from GEAK harness test discovery)
# ============================================================================

# (T, QH_per_KH, KH, D) — from benchmark_baseline.txt
EVAL_CONFIGS = [
    (1, 1, 1, 64),
    (1, 1, 1, 128),
    (4, 1, 1, 64),
    (2, 4, 1, 64),
    (4, 1, 1, 128),
    (2, 4, 1, 128),
    (1, 1, 8, 128),
    (1, 16, 1, 64),
    (2, 16, 1, 64),
    (4, 1, 8, 64),
    (1, 16, 1, 128),
    (1, 4, 8, 128),
    (4, 1, 8, 128),
    (2, 4, 8, 128),
    (128, 1, 1, 64),
    (4, 16, 1, 128),
    (4, 4, 8, 128),
    (1, 16, 8, 128),
    (128, 4, 1, 64),
    (128, 1, 8, 64),
    (4, 16, 8, 128),
    (128, 1, 8, 128),
    (128, 4, 8, 64),
    (128, 4, 8, 128),
    (2048, 16, 1, 64),
]

PROFILE_CONFIGS = [
    (1, 1, 1, 64),
    (1, 1, 8, 128),
    (4, 1, 8, 128),
    (128, 4, 1, 64),
    (2048, 16, 1, 64),
]

RTOL, ATOL = 1e-1, 1e-1


# ============================================================================
# TEST HARNESS
# ============================================================================


def get_inputs(T, QH_per_KH, KH, D, device="cuda"):
    return _generate_llama_inputs(T, QH_per_KH, KH, D, device=device)


def check_correctness(T, QH_per_KH, KH, D) -> dict:
    try:
        inputs = get_inputs(T, QH_per_KH, KH, D)
        res = triton_op(*inputs)
        ref = torch_op(*inputs)
        correct = torch.allclose(res, ref, rtol=RTOL, atol=ATOL)
        max_diff = torch.max(torch.abs(res - ref)).item() if not correct else 0.0
        return {"correct": correct, "max_diff": max_diff, "error": None}
    except Exception as e:
        return {"correct": False, "max_diff": float("inf"), "error": str(e)}


def _config_label(T, QH_per_KH, KH, D):
    return f"(T={T},QH/KH={QH_per_KH},KH={KH},D={D})"


BASELINE_LATENCIES = {
    (1, 1, 1, 64): 0.052,
    (1, 1, 1, 128): 0.0519,
    (4, 1, 1, 64): 0.0516,
    (2, 4, 1, 64): 0.0523,
    (4, 1, 1, 128): 0.0518,
    (2, 4, 1, 128): 0.0527,
    (1, 1, 8, 128): 0.0516,
    (1, 16, 1, 64): 0.0513,
    (2, 16, 1, 64): 0.0522,
    (4, 1, 8, 64): 0.052,
    (1, 16, 1, 128): 0.0519,
    (1, 4, 8, 128): 0.0524,
    (4, 1, 8, 128): 0.0524,
    (2, 4, 8, 128): 0.0518,
    (128, 1, 1, 64): 0.0528,
    (4, 16, 1, 128): 0.0518,
    (4, 4, 8, 128): 0.0528,
    (1, 16, 8, 128): 0.0523,
    (128, 4, 1, 64): 0.0543,
    (128, 1, 8, 64): 0.0521,
    (4, 16, 8, 128): 0.052,
    (128, 1, 8, 128): 0.0518,
    (128, 4, 8, 64): 0.0523,
    (128, 4, 8, 128): 0.0524,
    (2048, 16, 1, 64): 0.4408,
}


def benchmark_config(T, QH_per_KH, KH, D, warmup=100, iters=500) -> dict:
    import time

    cfg_key = (T, QH_per_KH, KH, D)
    inputs = get_inputs(T, QH_per_KH, KH, D)

    for _ in range(warmup):
        triton_op(*inputs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        triton_op(*inputs)
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - start) * 1000 / iters

    baseline_ms = BASELINE_LATENCIES.get(cfg_key, triton_ms)
    return {"torch_ms": baseline_ms, "triton_ms": triton_ms, "speedup": baseline_ms / triton_ms if triton_ms > 0 else 1.0}


def evaluate(configs=None, warmup=100, iters=500, verbose=True) -> dict:
    configs = configs or EVAL_CONFIGS
    results, failures = [], []

    if verbose:
        print(f"{'Config':<35} {'Correct':>8} {'Torch':>10} {'Triton':>10} {'Speedup':>10}")
        print("-" * 75)

    for cfg in configs:
        T, QH_per_KH, KH, D = cfg
        label = _config_label(*cfg)
        corr = check_correctness(*cfg)
        if not corr["correct"]:
            failures.append({"config": cfg, **corr})
            if verbose:
                err = corr["error"] or f"max_diff={corr['max_diff']:.2e}"
                print(f"{label:<35} {'FAIL':>8}   {err[:25]}")
            continue

        bench = benchmark_config(*cfg, warmup=warmup, iters=iters)
        results.append({"config": cfg, "correct": True, **bench})
        if verbose:
            marker = " *" if bench["speedup"] > 1.0 else ""
            print(f"{label:<35} {'PASS':>8} {bench['torch_ms']:>8.4f}ms {bench['triton_ms']:>8.4f}ms {bench['speedup']:>8.2f}x{marker}")

    total_baseline = sum(r["torch_ms"] for r in results)
    total_evolved = sum(r["triton_ms"] for r in results)
    speedup = total_baseline / total_evolved if total_evolved > 0 else 0.0

    if verbose:
        print("-" * 75)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(configs)})"
        print(f"{'Status:':<35} {status}")
        if results:
            print(f"{'Speedup (total):':<35} {speedup:.2f}x")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
        "results": results,
        "speedup_geomean": speedup,
    }


def run_profile(configs=None, warmup=3, iters=1, verbose=True):
    configs = configs or PROFILE_CONFIGS
    if verbose:
        print(f"Profile: {len(configs)} config(s)")
    for cfg in configs:
        T, QH_per_KH, KH, D = cfg
        inputs = get_inputs(*cfg)
        for _ in range(warmup):
            triton_op(*inputs)
        torch.cuda.synchronize()
        for _ in range(iters):
            triton_op(*inputs)
        torch.cuda.synchronize()
        if verbose:
            print(f"  {_config_label(*cfg)} done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fused QK RoPE + KV Cache MLA Kernel Test Harness")
    parser.add_argument("--profile", action="store_true", help="Run minimal profiling workload")
    args = parser.parse_args()

    print("=" * 75)
    print("Fused QK RoPE + KV Cache MLA Kernel")
    print("=" * 75)

    if args.profile:
        print("\n[Profile Mode]")
        run_profile()
    else:
        print("\n[Evaluation]")
        evaluate()

    print("=" * 75)
