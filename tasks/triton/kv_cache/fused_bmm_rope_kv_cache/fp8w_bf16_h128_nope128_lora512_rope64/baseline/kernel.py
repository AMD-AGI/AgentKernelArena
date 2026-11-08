# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused FP8 batched GEMM + RoPE + concat + MLA KV-cache write, vendored from ROCm/aiter.

Kernel: aiter/ops/triton/_triton_kernels/fusions/fused_bmm_rope_kv_cache.py
(_fused_fp8_bmm_rope_cat_and_cache_mla_kernel, _unit_rope). Helpers: the 1D rotation helpers of
aiter/ops/triton/_triton_kernels/rope/rope.py (SPDX MIT, Copyright (C) 2024-2026, Advanced Micro Devices,
Inc.) and make_kernel_repr of aiter/ops/triton/utils/_triton/kernel_repr.py (no header upstream; the
repository MIT license applies). Launch code follows fused_fp8_bmm_rope_cat_and_cache_mla in
aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py. The tuned table is the N=512, K=128 file
aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=512-K=128.json
inlined in the lookup order of aiter/ops/triton/utils/gemm_config_utils.py.
"""
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


def _sanitize_constexpr_value(value):
    if value is None:
        return "NONE"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)

    # for lists, tuples, sets - recursively join each
    if isinstance(value, (list, tuple, set)):
        items = sorted(value, key=str) if isinstance(value, set) else value
        sanitized_items = [_sanitize_constexpr_value(item) for item in items]
        joined = "_".join(sanitized_items)
        return joined if joined else "NONE"

    if isinstance(value, str):
        cleaned_value = "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")
        return cleaned_value.upper() if cleaned_value else "NONE"

    cleaned_value = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return cleaned_value.upper() if cleaned_value else "NONE"


def make_kernel_repr(base_name, config_keys, name_key=None):
    # When name_key is set, the base name is taken from the matching constexpr
    # kwarg at call time (falling back to base_name if missing/empty). Lets a
    # single shared kernel produce caller-specific names in compiled artifacts.
    def _repr(specialization):
        constants = specialization.constants

        name = base_name
        if name_key is not None:
            override = constants.get(name_key, None)
            if override:
                cleaned = "".join(
                    ch if ch.isalnum() or ch == "_" else "_" for ch in str(override)
                )
                if cleaned:
                    name = cleaned

        name_parts = []
        for key in config_keys:
            value = constants.get(key, None)
            symbol = _sanitize_constexpr_value(value)
            name_parts.append(f"{key}_{symbol}")

        if not name_parts:
            return name

        suffix = "_".join(name_parts)
        return f"{name}_{suffix}"

    return _repr


_fused_fp8_bmm_rope_cat_and_cache_mla_repr = make_kernel_repr(
    "_fused_fp8_bmm_rope_cat_and_cache_mla_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "QH_PER_KH",
        "REUSE_FREQS_FRONT_PART",
        "IS_NEOX",
        "BLOCK_D_nope",
        "BLOCK_DK_nope",
        "BLOCK_D_pe",
        "BLOCK_D_HALF_pe",
        "TRANSPOSE_BM",
        "OUTPUT_Q_NOPE_ZEROS",
        "HAVE_K_SCALE",
        "EVEN_K",
    ],
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
    """Apply RoPE to a vector. Copied from fused_kv_cache.py"""
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


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_fused_fp8_bmm_rope_cat_and_cache_mla_repr)
def _fused_fp8_bmm_rope_cat_and_cache_mla_kernel(
    a_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    q_pe_ptr,
    k_nope_ptr,
    k_pe_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    decode_q_pe_out_ptr,
    k_pe_out_ptr,
    q_nope_zeros_out_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    M,
    N,
    K,
    B,
    B_slot,
    num_decode_toks_for_zeros,
    QH,
    KH,
    bmm_programs,
    grid_mn,
    num_pid_m,
    num_pid_n,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    QH_PER_KH: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_DK_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    OUTPUT_Q_NOPE_ZEROS: tl.constexpr,
    HAVE_K_SCALE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr = ".ca",
):
    """
    Fused kernel for FP8 BMM + RoPE + KV cache write.

    These are INDEPENDENT operations fused to reduce kernel launch overhead:
    - BMM writes to q_out[:, :, :kv_lora_rank]
    - RoPE writes to q_out[:, :, kv_lora_rank:] and handles KV cache

    Note: FP8 does not support split-K.

    Grid structure:
    - Phase 1: pid < bmm_programs → BMM (tiled over heads, M, N)
    - Phase 2: pid in [bmm_programs, bmm_programs + B*QH) → RoPE + KV cache for decode
    - Phase 3: pid >= bmm_programs + B*QH → KV cache only for prefill tokens
    """

    pid = tl.program_id(0)
    tl.assume(pid >= 0)

    rope_start = bmm_programs
    prefill_start = bmm_programs + B * QH

    if pid < bmm_programs:
        stride_ab_i64 = tl.cast(stride_ab, tl.int64)
        stride_am_i64 = tl.cast(stride_am, tl.int64)
        stride_ak_i64 = tl.cast(stride_ak, tl.int64)
        stride_bb_i64 = tl.cast(stride_bb, tl.int64)
        stride_bk_i64 = tl.cast(stride_bk, tl.int64)
        stride_bn_i64 = tl.cast(stride_bn, tl.int64)
        stride_cb_i64 = tl.cast(stride_cb, tl.int64)
        stride_cm_i64 = tl.cast(stride_cm, tl.int64)
        stride_cn_i64 = tl.cast(stride_cn, tl.int64)

        tl.assume(stride_ab_i64 > 0)
        tl.assume(stride_am_i64 > 0)
        tl.assume(stride_ak_i64 > 0)
        tl.assume(stride_bb_i64 > 0)
        tl.assume(stride_bk_i64 > 0)
        tl.assume(stride_bn_i64 > 0)
        tl.assume(stride_cb_i64 > 0)
        tl.assume(stride_cm_i64 > 0)
        tl.assume(stride_cn_i64 > 0)

        pid_head = pid // GRID_MN
        pid_tile = pid % GRID_MN

        pid_head_i64 = tl.cast(pid_head, tl.int64)

        if GROUP_SIZE_M == 1:
            pid_m = pid_tile // num_pid_n
            pid_n = pid_tile % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid_tile // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (pid_tile % group_size_m)
            pid_n = (pid_tile % num_pid_in_group) // group_size_m

        pid_m_i64 = tl.cast(pid_m, tl.int64)
        pid_n_i64 = tl.cast(pid_n, tl.int64)

        tl.assume(pid_m_i64 >= 0)
        tl.assume(pid_n_i64 >= 0)
        tl.assume(pid_head_i64 >= 0)

        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        a_ptrs = a_ptr + (
            pid_head_i64 * stride_ab_i64
            + offs_am[:, None] * stride_am_i64
            + offs_k[None, :] * stride_ak_i64
        )
        b_ptrs = b_ptr + (
            pid_head_i64 * stride_bb_i64
            + offs_k[:, None] * stride_bk_i64
            + offs_bn[None, :] * stride_bn_i64
        )

        one_over_DTYPE_MAX = 1.0 / DTYPE_MAX
        b_scale = tl.load(b_scale_ptr)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_idx in range(tl.cdiv(K, BLOCK_SIZE_K)):
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < K - k_idx * BLOCK_SIZE_K, other=0.0
                )
                b = tl.load(
                    b_ptrs, mask=offs_k[:, None] < K - k_idx * BLOCK_SIZE_K, other=0.0
                )

            # Per-token group quantization
            m = tl.maximum(tl.max(tl.abs(a), axis=-1), 1e-10)[:, None]
            a_scale = m.to(tl.float32) * one_over_DTYPE_MAX
            a_scale_recip = 1.0 / a_scale
            a = tl.clamp(a * a_scale_recip, -DTYPE_MAX, DTYPE_MAX).to(
                b_ptr.dtype.element_ty
            )

            accumulator += tl.dot(a, b) * a_scale

            a_ptrs += BLOCK_SIZE_K * stride_ak_i64
            b_ptrs += BLOCK_SIZE_K * stride_bk_i64

        accumulator *= b_scale

        c = accumulator.to(c_ptr.type.element_ty)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        c_ptrs = (
            c_ptr
            + pid_head_i64 * stride_cb_i64
            + offs_cm[:, None] * stride_cm_i64
            + offs_cn[None, :] * stride_cn_i64
        )

        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    elif pid < prefill_start:
        pid_adjusted = pid - rope_start
        pid_b = pid_adjusted // QH
        pid_hq = pid_adjusted % QH

        tl.assume(pid_b >= 0)
        tl.assume(pid_hq >= 0)

        dk_nope_offs = tl.arange(0, BLOCK_DK_nope).to(tl.int64)
        d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

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

        q_pe_ptrs = (
            q_pe_ptr
            + pid_b * q_pe_stride_b
            + pid_hq * q_pe_stride_h
            + d_pe_offs * q_pe_stride_d
        )
        q_pe = _unit_rope(
            q_pe_ptrs,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            BLOCK_D_pe,
            BLOCK_D_HALF_pe,
        )

        q_out_base = c_ptr + pid_b * q_out_stride_b + pid_hq * q_out_stride_h
        tl.store(
            q_out_base + (d_pe_offs + BLOCK_D_nope) * q_out_stride_d,
            q_pe.to(c_ptr.dtype.element_ty),
        )

        if pid_adjusted < num_decode_toks_for_zeros * QH:
            decode_q_pe_out_ptrs = (
                decode_q_pe_out_ptr
                + pid_b * decode_q_pe_out_stride_b
                + pid_hq * decode_q_pe_out_stride_h
                + d_pe_offs * decode_q_pe_out_stride_d
            )
            tl.store(
                decode_q_pe_out_ptrs, q_pe.to(decode_q_pe_out_ptr.dtype.element_ty)
            )

        if OUTPUT_Q_NOPE_ZEROS and pid_adjusted < num_decode_toks_for_zeros * QH:
            z = tl.zeros((BLOCK_DK_nope,), dtype=q_nope_zeros_out_ptr.dtype.element_ty)
            tl.store(
                q_nope_zeros_out_ptr
                + pid_b * q_nope_zeros_out_stride_b
                + pid_hq * q_nope_zeros_out_stride_h
                + dk_nope_offs * q_nope_zeros_out_stride_d,
                z,
            )

        if pid_hq % QH_PER_KH == 0:
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
            if pid_slot >= 0:
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1

                pid_hk = pid_hq // QH_PER_KH

                k_nope_ptrs = (
                    k_nope_ptr
                    + pid_b * k_nope_stride_b
                    + pid_hk * k_nope_stride_h
                    + dk_nope_offs * k_nope_stride_d
                )
                k_nope = tl.load(k_nope_ptrs)

                k_pe_load_ptrs = (
                    k_pe_ptr
                    + pid_b * k_pe_stride_b
                    + pid_hk * k_pe_stride_h
                    + d_pe_offs * k_pe_stride_d
                )
                k_pe = _unit_rope(
                    k_pe_load_ptrs,
                    cos,
                    sin,
                    d_pe_offs,
                    IS_NEOX,
                    BLOCK_D_pe,
                    BLOCK_D_HALF_pe,
                )

                k_pe_out_ptrs = (
                    k_pe_out_ptr
                    + pid_b * k_pe_out_stride_b
                    + pid_hk * k_pe_out_stride_h
                    + d_pe_offs * k_pe_out_stride_d
                )
                tl.store(k_pe_out_ptrs, k_pe.to(k_pe_out_ptr.dtype.element_ty))

                k_scale_rcprl = (1 / k_scale).to(tl.float32)
                k_nope_scaled = (k_nope.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )
                k_pe_scaled = (k_pe.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )

                kv_cache_ptrs = (
                    kv_cache_ptr
                    + pid_slot * kv_cache_stride_b
                    + pid_hk * kv_cache_stride_h
                )

                tl.store(
                    kv_cache_ptrs + dk_nope_offs * kv_cache_stride_d, k_nope_scaled
                )
                tl.store(
                    kv_cache_ptrs + (d_pe_offs + BLOCK_DK_nope) * kv_cache_stride_d,
                    k_pe_scaled,
                )

    else:
        pid_adjusted = pid - prefill_start
        pid_b = pid_adjusted // KH + B
        pid_hk = pid_adjusted % KH

        if pid_b < B_slot:
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)

            if pid_slot >= 0:
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1

                dk_nope_offs = tl.arange(0, BLOCK_DK_nope).to(tl.int64)
                d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

                k_nope_ptrs = (
                    k_nope_ptr
                    + pid_b * k_nope_stride_b
                    + pid_hk * k_nope_stride_h
                    + dk_nope_offs * k_nope_stride_d
                )
                k_nope = tl.load(k_nope_ptrs)

                k_pe_load_ptrs = (
                    k_pe_ptr
                    + pid_b * k_pe_stride_b
                    + pid_hk * k_pe_stride_h
                    + d_pe_offs * k_pe_stride_d
                )
                k_pe = tl.load(k_pe_load_ptrs)

                k_pe_out_ptrs = (
                    k_pe_out_ptr
                    + pid_b * k_pe_out_stride_b
                    + pid_hk * k_pe_out_stride_h
                    + d_pe_offs * k_pe_out_stride_d
                )
                tl.store(k_pe_out_ptrs, k_pe.to(k_pe_out_ptr.dtype.element_ty))

                k_scale_rcprl = (1 / k_scale).to(tl.float32)
                k_nope_scaled = (k_nope.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )
                k_pe_scaled = (k_pe.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )

                kv_cache_ptrs = (
                    kv_cache_ptr
                    + pid_slot * kv_cache_stride_b
                    + pid_hk * kv_cache_stride_h
                )

                tl.store(
                    kv_cache_ptrs + dk_nope_offs * kv_cache_stride_d, k_nope_scaled
                )
                tl.store(
                    kv_cache_ptrs + (d_pe_offs + BLOCK_DK_nope) * kv_cache_stride_d,
                    k_pe_scaled,
                )


_CONFIGS = {
    "M_LEQ_16": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, GROUP_SIZE_M=1, num_warps=4, num_stages=2, waves_per_eu=6, matrix_instr_nonkdim=16, cache_modifier=".cg"),
    "M_LEQ_32": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=256, GROUP_SIZE_M=1, num_warps=8, num_stages=1, waves_per_eu=1, matrix_instr_nonkdim=32, cache_modifier=".cg"),
    "M_LEQ_64": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=256, GROUP_SIZE_M=1, num_warps=8, num_stages=1, waves_per_eu=1, matrix_instr_nonkdim=32, cache_modifier=".cg"),
    "M_LEQ_128": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=256, GROUP_SIZE_M=1, num_warps=8, num_stages=1, waves_per_eu=1, matrix_instr_nonkdim=32, cache_modifier=".cg"),
    "M_LEQ_256": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=256, GROUP_SIZE_M=1, num_warps=8, num_stages=1, waves_per_eu=1, matrix_instr_nonkdim=32, cache_modifier=".cg"),
    "any": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, GROUP_SIZE_M=1, num_warps=4, num_stages=1, waves_per_eu=6, matrix_instr_nonkdim=16, cache_modifier=".cg"),
}

_STANDARD_M_BOUNDS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)

_GROUP_SIZE = 128  # K group of the per-token activation quantization (the launcher's group_size default)
_DTYPE_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max, the launcher's DTYPE_MAX for the e4m3fn weights


def _get_config(M: int):
    """Row of the tuned table for M, in the lookup order of AITER's get_gemm_config."""
    for bound in _STANDARD_M_BOUNDS:
        key = f"M_LEQ_{bound}"
        if M <= bound and key in _CONFIGS:
            return dict(_CONFIGS[key])
    for bound in reversed(_STANDARD_M_BOUNDS):
        key = f"M_GEQ_{bound}"
        if M >= bound and key in _CONFIGS:
            return dict(_CONFIGS[key])
    return dict(_CONFIGS["any"])


def run(*, q_nope, w_k, w_k_scale, q_pe, k_nope, k_pe, positions, slot_mapping, cos, sin, kv_cache, q_out, k_pe_out) -> None:
    """MLA decode front end of fused_fp8_bmm_rope_cat_and_cache_mla (transpose_bm, no k_scale, bfloat16 cache).

    q_out[:, h, :512] = fp8-prequantized q_nope[h] @ w_k[h]^T * w_k_scale; q_out[:, :, 512:] = rope(q_pe); the
    latent head's [k_nope | rope(k_pe)] row goes to kv_cache[slot_mapping]; k_pe_out = rope(k_pe).
    """
    qh, b, p = q_nope.shape
    _, n, k = w_k.shape
    d_pe = q_pe.shape[-1]
    bk, kh, kv_lora_rank = k_nope.shape
    reuse_freqs_front_part = cos.shape[-1] == d_pe // 2

    w_k_t = w_k.transpose(1, 2)

    M = b
    N = kv_lora_rank
    K = k
    config = _get_config(M)
    config["BLOCK_SIZE_K"] = _GROUP_SIZE

    num_pid_m = triton.cdiv(M, config["BLOCK_SIZE_M"])
    num_pid_n = triton.cdiv(N, config["BLOCK_SIZE_N"])
    grid_mn = num_pid_m * num_pid_n
    bmm_programs = qh * grid_mn
    rope_programs = b * qh
    prefill_programs = (bk - b) * kh
    grid = (bmm_programs + rope_programs + prefill_programs, 1, 1)

    # No decode-copy or zeros outputs and no k_scale: the pointers below only need a valid tensor of
    # the right element type, because the matching branches are off (num_decode_toks_for_zeros = 0,
    # OUTPUT_Q_NOPE_ZEROS = HAVE_K_SCALE = False).
    _fused_fp8_bmm_rope_cat_and_cache_mla_kernel[grid](
        q_nope,
        w_k_t,
        w_k_scale,
        q_out,
        q_pe,
        k_nope,
        k_pe,
        positions,
        cos,
        sin,
        q_out,
        k_pe_out,
        q_out,
        kv_cache,
        slot_mapping,
        M,
        N,
        K,
        b,
        bk,
        0,
        qh,
        kh,
        bmm_programs,
        grid_mn,
        num_pid_m,
        num_pid_n,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        w_k_t.stride(0),
        w_k_t.stride(1),
        w_k_t.stride(2),
        q_out.stride(1),
        q_out.stride(0),
        q_out.stride(2),
        q_pe.stride(0),
        q_pe.stride(1),
        q_pe.stride(2),
        k_nope.stride(0),
        k_nope.stride(1),
        k_nope.stride(2),
        k_pe.stride(0),
        k_pe.stride(1),
        k_pe.stride(2),
        positions.stride(0),
        cos.stride(0),
        cos.stride(-1),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        0,
        0,
        0,
        k_pe_out.stride(0),
        k_pe_out.stride(1),
        k_pe_out.stride(2),
        0,
        0,
        0,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        k_scale_ptr=cos,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        QH_PER_KH=qh // kh,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=False,
        BLOCK_D_nope=kv_lora_rank,
        BLOCK_DK_nope=kv_lora_rank,
        BLOCK_D_pe=d_pe,
        BLOCK_D_HALF_pe=d_pe // 2,
        OUTPUT_Q_NOPE_ZEROS=False,
        HAVE_K_SCALE=False,
        DTYPE_MAX=_DTYPE_MAX,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        waves_per_eu=config["waves_per_eu"],
        matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
        cache_modifier=config["cache_modifier"],
    )
