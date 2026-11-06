# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 SageAttention v2 forward (Triton) from ROCm/aiter
(aiter/ops/triton/attention/fav3_sage_attention_mxfp4*.py and the sage quantization kernels),
wrapped in the run() interface. gfx950 only (tl.dot_scaled with e2m1).

Pipeline per call: per-channel V scale and FP8 V quantization; Hadamard rotation (block 64) of Q (with
the softmax scale, base-2 folded in, and per-256-row block mean smoothing) and of K (with mean-over-sequence
smoothing); two-component MXFP4 (E2M1 values, E8M0 scales per 32) expansion of Q and K; the attention
kernel with the Q-smoothing bias, FP8 probabilities and V descale.

Everything runs in Triton kernels: the host-side torch operations of the original wrapper (Hadamard
matrix construction, V abs-max, K mean and subtraction, concatenation of the expansion components) are
Triton kernels here.
"""
import torch
import triton
import triton.language as tl

_FP8_TYPE = torch.float8_e4m3fn  # gfx950 uses the OCP e4m3 format
_FP8_MAX = 448.0
_BLOCK_Q = 256  # rows of Q per smoothing block (BLOCK_M of the attention kernel)
_BLOCK_K_V = 64  # rows per V-quantization program
_BLOCK_R = 64  # Hadamard block
_LOG2E = 1.4426950408889634
_CONFIG = {
    "BLOCK_M": 256,
    "BLOCK_N": 128,
    "waves_per_eu": 2,
    "PRE_LOAD_V": False,
    "num_stages": 3,
    "num_warps": 8,
}


# ---------------------------------------------------------------------------
# Kernels that replace torch operations of the original host code.
# ---------------------------------------------------------------------------
@triton.jit
def _hadamard_kernel(R, scale, BLOCK_R: tl.constexpr):
    """R[i, j] = (-1)^popcount(i & j) * scale: the Sylvester-Hadamard matrix of the original."""
    i = tl.arange(0, BLOCK_R)[:, None]
    j = tl.arange(0, BLOCK_R)[None, :]
    x = i & j
    x = x ^ (x >> 16)
    x = x ^ (x >> 8)
    x = x ^ (x >> 4)
    x = x ^ (x >> 2)
    x = x ^ (x >> 1)
    sign = 1.0 - 2.0 * (x & 1).to(tl.float32)
    tl.store(R + i * BLOCK_R + j, (sign * scale).to(R.dtype.element_ty))


@triton.jit
def _v_scale_kernel(
    V,
    V_Scale,
    stride_vb,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_sb,
    stride_sh,
    seq_len,
    n_heads,
    fp8_max,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """V_Scale[b, h, d] = max_n |V[b, n, h, d]| / fp8_max."""
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    off_h = pid_bh % n_heads
    off_b = pid_bh // n_heads
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    amax = tl.zeros([BLOCK_D], dtype=tl.float32)
    for n in range(0, seq_len, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        v = tl.load(
            V
            + off_b * stride_vb
            + off_h * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd,
            mask=offs_n[:, None] < seq_len,
            other=0.0,
        ).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(v), axis=0))
    tl.store(V_Scale + off_b * stride_sb + off_h * stride_sh + offs_d, amax / fp8_max)


@triton.jit
def _k_mean_kernel(
    K,
    K_Mean,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    seq_len,
    n_heads,
    d_model,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """K_Mean[b, h, d] = mean_n K[b, n, h, d] (fp32 accumulation, rounded to the input dtype)."""
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    off_h = pid_bh % n_heads
    off_b = pid_bh // n_heads
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for n in range(0, seq_len, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        k = tl.load(
            K
            + off_b * stride_kb
            + off_h * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd,
            mask=offs_n[:, None] < seq_len,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(k, axis=0)
    mean = acc / seq_len
    tl.store(K_Mean + pid_bh * d_model + offs_d, mean.to(K_Mean.dtype.element_ty))


@triton.jit
def _k_sub_mean_kernel(
    K,
    K_Mean,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    seq_len,
    n_heads,
    d_model,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """K[b, n, h, :] -= K_Mean[b, h, :] in place (computed in fp32, rounded to the input dtype)."""
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    off_h = pid_bh % n_heads
    off_b = pid_bh // n_heads
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    ptrs = (
        K
        + off_b * stride_kb
        + off_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )
    mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < d_model)
    k = tl.load(ptrs, mask=mask, other=0.0)
    mean = tl.load(K_Mean + pid_bh * d_model + offs_d, mask=offs_d < d_model, other=0.0)
    tl.store(ptrs, (k.to(tl.float32) - mean.to(tl.float32)[None, :]).to(k.dtype), mask=mask)


# ===========================================================================
# MXFP4 downcast / upcast kernels and the Sage V2 quantization kernels (aiter)
# ===========================================================================
@triton.jit
def pid_grid_3d(pid, num_pid_m, num_pid_n, num_pid_k):
    """Maps 1D pid to 3D grid coords (pid_m, pid_n, pid_k)."""
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n
    pid_k = pid // (num_pid_m * num_pid_n) % num_pid_k
    return pid_m, pid_n, pid_k



@triton.jit
def _get_max_quant_val(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 6.0
    elif dtype == tl.float8e5:
        return 57344.0
    elif dtype == tl.float8e4nv:
        return 448.0
    else:
        tl.static_assert(False, f"Invalid {dtype=}")


@triton.jit
def _e2m1_rne_magnitude(magnitude):
    """E2M1 magnitude codes; midpoint ties select the even code."""
    return ((magnitude > 0.25).to(tl.uint32)
            + (magnitude >= 0.75).to(tl.uint32)
            + (magnitude > 1.25).to(tl.uint32)
            + (magnitude >= 1.75).to(tl.uint32)
            + (magnitude > 2.5).to(tl.uint32)
            + (magnitude >= 3.5).to(tl.uint32)
            + (magnitude > 5.0).to(tl.uint32))


@triton.jit
def _compute_mx_quant_and_scale(
    src_tensor,
    valid_src_mask,
    mx_tensor_dtype: tl.constexpr,
    DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr = 0,
):
    is_fp8: tl.constexpr = (
        mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    )
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // 32

    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(valid_src_mask, abs_tensor, -1.0)
    abs_tensor = tl.reshape(
        abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
    )
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
    if DEQUANT_SCALE_ROUNDING_MODE == 2:
        # E2M1 maximum power-of-two magnitude is 4. Zero blocks retain zero codes.
        maximum_exponent = ((max_val.to(tl.uint32, bitcast=True) >> 23) & 0xFF).to(tl.int32)
        scale_exponent = tl.maximum(0, maximum_exponent - 2)
        dequant_scale_exponent = scale_exponent.to(tl.uint32) << 23
        quant_scale = tl.exp2((127 - scale_exponent).to(tl.float32))
    elif DEQUANT_SCALE_ROUNDING_MODE == 0:
        dequant_scale_exponent = (
            dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF
        ) & 0x7F800000
    else:
        assert DEQUANT_SCALE_ROUNDING_MODE == 1
        dequant_scale_exponent = dequant_scale.to(tl.uint32, bitcast=True) & 0x7F800000
    if DEQUANT_SCALE_ROUNDING_MODE != 2:
        dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
        quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    f32_tensor = tl.reshape(
        f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
    )
    quant_tensor = f32_tensor * quant_scale

    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)
    dequant_scale_exponent = dequant_scale_exponent.reshape(
        [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE]
    )

    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)
    if is_fp8:
        out_tensor = quant_tensor.to(mx_tensor_dtype)
    else:
        signs = quant_tensor.to(tl.uint32, bitcast=True) & 0x80000000
        magnitude = tl.abs(quant_tensor)
        code = _e2m1_rne_magnitude(magnitude)
        e2m1_value = ((signs >> 28) | code).to(tl.uint8)

        e2m1_value = tl.reshape(
            e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2]
        )
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)

    return out_tensor, dequant_scale_exponent


@triton.jit
def _downcast_to_mxfp(
    mx_tensor_ptr,
    stride_mxt_outer,
    stride_mxt_quant: tl.constexpr,
    mx_scale_ptr,
    stride_mx_scale_outer,
    stride_mx_scale_quant,
    src_ptr,
    stride_src_outer,
    stride_src_quant,
    outer_dim,
    quant_dim,
    mx_tensor_ptr2,
    mx_scale_ptr2,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr,
):
    tl.static_assert(
        stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1."
    )
    tl.static_assert(
        BLOCK_SIZE_QUANT_DIM % 32 == 0,
        f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32",
    )

    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    tl.static_assert(
        mx_tensor_dtype == tl.uint8
        or (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5),
        f"Invalid {mx_tensor_dtype=}. Must be uint8 or float8.",
    )

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(
        mx_scale_ptr.dtype.element_ty == tl.uint8,
        f"{mx_scale_ptr.dtype.element_ty=} must be uint8",
    )
    tl.static_assert(
        (src_dtype == tl.bfloat16) or (src_dtype == tl.float16),
        f"{src_dtype=} must be bfloat16 or float16",
    )
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 32
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
    mx_scale_ptr += (
        start_mx_scale_quant * stride_mx_scale_quant + start_out * stride_mx_scale_outer
    )
    mx_tensor_ptr += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_scale_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

    mask_src_quant = start_src_quant + offs_src_quant < quant_dim
    mask_n = start_out + offs_outer < outer_dim
    full_mask_src = mask_src_quant & mask_n

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_mxt = mask_mxt_quant & mask_n

    scale_mask_k = start_mx_scale_quant + offs_scale_quant < tl.cdiv(quant_dim, 32)
    full_scale_mask = scale_mask_k & mask_n

    src_tensor_offsets = (
        offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
    )
    mx_scale_offsets = (
        offs_scale_quant * stride_mx_scale_quant + offs_outer * stride_mx_scale_outer
    )
    mx_tensor_offsets = offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer
    src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src)

    out_tensor, scale_tensor = _compute_mx_quant_and_scale(
        src_tensor, full_mask_src, mx_tensor_dtype, DEQUANT_SCALE_ROUNDING_MODE
    )

    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)

    # Second destination (same row strides, other column slot of the expanded operand).
    mx_scale_ptr2 += (
        start_mx_scale_quant * stride_mx_scale_quant + start_out * stride_mx_scale_outer
    )
    mx_tensor_ptr2 += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer
    tl.store(mx_scale_ptr2 + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr2 + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)


@triton.jit
def _upcast_from_mxfp(
    out_ptr,
    stride_o_outer,
    stride_o_quant: tl.constexpr,
    mx_scale_ptr,
    stride_scale_outer,
    stride_scale_quant,
    mx_tensor_ptr,
    stride_tensor_outer,
    stride_tensor_quant: tl.constexpr,
    outer_dim,
    quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
):
    tl.static_assert(
        stride_o_quant == 1, "the weight must be contiguous in the k dimension for mx"
    )
    tl.static_assert(
        BLOCK_SIZE_QUANT_DIM % 32 == 0, "BLOCK_SIZE_K must be a multiple of 32"
    )
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    dst_dtype: tl.constexpr = out_ptr.dtype.element_ty
    tl.static_assert(dst_dtype == tl.float16 or dst_dtype == tl.bfloat16)
    tl.static_assert(
        mx_tensor_dtype == tl.uint8
        or (
            (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5)
            or mx_tensor_dtype == dst_dtype
        ),
        "mx_tensor_ptr must be uint8 or float8 or dst_dtype",
    )
    tl.static_assert(
        mx_scale_ptr.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8"
    )

    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8
    is_fp8: tl.constexpr = (
        mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    )
    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 32
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    start_mxt_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    mx_tensor_ptr += (
        start_mxt_quant * stride_tensor_quant + start_out * stride_tensor_outer
    )
    mx_scale_ptr += (
        start_mx_scale_quant * stride_scale_quant + start_out * stride_scale_outer
    )
    out_ptr += start_out * stride_o_outer + start_out_quant * stride_o_quant

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_out_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)
    offs_scale = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)

    mask_outer = start_out + offs_outer < outer_dim
    mask_out_quant = start_out_quant + offs_out_quant < quant_dim
    full_mask_out = mask_out_quant & mask_outer

    mask_src_quant = start_mxt_quant + offs_src_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_src = mask_src_quant & mask_outer

    mask_scale = start_mx_scale_quant + offs_scale < tl.cdiv(quant_dim, 32)
    full_scale_mask = mask_scale & mask_outer

    tensor_offsets = (
        offs_src_quant * stride_tensor_quant + offs_outer * stride_tensor_outer
    )
    scale_offsets = offs_scale * stride_scale_quant + offs_outer * stride_scale_outer
    out_offsets = offs_out_quant * stride_o_quant + offs_outer * stride_o_outer

    tensor = tl.load(mx_tensor_ptr + tensor_offsets, mask=full_mask_src)
    scale = tl.load(mx_scale_ptr + scale_offsets, mask=full_scale_mask)

    if dst_dtype == tl.bfloat16:
        dst_scale = (scale.to(tl.uint16) << 7).to(dst_dtype, bitcast=True)
    else:
        tl.static_assert(dst_dtype == tl.float16)
        dst_scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        dst_scale = dst_scale.to(tl.float16)

    if is_fp8:
        dst_tensor = tensor.to(dst_dtype)
        if tensor.dtype == tl.float8e5:
            from_e_bits: tl.constexpr = 5
            from_m_bits: tl.constexpr = 2
            to_e_bits: tl.constexpr = 8 if dst_dtype == tl.bfloat16 else 5
            to_m_bits: tl.constexpr = 7 if dst_dtype == tl.bfloat16 else 10

            non_finite_mask_src: tl.constexpr = ((1 << from_e_bits) - 1) << from_m_bits
            non_finite_mask_dst: tl.constexpr = ((1 << to_e_bits) - 1) << to_m_bits
            dst_tensor = tl.where(
                (tensor.to(tl.uint8, bitcast=True) & non_finite_mask_src)
                == non_finite_mask_src,
                (dst_tensor.to(tl.uint16, bitcast=True) | non_finite_mask_dst).to(
                    dst_dtype, bitcast=True
                ),
                dst_tensor,
            )
    else:
        assert is_fp4
        dst_bias: tl.constexpr = 127 if dst_dtype == tl.bfloat16 else 15
        dst_0p5: tl.constexpr = 16128 if dst_dtype == tl.bfloat16 else 0x3800
        dst_m_bits: tl.constexpr = 7 if dst_dtype == tl.bfloat16 else 10
        em0 = tensor & 0x07
        em1 = tensor & 0x70
        x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | (
            (tensor & 0x08).to(tl.uint16) << 12
        )
        x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | (
            (tensor & 0x80).to(tl.uint16) << 8
        )
        x0 = tl.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
        x1 = tl.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)
        x0 = tl.where(em0 == 0x01, dst_0p5 | (x0 & 0x8000), x0)
        x1 = tl.where(em1 == 0x10, dst_0p5 | (x1 & 0x8000), x1)
        dst_tensor = tl.interleave(x0, x1).to(dst_dtype, bitcast=True)

    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32])
    dst_scale = dst_scale.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])
    scale = scale.reshape(dst_scale.shape)

    out_tensor = dst_tensor * dst_scale
    out_tensor = tl.where(scale == 0xFF, float("nan"), out_tensor)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    tl.store(out_ptr + out_offsets, out_tensor, mask=full_mask_out)


@triton.jit
def sage_quant_v_kernel(
    V_Input,
    V_Output,
    V_Scale,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vsz,
    stride_vsh,
    BATCH,
    K_HEAD,
    K_NUM_BLKS,
    SEQLEN_K,
    D: tl.constexpr,
    BLK_K: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    offs_blk_k = tl.arange(0, BLK_K)
    offs_d = tl.arange(0, D)

    # V
    off_blk, off_h, off_b = pid_grid_3d(pid, K_NUM_BLKS, K_HEAD, BATCH)
    offs_kn = off_blk * BLK_K + offs_blk_k

    v_offs = (
        off_b * stride_kz
        + off_h * stride_kh
        + offs_kn[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )

    v_input_ptrs = V_Input + v_offs
    v_output_ptrs = V_Output + v_offs

    # just apply the per channel v_scales that have been computed outside
    v_scale_ptrs = V_Scale + off_b * stride_vsz + off_h * stride_vsh + offs_d[None, :]
    v = tl.load(v_input_ptrs, mask=offs_kn[:, None] < SEQLEN_K, other=0.0)
    v = v.to(tl.float32)
    v_scales = tl.load(v_scale_ptrs)
    v_quant = v / tl.where(v_scales == 0.0, 1.0, v_scales)
    v_quant = v_quant.to(v_output_ptrs.dtype.element_ty)
    tl.store(v_output_ptrs, v_quant, mask=offs_kn[:, None] < SEQLEN_K)


@triton.jit
def _rot_q_kernel(
    Q,
    Q_rot,
    Q_mean,
    R,  # Hadamard matrix
    sm_scale: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_qob,
    stride_qoh,
    stride_qom,
    stride_qod,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_md,
    stride_rm,
    stride_rd,
    n_heads,
    seq_len,
    d_model,
    q_smoothing: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,  # BLOCK_D is 32
):
    # Grid: (batch * n_heads, seq_len // BLOCK_M, d_model // BLOCK_D)
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1).to(tl.int64)
    pid_d = tl.program_id(2).to(tl.int64)

    pid_h = pid_bh % n_heads
    pid_b = pid_bh // n_heads

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # Load Q block and R (Hadamard)
    # Q block shape: [BLOCK_M, BLOCK_D]
    q_ptr = (
        Q
        + (pid_b * stride_qb)
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    r_ptr = (
        R
        + tl.arange(0, BLOCK_D)[:, None] * stride_rm
        + tl.arange(0, BLOCK_D)[None, :] * stride_rd
    )
    q_tile = tl.load(
        q_ptr, mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < d_model), other=0.0
    )
    r_mat = tl.load(r_ptr)  # 32x32

    # Rotate: Q_rot = Q @ R
    q_rot_tile = tl.dot(q_tile.to(r_mat.dtype), r_mat)
    if sm_scale is not None:
        q_rot_tile *= sm_scale

    # Store rotated Q
    rot_ptr = (
        Q_rot
        + (pid_b * stride_qob)
        + pid_h * stride_qoh
        + offs_m[:, None] * stride_qom
        + offs_d[None, :] * stride_qod
    )

    # Calculate mean for the block (reduction over d within the BLOCK_M)
    # q_mean shape: [B, H, Q_NUM_BLKS, D]
    if q_smoothing:
        m_row_mean = (
            tl.sum(q_rot_tile, axis=0) / BLOCK_M
        )  # Sum over BLOCK_M -> shape [BLOCK_D]

        q_rot_tile -= m_row_mean[None, :]
        # Store mean (Atomic add or structured store)
        # For simplicity in this layout, we store the block-sum
        # and divide by BLOCK_M in the host or final step
        mean_ptr = (
            Q_mean
            + (pid_b * stride_mb)
            + pid_h * stride_mh
            + pid_m * stride_mm
            + offs_d * stride_md
        )
        tl.store(mean_ptr, m_row_mean)

    tl.store(
        rot_ptr,
        q_rot_tile,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < d_model),
    )


@triton.jit
def _rot_k_only_kernel(
    K,
    K_rot,
    R,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_kob,
    stride_koh,
    stride_kon,
    stride_kod,
    stride_rm,
    stride_rd,
    n_heads,
    seq_k,
    d_model,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    pid_d = tl.program_id(2).to(tl.int64)

    pid_h = pid_bh % n_heads
    pid_b = pid_bh // n_heads

    offs_n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # Load K block and R
    k_ptr = (
        K
        + (pid_b * stride_kb)
        + (pid_h * stride_kh)
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )
    r_ptr = (
        R
        + tl.arange(0, BLOCK_D)[:, None] * stride_rm
        + tl.arange(0, BLOCK_D)[None, :] * stride_rd
    )

    k_tile = tl.load(
        k_ptr, mask=(offs_n[:, None] < seq_k) & (offs_d[None, :] < d_model), other=0.0
    )
    r_mat = tl.load(r_ptr)

    # Rotate K
    k_rot_tile = tl.dot(k_tile.to(r_mat.dtype), r_mat)

    # Store
    rot_ptr = (
        K_rot
        + (pid_b * stride_kob)
        + pid_h * stride_koh
        + offs_n[:, None] * stride_kon
        + offs_d[None, :] * stride_kod
    )
    tl.store(
        rot_ptr,
        k_rot_tile,
        mask=(offs_n[:, None] < seq_k) & (offs_d[None, :] < d_model),
    )


@triton.jit
def _compute_delta_s_kernel(
    Q_mean,
    K_rot,
    Delta_S,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_md,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_sb,
    stride_sh,
    stride_sm,
    stride_sn,
    n_heads_q,
    n_heads_k,
    seq_k,
    d_model,
    BLOCK_N: tl.constexpr,  # Number of K-tokens to process
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_m_q = tl.program_id(1).to(tl.int64)  # The Q-block index
    pid_n_k = tl.program_id(2).to(tl.int64)  # The K-block index

    pid_hq = pid_bh % n_heads_q
    pid_b = pid_bh // n_heads_q

    pid_hk = pid_hq // (n_heads_q // n_heads_k)

    offs_n = pid_n_k * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulate dot product across the whole d_model
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over d_model in steps of 32 (our block_size)
    for d_offset in range(0, d_model, 32):
        offs_d = d_offset + tl.arange(0, 32)

        # Load Q_mean segment: [32]
        qm_ptr = (
            Q_mean
            + pid_b * stride_mb
            + pid_hq * stride_mh
            + pid_m_q * stride_mm
            + offs_d * stride_md
        )
        qm_val = tl.load(qm_ptr)

        # Load K_rot segment: [BLOCK_N, 32]
        kn_ptr = (
            K_rot
            + pid_b * stride_kb
            + pid_hk * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        kn_val = tl.load(kn_ptr, mask=offs_n[:, None] < seq_k, other=0.0)

        # Compute dot product for this d-segment
        acc += tl.sum(qm_val[None, :] * kn_val, axis=1)

    # Store to Delta_S [B, H, Q_BLKS, seq_k]
    s_ptr = (
        Delta_S
        + pid_b * stride_sb
        + pid_hq * stride_sh
        + pid_m_q * stride_sm
        + offs_n * stride_sn
    )
    tl.store(s_ptr, acc, mask=offs_n < seq_k)


@triton.jit
def _fp4_residual_subtract(Source, Decoded, Residual, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    original = tl.load(Source + offsets, mask=offsets < N, other=0).to(tl.float32)
    decoded = tl.load(Decoded + offsets, mask=offsets < N, other=0).to(tl.float32)
    tl.store(Residual + offsets, original - decoded, mask=offsets < N)


# ===========================================================================
# MXFP4 SageAttention forward kernel (aiter)
# ===========================================================================
@triton.jit
def compute_padding_info(seqlen_k, BLOCK_N: tl.constexpr):
    """Calculate padding information for the last K block."""
    # check if we will need to do masking due either BLOCK_N being bigger than seqlen_k or seqlen_k not being a factor of BLOCK_N
    # n_extra_tokens = 10 % 4 = 2
    # This means the last K block has 2 valid tokens and 2 padding positions
    # K blocks visualization:
    #         Block 0         Block 1         Block 2 (last)
    #         K0 K1 K2 K3    K4 K5 K6 K7     K8 K9 ?? ??
    #         ↑---------↑    ↑---------↑     ↑---↑ ↑---↑
    #         full block     full block      valid  pad
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    else:
        n_extra_tokens = 0
    return n_extra_tokens


@triton.jit
def compute_block_masking(
    seqlen_k,
    seqlen_q,
    start_m,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Classify K blocks for attention computation with sliding window support.

    Returns:
        - n_front_skip_blocks: Blocks completely before the window
        - n_front_masked_blocks: Blocks partially overlapping window front
        - n_full_blocks: Blocks completely inside the window
        - n_back_masked_blocks: Blocks partially overlapping window back
        - n_extra_tokens: Padding tokens in last K block
    """

    # common
    # q_start = start_m * BLOCK_M
    q_end = tl.minimum((start_m + 1) * BLOCK_M - 1, seqlen_q - 1)
    diag = seqlen_k - seqlen_q
    total_k_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    n_extra_tokens = compute_padding_info(seqlen_k, BLOCK_N)

    if IS_CAUSAL:
        # ========== CAUSAL MODE: Classify K Blocks ==========
        # Calculate causal boundary for this Q block
        #          [K0 K1 K2 K3] [K4 K5 K6 K7] [K8 K9 ?? ??]
        # Q0-Q3:   [ 1  0  0  0] [ 0  0  0  0] [ 0  0 -- --]  ← Q0
        #          [ 1  1  0  0] [ 0  0  0  0] [ 0  0 -- --]  ← Q1
        #          [ 1  1  1  0] [ 0  0  0  0] [ 0  0 -- --]  ← Q2
        #          [ 1  1  1  1] [ 1  1  0  0] [ 0  0 -- --]  ← Q3
        #                            ↑ can see up to K5
        #
        # Q4-Q7:   [ 1  1  1  1] [ 1  1  1  0] [ 0  0 -- --]  ← Q4
        #          [ 1  1  1  1] [ 1  1  1  1] [ 0  0 -- --]  ← Q5
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  0 -- --]  ← Q6
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -- --]  ← Q7

        # ------------------------------------------------------------
        # 1. figure out, in tokens, the right-most K position
        #    this Q-block may attend to
        # ------------------------------------------------------------
        k_max_token = q_end + diag  # last visible K index

        # this Q-block is entirely above the diagonal ⇒ nothing to do
        if k_max_token < 0:
            return 0, 0, 0, 0, n_extra_tokens

        k_max_token = tl.minimum(k_max_token, seqlen_k - 1)

        # ------------------------------------------------------------
        # 2. translate token indices into K-block indices
        # ------------------------------------------------------------
        last_visible_k_block = k_max_token // BLOCK_N
        n_visible_k_blocks = tl.minimum(last_visible_k_block + 1, total_k_blocks)

        # ------------------------------------------------------------
        # 3. classify those visible blocks
        #    – we *never* skip or mask blocks in front, because causal
        #      attention always starts at K0
        #    – the back side can require several masked blocks:
        #         • intersection of the causal diagonal with K-grid
        #           (at most  ⌈BLOCK_M / BLOCK_N⌉ blocks)
        #         • plus one extra block if this Q-block stops in the
        #           middle of a K-block or the last K-block is padded
        # ------------------------------------------------------------
        padded_last_k = n_extra_tokens != 0
        is_modulo_mn = (not padded_last_k) & (seqlen_q % BLOCK_M == 0)

        n_back_masked_blocks = BLOCK_M // BLOCK_N + tl.where(is_modulo_mn, 0, 1)
        n_back_masked_blocks = tl.minimum(n_back_masked_blocks, n_visible_k_blocks)

        n_front_skip_blocks = 0  # causal never skips the left side
        n_front_masked_blocks = 0  # ditto
        n_full_blocks = n_visible_k_blocks - n_back_masked_blocks
    else:
        # ========== NON-CAUSAL MODE ==========
        # Without causal mask, all positions can attend to all positions
        # Only need to handle the padding in the last block
        #          [K0 K1 K2 K3] [K4 K5 K6 K7] [K8 K9 ?? ??]
        # Q0-Q3:   [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #
        # Q4-Q7:   [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]

        n_front_skip_blocks = 0  # never skips the left side
        n_front_masked_blocks = 0  # ditto
        if n_extra_tokens != 0:
            n_back_masked_blocks = 1  # Last block needs padding mask
            n_full_blocks = total_k_blocks - 1
        else:
            n_back_masked_blocks = 0  # All blocks are aligned
            n_full_blocks = total_k_blocks

    return (
        n_front_skip_blocks,
        n_front_masked_blocks,
        n_full_blocks,
        n_back_masked_blocks,
        n_extra_tokens,
    )


@triton.jit
def _sage_fwd_no_mask_mxfp4(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
    bias_base_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    seqlen_k,
    seqlen_q,
    offs_m,
    offs_d_k,
    offs_d_v,
    block_min,
    block_max,
    q_descale,
    k_descale_base_ptrs,
    stride_ksn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
    ACCUMULATOR_TYPE: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    for start_n in range(block_min, block_max, BLOCK_N):
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk
        k_descale_ptrs = k_descale_base_ptrs + start_n * stride_ksn
        kv_offs_n = start_n + tl.arange(0, BLOCK_N)

        # Refactored K Load
        if PADDED_HEAD_QK:
            k_mask = offs_d_k[:, None] < ACTUAL_BLOCK_DMODEL_QK
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        else:
            k = tl.load(k_ptrs)

        k_descale = tl.load(k_descale_ptrs)

        if PRE_LOAD_V:
            # Refactored V Load
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)
        qk = tl.dot_scaled(
            q, q_descale, Q_DTYPE_STR, k, k_descale, K_DTYPE_STR, fast_math=True, acc=qk
        )

        if USE_BIAS:
            bias_mask = kv_offs_n < seqlen_k
            bias = tl.load(
                bias_base_ptrs + start_n * stride_bn, mask=bias_mask, other=0.0
            )
            qk += bias[None, :]

        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        if USE_BIAS:
            q_shifted = tl.where(
                m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None]
            )
        else:
            q_shifted = qk - m_ij[:, None]

        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)

        if USE_BIAS:
            m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        else:
            m_diff = m_i - m_ij

        alpha = tl.math.exp2(m_diff)
        acc = acc * alpha[:, None]

        if not PRE_LOAD_V:
            # Refactored V Load (Lazy)
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)

    return acc, l_i, m_i


@triton.jit
def _sage_fwd_mask_mxfp4(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
    bias_base_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    seqlen_k,
    seqlen_q,
    offs_m,
    offs_n,
    offs_d_k,
    offs_d_v,
    block_min,
    block_max,
    n_extra_tokens,
    q_descale,
    k_descale_base_ptrs,
    stride_ksn,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
    ACCUMULATOR_TYPE: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    seqlen_delta_qk = seqlen_k - seqlen_q
    for start_n in range(block_min, block_max, BLOCK_N):
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk
        k_descale_ptrs = k_descale_base_ptrs + start_n * stride_ksn
        kv_offs_n = start_n + tl.arange(0, BLOCK_N)

        # Refactored K Load with mandatory boundary check + optional padding check
        k_mask = kv_offs_n[None, :] < seqlen_k
        if PADDED_HEAD_QK:
            k_mask &= offs_d_k[:, None] < ACTUAL_BLOCK_DMODEL_QK

        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        k_descale = tl.load(
            k_descale_ptrs, mask=kv_offs_n[:, None] < seqlen_k, other=0.0
        )

        if PRE_LOAD_V:
            # Refactored V Load
            v_mask = kv_offs_n[:, None] < seqlen_k
            if PADDED_HEAD_V:
                v_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)

        if (n_extra_tokens != 0) and (start_n + BLOCK_N == block_max):
            mask = (start_n + offs_n[None, :]) < seqlen_k
            qk = tl.where(mask, qk, float("-inf"))

        qk = tl.dot_scaled(
            q, q_descale, Q_DTYPE_STR, k, k_descale, K_DTYPE_STR, fast_math=True, acc=qk
        )

        if IS_CAUSAL:
            qk = tl.where(
                offs_m[:, None] >= (start_n + offs_n - seqlen_delta_qk)[None, :],
                qk,
                float("-inf"),
            )

        if USE_BIAS:
            bias_mask = kv_offs_n < seqlen_k
            bias = tl.load(
                bias_base_ptrs + start_n * stride_bn, mask=bias_mask, other=0.0
            )
            qk += bias[None, :]

        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        q_shifted = tl.where(
            m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None]
        )

        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)

        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        alpha = tl.math.exp2(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            # Refactored V Load (Lazy)
            v_mask = kv_offs_n[:, None] < seqlen_k
            if PADDED_HEAD_V:
                v_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)

    return acc, l_i, m_i


@triton.jit
def _sage_fwd_blocksparse_nomask_mxfp4(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
    bias_base_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    seqlen_k,
    seqlen_q,
    offs_m,
    offs_d_k,
    offs_d_v,
    q_descale,
    k_descale_base_ptrs,
    stride_ksn,
    kv_block_indices,
    lut_start_val,
    n_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
    ACCUMULATOR_TYPE: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    for i in range(n_blocks):
        start_b = tl.load(kv_block_indices + lut_start_val + i)
        start_n = start_b * BLOCK_N
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk
        k_descale_ptrs = k_descale_base_ptrs + start_n * stride_ksn
        kv_offs_n = start_n + tl.arange(0, BLOCK_N)

        if PADDED_HEAD_QK:
            k_mask = offs_d_k[:, None] < ACTUAL_BLOCK_DMODEL_QK
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        else:
            k = tl.load(k_ptrs)

        k_descale = tl.load(k_descale_ptrs)

        if PRE_LOAD_V:
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)
        qk = tl.dot_scaled(
            q, q_descale, Q_DTYPE_STR, k, k_descale, K_DTYPE_STR, fast_math=True, acc=qk
        )

        if USE_BIAS:
            bias_mask = kv_offs_n < seqlen_k
            bias = tl.load(
                bias_base_ptrs + start_n * stride_bn, mask=bias_mask, other=0.0
            )
            qk += bias[None, :]

        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        if USE_BIAS:
            q_shifted = tl.where(
                m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None]
            )
        else:
            q_shifted = qk - m_ij[:, None]

        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)

        if USE_BIAS:
            m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        else:
            m_diff = m_i - m_ij

        alpha = tl.math.exp2(m_diff)
        acc = acc * alpha[:, None]

        if not PRE_LOAD_V:
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)

    return acc, l_i, m_i


@triton.jit
def _sage_fwd_blocksparse_mask_mxfp4(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
    bias_base_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    seqlen_k,
    seqlen_q,
    offs_m,
    offs_n,
    offs_d_k,
    offs_d_v,
    q_descale,
    k_descale_base_ptrs,
    stride_ksn,
    kv_block_indices,
    lut_start_val,
    n_blocks,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
    ACCUMULATOR_TYPE: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    seqlen_delta_qk = seqlen_k - seqlen_q
    for i in range(n_blocks):
        start_b = tl.load(kv_block_indices + lut_start_val + i)
        start_n = start_b * BLOCK_N
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk
        k_descale_ptrs = k_descale_base_ptrs + start_n * stride_ksn
        kv_offs_n = start_n + tl.arange(0, BLOCK_N)

        k_mask = kv_offs_n[None, :] < seqlen_k
        if PADDED_HEAD_QK:
            k_mask &= offs_d_k[:, None] < ACTUAL_BLOCK_DMODEL_QK

        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        k_descale = tl.load(
            k_descale_ptrs, mask=kv_offs_n[:, None] < seqlen_k, other=0.0
        )

        if PRE_LOAD_V:
            v_mask = kv_offs_n[:, None] < seqlen_k
            if PADDED_HEAD_V:
                v_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)

        # Padding mask: mask out positions beyond seqlen_k
        boundary_mask = kv_offs_n[None, :] < seqlen_k
        qk = tl.where(boundary_mask, qk, float("-inf"))

        qk = tl.dot_scaled(
            q, q_descale, Q_DTYPE_STR, k, k_descale, K_DTYPE_STR, fast_math=True, acc=qk
        )

        if IS_CAUSAL:
            qk = tl.where(
                offs_m[:, None] >= (start_n + offs_n - seqlen_delta_qk)[None, :],
                qk,
                float("-inf"),
            )

        if USE_BIAS:
            bias_mask = kv_offs_n < seqlen_k
            bias = tl.load(
                bias_base_ptrs + start_n * stride_bn, mask=bias_mask, other=0.0
            )
            qk += bias[None, :]

        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        q_shifted = tl.where(
            m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None]
        )

        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)

        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        alpha = tl.math.exp2(m_diff)
        acc = acc * alpha[:, None]

        if not PRE_LOAD_V:
            v_mask = kv_offs_n[:, None] < seqlen_k
            if PADDED_HEAD_V:
                v_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)

    return acc, l_i, m_i


@triton.jit
def sage_fwd_mxfp4(
    Q,
    K,
    V,
    bias,
    Q_Descale,
    K_Descale,
    V_Descale,
    stride_qsz,
    stride_qsh,
    stride_qsm,
    stride_ksz,
    stride_ksh,
    stride_ksn,
    stride_vsz,
    stride_vsh,
    Out,
    LSE,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_lse_z,
    stride_lse_h,
    stride_lse_m,
    cu_seqlens_q,
    cu_seqlens_k,
    kv_block_indices,
    lut_start,
    lut_count,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    USE_BIAS: tl.constexpr,
    USE_BLOCK_SPARSE: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    # Constants
    Q_HEAD_DIV: tl.constexpr = 2 if Q_DTYPE_STR == "e2m1" else 1
    K_HEAD_DIV: tl.constexpr = 2 if K_DTYPE_STR == "e2m1" else 1
    SCALE_GROUP: tl.constexpr = 32
    ACC_TYPE: tl.constexpr = tl.float32
    # b*h*s*d can grow to be larger than int32 max, so turn to int64
    start_m, off_h_q, off_z = (
        tl.program_id(0).to(tl.int64),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2).to(tl.int64),
    )
    off_h_k = off_h_q // (HQ // HK)

    PADDED_HEAD_QK: tl.constexpr = ACTUAL_BLOCK_DMODEL_QK != BLOCK_DMODEL_QK
    PADDED_HEAD_V: tl.constexpr = ACTUAL_BLOCK_DMODEL_V != BLOCK_DMODEL_V

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_q = tl.arange(0, BLOCK_DMODEL_QK // Q_HEAD_DIV)
    offs_d_k = tl.arange(0, BLOCK_DMODEL_QK // K_HEAD_DIV)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)
    offs_d_scale = tl.arange(0, BLOCK_DMODEL_QK // SCALE_GROUP)

    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + off_z)
        seqlen_q = tl.load(cu_seqlens_q + off_z + 1) - q_start
        k_start = tl.load(cu_seqlens_k + off_z)
        seqlen_k = tl.load(cu_seqlens_k + off_z + 1) - k_start
        if start_m * BLOCK_M >= seqlen_q:
            return
    else:
        q_start, k_start = 0, 0
        seqlen_q, seqlen_k = MAX_SEQLENS_Q, MAX_SEQLENS_K

    # Masking logic
    if USE_BLOCK_SPARSE:
        num_q_blocks = (seqlen_q + BLOCK_M - 1) // BLOCK_M
        n_extra = compute_padding_info(seqlen_k, BLOCK_N)
        lut_idx = off_z * (HQ * num_q_blocks) + off_h_q * num_q_blocks + start_m
        n_blocks = tl.load(lut_count + lut_idx)
        has_any_range = n_blocks > 0
    else:
        mask_info = compute_block_masking(
            seqlen_k, seqlen_q, start_m.to(tl.int32), IS_CAUSAL, BLOCK_M, BLOCK_N
        )  # need to turn start_m to int32 for consistent return values
        n_front_skip, n_front_masked, n_full, n_back_masked, n_extra = mask_info
        has_any_range = True

    # ============================================================
    #          PROGRAM EARLY EXIT (All K Blocks Skipped)
    # ============================================================
    if not USE_BLOCK_SPARSE:
        total_visible_blocks = n_front_masked + n_full + n_back_masked
    # Early exit: no K blocks to process
    if USE_BLOCK_SPARSE:
        _no_blocks = not has_any_range
    else:
        _no_blocks = total_visible_blocks == 0
    if _no_blocks:
        """
        No K blocks visible - write zeros and exit.
        """
        o_ptr = (
            Out
            + off_z * stride_oz
            + off_h_q * stride_oh
            + (q_start + offs_m[:, None]) * stride_om
            + offs_d_v[None, :]
        )
        o_mask = offs_m[:, None] < seqlen_q
        if PADDED_HEAD_V:
            o_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
        tl.store(
            o_ptr,
            tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=Out.type.element_ty),
            mask=o_mask,
        )

        if RETURN_LSE:
            l_offset = (
                LSE
                + off_z * stride_lse_z
                + off_h_q * stride_lse_h
                + q_start * stride_lse_m
            )
            l_ptrs = l_offset + offs_m * stride_lse_m
            tl.store(
                l_ptrs,
                tl.full([BLOCK_M], float("-inf"), dtype=tl.float32),
                mask=offs_m < seqlen_q,
            )

        return

    # Pointers
    q_ptrs = (
        Q
        + off_z * stride_qz
        + off_h_q * stride_qh
        + (q_start + offs_m[:, None]) * stride_qm
        + offs_d_q[None, :]
    )
    k_ptrs = (
        K
        + off_z * stride_kz
        + off_h_k * stride_kh
        + (k_start + offs_n[None, :]) * stride_kn
        + offs_d_k[:, None]
    )
    v_ptrs = (
        V
        + off_z * stride_vz
        + off_h_k * stride_vh
        + (k_start + offs_n[:, None]) * stride_vk
        + offs_d_v[None, :]
    )

    qd_ptrs = (
        Q_Descale
        + off_z * stride_qsz
        + off_h_q * stride_qsh
        + (q_start + offs_m[:, None]) * stride_qsm
        + offs_d_scale[None, :]
    )
    kd_ptrs = (
        K_Descale
        + off_z * stride_ksz
        + off_h_k * stride_ksh
        + (k_start + offs_n[:, None]) * stride_ksn
        + offs_d_scale[None, :]
    )
    vd_ptr = V_Descale + off_z * stride_vsz + off_h_k * stride_vsh + offs_d_v

    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q), other=0.0)
    q_descale = tl.load(qd_ptrs, mask=(offs_m[:, None] < seqlen_q), other=0.0)

    # Bias is delta s
    bias_ptrs = (
        (
            bias
            + off_z * stride_bz
            + off_h_q * stride_bh
            + start_m * stride_bm
            + tl.cast(offs_n, tl.int64) * stride_bn
        )
        if USE_BIAS
        else None
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=ACC_TYPE)
    l_i = tl.full([BLOCK_M], 1.0, dtype=ACC_TYPE)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=ACC_TYPE)

    if not USE_BLOCK_SPARSE:
        if n_full > 0:
            b_min = (n_front_skip + n_front_masked) * BLOCK_N
            b_max = b_min + n_full * BLOCK_N
            acc, l_i, m_i = _sage_fwd_no_mask_mxfp4(
                acc,
                l_i,
                m_i,
                q,
                k_ptrs,
                v_ptrs,
                bias_ptrs,
                stride_kn,
                stride_vk,
                stride_bn,
                seqlen_k,
                seqlen_q,
                offs_m,
                offs_d_k,
                offs_d_v,
                b_min,
                b_max,
                q_descale,
                kd_ptrs,
                stride_ksn,
                BLOCK_M,
                BLOCK_N,
                PRE_LOAD_V,
                PADDED_HEAD_QK,
                PADDED_HEAD_V,
                ACTUAL_BLOCK_DMODEL_QK,
                ACTUAL_BLOCK_DMODEL_V,
                Q_DTYPE_STR,
                K_DTYPE_STR,
                ACC_TYPE,
                USE_BIAS,
            )

        if n_back_masked > 0:
            b_min = (n_front_skip + n_front_masked + n_full) * BLOCK_N
            b_max = b_min + n_back_masked * BLOCK_N
            acc, l_i, m_i = _sage_fwd_mask_mxfp4(
                acc,
                l_i,
                m_i,
                q,
                k_ptrs,
                v_ptrs,
                bias_ptrs,
                stride_kn,
                stride_vk,
                stride_bn,
                seqlen_k,
                seqlen_q,
                offs_m,
                offs_n,
                offs_d_k,
                offs_d_v,
                b_min,
                b_max,
                n_extra,
                q_descale,
                kd_ptrs,
                stride_ksn,
                IS_CAUSAL,
                BLOCK_M,
                BLOCK_N,
                PRE_LOAD_V,
                PADDED_HEAD_QK,
                PADDED_HEAD_V,
                ACTUAL_BLOCK_DMODEL_QK,
                ACTUAL_BLOCK_DMODEL_V,
                Q_DTYPE_STR,
                K_DTYPE_STR,
                ACC_TYPE,
                USE_BIAS,
            )
    else:
        lut_start_val = tl.load(lut_start + lut_idx)
        acc, l_i, m_i = _sage_fwd_blocksparse_nomask_mxfp4(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            seqlen_k,
            seqlen_q,
            offs_m,
            offs_d_k,
            offs_d_v,
            q_descale,
            kd_ptrs,
            stride_ksn,
            kv_block_indices,
            lut_start_val,
            n_blocks - 1,
            BLOCK_M,
            BLOCK_N,
            PRE_LOAD_V,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            Q_DTYPE_STR,
            K_DTYPE_STR,
            ACC_TYPE,
            USE_BIAS,
        )
        invalid_q_rows = offs_m >= seqlen_q
        m_i = tl.where(invalid_q_rows, float("-inf"), m_i)
        l_i = tl.where(invalid_q_rows, 1.0, l_i)
        acc = tl.where(invalid_q_rows[:, None], 0.0, acc)
        acc, l_i, m_i = _sage_fwd_blocksparse_mask_mxfp4(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            seqlen_k,
            seqlen_q,
            offs_m,
            offs_n,
            offs_d_k,
            offs_d_v,
            q_descale,
            kd_ptrs,
            stride_ksn,
            kv_block_indices,
            lut_start_val + (n_blocks - 1),
            1,
            False,  # IS_CAUSAL is not supported for block sparse
            BLOCK_M,
            BLOCK_N,
            PRE_LOAD_V,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            Q_DTYPE_STR,
            K_DTYPE_STR,
            ACC_TYPE,
            USE_BIAS,
        )

    # Epilogue
    invalid_mask = m_i == float("-inf")
    l_i_safe = tl.where(invalid_mask, 1.0, l_i)
    l_i_safe = tl.maximum(l_i_safe, 1e-7)
    l_recip = 1 / l_i_safe[:, None]
    v_descale = tl.load(vd_ptr, mask=offs_d_v < ACTUAL_BLOCK_DMODEL_V, other=0.0)
    acc = acc * l_recip * v_descale
    acc = tl.where(invalid_mask[:, None], 0.0, acc)

    if RETURN_LSE:
        # m_i / l_i are in base-2 (sm_scale was pre-multiplied by 1/ln(2)).
        # Convert back to natural units to match the int8 sage convention.
        LN2: tl.constexpr = 0.6931471824645996
        log_l_i = tl.where(invalid_mask, 0.0, tl.math.log2(l_i_safe))
        softmax_lse = tl.where(invalid_mask, float("-inf"), (m_i + log_l_i) * LN2)
        l_offset = (
            LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + q_start * stride_lse_m
        )
        l_ptrs = l_offset + offs_m * stride_lse_m
        tl.store(l_ptrs, softmax_lse, mask=offs_m < seqlen_q)

    o_ptr = (
        Out
        + off_z * stride_oz
        + off_h_q * stride_oh
        + (q_start + offs_m[:, None]) * stride_om
        + offs_d_v[None, :]
    )
    o_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD_V:
        o_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
    tl.store(o_ptr, acc.to(Out.dtype.element_ty), mask=o_mask)


# ---------------------------------------------------------------------------
# Host launch code (graph-capturable: allocation, views and kernel launches only).
# ---------------------------------------------------------------------------
def _quantize_expand(x, codes, scales, first_slots, second_slots):
    """Two-component MXFP4 expansion of ``x`` ([b, s, h, d], contiguous).

    The first component's codes and scales go to the column slots ``first_slots`` of the expanded
    operand, the quantized residual (x minus the decoded first component) to ``second_slots``.
    """
    d = x.shape[-1]
    half = d // 2
    groups = d // 32
    rows = x.numel() // d
    row_codes = codes.shape[-1]
    row_scales = scales.shape[-1]
    grid = (triton.cdiv(rows, 128), triton.cdiv(d, 32))

    def slot_codes(slot):
        return codes[..., slot * half:(slot + 1) * half]

    def slot_scales(slot):
        return scales[..., slot * groups:(slot + 1) * groups]

    def downcast(source, slots):
        _downcast_to_mxfp[grid](
            slot_codes(slots[0]), row_codes, 1,
            slot_scales(slots[0]), row_scales, 1,
            source, d, 1,
            rows, d,
            slot_codes(slots[1]), slot_scales(slots[1]),
            128, 32, 2,
            num_warps=8,
        )

    downcast(x, first_slots)
    decoded = torch.empty_like(x)
    _upcast_from_mxfp[grid](
        decoded, d, 1,
        slot_scales(first_slots[0]), row_scales, 1,
        slot_codes(first_slots[0]), row_codes, 1,
        rows, d,
        128, 32,
        num_warps=8,
    )
    residual = torch.empty_like(x)
    size = x.numel()
    _fp4_residual_subtract[(triton.cdiv(size, 256),)](x, decoded, residual, size, 256)
    downcast(residual, second_slots)


def run(*, q, k, v, out):
    b, s_q, h_q, d = q.shape
    _, s_k, h_k, _ = k.shape
    device = q.device
    q_blocks = triton.cdiv(s_q, _BLOCK_Q)
    k_blocks = triton.cdiv(s_k, _BLOCK_Q)
    sm_scale = d ** -0.5

    # Hadamard matrix (block _BLOCK_R), normalized.
    rotation = torch.empty((_BLOCK_R, _BLOCK_R), dtype=q.dtype, device=device)
    _hadamard_kernel[(1,)](rotation, _BLOCK_R ** -0.5, _BLOCK_R)

    # Per-channel V scale and FP8 V.
    v_scale = torch.empty((b, h_k, d), dtype=torch.float32, device=device)
    _v_scale_kernel[(b * h_k, triton.cdiv(d, 32))](
        v, v_scale,
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        v_scale.stride(0), v_scale.stride(1),
        s_k, h_k, _FP8_MAX,
        64, 32,
    )
    v_fp8 = torch.empty_like(v, dtype=_FP8_TYPE)
    _v_blocks = triton.cdiv(s_k, _BLOCK_K_V)
    sage_quant_v_kernel[(b * h_k * _v_blocks,)](
        v, v_fp8, v_scale,
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        v_scale.stride(0), v_scale.stride(1),
        b, h_k, _v_blocks, s_k,
        D=d,
        BLK_K=_BLOCK_K_V,
        num_stages=3,
        num_warps=8,
    )

    # Rotation (Q also scaled by sm_scale * log2(e) and block-mean smoothed; K mean-smoothed).
    q_rot = torch.empty_like(q)
    k_rot = torch.empty_like(k)
    q_mean = torch.empty((b, h_q, q_blocks, d), dtype=torch.float32, device=device)
    delta_s = torch.empty((b, h_q, q_blocks, s_k), dtype=torch.float32, device=device)
    _rot_q_kernel[(b * h_q, q_blocks, d // _BLOCK_R)](
        q, q_rot, q_mean, rotation,
        sm_scale * _LOG2E,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        q_rot.stride(0), q_rot.stride(2), q_rot.stride(1), q_rot.stride(3),
        q_mean.stride(0), q_mean.stride(1), q_mean.stride(2), q_mean.stride(3),
        rotation.stride(0), rotation.stride(1),
        h_q, s_q, d,
        q_smoothing=True,
        BLOCK_M=_BLOCK_Q,
        BLOCK_D=_BLOCK_R,
    )
    _rot_k_only_kernel[(b * h_k, k_blocks, d // _BLOCK_R)](
        k, k_rot, rotation,
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        k_rot.stride(0), k_rot.stride(2), k_rot.stride(1), k_rot.stride(3),
        rotation.stride(0), rotation.stride(1),
        h_k, s_k, d,
        BLOCK_M=_BLOCK_Q,
        BLOCK_D=_BLOCK_R,
    )
    k_mean = torch.empty((b, h_k, d), dtype=k.dtype, device=device)
    _k_mean_kernel[(b * h_k, triton.cdiv(d, 32))](
        k_rot, k_mean,
        k_rot.stride(0), k_rot.stride(1), k_rot.stride(2), k_rot.stride(3),
        s_k, h_k, d,
        64, 32,
    )
    _k_sub_mean_kernel[(b * h_k, triton.cdiv(s_k, 64))](
        k_rot, k_mean,
        k_rot.stride(0), k_rot.stride(1), k_rot.stride(2), k_rot.stride(3),
        s_k, h_k, d,
        64, triton.next_power_of_2(d),
    )
    _compute_delta_s_kernel[(b * h_q, q_blocks, k_blocks)](
        q_mean, k_rot, delta_s,
        q_mean.stride(0), q_mean.stride(1), q_mean.stride(2), q_mean.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        delta_s.stride(0), delta_s.stride(1), delta_s.stride(2), delta_s.stride(3),
        h_q, h_k, s_k, d,
        BLOCK_N=_BLOCK_Q,
    )

    # Two-component MXFP4 expansion: (Q0 + Q1) @ (K0 + K1)^T as four feature products,
    # Q slots (Q0, Q1, Q0, Q1) and K slots (K0, K0, K1, K1).
    q_codes = torch.empty((b, s_q, h_q, 2 * d), dtype=torch.uint8, device=device)
    q_scales = torch.empty((b, s_q, h_q, 4 * (d // 32)), dtype=torch.uint8, device=device)
    k_codes = torch.empty((b, s_k, h_k, 2 * d), dtype=torch.uint8, device=device)
    k_scales = torch.empty((b, s_k, h_k, 4 * (d // 32)), dtype=torch.uint8, device=device)
    _quantize_expand(q_rot, q_codes, q_scales, (0, 2), (1, 3))
    _quantize_expand(k_rot, k_codes, k_scales, (0, 1), (2, 3))

    # Attention (causal), bshd layout.
    head_size_qk = q_codes.shape[-1] * 2
    padded_d_qk = max(16, 1 << (head_size_qk - 1).bit_length())
    padded_d_v = max(16, 1 << (d - 1).bit_length())
    kv_block_indices = torch.zeros(1, dtype=torch.int32, device=device)
    lut_start = torch.zeros(1, dtype=torch.int32, device=device)
    lut_count = torch.zeros(1, dtype=torch.int32, device=device)
    stride_bz, stride_bh, stride_bm, stride_bn = delta_s.stride()
    sage_fwd_mxfp4[(triton.cdiv(s_q, _CONFIG["BLOCK_M"]), h_q, b)](
        Q=q_codes,
        K=k_codes,
        V=v_fp8,
        bias=delta_s,
        Q_Descale=q_scales,
        K_Descale=k_scales,
        V_Descale=v_scale,
        stride_qsz=q_scales.stride(0),
        stride_qsh=q_scales.stride(2),
        stride_qsm=q_scales.stride(1),
        stride_ksz=k_scales.stride(0),
        stride_ksh=k_scales.stride(2),
        stride_ksn=k_scales.stride(1),
        stride_vsz=v_scale.stride(0),
        stride_vsh=v_scale.stride(1),
        Out=out,
        LSE=None,
        stride_qz=q_codes.stride(0),
        stride_qh=q_codes.stride(2),
        stride_qm=q_codes.stride(1),
        stride_kz=k_codes.stride(0),
        stride_kh=k_codes.stride(2),
        stride_kn=k_codes.stride(1),
        stride_vz=v_fp8.stride(0),
        stride_vh=v_fp8.stride(2),
        stride_vk=v_fp8.stride(1),
        stride_oz=out.stride(0),
        stride_oh=out.stride(2),
        stride_om=out.stride(1),
        stride_bz=stride_bz,
        stride_bh=stride_bh,
        stride_bm=stride_bm,
        stride_bn=stride_bn,
        stride_lse_z=0,
        stride_lse_h=0,
        stride_lse_m=0,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        kv_block_indices=kv_block_indices,
        lut_start=lut_start,
        lut_count=lut_count,
        Q_DTYPE_STR="e2m1",
        K_DTYPE_STR="e2m1",
        HQ=h_q,
        HK=h_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=d,
        MAX_SEQLENS_Q=s_q,
        MAX_SEQLENS_K=s_k,
        IS_VARLEN=False,
        IS_CAUSAL=True,
        BLOCK_DMODEL_QK=padded_d_qk,
        BLOCK_DMODEL_V=padded_d_v,
        USE_BIAS=True,
        USE_BLOCK_SPARSE=False,
        RETURN_LSE=False,
        **_CONFIG,
    )
