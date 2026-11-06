# SPDX-License-Identifier: MIT
# Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
"""Fused RMSNorm and FP8 per-group quantization, from AITER's ``fused_fp8_quant`` Triton kernels."""
import triton
import triton.language as tl


@triton.jit
def _rmsmorm_op(row, weight, n_cols, epsilon):
    row_norm = row * row
    row_norm = tl.sum(row_norm, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

    rms_norm = row * norm_factor * weight
    return rms_norm


@triton.jit
def _fp8_quant_op(
    x,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
):
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, QUANT_BLOCK_SIZE)
    m = tl.maximum(tl.max(tl.abs(x), axis=-1), 1e-10)
    scale_out = m.to(tl.float32) / DTYPE_MAX
    scale_recip = 1.0 / scale_out.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, 1)
    x = tl.clamp(x * scale_recip, DTYPE_MIN, DTYPE_MAX)

    return x, scale_out


@triton.jit
def _fused_rms_fp8_group_quant_kernel(
    inp1_ptr,
    weight1_ptr,
    inp2_ptr,
    weight2_ptr,
    res1_ptr,
    out1_fp8_ptr,
    out1_bs_ptr,
    out2_ptr,
    out_res1_ptr,
    out1_ptr,
    eps1,
    eps2,
    n_rows,
    inp1_n_cols,
    inp2_n_cols,
    inp1_row_stride,
    inp2_row_stride,
    inp1_col_stride,
    inp2_col_stride,
    res1_row_stride,
    res1_col_stride,
    out1_fp8_row_stride,
    out1_fp8_col_stride,
    out1_bs_row_stride,
    out1_bs_col_stride,
    out2_row_stride,
    out2_col_stride,
    out_res1_row_stride,
    out_res1_col_stride,
    out1_row_stride,
    out1_col_stride,
    gate_ptr,
    linear_bias_ptr,
    stride_gate_row,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
    HAVE_SECOND_INPUT: tl.constexpr,
    FIRST_INPUT_RES: tl.constexpr,
    FIRST_INPUT_OUT: tl.constexpr,
    GATED_RMS_FP8: tl.constexpr,
    RMS_TILE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    GROUP_SIZE_GATED: tl.constexpr,
    NUM_GROUPS_GATED: tl.constexpr,
    BLOCK_G: tl.constexpr,
    HAS_BIAS_GATED: tl.constexpr,
    HAS_Z_GATED: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    FP8_MIN_SCALING_FACTOR: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """RMSNorm + FP8 row/group quant (classic) or gated RMSNorm + FP8 (vLLM-style).

    When ``GATED_RMS_FP8`` is True, use grid ``(cdiv(M, ROWS_PER_BLOCK),)`` and batch
    ``ROWS_PER_BLOCK`` rows per program (vLLM ``calc_rows_per_block`` heuristic on the host).
    Extra pointer args are unused in the classic path but must refer to valid tensors.
    """
    if GATED_RMS_FP8:
        # --- Gated path (adapted from vLLM / ROCm gated RMSNorm FP8 kernel) ---
        X = inp1_ptr
        W = weight1_ptr
        Bptr = linear_bias_ptr
        Z = gate_ptr
        Y_quant = out1_fp8_ptr
        Scales = out1_bs_ptr
        stride_x_row = inp1_row_stride
        stride_z_row = stride_gate_row
        stride_y_row = out1_fp8_row_stride
        stride_s_row = out1_bs_row_stride
        stride_s_g = out1_bs_col_stride
        M = n_rows
        N = inp1_n_cols
        eps = eps1

        row_start = tl.program_id(0) * ROWS_PER_BLOCK
        rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
        row_mask_1d = rows < M

        sumsq = tl.zeros([ROWS_PER_BLOCK], dtype=tl.float32)
        off_rms = 0
        while off_rms < N:
            cols = tl.arange(0, RMS_TILE) + off_rms
            col_mask = cols < N
            mask_r = row_mask_1d[:, None] & col_mask[None, :]
            row_offsets = rows[:, None] * stride_x_row
            col_offsets = cols[None, :]
            X_base = X + row_offsets + col_offsets
            x_el = tl.load(X_base, mask=mask_r, other=0.0).to(tl.float32)
            if HAS_Z_GATED and (not NORM_BEFORE_GATE):
                Z_base = Z + rows[:, None] * stride_z_row + col_offsets
                z_el = tl.load(Z_base, mask=mask_r, other=0.0).to(tl.float32)
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    x_el = x_el * (z_el * tl.sigmoid(z_el))
                elif ACTIVATION == "sigmoid":
                    x_el = x_el * tl.sigmoid(z_el)
            xbar_sq = tl.where(mask_r, x_el, 0.0)
            sumsq = sumsq + tl.sum(xbar_sq * xbar_sq, axis=1)
            off_rms += RMS_TILE

        var = sumsq / N
        rstd = tl.math.rsqrt(var + eps)

        for g in range(NUM_GROUPS_GATED):
            col0 = g * GROUP_SIZE_GATED
            cols = tl.arange(0, BLOCK_G) + col0
            col_mask = (cols < N) & (cols < col0 + GROUP_SIZE_GATED)
            mask_g = row_mask_1d[:, None] & col_mask[None, :]
            row_offsets = rows[:, None] * stride_x_row
            col_offsets = cols[None, :]
            X_base = X + row_offsets + col_offsets
            x_el = tl.load(X_base, mask=mask_g, other=0.0).to(tl.float32)

            if HAS_Z_GATED and (not NORM_BEFORE_GATE):
                Z_base = Z + rows[:, None] * stride_z_row + col_offsets
                z_el = tl.load(Z_base, mask=mask_g, other=0.0).to(tl.float32)
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    x_el = x_el * (z_el * tl.sigmoid(z_el))
                elif ACTIVATION == "sigmoid":
                    x_el = x_el * tl.sigmoid(z_el)

            x_hat = x_el * rstd[:, None]

            w_mask = col_mask
            w_el = tl.load(W + cols, mask=w_mask, other=0.0).to(tl.float32)
            if HAS_BIAS_GATED:
                b_el = tl.load(Bptr + cols, mask=w_mask, other=0.0).to(tl.float32)
                y_el = x_hat * w_el[None, :] + b_el[None, :]
            else:
                y_el = x_hat * w_el[None, :]

            if HAS_Z_GATED and NORM_BEFORE_GATE:
                Z_base = Z + rows[:, None] * stride_z_row + col_offsets
                z_el = tl.load(Z_base, mask=mask_g, other=0.0).to(tl.float32)
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    y_el = y_el * (z_el * tl.sigmoid(z_el))
                elif ACTIVATION == "sigmoid":
                    y_el = y_el * tl.sigmoid(z_el)

            abs_y = tl.where(mask_g, tl.abs(y_el), 0.0)
            absmax = tl.max(abs_y, axis=1)
            scales_raw = absmax / FP8_MAX
            if USE_UE8M0:
                scales_raw = tl.exp2(tl.ceil(tl.log2(scales_raw)))
            scales = tl.maximum(scales_raw, FP8_MIN_SCALING_FACTOR)

            y_scaled = y_el / scales[:, None]
            y_q = tl.maximum(tl.minimum(y_scaled, FP8_MAX), FP8_MIN)

            Y_base = Y_quant + rows[:, None] * stride_y_row + col_offsets
            tl.store(Y_base, y_q.to(Y_quant.dtype.element_ty), mask=mask_g)

            S_ptr = Scales + rows * stride_s_row + g * stride_s_g
            tl.store(S_ptr, scales, mask=row_mask_1d)
    else:
        m_pid = tl.program_id(0)
        n_offs = tl.arange(0, BLOCK_SIZE_N)
        NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE

        mask1 = n_offs < inp1_n_cols
        inp1 = tl.load(
            inp1_ptr + m_pid * inp1_row_stride + n_offs * inp1_col_stride,
            mask=mask1,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        if FIRST_INPUT_RES:
            res1 = tl.load(
                res1_ptr + m_pid * res1_row_stride + n_offs * res1_col_stride,
                mask=mask1,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            inp1 = inp1 + res1

        w1 = tl.load(weight1_ptr + n_offs, mask=mask1, other=0.0).to(tl.float32)

        norm1 = _rmsmorm_op(inp1, w1, inp1_n_cols, eps1)

        if FIRST_INPUT_OUT:
            mask1 = n_offs < inp1_n_cols
            tl.store(
                out1_ptr + m_pid * out1_row_stride + n_offs * out1_col_stride,
                norm1,
                mask=mask1,
            )

        out1_fp8_t, out1_block_scales = _fp8_quant_op(
            norm1, 1, BLOCK_SIZE_N, QUANT_BLOCK_SIZE, DTYPE_MAX, DTYPE_MIN
        )
        out1_fp8_t = tl.ravel(out1_fp8_t)
        out1_block_scales = tl.ravel(out1_block_scales)

        tl.store(
            out1_fp8_ptr + m_pid * out1_fp8_row_stride + n_offs * out1_fp8_col_stride,
            out1_fp8_t.to(out1_fp8_ptr.dtype.element_ty),
            mask=mask1,
        )
        g_offs = tl.arange(0, NUM_QUANT_BLOCKS)
        num_bs_cols = (inp1_n_cols + QUANT_BLOCK_SIZE - 1) // QUANT_BLOCK_SIZE
        tl.store(
            out1_bs_ptr + m_pid * out1_bs_row_stride + g_offs * out1_bs_col_stride,
            out1_block_scales.to(out1_bs_ptr.dtype.element_ty),
            mask=g_offs < num_bs_cols,
        )
        if HAVE_SECOND_INPUT:
            mask2 = n_offs < inp2_n_cols
            inp2 = tl.load(
                inp2_ptr + m_pid * inp2_row_stride + n_offs * inp2_col_stride,
                mask=mask2,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            w2 = tl.load(weight2_ptr + n_offs, mask=mask2, other=0.0).to(tl.float32)
            norm2 = _rmsmorm_op(inp2, w2, inp2_n_cols, eps2)
            tl.store(
                out2_ptr + m_pid * out2_row_stride + n_offs * out2_col_stride,
                norm2,
                mask=mask2,
            )

        if FIRST_INPUT_RES:
            inp1 = inp1.to(out_res1_ptr.dtype.element_ty)
            tl.store(
                out_res1_ptr
                + m_pid * out_res1_row_stride
                + n_offs * out_res1_col_stride,
                inp1,
                mask=mask1,
            )



def run(*, x1, weight1, x2, weight2, epsilon1, epsilon2, out1_q, out1_scale, out2):
    M, N1 = x1.shape
    N2 = x2.shape[1]
    group_size = 128
    # One program per row; the block covers both rows' widths (the upstream launcher's heuristic).
    BLOCK_SIZE_N = max(triton.next_power_of_2(N1), group_size)
    BLOCK_SIZE_N = max(triton.next_power_of_2(N2), BLOCK_SIZE_N)
    if BLOCK_SIZE_N <= 512:
        num_warps = 1
    elif BLOCK_SIZE_N <= 2048:
        num_warps = 4
    elif BLOCK_SIZE_N <= 4096:
        num_warps = 8
    else:
        num_warps = 16
    DTYPE_MAX = 448.0  # float8_e4m3fn

    _fused_rms_fp8_group_quant_kernel[(M,)](
        inp1_ptr=x1,
        weight1_ptr=weight1,
        inp2_ptr=x2,
        weight2_ptr=weight2,
        res1_ptr=None,
        out1_fp8_ptr=out1_q,
        out1_bs_ptr=out1_scale,
        out2_ptr=out2,
        out_res1_ptr=None,
        out1_ptr=None,
        eps1=epsilon1,
        eps2=epsilon2,
        n_rows=M,
        inp1_n_cols=N1,
        inp2_n_cols=N2,
        inp1_row_stride=x1.stride(0),
        inp2_row_stride=x2.stride(0),
        inp1_col_stride=x1.stride(1),
        inp2_col_stride=x2.stride(1),
        res1_row_stride=0,
        res1_col_stride=0,
        out1_fp8_row_stride=out1_q.stride(0),
        out1_fp8_col_stride=out1_q.stride(1),
        out1_bs_row_stride=out1_scale.stride(0),
        out1_bs_col_stride=out1_scale.stride(1),
        out2_row_stride=out2.stride(0),
        out2_col_stride=out2.stride(1),
        out_res1_row_stride=0,
        out_res1_col_stride=0,
        out1_row_stride=0,
        out1_col_stride=0,
        gate_ptr=x1,
        linear_bias_ptr=weight1,
        stride_gate_row=x1.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        QUANT_BLOCK_SIZE=group_size,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        HAVE_SECOND_INPUT=True,
        FIRST_INPUT_RES=False,
        FIRST_INPUT_OUT=False,
        GATED_RMS_FP8=False,
        RMS_TILE=512,
        ROWS_PER_BLOCK=1,
        GROUP_SIZE_GATED=1,
        NUM_GROUPS_GATED=1,
        BLOCK_G=1,
        HAS_BIAS_GATED=False,
        HAS_Z_GATED=False,
        NORM_BEFORE_GATE=False,
        FP8_MIN=-DTYPE_MAX,
        FP8_MAX=DTYPE_MAX,
        USE_UE8M0=False,
        FP8_MIN_SCALING_FACTOR=1.0,
        ACTIVATION="silu",
        num_warps=num_warps,
    )
