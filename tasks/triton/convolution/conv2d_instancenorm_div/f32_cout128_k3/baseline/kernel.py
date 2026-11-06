# SPDX-License-Identifier: BSD-3-Clause
# Adapted from PyTorch (Copyright (c) 2016- Facebook, Inc and the PyTorch contributors; see LICENSE):
#   torch/_inductor/kernel/conv.py            (conv2d_template, the Triton implicit-GEMM convolution)
#   torch/_inductor/runtime/triton_helpers.py (welford_reduce, welford_combine)
"""conv2d -> InstanceNorm2d (no affine, no running stats) -> / divide_by, float32, NCHW.

A standalone Triton rewrite of what TorchInductor generates for this module with the
Triton convolution template selected (max-autotune): the conv2d template with the bias
fused into its store, then one kernel per (sample, channel) plane that computes the
Welford mean and biased variance in a first loop and normalizes and divides in a second.
The convolution result is staged in the output tensor and normalized in place.
"""
import triton
import triton.language as tl

# Conv template configuration, fixed instead of autotuned: the ROCm conv_configs entry
# ConvConfig(128, 128, 32, 2, 8) (torch/_inductor/heuristics/template/triton.py) with the
# warp count capped at 4 as the conv lowering does (torch/_inductor/kernel/conv.py: the
# non-unrolled loop of a non-1x1 kernel miscompiles with 8 warps, triton#1254).
CONV_BLOCK_M = 128
CONV_BLOCK_N = 128
CONV_BLOCK_K = 32
CONV_NUM_WARPS = 4
CONV_NUM_STAGES = 2
NORM_RBLOCK = 1024


@triton.jit
def _welford_reduce(value, mean, m2, weight, first_iteration):
    if first_iteration:
        new_weight = tl.full(weight.shape, 1, weight.dtype)
        new_mean = value
        new_m2 = tl.zeros_like(m2)
    else:
        delta = value - mean
        new_weight = weight + 1
        new_mean = mean + delta / new_weight
        new_m2 = m2 + delta * (value - new_mean)
    return new_mean, new_m2, new_weight


@triton.jit
def _welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = tl.where(mean_1 == mean_2, 0.0, mean_2 - mean_1)
    new_weight = weight_1 + weight_2
    w2_over_w = tl.where(new_weight == 0.0, 0.0, weight_2 / new_weight)
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )


@triton.jit
def _conv2d_bias_kernel(
    X, W, BIAS, Y,
    BATCH, IN_H, IN_W, OUT_H, OUT_W,
    IN_C: tl.constexpr, OUT_C: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PADDING_H: tl.constexpr, PADDING_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Contiguous NCHW strides of X, W and Y.
    stride_xw = 1
    stride_xh = IN_W
    stride_xc = IN_H * IN_W
    stride_xn = IN_C * stride_xc
    stride_ww = 1
    stride_wh = KERNEL_W
    stride_wc_in = KERNEL_H * KERNEL_W
    stride_wc_out = IN_C * stride_wc_in

    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    x_base = X + (idx_n * stride_xn)[:, None]
    w_base = W + (idx_y_c * stride_wc_out)[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    BLOCK_K_COUNT: tl.constexpr = (IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < IN_C) & (idx_y_c[None, :] < OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, input_precision="ieee")

    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :]
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # Inductor's fused epilogue: the convolution bias.
    bias = tl.load(BIAS + idx_c, mask=idx_c < OUT_C, other=0.0)
    value = acc + bias
    y_index = idx_w + OUT_W * idx_h + OUT_W * OUT_H * idx_c + OUT_W * OUT_H * OUT_C * idx_n
    tl.store(Y + y_index, value, mask)


@triton.jit
def _instance_norm_div_kernel(Y, PLANE_NUMEL, eps, divide_by, RBLOCK: tl.constexpr):
    # One program per (sample, channel) plane, normalized in place.
    row = tl.program_id(0)
    base = Y + row * PLANE_NUMEL
    rbase = tl.arange(0, RBLOCK)
    mean = tl.zeros([RBLOCK], tl.float32)
    m2 = tl.zeros([RBLOCK], tl.float32)
    weight = tl.zeros([RBLOCK], tl.float32)
    for roffset in range(0, PLANE_NUMEL, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < PLANE_NUMEL
        value = tl.load(base + rindex, rmask, other=0.0)
        mean_next, m2_next, weight_next = _welford_reduce(value, mean, m2, weight, roffset == 0)
        mean = tl.where(rmask, mean_next, mean)
        m2 = tl.where(rmask, m2_next, m2)
        weight = tl.where(rmask, weight_next, weight)
    plane_mean, plane_m2, plane_weight = tl.reduce((mean, m2, weight), 0, _welford_combine)
    rstd = tl.rsqrt(plane_m2 / plane_weight + eps)
    for roffset in range(0, PLANE_NUMEL, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < PLANE_NUMEL
        value = tl.load(base + rindex, rmask, other=0.0)
        tl.store(base + rindex, (value - plane_mean) * rstd / divide_by, rmask)


def run(*, x, conv_weight, conv_bias, epsilon, divide_by, out):
    batch, in_c, in_h, in_w = x.shape
    out_c, _, kernel_h, kernel_w = conv_weight.shape
    out_h = in_h - kernel_h + 1
    out_w = in_w - kernel_w + 1

    grid = (triton.cdiv(batch * out_h * out_w, CONV_BLOCK_M), triton.cdiv(out_c, CONV_BLOCK_N), 1)
    _conv2d_bias_kernel[grid](
        x, conv_weight, conv_bias, out,
        batch, in_h, in_w, out_h, out_w,
        IN_C=in_c, OUT_C=out_c, KERNEL_H=kernel_h, KERNEL_W=kernel_w,
        STRIDE_H=1, STRIDE_W=1, PADDING_H=0, PADDING_W=0,
        BLOCK_M=CONV_BLOCK_M, BLOCK_N=CONV_BLOCK_N, BLOCK_K=CONV_BLOCK_K,
        num_warps=CONV_NUM_WARPS, num_stages=CONV_NUM_STAGES,
    )
    _instance_norm_div_kernel[(batch * out_c,)](
        out, out_h * out_w, epsilon, divide_by, RBLOCK=NORM_RBLOCK, num_warps=4,
    )
