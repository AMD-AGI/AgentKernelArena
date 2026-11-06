# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2016- Facebook, Inc and PyTorch contributors. See baseline/LICENSE.
"""Conv2d + Mish + eval-mode BatchNorm2d (float32 NCHW) as one Triton kernel.

A standalone rewrite of what PyTorch Inductor generates for
``bn(x_conv * tanh(softplus(x_conv)))`` with ``x_conv = conv(x)`` in eval mode:
its ``convolution2d`` Triton template (torch/_inductor/kernel/conv.py, the
non-unrolled loop used for kernels larger than 1x1) with the pointwise tail
fused into the template epilogue. Inductor peels the convolution bias off the
template and adds it back in the epilogue, lowers softplus as
``where(v > 20, v, log1p(exp(v)))`` and eval-mode batch norm as
``(v - running_mean) * rsqrt(running_var + eps) * weight + bias``
(torch/_decomp/decompositions.py). Each program computes a BLOCK_M x BLOCK_N
tile of the implicit GEMM [batch * out_h * out_w, out_c] x [in_c * kh * kw]
with IEEE float32 products and float32 accumulation (ALLOW_TF32=False), applies
the epilogue in float32 and writes the NCHW output in place. The tile
configuration is fixed to one of Inductor's ROCm conv configs instead of being
autotuned.
"""
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Inductor ROCm conv config (128, 128, 32, num_stages=2); the non-unrolled template caps
# num_warps at 4.
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
NUM_STAGES = 2
NUM_WARPS = 4
MATRIX_INSTR_NONKDIM = 16
SOFTPLUS_THRESHOLD = 20.0


@triton.jit
def _conv2d_mish_batchnorm_kernel(
    X, W, CONV_BIAS, RUNNING_MEAN, RUNNING_VAR, BN_WEIGHT, BN_BIAS, Y,
    eps,
    BATCH, IN_H, IN_W, OUT_H, OUT_W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wc_out, stride_wc_in, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    IN_C: tl.constexpr, OUT_C: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PADDING_H: tl.constexpr, PADDING_W: tl.constexpr,
    THRESHOLD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    INDEX_64: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if INDEX_64:
        pid_m = pid_m.to(tl.int64)
        pid_n = pid_n.to(tl.int64)

    nhw = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_base = X + (idx_n * stride_xn)[:, None]
    w_base = W + (idx_y_c * stride_wc_out)[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    BLOCK_K_COUNT = (IN_C + BLOCK_K - 1) // BLOCK_K
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

    # Fused epilogue: conv bias, Mish (x * tanh(softplus(x))), eval-mode batch norm.
    mask_c = idx_y_c < OUT_C
    conv_bias = tl.load(CONV_BIAS + idx_y_c, mask=mask_c, other=0.0)
    mean = tl.load(RUNNING_MEAN + idx_y_c, mask=mask_c, other=0.0)
    var = tl.load(RUNNING_VAR + idx_y_c, mask=mask_c, other=1.0)
    gamma = tl.load(BN_WEIGHT + idx_y_c, mask=mask_c, other=0.0)
    beta = tl.load(BN_BIAS + idx_y_c, mask=mask_c, other=0.0)

    v = acc + conv_bias[None, :]
    softplus = tl.where(v > THRESHOLD, v, libdevice.log1p(tl.exp(v)))
    mish = libdevice.tanh(softplus) * v
    invstd = libdevice.rsqrt(var + eps)
    out = (mish - mean[None, :]) * invstd[None, :]
    out = out * gamma[None, :]
    out = out + beta[None, :]

    y_ptrs = Y + (
        (idx_n * stride_yn)[:, None]
        + (idx_y_c * stride_yc)[None, :]
        + (idx_y_h * stride_yh)[:, None]
        + (idx_y_w * stride_yw)[:, None]
    )
    tl.store(y_ptrs, out, mask=mask)


def run(*, x, conv_weight, conv_bias, bn_weight, bn_bias, running_mean, running_var, eps, out):
    batch, in_c, in_h, in_w = x.shape
    out_c, _, kernel_h, kernel_w = conv_weight.shape
    out_h, out_w = out.shape[2], out.shape[3]
    index_64 = max(x.numel(), conv_weight.numel(), out.numel()) >= 2 ** 31
    grid = (triton.cdiv(batch * out_h * out_w, BLOCK_M), triton.cdiv(out_c, BLOCK_N), 1)
    _conv2d_mish_batchnorm_kernel[grid](
        x, conv_weight, conv_bias, running_mean, running_var, bn_weight, bn_bias, out,
        eps,
        batch, in_h, in_w, out_h, out_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2), conv_weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        IN_C=in_c, OUT_C=out_c, KERNEL_H=kernel_h, KERNEL_W=kernel_w,
        STRIDE_H=1, STRIDE_W=1, PADDING_H=0, PADDING_W=0,
        THRESHOLD=SOFTPLUS_THRESHOLD,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, INDEX_64=index_64,
        num_stages=NUM_STAGES, num_warps=NUM_WARPS, matrix_instr_nonkdim=MATRIX_INSTR_NONKDIM,
    )
