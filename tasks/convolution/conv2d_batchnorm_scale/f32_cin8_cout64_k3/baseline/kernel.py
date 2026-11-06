# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2016- Facebook, Inc. and the PyTorch contributors (see LICENSE).
#
# Triton rewrite of the kernel TorchInductor generates for
#     conv2d(x, w, b) -> batch_norm(training=False) -> * scaling_factor
# under torch.compile(mode="max-autotune") with the Triton convolution backend:
# the `convolution2d` TritonTemplate of torch/_inductor/kernel/conv.py (implicit GEMM over
# NCHW, one BLOCK_M x BLOCK_N tile of (n*h*w) x out_channels per program, fp32 accumulator)
# with the epilogue Inductor fuses into the template's store: the peeled-off conv bias and
# torch._decomp's inference batch norm, (z - running_mean) * rsqrt(running_var + eps) *
# weight + bias, followed by the scale. Standalone: no torch._inductor imports, the template
# rendered for this scenario (groups 1, stride 1, padding 0) with the taps unrolled (the
# template's UNROLL form, which Inductor itself applies only to 1x1 kernels), IEEE float32 dots
# (TF32 disabled) and one tile configuration from Inductor's conv config list,
# (BLOCK_M, BLOCK_N, BLOCK_K) = (128, 64, 16), 4 warps, 2 stages, with the 16-wide MFMA
# (matrix_instr_nonkdim = 16) that Inductor's ROCm config finalization sets.
import triton
import triton.language as tl


BLOCK_M = 128
BLOCK_N = 64
BLOCK_K = 16
NUM_WARPS = 4
NUM_STAGES = 2
MATRIX_INSTR_NONKDIM = 16


@triton.jit
def _conv2d_bn_scale_kernel(
    X, W, CONV_BIAS, BN_WEIGHT, BN_BIAS, BN_MEAN, BN_VAR, Y,
    BATCH, IN_H, IN_W, OUT_H, OUT_W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wc_out, stride_wc_in, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    bn_eps, scaling_factor,
    IN_C: tl.constexpr, OUT_C: tl.constexpr, KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    x_base = X + (idx_n.to(tl.int64) * stride_xn)[:, None]
    w_base = W + (idx_y_c * stride_wc_out)[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in tl.static_range(KERNEL_H):
        for j in tl.static_range(KERNEL_W):
            for k in tl.static_range(0, IN_C, BLOCK_K):
                idx_x_h = i + idx_y_h
                idx_x_w = j + idx_y_w
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

                w_ptrs = w_base + ((idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww))
                mask_w = (idx_x_c[:, None] < IN_C) & (idx_y_c[None, :] < OUT_C)
                matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
                acc += tl.dot(matrix_x, matrix_w, input_precision="ieee")

    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < OUT_C)[None, :]
    )

    # Fused epilogue: conv bias, inference batch norm, scale (per output channel).
    c_mask = idx_y_c < OUT_C
    conv_bias = tl.load(CONV_BIAS + idx_y_c, mask=c_mask, other=0.0)
    running_mean = tl.load(BN_MEAN + idx_y_c, mask=c_mask, other=0.0)
    running_var = tl.load(BN_VAR + idx_y_c, mask=c_mask, other=1.0)
    bn_weight = tl.load(BN_WEIGHT + idx_y_c, mask=c_mask, other=0.0)
    bn_bias = tl.load(BN_BIAS + idx_y_c, mask=c_mask, other=0.0)
    invstd = tl.rsqrt(running_var + bn_eps)
    tmp = acc + conv_bias[None, :]
    tmp = tmp - running_mean[None, :]
    tmp = tmp * invstd[None, :]
    tmp = tmp * bn_weight[None, :]
    tmp = tmp + bn_bias[None, :]
    tmp = tmp * scaling_factor

    y_ptrs = Y + (
        (idx_n.to(tl.int64) * stride_yn)[:, None]
        + (idx_y_c * stride_yc)[None, :]
        + (idx_y_h * stride_yh)[:, None]
        + (idx_y_w * stride_yw)[:, None]
    )
    tl.store(y_ptrs, tmp, mask=mask)


def run(*, x, conv_weight, conv_bias, bn_weight, bn_bias, bn_running_mean, bn_running_var,
        bn_eps, scaling_factor, out) -> None:
    batch, in_c, in_h, in_w = x.shape
    out_c, _, kernel_h, kernel_w = conv_weight.shape
    out_h, out_w = out.shape[2], out.shape[3]
    grid = (triton.cdiv(batch * out_h * out_w, BLOCK_M), triton.cdiv(out_c, BLOCK_N))
    _conv2d_bn_scale_kernel[grid](
        x, conv_weight, conv_bias, bn_weight, bn_bias, bn_running_mean, bn_running_var, out,
        batch, in_h, in_w, out_h, out_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2), conv_weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        float(bn_eps), float(scaling_factor),
        IN_C=in_c, OUT_C=out_c, KERNEL_H=kernel_h, KERNEL_W=kernel_w,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES, matrix_instr_nonkdim=MATRIX_INSTR_NONKDIM,
    )
