# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2016- Facebook, Inc. and the PyTorch contributors (see LICENSE).
#
# Triton rewrite of TorchInductor's direct depthwise convolution template
# (torch/_inductor/kernel/templates/triton_depthwise_conv.py.jinja, selected by
# torch/_inductor/kernel/conv.py for groups == in_channels == out_channels): no implicit GEMM,
# a float32 tile accumulator updated by one elementwise multiply-add per kernel tap from the
# masked shifted input tile and the channel's tap weight. Inductor ships the template for 1-D
# channels-last inputs; this version applies the same scheme to a 2-D NCHW feature map, one
# BLOCK_H x BLOCK_W output tile of one (image, channel) plane per program, taps unrolled.
# Standalone: no torch._inductor imports; fixed tile configuration (16 x 64, 4 warps).
import triton
import triton.language as tl


BLOCK_H = 16
BLOCK_W = 64
NUM_WARPS = 4


@triton.jit
def _depthwise_conv2d_kernel(
    X, W, Y,
    OUT_H, OUT_W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    CHANNELS: tl.constexpr, KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nc = tl.program_id(2)
    idx_n = (pid_nc // CHANNELS).to(tl.int64)
    idx_c = pid_nc % CHANNELS

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (h_offs < OUT_H)[:, None] & (w_offs < OUT_W)[None, :]

    x_base = X + idx_n * stride_xn + idx_c * stride_xc
    w_base = W + idx_c * stride_wc

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for i in tl.static_range(KERNEL_H):
        for j in tl.static_range(KERNEL_W):
            wk = tl.load(w_base + i * stride_wh + j * stride_ww).to(tl.float32)
            x_ptrs = x_base + ((h_offs + i) * stride_xh)[:, None] + ((w_offs + j) * stride_xw)[None, :]
            x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
            acc += x_vals * wk

    y_ptrs = (Y + idx_n * stride_yn + idx_c * stride_yc
              + (h_offs * stride_yh)[:, None] + (w_offs * stride_yw)[None, :])
    tl.store(y_ptrs, acc, mask=mask)


def run(*, x, weight, out) -> None:
    batch, channels = x.shape[0], x.shape[1]
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    out_h, out_w = out.shape[2], out.shape[3]
    grid = (triton.cdiv(out_w, BLOCK_W), triton.cdiv(out_h, BLOCK_H), batch * channels)
    _depthwise_conv2d_kernel[grid](
        x, weight, out,
        out_h, out_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        CHANNELS=channels, KERNEL_H=kernel_h, KERNEL_W=kernel_w,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=NUM_WARPS,
    )
