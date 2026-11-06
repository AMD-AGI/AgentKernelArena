# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2016- Facebook, Inc and PyTorch contributors. See baseline/LICENSE.
"""conv2d(3x3) -> - subtract1 -> tanh -> - subtract2 -> avg_pool2d(2), float32 NCHW, in Triton.

A standalone rewrite of what torch.compile(mode="max-autotune") builds for this module with
PyTorch Inductor's ``convolution2d`` Triton template (torch/_inductor/kernel/conv.py): an
implicit GEMM [output pixels, out_c] x [in_c * kh * kw] whose reduction gathers the shifted
input pixels and the matching filter slice and accumulates ``tl.dot`` with IEEE float32
products (Inductor's ALLOW_TF32=False). Here the reduction runs over (input channel, filter
row, filter column) in the OIHW weight order, and the bias, both subtractions, tanh
(libdevice, as Inductor's pointwise codegen) and the 2x2 average pool are fused into the
epilogue; Inductor emits the pool as a second kernel over a materialized intermediate.

Each program owns one pooled output row of one image, BLOCK_PW pooled columns and BLOCK_N
output channels. Its BLOCK_M = 2 * 2 * BLOCK_PW implicit-GEMM rows are the two convolution
rows under that pooled row, ordered (row, pooled column, column in window), so the epilogue
reshapes the tile and sums the 2x2 windows in registers. The tile configuration is fixed
instead of autotuned.
"""
import triton
import triton.language as tl
from triton.language.extra import libdevice

POOL = 2
BLOCK_PW = 32   # pooled columns per program: BLOCK_M = 2 * 2 * 32 = 128 convolution outputs
BLOCK_N = 64
BLOCK_K = 32
NUM_WARPS = 4
NUM_STAGES = 2


@triton.jit
def _conv_sub_tanh_sub_avgpool_kernel(
    X, W, B, Y,
    subtract1, subtract2,
    POOL_H, POOL_W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wc_out, stride_wc_in, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    IN_C: tl.constexpr, OUT_C: tl.constexpr, KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    POOL: tl.constexpr, BLOCK_PW: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    BLOCK_M: tl.constexpr = POOL * POOL * BLOCK_PW
    TILE_W: tl.constexpr = POOL * BLOCK_PW
    TAPS: tl.constexpr = KERNEL_H * KERNEL_W
    RED_K: tl.constexpr = IN_C * TAPS

    pid_row = tl.program_id(0)
    pid_pw = tl.program_id(1)
    pid_c = tl.program_id(2)
    idx_n = pid_row // POOL_H
    idx_ph = pid_row % POOL_H

    # Implicit-GEMM rows: convolution pixels (idx_y_h, idx_y_w) under this pooled row.
    offs_m = tl.arange(0, BLOCK_M)
    idx_y_h = idx_ph * POOL + offs_m // TILE_W
    idx_y_w = pid_pw * TILE_W + offs_m % TILE_W
    mask_m = idx_y_w < POOL_W * POOL
    idx_y_c = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_c = idx_y_c < OUT_C

    x_base = X + idx_n.to(tl.int64) * stride_xn
    x_m = idx_y_h * stride_xh + idx_y_w * stride_xw
    w_c = idx_y_c * stride_wc_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, RED_K, BLOCK_K):
        idx_k = k + tl.arange(0, BLOCK_K)
        mask_k = idx_k < RED_K
        idx_x_c = idx_k // TAPS
        tap = idx_k % TAPS
        i = tap // KERNEL_W
        j = tap % KERNEL_W

        x_k = idx_x_c * stride_xc + i * stride_xh + j * stride_xw
        matrix_x = tl.load(x_base + (x_m[:, None] + x_k[None, :]),
                           mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w_k = idx_x_c * stride_wc_in + i * stride_wh + j * stride_ww
        matrix_w = tl.load(W + (w_k[:, None] + w_c[None, :]),
                           mask=mask_k[:, None] & mask_c[None, :], other=0.0)
        acc += tl.dot(matrix_x, matrix_w, input_precision="ieee")

    # Fused epilogue: conv bias, x - subtract1, tanh, - subtract2, 2x2 average pool.
    bias = tl.load(B + idx_y_c, mask=mask_c, other=0.0)
    v = libdevice.tanh(acc + bias[None, :] - subtract1) - subtract2
    v = tl.reshape(v, (POOL, BLOCK_PW, POOL, BLOCK_N))
    v = tl.sum(tl.sum(v, axis=2), axis=0) * (1.0 / (POOL * POOL))

    idx_pw = pid_pw * BLOCK_PW + tl.arange(0, BLOCK_PW)
    y_ptrs = (Y + idx_n.to(tl.int64) * stride_yn + idx_ph * stride_yh
              + (idx_pw * stride_yw)[:, None] + (idx_y_c * stride_yc)[None, :])
    tl.store(y_ptrs, v, mask=(idx_pw < POOL_W)[:, None] & mask_c[None, :])


def run(*, x, conv_weight, conv_bias, subtract1_value, subtract2_value, out):
    batch, in_c = x.shape[0], x.shape[1]
    out_c, _, kernel_h, kernel_w = conv_weight.shape
    pool_h, pool_w = out.shape[2], out.shape[3]
    grid = (batch * pool_h, triton.cdiv(pool_w, BLOCK_PW), triton.cdiv(out_c, BLOCK_N))
    _conv_sub_tanh_sub_avgpool_kernel[grid](
        x, conv_weight, conv_bias, out,
        subtract1_value, subtract2_value,
        pool_h, pool_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2), conv_weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        IN_C=in_c, OUT_C=out_c, KERNEL_H=kernel_h, KERNEL_W=kernel_w,
        POOL=POOL, BLOCK_PW=BLOCK_PW, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
