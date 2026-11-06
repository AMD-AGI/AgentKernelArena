# SPDX-License-Identifier: BSD-3-Clause
"""Conv3d (3 -> 16 channels, 3x3x3, bias) -> channel softmax -> two 2x2x2 max pools, in Triton.

Standalone Triton rewrite of what PyTorch Inductor builds for this module with its
Triton convolution template (torch/_inductor/kernel/conv.py, ``conv3d_template``):

1. ``_conv3d_bias_softmax_kernel``: an implicit GEMM over (output position) x
   (output channel), one program per BLOCK_M output positions and all 16 output
   channels, float32 accumulation with IEEE float32 dot products. Because the
   input has only 3 channels, the reduction runs over the flattened
   (in_channel, kd, kh, kw) axis of the NCDHW input instead of per-tap channel
   blocks. The bias add and the softmax over the 16 channels run in the epilogue;
   the probabilities are written to a float32 intermediate.
2. ``_max_pool_kernel``: two stride-2 2x2x2 floor-mode max pools compose exactly
   to one stride-4 4x4x4 window, so one kernel takes the maximum of the 64
   probabilities of each window.
"""
import torch
import triton
import triton.language as tl

C_IN = 3
C_OUT = 16
KERNEL = 3
WINDOW = 4  # two 2x2x2 stride-2 pools == one 4x4x4 stride-4 window
BLOCK_M = 128
BLOCK_K = 32
BLOCK_POOL = 256


@triton.jit
def _conv3d_bias_softmax_kernel(
    x_ptr, w_ptr, b_ptr, p_ptr,
    M, IN_D, IN_H, IN_W, CONV_D, CONV_H, CONV_W,
    C_IN: tl.constexpr, C_OUT: tl.constexpr, KS: tl.constexpr, K_TOTAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m < M
    ow = m % CONV_W
    t = m // CONV_W
    oh = t % CONV_H
    t = t // CONV_H
    od = t % CONV_D
    n = (t // CONV_D).to(tl.int64)

    in_hw = IN_H * IN_W
    in_dhw = IN_D * in_hw
    x_row = n * (C_IN * in_dhw) + (od * in_hw + oh * IN_W + ow)
    oc = tl.arange(0, C_OUT)

    acc = tl.zeros((BLOCK_M, C_OUT), dtype=tl.float32)
    for k0 in tl.static_range(0, K_TOTAL, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K_TOTAL
        c = k // (KS * KS * KS)
        r = k % (KS * KS * KS)
        kd = r // (KS * KS)
        kh = (r // KS) % KS
        kw = r % KS
        x_col = c * in_dhw + kd * in_hw + kh * IN_W + kw
        a = tl.load(x_ptr + (x_row[:, None] + x_col[None, :]),
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptr + (oc[None, :] * K_TOTAL + k[:, None]), mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, w, acc, input_precision="ieee")

    acc += tl.load(b_ptr + oc)[None, :]
    row_max = tl.max(acc, axis=1)
    e = tl.exp(acc - row_max[:, None])
    prob = e / tl.sum(e, axis=1)[:, None]

    conv_hw = CONV_H * CONV_W
    conv_dhw = CONV_D * conv_hw
    p_row = n * (C_OUT * conv_dhw) + (od * conv_hw + oh * CONV_W + ow)
    tl.store(p_ptr + (p_row[:, None] + oc[None, :] * conv_dhw), prob, mask=m_mask[:, None])


@triton.jit
def _max_pool_kernel(
    p_ptr, out_ptr, TOTAL, CONV_D, CONV_H, CONV_W, POOL_D, POOL_H, POOL_W,
    WINDOW: tl.constexpr, BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < TOTAL
    pw = idx % POOL_W
    t = idx // POOL_W
    ph = t % POOL_H
    t = t // POOL_H
    pd = t % POOL_D
    nc = (t // POOL_D).to(tl.int64)

    conv_hw = CONV_H * CONV_W
    base = nc * (CONV_D * conv_hw) + (pd * WINDOW * conv_hw + ph * WINDOW * CONV_W + pw * WINDOW)
    best = tl.full((BLOCK,), float("-inf"), tl.float32)
    for dd in tl.static_range(WINDOW):
        for hh in tl.static_range(WINDOW):
            for ww in tl.static_range(WINDOW):
                v = tl.load(p_ptr + (base + (dd * conv_hw + hh * CONV_W + ww)), mask=mask,
                            other=float("-inf"))
                best = tl.maximum(best, v)
    tl.store(out_ptr + idx, best, mask=mask)


def run(*, x, weight, bias, out):
    batch, _, in_d, in_h, in_w = x.shape
    conv_d, conv_h, conv_w = in_d - KERNEL + 1, in_h - KERNEL + 1, in_w - KERNEL + 1
    probs = torch.empty((batch, C_OUT, conv_d, conv_h, conv_w), device=x.device, dtype=torch.float32)
    m = batch * conv_d * conv_h * conv_w
    _conv3d_bias_softmax_kernel[(triton.cdiv(m, BLOCK_M),)](
        x, weight, bias, probs, m, in_d, in_h, in_w, conv_d, conv_h, conv_w,
        C_IN=C_IN, C_OUT=C_OUT, KS=KERNEL, K_TOTAL=C_IN * KERNEL ** 3,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, num_warps=4)
    pool_d, pool_h, pool_w = out.shape[2], out.shape[3], out.shape[4]
    total = batch * C_OUT * pool_d * pool_h * pool_w
    _max_pool_kernel[(triton.cdiv(total, BLOCK_POOL),)](
        probs, out, total, conv_d, conv_h, conv_w, pool_d, pool_h, pool_w,
        WINDOW=WINDOW, BLOCK=BLOCK_POOL, num_warps=4)
