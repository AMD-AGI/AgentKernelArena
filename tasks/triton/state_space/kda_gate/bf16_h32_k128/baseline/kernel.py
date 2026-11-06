# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project (MIT, see LICENSE;
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang).
# Adapted from vllm-project/vllm, vllm/third_party/flash_linear_attention/ops/kda.py
# (kda_gate_fwd_kernel and fused_kda_gate).
"""Kimi Delta Attention decay gate in Triton.

    out[t, h, :] = -exp(A_log[h]) * softplus(g[t, h, :] + dt_bias[h, :])

with the softplus of beta = 1 and threshold 20 (linear above the threshold). One program handles a
block of tokens for one head and all of its channels; the result is float32.
"""
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=nw, num_stages=ns)
        for bt in [32, 64, 128]
        for nw in [2, 4, 8, 16]
        for ns in [2, 3]
    ],
    key=["H", "D"],
)
@triton.jit
def kda_gate_fwd_kernel(
    g,
    A,
    y,
    g_bias,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    T,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    n_t = i_t * BT

    b_a = tl.exp(tl.load(A + i_h).to(tl.float32))

    stride_row = H * D
    stride_col = 1

    g_ptr = tl.make_block_ptr(
        base=g + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    y_ptr = tl.make_block_ptr(
        base=y + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    b_g = tl.load(g_ptr, boundary_check=(0, 1)).to(tl.float32)

    if HAS_BIAS:
        n_d = tl.arange(0, BD)
        bias_mask = n_d < D
        b_bias = tl.load(g_bias + i_h * D + n_d, mask=bias_mask, other=0.0).to(tl.float32)
        b_g = b_g + b_bias[None, :]

    g_scaled = b_g * beta
    use_linear = g_scaled > threshold
    sp = tl.where(use_linear, b_g, (1.0 / beta) * tl.log(1.0 + tl.exp(g_scaled)))
    b_y = -b_a * sp

    tl.store(y_ptr, b_y.to(y.dtype.element_ty), boundary_check=(0, 1))


def run(*, g, A_log, dt_bias, out):
    T, H, D = g.shape
    grid = lambda meta: (triton.cdiv(T, meta["BT"]), H)
    kda_gate_fwd_kernel[grid](
        g, A_log, out, dt_bias,
        1.0, 20.0,
        T, H, D,
        BD=triton.next_power_of_2(D),
        HAS_BIAS=True,
    )
