# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dynamic MXFP8 (E4M3, per-1x32 E8M0 scale) quantization in Triton, from AITER."""
import triton
import triton.language as tl

_MXFP8_QUANT_BLOCK_SIZE = 32


# MXFP8 (1x32 e8m0) quant: derives a per-block uint8 e8m0 scale + FP8 e4m3
# values. The bit-trick (bitcast amax to int32, add 0x200000, mask 0xFF800000,
# bitcast back to fp32) rounds amax up to a power of 2; log2(amax).floor() - 8
# is the unbiased e8m0 exponent (dtypeMax = 2**8).


@triton.jit
def _mxfp8_quant_op(x_grouped, QUANT_AXIS: tl.constexpr):
    """Shared MXFP8 (1x32 e8m0) scale derivation.

    Given a fp32 tile where the QUANT_AXIS dim is sized QUANT_BLOCK_SIZE (=32),
    returns (scale_e8m0, quant_scale): the per-group uint8 e8m0 scale and the
    matching fp32 multiplicative scale. Both outputs keep QUANT_AXIS with size 1
    so they broadcast against the input for in-place quantization.
    """
    amax = tl.max(tl.abs(x_grouped), axis=QUANT_AXIS, keep_dims=True)
    amax_i32 = amax.to(tl.int32, bitcast=True)
    amax_i32 = (amax_i32 + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax_p2 = amax_i32.to(tl.float32, bitcast=True)
    scale_unbiased = tl.log2(amax_p2).floor() - 8
    scale_unbiased = tl.clamp(scale_unbiased, min=-127, max=127)
    scale_e8m0 = (scale_unbiased.to(tl.int32) + 127).to(tl.uint8)
    quant_scale = tl.exp2(-scale_unbiased)
    return scale_e8m0, quant_scale


@triton.jit
def _dynamic_mxfp8_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    BLOCK_SIZE_N: tl.constexpr,  # power-of-2 covering full N
    QUANT_BLOCK_SIZE: tl.constexpr,  # =32
    NUM_PRGMS: tl.constexpr,  # row-loop range (usually =M)
):
    """
    Per-1x32 MXFP8 quant. One program per row, holding the full row in
    registers so a single launch handles all K-groups. Mirrors
    _fused_rms_mxfp8_kernel shape (in fused_mxfp8_quant.py) and minimizes
    grid overhead.
    """
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = col_offsets < N
    n_groups: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE

    for row_idx in tl.range(row_start, M, NUM_PRGMS, num_stages=2):
        x = tl.load(
            x_ptr + row_idx * stride_xm + col_offsets * stride_xn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        # (BLOCK_SIZE_N,) -> (n_groups, QUANT_BLOCK_SIZE)
        x_2d = tl.reshape(x, (n_groups, QUANT_BLOCK_SIZE))
        scale_e8m0, quant_scale = _mxfp8_quant_op(x_2d, QUANT_AXIS=1)

        qx_2d = x_2d * quant_scale
        qx = tl.reshape(qx_2d, (BLOCK_SIZE_N,))
        y = qx.to(y_ptr.type.element_ty)

        tl.store(
            y_ptr + row_idx * stride_ym + col_offsets * stride_yn,
            y,
            mask=mask,
        )

        group_offsets = tl.arange(0, n_groups)
        group_mask = group_offsets < (N // QUANT_BLOCK_SIZE)
        scale_flat = tl.reshape(scale_e8m0, (n_groups,))
        tl.store(
            s_ptr + row_idx * stride_sm + group_offsets * stride_sn,
            scale_flat,
            mask=group_mask,
        )


def run(*, x, y, scale):
    M, K = x.shape
    assert (
        K % _MXFP8_QUANT_BLOCK_SIZE == 0
    ), f"last dim K={K} must be a multiple of {_MXFP8_QUANT_BLOCK_SIZE}"

    BLOCK_SIZE_N = triton.next_power_of_2(K)
    NUM_PRGMS = M
    grid = (NUM_PRGMS,)

    _dynamic_mxfp8_quant_kernel[grid](
        x,
        y,
        scale,
        M,
        K,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        scale.stride(0),
        scale.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        QUANT_BLOCK_SIZE=_MXFP8_QUANT_BLOCK_SIZE,
        NUM_PRGMS=NUM_PRGMS,
    )
