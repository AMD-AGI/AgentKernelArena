# SPDX-License-Identifier: Apache-2.0
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/minimax_m31/a16w8.py). Modified for
# AgentKernelArena: imports are package-relative; the host-side ``layout`` (a load-time weight
# transform) moved to the task's inputs.py (``a16w8_layout``), so this file holds device code only.
"""MXFP8 weights for the shared expert's GEMVs: ``a16w4``'s lane-to-K mapping (lane l
holds row l % 16 over K ``step * 128 + (l // 16) * 32 ..``, MFMA d its K ``.. + 8 d``),
with 32 FP8 bytes a lane a step and the lane's group-32 scale as an f32, both in
``layout``'s order. The checkpoint stores the shared expert in MXFP8, so these are its
own values; an MXFP4 copy has about 4.6x the output error and costs DSpark acceptance.

The weight layout (``inputs.a16w8_layout`` here, ``layout`` upstream): ``weight`` [R, K] FP8 and its E8M0
group-32 ``scale`` [R, K / 32] -> (uint8 [R/16, K/128, 64, 32], f32 [R/16, K/128, 64]), lane =
kg * 16 + row % 16 for the K group kg = (k % 128) // 32 of the step."""

import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

from . import buffer_ops as bo
from .ops import CM_NT, mfma_bf16, rsrc

STEP = 128  # K a 16-row group covers in one step, as in a16w4
LANE_DWORDS = 8  # 32 FP8 a lane a step


def weight_loads(w, row_group, k, st0, steps, cm=CM_NT):
    """The lane's two 16-byte words of each step ``st0 ..`` of 16-row group
    ``row_group``; loads only."""
    lane = fx.thread_idx.x % 64
    per_group = k // STEP
    return [
        [
            fx.Vector(
                bo.buffer_load(
                    rsrc(w),
                    ((row_group * per_group + st0 + i) * 64 + lane) * LANE_DWORDS
                    + 4 * h,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=cm,
                )
            )
            for h in range(2)
        ]
        for i in range(steps)
    ]


def scale_loads(ws, row_group, k, st0, steps, cm=CM_NT):
    """The lane's f32 scale of each step ``st0 ..``."""
    lane = fx.thread_idx.x % 64
    per_group = k // STEP
    return [
        fx.Float32(
            bo.buffer_load(
                rsrc(ws),
                (row_group * per_group + st0 + i) * 64 + lane,
                vec_width=1,
                dtype=T.f32,
                cache_modifier=cm,
            )
        )
        for i in range(steps)
    ]


def dequant(lo, hi, scale):
    """Two FP8 dwords (8 values, byte 0 first) -> 8 BF16 (as 4 i32)."""
    pairs = [
        fx.Vector(
            rocdl.cvt_scalef32_pk_bf16_fp8(
                T.vec(2, T.bf16), word.ir_value(), scale.ir_value(), half
            )
        ).bitcast(fx.Int32)[0]
        for word in (lo, hi)
        for half in (False, True)
    ]
    return fx.Vector.from_elements(pairs, fx.Int32)


def step_mfma(acc, wv, scale, xs):
    """``acc`` += one step: the lane's words ``wv`` (scale ``scale``) times its four
    B operands ``xs``, MFMA d over dwords 2 d and 2 d + 1."""
    for d in range_constexpr(4):
        words = wv[d // 2]
        lo, hi = fx.Int32(words[(d % 2) * 2]), fx.Int32(words[(d % 2) * 2 + 1])
        a = dequant(lo, hi, scale).bitcast(fx.BFloat16)
        acc = mfma_bf16(a, xs[d].bitcast(fx.BFloat16), acc)
    return acc
