# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/mx.py
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/common/mx.py). Modified for
# AgentKernelArena: only the scaled MFMA and its operand formats are kept (the FP8 clamps, E8M0
# code helpers and the shuffled FP4 weight load, which the K2i scorer does not use, are not).
"""FP8 / MX arithmetic shared by the mono kernels."""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

# the f8f6f4 MFMA's operand formats (cbsz / blgp)
FP8, FP4 = 0, 4


def mfma_scaled(a, b, c, sa, sb, a_fmt=FP8, b_fmt=FP8, sa_byte=0, sb_byte=0):
    """One 16x16x128 scaled MFMA (gfx950 f8f6f4)."""
    return fx.Vector(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32), [a, b, c, a_fmt, b_fmt, sa_byte, sa, sb_byte, sb]
        )
    )
