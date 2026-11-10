# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/execution.py
# atom/mono/plan/arith.py
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/common/plan.py). Modified for
# AgentKernelArena: only the execution model and the arithmetic helpers the K2i scorer uses are
# kept (the mailbox layouts, TP shards and the build-key hashing, which reads source files and
# imports msgspec, are not).
"""Plain-Python planning of the mono kernels: execution model and arithmetic."""

BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64


def sel(pred, a, b):
    """Return a if pred else b, as ``select`` on traced values."""
    if isinstance(pred, bool):
        return a if pred else b
    out = pred.select(a, b)
    traced = [x for x in (a, b) if not isinstance(x, int)]
    return type(traced[0])(out) if traced else out


def i32(x):
    """``x``, asserted inside the device's Int32 range when it is a Python int."""
    if isinstance(x, int):
        assert -(2**31) <= x < 2**31, x
    return x


def cdiv(a, b):
    """ceil(a / b) for a >= 0, b > 0."""
    return i32((a + b - 1) // b)
