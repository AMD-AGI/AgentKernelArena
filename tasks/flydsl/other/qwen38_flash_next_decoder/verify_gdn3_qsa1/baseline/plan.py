# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/execution.py
# atom/mono/plan/arith.py
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/common/plan.py). Modified for
# AgentKernelArena: only the execution model and the arithmetic helpers the Qwen3.8-Flash-Next
# launch uses are kept (the build-key hashing, which reads source files, and the TP shards are not).
"""Plain-Python planning of the mono kernels: execution model and arithmetic."""

BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64


def i32(x):
    """``x``, asserted inside the device's Int32 range when it is a Python int."""
    if isinstance(x, int):
        assert -(2**31) <= x < 2**31, x
    return x


def cdiv(a, b):
    """ceil(a / b) for a >= 0, b > 0."""
    return i32((a + b - 1) // b)


def padded(n: int, align: int) -> int:
    return cdiv(n, align) * align
