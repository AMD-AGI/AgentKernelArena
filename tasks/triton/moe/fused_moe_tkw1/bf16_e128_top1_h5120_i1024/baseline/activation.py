# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# (ROCm/aiter aiter/ops/triton/_triton_kernels/activation.py: _silu_exp2.)

import triton
import triton.language as tl


@triton.jit
def _silu_exp2(x):
    return x / (1.0 + tl.exp2(-(x * 1.44269504089)))
