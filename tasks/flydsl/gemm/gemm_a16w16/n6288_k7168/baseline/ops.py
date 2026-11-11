# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/ops.py
# atom/mono/device/ranks.py
# atom/mono/device/stamps.py
# atom/mono/plan/execution.py
# Shared GLM helpers adapted from ROCm/aiter#6173 at 56d6390def and
# vllm-project/vllm at ca328e297b (Apache-2.0), originally
# aiter/ops/flydsl/kernels/glm5_mono/ops.py:
# Copyright (c) 2025 FlyDSL Project Contributors
"""The device primitives and execution model of sgl-amd-mono that the tile GEMM uses.

Vendored from sgl-project/sgl-amd-mono at 52486703 (``sgl_amd_mono/common/ops.py``:
``CM_NT``, ``rsrc``, ``mfma_bf16``, ``bf16_pair``; ``sgl_amd_mono/common/plan.py``:
``THREADS``, ``WAVES``), unchanged apart from the imports (``buffer_ops`` is the
bundled copy).
"""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from . import buffer_ops as bo

# ---------------------------------------------------------------- execution model
THREADS = 512
WAVES = THREADS // 64


# NT = 2: nontemporal, for weights a layer reads once
CM_NT = 2


def rsrc(addr, nbytes=None):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def mfma_bf16(a, b, c):
    return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))


def bf16_pair(a, b):
    """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
    return (
        fx.Vector.from_elements([a, b], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Float32)[0]
    )
