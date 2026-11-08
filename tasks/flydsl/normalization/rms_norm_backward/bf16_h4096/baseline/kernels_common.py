# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Common helpers shared by kernel modules.

Keep helper naming consistent with other kernel helpers (e.g. `mfma_preshuffle_pipeline.py`),
but this module is intentionally small and MLIR-dialect facing.
"""

import flydsl.expr as fx
from flydsl.runtime.device import get_warp_size as get_warp_size


def dtype_to_elem_type(dtype_str: str):
    """Map a dtype string to its FlyDSL numeric type.

    Supported: 'f32', 'f16', 'bf16', 'fp8' (OCP e4m3fn, not the fnuz variant).
    """
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    if dtype_str == "fp8":
        return fx.Float8E4M3FN
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f32', 'f16', 'bf16', or 'fp8')")
