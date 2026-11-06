# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Vendored from ROCm/aiter (commit 652f2d4b768239aefd515c929e84817681ff598d),
# aiter/ops/flydsl/kernels/tensor_shim.py: ptr_arg only; the rest of that module is not used.

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch


def ptr_arg(t: torch.Tensor):
    """Wrap a torch.Tensor as an fx.Pointer (PointerJitArg) for kernel launch."""
    type_name = type(t).__name__
    module_name = type(t).__module__
    if type_name == "FakeTensor" or "fake_tensor" in module_name:
        return flyc.from_c_void_p(fx.Uint8, 0)
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())
