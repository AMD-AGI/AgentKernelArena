# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Correctness reference for this workload, copied from the schema bundle.

Everything below this docstring is the definition's ``reference`` callback, taken
without edit from schema v2 so that this task and the acceptance run that
verifies its result apply one implementation rather than two that agree today.
Do not edit it here: change it in the bundle and copy it down again, or the two
silently diverge -- which is exactly the failure this file exists to prevent.

``run`` is the entry point the bundle exports.
"""

from __future__ import annotations
from functools import partial as partial
from typing import Optional as Optional
import torch as torch

def _gemm_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    trans_b: bool = True,
) -> torch.Tensor:
    """Plain fp32 matmul; b is (n, k) when trans_b, else (k, n)."""
    rhs = b.transpose(-1, -2) if trans_b else b
    out = a.to(torch.float32) @ rhs.to(torch.float32)
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(a.dtype)


_callable = partial(_gemm_reference, **{'trans_b': True})


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
