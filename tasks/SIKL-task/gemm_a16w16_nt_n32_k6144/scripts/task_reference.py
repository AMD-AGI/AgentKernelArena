# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Correctness reference for the a16w16 GEMM workload.

Transcribed from the workload schema's ``reference`` solution for this operator
(``solutions/reference/gemm/*/gemm_torch_reference_*.json``, entry point
``main.py::run``): a plain fp32 matmul. It says what the answer is, not how fast
it should be -- an fp32 accumulation of the whole reduction is slower than any
kernel worth submitting.

Two documented divergences from the schema source, so a future comparison
against the bundle knows what to expect:

1. ``from typing import Optional`` is added. The schema source annotates ``bias``
   with it and never imports it; that survives there only because
   ``from __future__ import annotations`` keeps annotations unevaluated.
2. The schema source's unused ``math`` and ``torch.nn.functional`` imports are
   dropped.
"""

from __future__ import annotations

from typing import Optional

import torch


def _gemm_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_b: bool = True,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Plain fp32 matmul; b is (n, k) when trans_b, else (k, n)."""
    rhs = b.transpose(-1, -2) if trans_b else b
    out = a.to(torch.float32) @ rhs.to(torch.float32)
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(a.dtype)


# ----- entry point -----
def run(*args, **kwargs):
    return _gemm_reference(*args, **kwargs)
