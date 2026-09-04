# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Performance baseline for the a16w16 GEMM workload.

Transcribed verbatim from the workload schema's ``baseline`` solution for this
operator (``solutions/baseline/gemm/*/gemm_aiter_baseline_*.json``, entry point
``main.py::run``). This is the production implementation the ported FlyDSL
kernel is scored against.

``gemm_a16w16`` is a tuned dispatch, not a single kernel: it looks the shape up
in aiter's merged bf16 tuned table and picks a libtype per M bucket. On
gfx950/256CU this shape family resolves to aiter's own FlyDSL kernels over much
of the small-M range and to torch's native matmul at the larger M, so the bar a
port has to clear is not the same kind of implementation at every case. Which
one a given case faces is resolved at run time by
``aiter.tuned_gemm.get_GEMM_A16W16_config`` and is deliberately not recorded in
the task.
"""

from __future__ import annotations

import torch  # noqa: F401  (kept: the schema source imports it)


# ----- baseline -----
# Generated from @sikl_proxy: the annotated entry point is the ground truth.
def run(a, b):
    from aiter.tuned_gemm import gemm_a16w16

    return gemm_a16w16(A=a, B=b)
