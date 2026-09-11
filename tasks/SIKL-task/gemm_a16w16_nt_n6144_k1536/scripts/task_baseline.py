# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Performance baseline for this workload, copied from the schema bundle.

Everything below this docstring is the bundle's ``baseline`` solution, taken
without edit from schema v2. It is the production implementation a ported FlyDSL
kernel is scored against, so the task measures the same thing the acceptance run
measures. Do not edit it here: change it in the bundle and copy it down again.

``run`` is the entry point the bundle exports.
"""

from __future__ import annotations
import torch

# ----- baseline -----
# Generated from @sikl_proxy: the annotated entry point is the ground truth.
def run(a, b):
    from aiter.tuned_gemm import gemm_a16w16
    return gemm_a16w16(A=a, B=b)
