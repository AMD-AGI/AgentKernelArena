# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Performance baseline for the a16w16 GEMM workload.

Transcribed from the workload schema's ``baseline`` solution for this operator
(``solutions/baseline/gemm/*/gemm_aiter_baseline_*.json``, entry point
``main.py::run``). This is the production implementation the ported FlyDSL
kernel is scored against. The function body is byte-identical to the schema
source; the only divergences are this file header docstring and the ``noqa``
comment on the torch import, neither of which is executable.

``gemm_a16w16`` is a tuned dispatch, not a single kernel: it looks the shape up
in aiter's merged bf16 tuned table and picks a libtype per M bucket, so the bar
a port has to clear is not the same kind of implementation at every case. The
selection is NOT monotonic in M -- on gfx950/256CU at n=k=6144 it resolves to
aiter's own FlyDSL kernels at m=1..128 and again at m=512, and to torch's native
matmul (hipBLASLt) at m=256, 1024, 2048 and 4096. Which one a given case faces
is resolved at run time by ``aiter.tuned_gemm.get_GEMM_A16W16_config`` and is
deliberately not recorded in the task.

The libtype is also not what sets a case's distance from the fp32 reference:
that tracks whether the selected kernel splits the reduction. The FlyDSL kernel
at m=1 splits k two ways and lands around 2.4e-3, while the FlyDSL kernel at
m=512 does not split and lands near 7e-7, the same order as the hipBLASLt cases.
"""

from __future__ import annotations

import torch  # noqa: F401  (kept: the schema source imports it)


# ----- baseline -----
# Generated from @sikl_proxy: the annotated entry point is the ground truth.
def run(a, b):
    from aiter.tuned_gemm import gemm_a16w16
    return gemm_a16w16(A=a, B=b)
