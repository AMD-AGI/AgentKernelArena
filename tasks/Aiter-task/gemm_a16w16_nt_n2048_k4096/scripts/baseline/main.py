from __future__ import annotations
import torch

# ----- baseline -----
# Generated from @sikl_proxy: the annotated entry point is the ground truth.
def run(a, b):
    from aiter.tuned_gemm import gemm_a16w16
    return gemm_a16w16(A=a, B=b)
