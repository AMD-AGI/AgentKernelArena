"""High-precision GEMM reference used only for correctness."""
import torch


def reference(*, a, b):
    return a.double() @ b.double()
