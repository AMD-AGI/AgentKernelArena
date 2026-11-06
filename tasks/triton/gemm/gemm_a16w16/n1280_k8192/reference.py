"""High-precision GEMM reference used only for correctness."""
import torch


def reference(*, x, w):
    return x.double() @ w.double().T
