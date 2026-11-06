"""High-precision row-maximum reference used only for correctness."""
import torch


def reference(*, x):
    # The maximum of bfloat16 values is a bfloat16 value, so the float64 result is exact.
    return x.double().max(dim=1).values
