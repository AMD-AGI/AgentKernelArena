"""High-precision row-maximum reference used only for correctness."""
import torch


def reference(*, x):
    # The maximum of float16 values is a float16 value, so the float64 result is exact.
    return x.double().max(dim=1).values
