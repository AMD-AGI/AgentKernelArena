"""High-precision row-maximum reference used only for correctness."""
import torch


def reference(*, x):
    # The maximum of float32 values is a float32 value, so the float64 result is exact.
    return x.double().max(dim=1).values
