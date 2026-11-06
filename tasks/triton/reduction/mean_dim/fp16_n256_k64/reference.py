"""High-precision mean-over-axis reference used only for correctness."""
import torch


def reference(*, x):
    return x.double().mean(dim=1)
