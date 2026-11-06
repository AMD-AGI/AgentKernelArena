"""Float64 W8A8 int8 GEMM reference used only for correctness."""
import torch


def reference(*, a, w, scale_a, scale_b, bias):
    # int8 products summed in float64 are exact, matching an integer accumulation.
    acc = a.double() @ w.double().T
    return acc * (scale_a.double() * scale_b.double().T) + bias.double()
