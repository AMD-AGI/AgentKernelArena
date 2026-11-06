"""Float64 FP8 row-wise-scaled GEMM reference used only for correctness."""
import torch


def reference(*, x, w, x_scale, w_scale):
    # float8 values are exactly representable in float32 and float64; the float64 products are exact.
    acc = x.float().double() @ w.float().double().T
    return acc * (x_scale.double() * w_scale.double())
