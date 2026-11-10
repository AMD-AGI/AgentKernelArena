"""Float64 reference: out = dequant(a) @ dequant(b)^T for 1x128-scaled activations and 128x128-scaled weights.

Receives the logical operands: a [m, k], a_scale [k / 128, m] (k-group-major, as the activation quantizer writes it),
b [n, k] in natural nn.Linear order (not preshuffled), b_scale [n / 128, k / 128].
"""
import torch

_BLOCK = 128


def reference(*, a, a_scale, b, b_scale):
    # float8 values and float32 scales are exact in float64.
    b_dequant = b.float().double() * b_scale.double().repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    a_dequant = a.float().double() * a_scale.double().T.repeat_interleave(_BLOCK, dim=1)
    return a_dequant @ b_dequant.T
