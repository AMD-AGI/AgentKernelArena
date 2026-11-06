"""Float64 reference: out = dequant(a) @ dequant(b)^T with one scale per 1x128 block of a and per 128x128 block of b."""
import torch

_BLOCK = 128


def reference(*, a, b, a_scale, b_scale):
    a_dequant = a.double() * a_scale.double().repeat_interleave(_BLOCK, dim=1)
    b_scale_full = b_scale.double().repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    return a_dequant @ (b.double() * b_scale_full).T
