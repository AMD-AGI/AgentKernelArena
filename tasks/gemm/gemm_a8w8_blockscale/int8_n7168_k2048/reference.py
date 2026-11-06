"""Float64 reference: out = (a * a_scale) @ (b * b_scale)^T with 1x128 activation and 128x128 weight scales."""
import torch

_BLOCK = 128


def reference(*, a, a_scale, b, b_scale):
    m, k = a.shape
    a_dequant = (a.double().view(m, k // _BLOCK, _BLOCK) * a_scale.double().unsqueeze(-1)).view(m, k)
    scale = b_scale.double().repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    b_dequant = b.double() * scale
    return a_dequant @ b_dequant.T
