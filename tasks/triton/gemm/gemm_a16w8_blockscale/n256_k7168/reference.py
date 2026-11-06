"""Float64 reference: out = x @ dequant(w)^T with a float32 scale per 128x128 weight block."""
import torch

_BLOCK = 128


def reference(*, x, w, w_scale):
    scale = w_scale.double().repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    w_dequant = w.double() * scale
    return x.double() @ w_dequant.T
