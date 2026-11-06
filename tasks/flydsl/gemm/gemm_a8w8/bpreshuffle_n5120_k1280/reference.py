"""Float64 reference: out = (xq @ w^T) * x_scale * w_scale with per-token and per-channel scales (logical weight layout)."""
import torch


def reference(*, xq, x_scale, w, w_scale):
    product = xq.double() @ w.double().T
    return product * x_scale.double() * w_scale.double().T
