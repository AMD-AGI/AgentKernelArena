"""High-precision fused bias + leaky ReLU + scale reference used only for correctness."""
import torch

_CHUNK_IMAGES = 16


def reference(*, x, bias, negative_slope, scale):
    out = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    b64 = bias.double().reshape(1, -1, 1, 1)
    slope, gain = float(negative_slope), float(scale)
    for start in range(0, x.shape[0], _CHUNK_IMAGES):
        v = x[start:start + _CHUNK_IMAGES].double() + b64
        out[start:start + _CHUNK_IMAGES] = gain * torch.where(v >= 0, v, v * slope)
    return out
