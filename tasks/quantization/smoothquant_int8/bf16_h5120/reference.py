"""Reference for SmoothQuant dynamic per-token INT8 quantization, used only for correctness.

The codes are compared bit-for-bit, so the operator's arithmetic is reproduced in IEEE float32: the
smoothed value is x * x_scale, the row scale is max|smoothed| / 127 (1 for an all-zero row), and the
code is the float32 quotient smoothed / scale, saturated to [-128, 127] and truncated toward zero.
The bfloat16 activation is exact in float32, and the product and maximum introduce no accumulation.
"""
import torch

INT8_MAX = 127
INT8_MIN = -128
_CHUNK_ROWS = 4096


def reference(*, x, x_scale):
    rows = x.shape[0]
    codes = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scales = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    s = x_scale.float()
    for start in range(0, rows, _CHUNK_ROWS):
        y = x[start:start + _CHUNK_ROWS].float() * s
        amax = y.abs().amax(dim=-1, keepdim=True)
        # Divide by a tensor, not a Python scalar: on GPU torch turns `tensor / scalar` into a
        # multiplication by the float32 reciprocal, which differs from the IEEE quotient by 1 ulp.
        scale = torch.div(amax, torch.full_like(amax, INT8_MAX))
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        quotient = (y / scale).clamp(INT8_MIN, INT8_MAX)
        codes[start:start + _CHUNK_ROWS] = quotient.trunc().to(torch.int8)
        scales[start:start + _CHUNK_ROWS] = scale
    return codes, scales
