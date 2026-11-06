"""Reference for per-token INT8 quantization, used only for correctness.

The codes are compared bit-for-bit, so the operator's arithmetic is reproduced in IEEE float32:
scale = max(max|x|, 1e-10) / 127, quotient = x * (127 / max(max|x|, 1e-10)) (a reciprocal multiply,
two correctly rounded operations), and the quotient is rounded to the nearest integer with ties away
from zero (the device `round`). The float16 inputs and the absolute maximum are exact in float32.
"""
import torch

INT8_MAX = 127
INT8_MIN = -128
EPS = 1e-10
_CHUNK_ROWS = 4096


def _round_half_away(v):
    # floor(|v|) and the fractional part are exact in float32, unlike floor(|v| + 0.5).
    a = v.abs()
    r = a.floor()
    r = r + ((a - r) >= 0.5).to(v.dtype)
    return torch.copysign(r, v)


def reference(*, x):
    rows = x.shape[0]
    codes = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scales = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    for start in range(0, rows, _CHUNK_ROWS):
        v = x[start:start + _CHUNK_ROWS].float()
        absmax = v.abs().amax(dim=-1, keepdim=True).clamp(min=EPS)
        inverse = torch.full_like(absmax, float(INT8_MAX)) / absmax
        quotient = _round_half_away(v * inverse).clamp(INT8_MIN, INT8_MAX)
        codes[start:start + _CHUNK_ROWS] = quotient.to(torch.int8)
        scales[start:start + _CHUNK_ROWS] = absmax / INT8_MAX
    return codes, scales
