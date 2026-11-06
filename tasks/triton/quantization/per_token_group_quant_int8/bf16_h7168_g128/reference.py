"""Reference for per-token-group INT8 quantization, used only for correctness.

The codes are compared bit-for-bit, and the operator truncates the quotient toward zero, so the
quotient must be formed exactly as the operator defines it: in IEEE float32 (correctly rounded
division by the float32 scale). Float64 would flip codes whose float32 quotient lands exactly on an
integer, so the scale and quotient are computed in float32 here; the absolute maximum and the
bfloat16 inputs are exact in float32.
"""
import torch

INT8_MAX = 127
INT8_MIN = -128
GROUP_SIZE = 128
EPS = 1e-10
_CHUNK_ROWS = 4096


def reference(*, x):
    rows, width = x.shape
    groups = width // GROUP_SIZE
    codes = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scales = torch.empty((rows, groups), dtype=torch.float32, device=x.device)
    for start in range(0, rows, _CHUNK_ROWS):
        g = x[start:start + _CHUNK_ROWS].float().reshape(-1, groups, GROUP_SIZE)
        absmax = g.abs().amax(dim=-1, keepdim=True).clamp(min=EPS)
        # Divide by a tensor, not a Python scalar: on GPU torch turns `tensor / scalar` into a
        # multiplication by the float32 reciprocal, which differs from the IEEE quotient by 1 ulp.
        scale = torch.div(absmax, torch.full_like(absmax, INT8_MAX))
        quotient = (g / scale).clamp(INT8_MIN, INT8_MAX)
        codes[start:start + _CHUNK_ROWS] = quotient.trunc().to(torch.int8).reshape(-1, width)
        scales[start:start + _CHUNK_ROWS] = scale.squeeze(-1)
    return codes, scales
