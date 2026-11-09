"""Float64 reference for the batched a16wfp4 GEMM.

Per batch element (attention head) the bfloat16 activations are quantized to MXFP4 exactly as the operator does (one
e8m0 scale per 32 values of K, round to nearest with ties to the even e2m1 code, saturating at 6), dequantized, and
multiplied with the dequantized weights in float64. Activations and the output are token-major: x [m, batch, k],
out [m, batch, n].
"""
import torch

_GROUP = 32
_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (False, True, False, True, False, True, False)
_ROW_CHUNK = 4096


def _grid_value(code, device):
    table = torch.tensor(_GRID + tuple(-v for v in _GRID), dtype=torch.float64, device=device)
    return table[code.long()]


def _quantize_dequantize_activation(x):
    """Activation after the operator's per-1x32 MXFP4 quantization, as float64."""
    m, k = x.shape
    groups = x.float().reshape(m, k // _GROUP, _GROUP)
    amax = groups.abs().amax(dim=-1, keepdim=True)
    # The group maximum is rounded to a power of two (mantissa >= 0.75 rounds the exponent up); the scale
    # exponent is that power's exponent minus the exponent of the largest e2m1 value (2).
    bits = amax.contiguous().view(torch.int32)
    rounded = (bits + 0x200000) & -0x800000
    exponent = ((rounded >> 23) & 0xFF) - 127
    exponent = (exponent - 2).clamp(-127, 127)
    scaled = (groups.double() * torch.exp2(-exponent.double())).float()
    magnitude = scaled.abs()
    code = torch.zeros(scaled.shape, dtype=torch.uint8, device=x.device)
    for bound, up in zip(_BOUNDS, _TIE_UP):
        code += ((magnitude >= bound) if up else (magnitude > bound)).to(torch.uint8)
    code = code | ((scaled < 0).to(torch.uint8) << 3)
    return (_grid_value(code, x.device) * torch.exp2(exponent.double())).reshape(m, k)


def _dequantize_weight(w, w_scales):
    n, half_k = w.shape
    k = half_k * 2
    codes = torch.empty(n, k, dtype=torch.uint8, device=w.device)
    codes[:, 0::2] = w & 0xF
    codes[:, 1::2] = w >> 4
    values = _grid_value(codes, w.device).view(n, k // _GROUP, _GROUP)
    return (values * torch.exp2(w_scales.double() - 127).unsqueeze(-1)).view(n, k)


def reference(*, x, w, w_scales):
    m, batch, _ = x.shape
    n = w.shape[1]
    out = torch.empty(m, batch, n, dtype=torch.float64, device=x.device)
    for b in range(batch):
        w_dequant = _dequantize_weight(w[b], w_scales[b])
        for start in range(0, m, _ROW_CHUNK):
            chunk = _quantize_dequantize_activation(x[start:start + _ROW_CHUNK, b])
            out[start:start + _ROW_CHUNK, b] = chunk @ w_dequant.T
    return out
