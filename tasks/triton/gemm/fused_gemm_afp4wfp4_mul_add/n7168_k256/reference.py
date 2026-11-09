"""Float64 reference: out = a * b + dequant_mxfp4(x, x_scales) @ dequant_mxfp4(w, w_scales)^T."""
import torch

_GROUP = 32
_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_ROW_CHUNK = 4096


def _dequantize(packed, scales):
    """MXFP4 [rows, k // 2] bytes (even K index in the low nibble) with e8m0 scales [rows, k // 32] -> float64."""
    rows, half_k = packed.shape
    k = half_k * 2
    table = torch.tensor(_GRID + tuple(-v for v in _GRID), dtype=torch.float64, device=packed.device)
    codes = torch.empty(rows, k, dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    values = table[codes.long()].view(rows, k // _GROUP, _GROUP)
    return (values * torch.exp2(scales.double() - 127).unsqueeze(-1)).view(rows, k)


def reference(*, x, x_scales, w, w_scales, a, b):
    m = x.shape[0]
    w_dequant = _dequantize(w, w_scales)
    out = torch.empty(m, w.shape[0], dtype=torch.float64, device=x.device)
    for start in range(0, m, _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, m)
        product = _dequantize(x[start:stop], x_scales[start:stop]) @ w_dequant.T
        out[start:stop] = float(a) * b[start:stop].double() + product
    return out
