"""Float64 reference: out = dequant_mxfp4(a, a_scales) @ dequant_mxfp4(w, w_scales)^T."""
import torch

_GROUP = 32
_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_ROW_CHUNK = 4096


def _dequantize(packed, scales):
    rows, half_k = packed.shape
    k = half_k * 2
    table = torch.tensor(_GRID + tuple(-v for v in _GRID), dtype=torch.float64, device=packed.device)
    codes = torch.empty(rows, k, dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    values = table[codes.long()].view(rows, k // _GROUP, _GROUP)
    return (values * torch.exp2(scales.double() - 127).unsqueeze(-1)).view(rows, k)


def reference(*, a, a_scales, w, w_scales):
    w_dequant = _dequantize(w, w_scales)
    out = torch.empty(a.shape[0], w.shape[0], dtype=torch.float64, device=a.device)
    for start in range(0, a.shape[0], _ROW_CHUNK):
        chunk = _dequantize(a[start:start + _ROW_CHUNK], a_scales[start:start + _ROW_CHUNK])
        out[start:start + _ROW_CHUNK] = chunk @ w_dequant.T
    return out
