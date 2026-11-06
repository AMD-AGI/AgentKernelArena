"""Float64 reference: out = (a * a_scales) @ dequant_mxfp4(w, w_scales)^T."""
import torch

_GROUP = 32
_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_ROW_CHUNK = 8192


def _dequantize_weight(w, w_scales):
    n, half_k = w.shape
    k = half_k * 2
    table = torch.tensor(_GRID + tuple(-v for v in _GRID), dtype=torch.float64, device=w.device)
    codes = torch.empty(n, k, dtype=torch.uint8, device=w.device)
    codes[:, 0::2] = w & 0xF
    codes[:, 1::2] = w >> 4
    values = table[codes.long()].view(n, k // _GROUP, _GROUP)
    return (values * torch.exp2(w_scales.double() - 127).unsqueeze(-1)).view(n, k)


def reference(*, a, a_scales, w, w_scales):
    w_dequant = _dequantize_weight(w, w_scales)
    out = torch.empty(a.shape[0], w.shape[0], dtype=torch.float64, device=a.device)
    for start in range(0, a.shape[0], _ROW_CHUNK):
        chunk = a[start:start + _ROW_CHUNK].double() * a_scales[start:start + _ROW_CHUNK].double()
        out[start:start + _ROW_CHUNK] = chunk @ w_dequant.T
    return out
