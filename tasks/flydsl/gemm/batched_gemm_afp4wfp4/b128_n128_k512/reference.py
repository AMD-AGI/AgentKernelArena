"""Float64 reference: out[b] = dequant_mxfp4(x[b], x_scales[b]) @ dequant_mxfp4(w[b], w_scales[b])^T."""
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


def reference(*, x, x_scales, w, w_scales):
    batch, m, _ = x.shape
    n = w.shape[1]
    out = torch.empty(batch, m, n, dtype=torch.float64, device=x.device)
    for b in range(batch):
        w_dequant = _dequantize(w[b], w_scales[b])
        for start in range(0, m, _ROW_CHUNK):
            chunk = _dequantize(x[b, start:start + _ROW_CHUNK], x_scales[b, start:start + _ROW_CHUNK])
            out[b, start:start + _ROW_CHUNK] = chunk @ w_dequant.T
    return out
