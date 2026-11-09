"""Float64 reference: out = a * b + dequant(x) @ dequant(w)^T with 1x128 activation and 128x128 weight scales."""
import torch

_BLOCK = 128
_ROW_CHUNK = 4096


def _dequantize_rows(codes, scales):
    """float8 codes [rows, k] with one float32 scale per 1x128 block -> float64 [rows, k]."""
    rows, k = codes.shape
    return (codes.double().view(rows, k // _BLOCK, _BLOCK) * scales.double().unsqueeze(-1)).view(rows, k)


def _dequantize_weight(codes, scales):
    """float8 codes [n, k] with one float32 scale per 128x128 block -> float64 [n, k]."""
    full = scales.double().repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    return codes.double() * full


def reference(*, x, x_scales, w, w_scales, a, b):
    m = x.shape[0]
    w_dequant = _dequantize_weight(w, w_scales)
    out = torch.empty(m, w.shape[0], dtype=torch.float64, device=x.device)
    for start in range(0, m, _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, m)
        product = _dequantize_rows(x[start:stop], x_scales[start:stop]) @ w_dequant.T
        out[start:stop] = float(a) * b[start:stop].double() + product
    return out
