"""Float64 reference: c = dequant(x) @ dequant(w)^T viewed as [m, D, S1 + S2]; k = cat(c[..., :S1], k_pe broadcast), v = c[..., S1:]."""
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
_S1 = 128  # per-head no-position key width
_S2 = 128  # per-head value width


def reference(*, x, x_scale, w, w_scale, k_pe):
    m = x.shape[0]
    d = w.shape[0] // (_S1 + _S2)
    s3 = k_pe.shape[1]
    w_dequant = _dequantize_weight(w, w_scale)
    k = torch.empty(m, d, _S1 + s3, dtype=torch.float64, device=x.device)
    v = torch.empty(m, d, _S2, dtype=torch.float64, device=x.device)
    for start in range(0, m, _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, m)
        c = (_dequantize_rows(x[start:stop], x_scale[start:stop]) @ w_dequant.T).view(stop - start, d, _S1 + _S2)
        k[start:stop, :, :_S1] = c[..., :_S1]
        k[start:stop, :, _S1:] = k_pe[start:stop].double().unsqueeze(1)
        v[start:stop] = c[..., _S1:]
    return {"k": k, "v": v}
