"""Float64 reference: c = dequant_mxfp4(x) @ dequant_mxfp4(w)^T viewed as [m, D, S1 + S2]; k = cat(c[..., :S1], k_pe broadcast), v = c[..., S1:]."""
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
_S1 = 128  # per-head no-position key width
_S2 = 128  # per-head value width


def reference(*, x, x_scale, w, w_scale, k_pe):
    m = x.shape[0]
    d = w.shape[0] // (_S1 + _S2)
    s3 = k_pe.shape[1]
    w_dequant = _dequantize(w, w_scale)
    k = torch.empty(m, d, _S1 + s3, dtype=torch.float64, device=x.device)
    v = torch.empty(m, d, _S2, dtype=torch.float64, device=x.device)
    for start in range(0, m, _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, m)
        c = (_dequantize(x[start:stop], x_scale[start:stop]) @ w_dequant.T).view(stop - start, d, _S1 + _S2)
        k[start:stop, :, :_S1] = c[..., :_S1]
        k[start:stop, :, _S1:] = k_pe[start:stop].double().unsqueeze(1)
        v[start:stop] = c[..., _S1:]
    return {"k": k, "v": v}
