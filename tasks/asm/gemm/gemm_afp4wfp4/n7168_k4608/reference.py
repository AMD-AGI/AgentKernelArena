"""Float64 reference: out = dequant_mxfp4(a, a_scales) @ dequant_mxfp4(w, w_scales)^T.

Receives the logical weight (w [n, k / 2] packed codes in nn.Linear order, w_scales [n, k / 32] row-major) and the
activation operands as the quantizer writes them: a [m, k / 2] packed codes, row-major, and a_scales
[ceil(m / 256) * 256, k / 32] in AITER's e8m0_shuffle layout, which this file undoes before dequantizing.
"""
import torch

_GROUP = 32
_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_ROW_CHUNK = 4096


def _unshuffle_scales(shuffled, rows):
    """Inverse of e8m0_shuffle: (rows / 32, k / 256, 4, 16, 2, 2) back to (rows / 32, 2, 16, k / 256, 2, 4)."""
    padded_rows, k_scales = shuffled.shape
    blocks = shuffled.reshape(padded_rows // 32, k_scales // 8, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2)
    return blocks.reshape(padded_rows, k_scales)[:rows]


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
    a_scales = _unshuffle_scales(a_scales, a.shape[0])
    w_dequant = _dequantize(w, w_scales)
    out = torch.empty(a.shape[0], w.shape[0], dtype=torch.float64, device=a.device)
    for start in range(0, a.shape[0], _ROW_CHUNK):
        chunk = _dequantize(a[start:start + _ROW_CHUNK], a_scales[start:start + _ROW_CHUNK])
        out[start:start + _ROW_CHUNK] = chunk @ w_dequant.T
    return out
