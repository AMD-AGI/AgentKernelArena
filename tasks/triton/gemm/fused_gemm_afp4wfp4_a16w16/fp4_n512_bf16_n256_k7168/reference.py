"""Float64 reference: y_a4w4 = dequant_mxfp4(x_fp4) @ dequant_mxfp4(w_fp4)^T and y_a16w16 = x_bf16 @ w_bf16^T."""
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


def reference(*, x_fp4, x_fp4_scale, w_fp4, w_fp4_scale, x_bf16, w_bf16):
    m = x_fp4.shape[0]
    w_dequant = _dequantize(w_fp4, w_fp4_scale)
    w16 = w_bf16.double()
    y_a4w4 = torch.empty(m, w_fp4.shape[0], dtype=torch.float64, device=x_fp4.device)
    y_a16w16 = torch.empty(m, w_bf16.shape[0], dtype=torch.float64, device=x_fp4.device)
    for start in range(0, m, _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, m)
        y_a4w4[start:stop] = _dequantize(x_fp4[start:stop], x_fp4_scale[start:stop]) @ w_dequant.T
        y_a16w16[start:stop] = x_bf16[start:stop].double() @ w16.T
    return {"y_a4w4": y_a4w4, "y_a16w16": y_a16w16}
