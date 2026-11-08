"""Float64 reference: y_a8w8 = dequant(x_fp8) @ dequant(w_fp8)^T and y_a16w16 = x_bf16 @ w_bf16^T."""
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


def reference(*, x_fp8, x_fp8_scale, w_fp8, w_fp8_scale, x_bf16, w_bf16):
    m = x_fp8.shape[0]
    w_dequant = _dequantize_weight(w_fp8, w_fp8_scale)
    w16 = w_bf16.double()
    y_a8w8 = torch.empty(m, w_fp8.shape[0], dtype=torch.float64, device=x_fp8.device)
    y_a16w16 = torch.empty(m, w_bf16.shape[0], dtype=torch.float64, device=x_fp8.device)
    for start in range(0, m, _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, m)
        y_a8w8[start:stop] = _dequantize_rows(x_fp8[start:stop], x_fp8_scale[start:stop]) @ w_dequant.T
        y_a16w16[start:stop] = x_bf16[start:stop].double() @ w16.T
    return {"y_a8w8": y_a8w8, "y_a16w16": y_a16w16}
