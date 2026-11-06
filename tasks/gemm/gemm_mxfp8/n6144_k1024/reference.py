"""float64 reference for the dense MXFP8 GEMM, used only for correctness."""
import torch

_BLOCK = 32
_ROW_CHUNK = 8192


def _dequantize(values, scales):
    """Exact dequantization: fp8 value times 2^(byte - 127), per 1x32 block along K."""
    return values.to(torch.float64) * torch.exp2(scales.to(torch.float64) - 127.0).repeat_interleave(_BLOCK, dim=-1)


def reference(*, x, x_scale, w, w_scale):
    w_dq = _dequantize(w, w_scale)
    rows = []
    for start in range(0, x.shape[0], _ROW_CHUNK):
        x_dq = _dequantize(x[start:start + _ROW_CHUNK], x_scale[start:start + _ROW_CHUNK])
        rows.append(x_dq @ w_dq.T)
    return torch.cat(rows, dim=0)
