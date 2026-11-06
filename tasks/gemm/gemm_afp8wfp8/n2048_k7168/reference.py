"""float64 reference for the MXFP8 x FP8 scaled GEMM, used only for correctness."""
import torch

_A_GROUP = 32
_W_BLOCK = 128
_ROW_CHUNK = 8192


def _e8m0(scale):
    """Decode unsigned-biased e8m0 bytes: 2^(byte - 127)."""
    return torch.exp2(scale.to(torch.float64) - 127.0)


def reference(*, a, a_scale, w, w_scale):
    w_dq = w.to(torch.float64) * _e8m0(w_scale).repeat_interleave(_W_BLOCK, dim=0).repeat_interleave(_W_BLOCK, dim=1)
    rows = []
    for start in range(0, a.shape[0], _ROW_CHUNK):
        a_dq = a[start:start + _ROW_CHUNK].to(torch.float64) * _e8m0(a_scale[start:start + _ROW_CHUNK]).repeat_interleave(
            _A_GROUP, dim=1)
        rows.append(a_dq @ w_dq.T)
    return torch.cat(rows, dim=0)
