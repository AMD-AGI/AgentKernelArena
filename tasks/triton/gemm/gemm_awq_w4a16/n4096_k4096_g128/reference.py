"""float64 reference for the AWQ int4 GEMM, used only for correctness."""
import torch

# Output column 8 * j + i holds the nibble at bit offset 4 * _AWQ_ORDER[i] of packed word j.
_AWQ_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def _unpack(packed):
    nibbles = [(packed >> (4 * shift)) & 0xF for shift in _AWQ_ORDER]
    return torch.stack(nibbles, dim=-1).reshape(packed.shape[0], packed.shape[1] * 8)


def reference(*, x, qweight, qzeros, scales):
    k = qweight.shape[0]
    group = k // qzeros.shape[0]
    weight = _unpack(qweight).to(torch.float64)
    zeros = _unpack(qzeros).to(torch.float64)
    dequant = (weight - zeros.repeat_interleave(group, dim=0)) * scales.to(torch.float64).repeat_interleave(group, dim=0)
    return x.to(torch.float64) @ dequant
