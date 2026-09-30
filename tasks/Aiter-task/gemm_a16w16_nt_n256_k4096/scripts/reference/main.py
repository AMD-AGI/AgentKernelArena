from __future__ import annotations
from functools import partial as partial
import torch as torch

def _gemm_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    trans_b: bool = True,
) -> torch.Tensor:
    """Plain fp32 matmul; b is (n, k) when trans_b, else (k, n)."""
    rhs = b.transpose(-1, -2) if trans_b else b
    out = a.to(torch.float32) @ rhs.to(torch.float32)
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(a.dtype)


_callable = partial(_gemm_reference, **{'trans_b': True})


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
