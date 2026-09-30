from __future__ import annotations
from functools import partial as partial
import torch as torch

@torch.no_grad()
def _blockwise_scaled_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    trans_b: bool = True,
    block_size: tuple[int, int] = (128, 128),
    out_dtype: str = "bfloat16",
    scale_major_mode: str = "K",
    a_scale_storage: str = "plain",
) -> torch.Tensor:
    """Decode declared values (not arbitrary storage), dequantize, then fp32 GEMM.

    Inverse of aiter.ops.shuffle.shuffle_weight(layout=(16,16)), MIT licensed,
    source commit 21db2d9edeab6f9315cae49d3d3203c5a44ea928. No AITER import.
    """
    m, k = a.shape
    weight = b if trans_b else b.transpose(-1, -2)
    n = weight.shape[0]
    block_n, block_k = block_size
    if a_scale_storage != "plain":
        weight = (
            weight.reshape(n // 16, k // 32, 2, 16, 16)
            .permute(0, 3, 1, 2, 4)
            .reshape(n, k)
        )
    if a_scale_storage == "raw":
        a_scale = a_scale.reshape(-1, m).T
    elif scale_major_mode == "MN":
        a_scale = a_scale.T
    lhs_scale = a_scale.to(torch.float32).repeat_interleave(block_k, dim=1)[:, :k]
    lhs = a.to(torch.float32) * lhs_scale
    weight_scale = b_scale.to(torch.float32)
    weight_scale = weight_scale.repeat_interleave(block_n, dim=0).repeat_interleave(
        block_k, dim=1
    )
    weight = weight.to(torch.float32) * weight_scale[:n, :k]

    with torch.autocast(a.device.type, enabled=False):
        out = lhs @ weight.transpose(-1, -2)
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(getattr(torch, out_dtype))


_callable = partial(_blockwise_scaled_reference, **{'trans_b': True, 'block_size': (128, 128,), 'scale_major_mode': 'K', 'a_scale_storage': 'logical', 'out_dtype': 'bfloat16'})


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
