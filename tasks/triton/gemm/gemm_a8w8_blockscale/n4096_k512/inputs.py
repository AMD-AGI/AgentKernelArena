"""Block-scaled fp8 operands as the quantizing producers emit them, and the kernel-facing layout of the weight and the
activation scales.

Logical inputs: `a` [m, k] fp8 codes with one float32 scale per 1x128 block (`a_scale` [m, k // 128], block amax / 448);
`b` [n, k] fp8 codes with one float32 scale per 128x128 block (`b_scale` [n // 128, k // 128]).
"""
import torch

_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_BLOCK = 128


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    m, n, k = axes["m"], axes["n"], axes["k"]
    sn, sk = n // _BLOCK, k // _BLOCK

    a_float = torch.randn(m, k, generator=generator, device=device, dtype=torch.float32)
    a_blocks = a_float.view(m, sk, _BLOCK)
    a_scale = (a_blocks.abs().amax(dim=-1) / _FP8_MAX).clamp_min(1e-12)
    a = (a_blocks / a_scale.unsqueeze(-1)).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).view(m, k)
    del a_float, a_blocks

    b_float = torch.randn(n, k, generator=generator, device=device, dtype=torch.float32)
    b_blocks = b_float.view(sn, _BLOCK, sk, _BLOCK)
    b_scale = (b_blocks.abs().amax(dim=(1, 3)) / _FP8_MAX).clamp_min(1e-12)
    b_full = b_scale.repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    b = (b_float / b_full).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return {"a": a, "b": b, "a_scale": a_scale, "b_scale": b_scale}


def to_kernel_layout(case, inputs):
    """AITER shuffle_weight(layout=(16, 16)) for `b`; the column-major payload of `a_scale`.

    The weight is stored in 16 (n) x 32 (k) tiles, each tile as [k half (2)][n (16)][k within half (16)]. The
    activation scales leave the producing quantization kernel k-block-major (all rows of one k block are
    contiguous); the payload is the [k // 128, m] array stored in memory as an [m, k // 128] tensor.
    """
    inputs = dict(inputs)
    n, k = inputs["b"].shape
    m, sk = inputs["a_scale"].shape
    b = inputs["b"].view(n // 16, 16, k // 32, 2, 16).permute(0, 2, 3, 1, 4).contiguous().view(n, k)
    inputs["b"] = b
    inputs["a_scale"] = inputs["a_scale"].T.contiguous().view(m, sk)
    return inputs
