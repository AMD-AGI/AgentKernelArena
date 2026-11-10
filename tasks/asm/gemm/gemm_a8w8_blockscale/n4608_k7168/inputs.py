"""Quantized block-scaled operands as the production producers emit them, and the preshuffled weight layout.

Activations: float8_e4m3fn codes a [m, k] with one float32 scale per 1x128 group, amax / 448 of the group. The scales
are stored k-group-major, a_scale [k / 128, m], which is what AITER's per-1x128 quantization writes with
`transpose_scale=True` and what the asm kernel reads; generating that layout directly keeps the activation path free
of any untimed transform. Weights: float8_e4m3fn codes b [n, k] in nn.Linear order with one float32 scale per 128x128
block (b_scale [n / 128, k / 128]). Group and block magnitudes are drawn over a 16x range so that a kernel applying the
wrong scale to a group is caught.
"""
import torch

_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448
_BLOCK = 128


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    m, n, k = axes["m"], axes["n"], axes["k"]
    sk, sn = k // _BLOCK, n // _BLOCK

    values = torch.randn(m, sk, _BLOCK, generator=generator, device=device, dtype=torch.float32)
    values = values * torch.exp2(4.0 * torch.rand(m, sk, 1, generator=generator, device=device) - 2.0)
    scale = (values.abs().amax(dim=-1, keepdim=True) / _FP8_MAX).clamp_min(1e-12)
    a = (values / scale).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).view(m, k)
    a_scale = scale.view(m, sk).T.contiguous()

    values = torch.randn(sn, _BLOCK, sk, _BLOCK, generator=generator, device=device, dtype=torch.float32)
    values = values * torch.exp2(4.0 * torch.rand(sn, 1, sk, 1, generator=generator, device=device) - 2.0)
    b_scale = (values.abs().amax(dim=(1, 3)) / _FP8_MAX).clamp_min(1e-12)
    b = (values / b_scale[:, None, :, None]).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).view(n, k)
    return {"a": a, "a_scale": a_scale, "b": b, "b_scale": b_scale}


def to_kernel_layout(case, inputs):
    """AITER `shuffle_weight(b, layout=(16, 16))` (aiter/ops/shuffle.py), done once at weight-load time.

    For 1-byte elements: view b as [n / 16, 16, k / 32, 2, 16] and permute to [n / 16, k / 32, 2, 16, 16], i.e. 16 (n)
    x 32 (k) tiles, each stored as [k half][n row][16 consecutive k].
    """
    inputs = dict(inputs)
    b = inputs["b"]
    n, k = b.shape
    shuffled = b.view(torch.uint8).view(n // 16, 16, k // 32, 2, 16).permute(0, 2, 3, 1, 4).contiguous()
    inputs["b"] = shuffled.view(n, k).view(torch.float8_e4m3fn)
    return inputs
