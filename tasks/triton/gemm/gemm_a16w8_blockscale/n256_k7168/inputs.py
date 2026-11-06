"""Block-quantized fp8 weights: 128x128 blocks, one float32 scale per block (block amax / 448)."""
import torch

_FP8_MAX = 448.0  # e4m3fn saturation magnitude
_BLOCK = 128


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n, k = axes["n"], axes["k"]
    sn, sk = n // _BLOCK, k // _BLOCK
    w_float = 0.02 * torch.randn(n, k, generator=generator, device=device, dtype=torch.float32)
    blocks = w_float.view(sn, _BLOCK, sk, _BLOCK)
    scale = (blocks.abs().amax(dim=(1, 3)) / _FP8_MAX).clamp_min(1e-12)
    scale_full = scale.repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    w = (w_float / scale_full).to(torch.float8_e4m3fn)
    return {"w": w, "w_scale": scale}
