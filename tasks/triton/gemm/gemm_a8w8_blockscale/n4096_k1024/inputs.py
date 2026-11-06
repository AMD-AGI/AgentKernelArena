"""Quantized fp8 operands with consistent scales, and the kernel-facing layouts.

Activations carry one float32 scale per 1x128 group (group amax / 448); weights one per 128x128 block (block
amax / 448), so codes and scales are mutually consistent. The logical inputs are what the reference sees. The
layout transform produces what production hands the kernel: the weight in AITER's shuffle_weight(layout=(16, 16))
tiling and the activation scale as the column-major payload that the upstream activation-quantization kernel
writes (element (m, g) of the logical scale at offset g * M + m of a contiguous [M, K // 128] buffer).
"""
import torch

_FP8_MAX = 448.0  # e4m3fn saturation magnitude
_BLOCK = 128


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    m, n, k = axes["m"], axes["n"], axes["k"]
    sn, sk = n // _BLOCK, k // _BLOCK

    a_float = torch.randn(m, k, generator=generator, device=device, dtype=torch.float32)
    a_groups = a_float.view(m, sk, _BLOCK)
    a_scale = (a_groups.abs().amax(dim=2) / _FP8_MAX).clamp_min(1e-12)
    a = (a_groups / a_scale.unsqueeze(2)).view(m, k).to(torch.float8_e4m3fn)
    del a_float, a_groups

    b_float = 0.02 * torch.randn(n, k, generator=generator, device=device, dtype=torch.float32)
    b_blocks = b_float.view(sn, _BLOCK, sk, _BLOCK)
    b_scale = (b_blocks.abs().amax(dim=(1, 3)) / _FP8_MAX).clamp_min(1e-12)
    b_scale_full = b_scale.repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    b = (b_float / b_scale_full).to(torch.float8_e4m3fn)
    return {"a": a, "a_scale": a_scale, "b": b, "b_scale": b_scale}


def to_kernel_layout(case, inputs):
    inputs = dict(inputs)
    m, k = inputs["a"].shape
    n = inputs["b"].shape[0]
    # shuffle_weight(layout=(16, 16)): [n/16, 16, k/32, 2, 16] -> [n/16, k/32, 2, 16, 16]
    inputs["b"] = (
        inputs["b"].view(n // 16, 16, k // 32, 2, 16).permute(0, 2, 3, 1, 4).contiguous().view(n, k)
    )
    # Column-major payload: the transposed scale [sk, m] stored contiguously, read back as [m, sk].
    inputs["a_scale"] = inputs["a_scale"].T.contiguous().view(m, k // _BLOCK)
    return inputs
