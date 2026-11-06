"""Quantized block-scaled operands as a quantizing producer would emit them, and their kernel-facing layout.

Logical operands (what the reference sees): fp8 e4m3fn activations a [m, k] with one float32 scale per 1x128 group
(a_scale [m, k/128]), fp8 weights b [n, k] with one float32 scale per 128x128 block (b_scale [n/128, k/128]).
Scales are amax / 448 of the block they cover; block magnitudes are drawn over a 16x range so the scales differ
visibly between blocks and a kernel that mixes them up is caught.

Kernel-facing layout (outside timing, as production holds it): the weight is preshuffled once at load time with
the aiter shuffle_weight(layout=(16, 16)) permutation, and the activation scales are the column-major payload
of the upstream activation-quantization kernel: a buffer of shape [k/128, m] presented as [m, k/128], so that
the scale of activation row i and group j is payload_flat[j * m + i].
"""
import torch

_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448
_BLOCK = 128
_ROW_CHUNK = 16384


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    m, n, k = axes["m"], axes["n"], axes["k"]
    sk, sn = k // _BLOCK, n // _BLOCK

    a = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=device)
    a_scale = torch.empty(m, sk, dtype=torch.float32, device=device)
    for start in range(0, m, _ROW_CHUNK):
        rows = min(_ROW_CHUNK, m - start)
        values = torch.randn(rows, sk, _BLOCK, generator=generator, device=device, dtype=torch.float32)
        magnitude = torch.exp2(4.0 * torch.rand(rows, sk, 1, generator=generator, device=device) - 2.0)
        values = values * magnitude
        scale = (values.abs().amax(dim=-1, keepdim=True) / _FP8_MAX).clamp_min(1e-12)
        codes = (values / scale).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        a[start:start + rows] = codes.view(rows, k)
        a_scale[start:start + rows] = scale.view(rows, sk)

    values = torch.randn(sn, _BLOCK, sk, _BLOCK, generator=generator, device=device, dtype=torch.float32)
    magnitude = torch.exp2(4.0 * torch.rand(sn, 1, sk, 1, generator=generator, device=device) - 2.0)
    values = values * magnitude
    b_scale = (values.abs().amax(dim=(1, 3)) / _FP8_MAX).clamp_min(1e-12)
    b = (values / b_scale[:, None, :, None]).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).view(n, k)
    return {"a": a, "a_scale": a_scale, "b": b, "b_scale": b_scale}


def to_kernel_layout(case, inputs):
    inputs = dict(inputs)
    a_scale, b = inputs["a_scale"], inputs["b"]
    m, sk = a_scale.shape
    n, k = b.shape
    # Column-major activation scales: store the transpose contiguously, then present it as [m, sk].
    inputs["a_scale"] = a_scale.T.contiguous().view(m, sk)
    # aiter shuffle_weight(layout=(16, 16)) for one-byte elements: [n, k] -> [n/16, k/32, 2, 16, 16] tiles.
    inputs["b"] = (b.view(n // 16, 16, k // 32, 2, 16).permute(0, 2, 3, 1, 4).contiguous().view(n, k))
    return inputs
