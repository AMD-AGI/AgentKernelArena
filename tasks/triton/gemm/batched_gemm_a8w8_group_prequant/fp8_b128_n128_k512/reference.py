"""Float64 reference for the batched a8w8 GEMM with in-operator per-token group quantization of the activations.

For every token and batch element (attention head) the bfloat16 activations are quantized to fp8 (e4m3fn) per group of
128 values of K, as the operator does: group scale = max(group max |x|, 1e-10) * (1 / 448) in float32, codes =
fp8(clamp(x * (1 / scale), -448, 448)) with round to nearest even. The product of each group's fp8 codes with the fp8
weight codes is accumulated exactly (float64), multiplied by the group scale, summed over groups and multiplied by the
single per-tensor weight scale. Activations and the output are token-major: x [m, batch, k], out [m, batch, n].
"""
import torch

_GROUP = 128
_FP8_MAX = 448.0


def reference(*, x, wq, w_scale):
    m, batch, k = x.shape
    n = wq.shape[1]
    groups = k // _GROUP
    one_over_max = torch.tensor(1.0 / _FP8_MAX, dtype=torch.float32, device=x.device)
    out = torch.empty(m, batch, n, dtype=torch.float64, device=x.device)
    scale_w = w_scale.double().reshape(())
    for b in range(batch):
        x_group = x[:, b].float().view(m, groups, _GROUP)
        amax = x_group.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
        scale = amax * one_over_max
        codes = (x_group * (1.0 / scale)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        w_group = wq[b].double().view(n, groups, _GROUP)
        products = torch.einsum("mgk,ngk->mgn", codes.double(), w_group)
        out[:, b] = (products * scale.double()).sum(dim=1) * scale_w
    return out
