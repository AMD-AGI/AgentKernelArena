"""Task-owned weight, bias and weight layout of the 384 -> 384 channel 3x3x3 causal conv3d.

The weight is a bfloat16 `nn.Conv3d` parameter [K, C, 3, 3, 3] with a standard deviation of 0.01, which keeps the
output standard deviation near 1 for unit-variance activations (sqrt(C * 27) * 0.01 is about 1). The bias is a full-precision float32
vector (the dtype the kernel's bias argument takes); a checkpoint's bfloat16 bias is a special case of this domain.

`to_kernel_layout` packs the weight as `[K, T*R*S*C]` (channels fastest), the layout FlyDSL's conv3d reads and
which `_prep_weight` of kernels/conv/conv3d_implicit.py produces once per weight upstream.
"""
import torch

WEIGHT_STD = 0.01
BIAS_STD = 0.1


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    k, c = axes["k"], axes["c"]
    weight = (WEIGHT_STD * torch.randn(k, c, 3, 3, 3, generator=generator, device=device,
                                       dtype=torch.float32)).to(torch.bfloat16)
    bias = BIAS_STD * torch.randn(k, generator=generator, device=device, dtype=torch.float32)
    return {"weight": weight, "bias": bias}


def to_kernel_layout(case, inputs):
    inputs = dict(inputs)
    weight = inputs["weight"]
    k = weight.shape[0]
    inputs["weight"] = weight.permute(0, 2, 3, 4, 1).contiguous().reshape(k, -1)
    return inputs
