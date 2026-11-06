"""MXFP8 activation and block-scaled FP8 weight operands.

Codes are standard-normal values rounded to float8_e4m3fn. Scales are e8m0 bytes (value 2^(byte - 127)) drawn from
125..129, i.e. 2^-2..2^2, so operands span several binades without overflowing the bfloat16 output.
"""
import torch

_A_GROUP = 32  # activation scale group along k
_W_BLOCK = 128  # weight scale block along n and k
_FP8_MAX = 448.0


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    m, n, k = axes["m"], axes["n"], axes["k"]
    a = torch.randn(m, k, generator=generator, device=device, dtype=torch.float32)
    a = a.clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    w = torch.randn(n, k, generator=generator, device=device, dtype=torch.float32)
    w = w.clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    a_scale = torch.randint(125, 130, (m, k // _A_GROUP), generator=generator, device=device, dtype=torch.int32)
    w_scale = torch.randint(125, 130, (n // _W_BLOCK, k // _W_BLOCK), generator=generator, device=device,
                            dtype=torch.int32)
    return {"a": a, "a_scale": a_scale.to(torch.uint8), "w": w, "w_scale": w_scale.to(torch.uint8)}
