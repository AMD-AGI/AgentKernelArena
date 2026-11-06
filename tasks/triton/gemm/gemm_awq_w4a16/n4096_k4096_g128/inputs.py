"""AWQ int4 operands: fp16 activations, packed weights and zero points, fp16 group scales.

qweight [k, n // 8] and qzeros [k // 128, n // 8] are int32 words holding eight 4-bit values each; every bit pattern
is a valid weight, so the words are uniform over the whole int32 range. Scales are small positive fp16 values of the
magnitude AWQ produces (group range / 15), keeping the fp16 output well inside range.
"""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    m, n, k, g = axes["m"], axes["n"], axes["k"], axes["group_size"]
    low, high = -(2 ** 31), 2 ** 31 - 1
    x = torch.randn(m, k, generator=generator, device=device, dtype=torch.float32).to(torch.float16)
    qweight = torch.randint(low, high, (k, n // 8), generator=generator, device=device, dtype=torch.int32)
    qzeros = torch.randint(low, high, (k // g, n // 8), generator=generator, device=device, dtype=torch.int32)
    scales = torch.randn(k // g, n, generator=generator, device=device, dtype=torch.float32).abs() * 0.004 + 0.001
    return {"x": x, "qweight": qweight, "qzeros": qzeros, "scales": scales.to(torch.float16)}
