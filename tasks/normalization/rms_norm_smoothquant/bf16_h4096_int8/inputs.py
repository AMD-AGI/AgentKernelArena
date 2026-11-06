"""Activations whose INT8 codes are decided by the specified rounding, not by float32 noise.

The codes are compared exactly. The code of an element is round(127 * p / max|p|) with p = x * weight * x_scale (the
reciprocal RMS cancels), and an implementation computes it in float32 with a relative error of a few 1e-7, so an
element whose quotient lies within about 4e-4 of a rounding decision (a half integer) could legitimately round either
way. The generator moves each such element (a fraction of about 4e-3 with the margin below) to the next bfloat16
neighbour of larger magnitude and repeats until no element of any row is within MARGIN of a decision. The row
maximum itself quotients to exactly 127 and is never a decision. p is exact in float64 (8 + 8 + 24 significant bits).
"""
import torch

CHUNK_ROWS = 1 << 13
MARGIN = 2.0e-3          # distance of 127 * p / max|p| from a half integer below which an element is moved
INT8_MAX = 127.0


def _unsafe(x, gain):
    p = x.double() * gain
    amax = p.abs().amax(dim=-1, keepdim=True)
    q = (p.abs() / amax) * INT8_MAX
    distance = ((q % 1.0) - 0.5).abs()
    return distance < MARGIN


def _bump(value):
    """Next bfloat16 value of larger magnitude."""
    return (value.view(torch.int16) + 1).view(torch.bfloat16)


def make_inputs(case, inputs, generator, device):
    n_tokens = case["axes"]["n_tokens"]
    hidden = case["axes"]["hidden_size"]
    gain = inputs["weight"].double() * inputs["x_scale"].double()
    x = torch.empty((n_tokens, hidden), dtype=torch.bfloat16, device=device)
    for start in range(0, n_tokens, CHUNK_ROWS):
        rows = min(CHUNK_ROWS, n_tokens - start)
        block = torch.randn((rows, hidden), generator=generator, device=device, dtype=torch.float32).to(torch.bfloat16)
        for _ in range(256):
            bad = _unsafe(block, gain)
            if not bool(bad.any()):
                break
            block = torch.where(bad, _bump(block), block)
        else:
            raise RuntimeError("could not move all elements away from rounding decisions")
        x[start:start + rows] = block
    return {"x": x}
