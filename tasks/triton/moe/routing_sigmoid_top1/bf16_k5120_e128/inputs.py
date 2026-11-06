"""Router activations and weights on an exact grid.

Every value is a small multiple of 1/8 (activations) or 1/64 (weights), exactly representable in
bfloat16, so each router logit is an exact multiple of 1/512 whatever the accumulation order. Two
different logits therefore differ by at least 1/512, far more than any float32 sigmoid error, and equal
logits are tied exactly: the arg-max (first maximum wins) is the same for the float64 reference and for
every correct kernel, and expert ids compare exactly.
"""
import torch

CHUNK_ROWS = 16384


def make_inputs(case, inputs, generator, device):
    n_tokens = case["axes"]["n_tokens"]
    hidden = case["axes"]["hidden_size"]
    n_experts = case["axes"]["n_experts"]
    x = torch.empty((n_tokens, hidden), device=device, dtype=torch.bfloat16)
    for start in range(0, n_tokens, CHUNK_ROWS):
        rows = min(CHUNK_ROWS, n_tokens - start)
        draw = torch.randn((rows, hidden), generator=generator, device=device, dtype=torch.float32)
        x[start:start + rows] = (draw * 3.0).round().clamp(-8, 8).div(8.0).to(torch.bfloat16)
    draw = torch.randn((hidden, n_experts), generator=generator, device=device, dtype=torch.float32)
    w = (draw * 1.5).round().clamp(-8, 8).div(64.0).to(torch.bfloat16)
    return {"x": x, "w": w}
