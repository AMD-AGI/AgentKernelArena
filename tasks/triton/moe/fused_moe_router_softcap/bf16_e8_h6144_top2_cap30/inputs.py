"""Router activations whose top-2 selection is unambiguous."""
import torch

_CHUNK = 16384
_STD = 0.1
# Minimum capped-logit gap at the two decision points (1st vs 2nd, 2nd vs 3rd expert). Rows closer than
# this would let float32 accumulation order flip an id against the float64 reference; the exact id
# comparison cannot tolerate that, and production routing is indifferent to such ties.
MIN_GAP = 1.0e-3


def _draw(rows, hidden, generator, device):
    return (torch.randn((rows, hidden), generator=generator, device=device, dtype=torch.float32) * _STD).to(
        torch.bfloat16)


def _close_rows(x, weight, cap):
    logits = x.double() @ weight.double().T
    if cap != 0.0:
        logits = torch.tanh(logits / cap) * cap
    top = logits.topk(3, dim=-1).values
    return torch.minimum(top[:, 0] - top[:, 1], top[:, 1] - top[:, 2]) < MIN_GAP


def make_inputs(case, inputs, generator, device):
    n_tokens = case["axes"]["n_tokens"]
    hidden = case["axes"]["hidden_size"]
    cap = float(case["scalars"]["moe_softcapping"])
    weight = inputs["router_weight"]
    x = torch.empty((n_tokens, hidden), device=device, dtype=torch.bfloat16)
    for start in range(0, n_tokens, _CHUNK):
        stop = min(start + _CHUNK, n_tokens)
        chunk = _draw(stop - start, hidden, generator, device)
        for _ in range(64):
            bad = _close_rows(chunk, weight, cap).nonzero().flatten()
            if bad.numel() == 0:
                break
            chunk[bad] = _draw(bad.numel(), hidden, generator, device)
        x[start:stop] = chunk
    return {"x": x}
