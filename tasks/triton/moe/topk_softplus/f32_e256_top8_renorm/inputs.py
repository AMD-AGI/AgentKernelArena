"""Router logits for the sqrt-softplus top-k task."""
import torch

TOPK = 8
# Minimum gap, in selection-score units, between the 8th and 9th largest selection score. Rows closer
# than this would let float32 rounding or a fast exp/log approximation flip a selection against the
# float64 reference; the production routing tolerates such ties, the exact id comparison here cannot.
MIN_GAP = 1.0e-4


def _gaps(logits, bias):
    scores = torch.sqrt(torch.logaddexp(logits.double(), torch.zeros((), dtype=torch.float64,
                                                                      device=logits.device)))
    top = (scores + bias.double()).topk(TOPK + 1, dim=-1).values
    return top[:, TOPK - 1] - top[:, TOPK]


def make_inputs(case, inputs, generator, device):
    n_tokens = case["axes"]["n_tokens"]
    n_experts = case["axes"]["n_experts"]
    bias = inputs["correction_bias"]
    logits = 1.5 * torch.randn((n_tokens, n_experts), generator=generator, device=device,
                               dtype=torch.float32)
    for _ in range(64):
        bad = (_gaps(logits, bias) < MIN_GAP).nonzero().flatten()
        if bad.numel() == 0:
            break
        logits[bad] = 1.5 * torch.randn((bad.numel(), n_experts), generator=generator, device=device,
                                        dtype=torch.float32)
    return {"gating_output": logits}
