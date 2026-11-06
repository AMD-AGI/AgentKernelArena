"""Router logits for the biased grouped top-k task."""
import torch

N_GROUPS = 8
TOPK_GROUPS = 4
TOPK = 8
# Minimum gap, in selection-score units, at the two decision boundaries (4th vs 5th group score and
# 8th vs 9th expert score). Rows closer than this would let float32 rounding flip a selection against
# the float64 reference; the production routing tolerates such ties, the exact id comparison here cannot.
MIN_GAP = 1.0e-4


def _boundary_gaps(logits, bias):
    scores = torch.sigmoid(logits.double())
    choice = scores + bias.double()
    n_tokens, n_experts = choice.shape
    group_scores = choice.view(n_tokens, N_GROUPS, -1).topk(2, dim=-1).values.sum(dim=-1)
    top_groups = group_scores.topk(TOPK_GROUPS + 1, dim=-1).values
    group_gap = top_groups[:, TOPK_GROUPS - 1] - top_groups[:, TOPK_GROUPS]
    group_idx = group_scores.topk(TOPK_GROUPS, dim=-1).indices
    mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0)
    mask = mask.unsqueeze(-1).expand(n_tokens, N_GROUPS, n_experts // N_GROUPS).reshape(n_tokens, n_experts)
    masked = choice.masked_fill(mask == 0, 0.0)
    top = masked.topk(TOPK + 1, dim=-1).values
    return torch.minimum(group_gap, top[:, TOPK - 1] - top[:, TOPK])


def make_inputs(case, inputs, generator, device):
    n_tokens = case["axes"]["n_tokens"]
    n_experts = case["axes"]["n_experts"]
    bias = inputs["correction_bias"]
    logits = torch.randn((n_tokens, n_experts), generator=generator, device=device, dtype=torch.float32)
    for _ in range(64):
        bad = (_boundary_gaps(logits, bias) < MIN_GAP).nonzero().flatten()
        if bad.numel() == 0:
            break
        logits[bad] = torch.randn((bad.numel(), n_experts), generator=generator, device=device,
                                  dtype=torch.float32)
    return {"gating_output": logits}
