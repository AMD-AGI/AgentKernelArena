"""float64 DeepSeek-V3 biased grouped top-k reference, used only for correctness."""
import torch

N_GROUPS = 8
TOPK_GROUPS = 4


def reference(*, gating_output, correction_bias, route_scale):
    n_tokens, n_experts = gating_output.shape
    topk = 8
    scores = torch.sigmoid(gating_output.double())
    choice = scores + correction_bias.double().unsqueeze(0)
    group_scores = choice.view(n_tokens, N_GROUPS, -1).topk(2, dim=-1).values.sum(dim=-1)
    group_idx = torch.topk(group_scores, k=TOPK_GROUPS, dim=-1).indices
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0)
    expert_mask = group_mask.unsqueeze(-1).expand(n_tokens, N_GROUPS, n_experts // N_GROUPS)
    masked = choice.masked_fill(expert_mask.reshape(n_tokens, n_experts) == 0, 0.0)
    ids = torch.topk(masked, k=topk, dim=-1).indices
    weights = scores.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True) * float(route_scale)
    return weights, ids.to(torch.int32)
