"""float64 top-k softmax routing reference, used only for correctness."""
import torch


def reference(*, gating_output):
    topk = 8
    probs = torch.softmax(gating_output.double(), dim=-1)
    # Largest probabilities first; equal probabilities (equal logits) keep the lower expert id,
    # as the production kernel does. A stable descending sort makes the choice deterministic.
    ids = torch.sort(probs, dim=-1, descending=True, stable=True).indices[:, :topk]
    weights = probs.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, ids.to(torch.int32)
