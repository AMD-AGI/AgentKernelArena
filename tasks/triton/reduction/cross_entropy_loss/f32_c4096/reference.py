"""High-precision cross-entropy reference (per-row losses and their mean) used only for correctness."""
import torch

_CHUNK = 16384


def reference(*, logits, targets):
    rows = logits.shape[0]
    losses = torch.empty(rows, dtype=torch.float64, device=logits.device)
    for start in range(0, rows, _CHUNK):
        x = logits[start:start + _CHUNK].double()
        t = targets[start:start + _CHUNK]
        picked = x.gather(-1, t.unsqueeze(-1)).squeeze(-1)
        losses[start:start + _CHUNK] = torch.logsumexp(x, dim=-1) - picked
    return {"losses": losses, "out": losses.sum() / rows}
