"""High-precision label-smoothed cross entropy reference (per-row losses and their mean over the rows
that are not ignored) used only for correctness."""
import torch

_CHUNK = 131072


def reference(*, x, target, weight, ignore_index, smooth_eps):
    rows, num_classes = x.shape
    eps = float(smooth_eps)
    eps_sum = eps / num_classes if eps > 0 else 0.0
    eps_nll = 1.0 - eps_sum - eps
    w = weight.double()
    losses = torch.empty(rows, dtype=torch.float64, device=x.device)
    for start in range(0, rows, _CHUNK):
        logits = x[start:start + _CHUNK].double()
        labels = target[start:start + _CHUNK]
        lsm = torch.log_softmax(logits, dim=-1) * w
        likelihood = lsm.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        losses[start:start + _CHUNK] = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    if ignore_index >= 0:
        ignored = target == ignore_index
        losses.masked_fill_(ignored, 0.0)
        valid = int((~ignored).sum())
    else:
        valid = rows
    if valid == 0:
        return {"losses": losses, "out": torch.zeros((), dtype=torch.float64, device=x.device)}
    return {"losses": losses, "out": losses.sum() / valid}
