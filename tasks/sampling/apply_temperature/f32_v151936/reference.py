import torch

_CHUNK = 128


def reference(*, logits, expanded_idx_mapping, temperature):
    out = torch.empty(logits.shape, dtype=torch.float64, device=logits.device)
    for start in range(0, out.shape[0], _CHUNK):
        rows = logits[start:start + _CHUNK].double()
        t = temperature[expanded_idx_mapping[start:start + _CHUNK].long()].double()[:, None]
        out[start:start + _CHUNK] = torch.where((t == 0) | (t == 1), rows, rows / t)
    return out
