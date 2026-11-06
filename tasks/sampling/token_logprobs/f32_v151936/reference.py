import torch

_CHUNK = 128


def reference(*, logits, token_ids):
    out = torch.empty(token_ids.shape, dtype=torch.float64, device=logits.device)
    for start in range(0, logits.shape[0], _CHUNK):
        rows = logits[start:start + _CHUNK].double()
        selected = rows.gather(1, token_ids[start:start + _CHUNK])
        out[start:start + _CHUNK] = selected - torch.logsumexp(rows, dim=-1, keepdim=True)
    return out
