import torch

_CHUNK = 256


def reference(*, logits, sampled_token_ids):
    out = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
    for start in range(0, logits.shape[0], _CHUNK):
        rows = logits[start:start + _CHUNK].double()
        picked = rows.gather(1, sampled_token_ids[start:start + _CHUNK, None])
        out[start:start + _CHUNK] = (rows >= picked).sum(-1)
    return out
