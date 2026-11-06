import math

import torch

_CHUNK = 128


def reference(*, logits, expanded_idx_mapping, min_p):
    out = logits.clone()
    for start in range(0, out.shape[0], _CHUNK):
        rows = out[start:start + _CHUNK]
        p = min_p[expanded_idx_mapping[start:start + _CHUNK].long()].double()
        threshold = rows.max(-1).values.double() + torch.log(p)
        drop = (rows.double() < threshold[:, None]) & (p != 0)[:, None]
        rows.masked_fill_(drop, -math.inf)
    return out
