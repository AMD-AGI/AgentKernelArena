import math

import torch

_CHUNK = 16


def reference(*, logits, k, p, percentile_to_std_table, normal_cdf_to_sigma_table):
    """Top-k, then top-p over the survivors' softmax, as vLLM's PyTorch sampler defines them.

    A row keeps its k largest logits (all logits when k >= vocab_size); among those it keeps the
    smallest set of largest logits whose softmax mass (renormalised over the k survivors) reaches p
    (everything when p >= 1). Dropped logits become -inf and kept logits are unchanged. The pivot
    tables only parametrise the vLLM kernel's search and are not used here.
    """
    out = logits.clone()
    vocab = logits.shape[-1]
    for start in range(0, out.shape[0], _CHUNK):
        x = logits[start:start + _CHUNK].double()
        rows = torch.arange(x.shape[0], device=x.device)
        kk = k[start:start + _CHUNK].long().clamp(1, vocab)
        pp = p[start:start + _CHUNK].double()
        candidates = torch.where(x >= x.sort(-1, descending=True).values[rows, kk - 1][:, None], x, -math.inf)
        probs = torch.softmax(candidates, dim=-1)
        sorted_probs, _ = probs.sort(-1, descending=True)
        mass_before = sorted_probs.cumsum(-1) - sorted_probs
        n_keep = torch.where(pp >= 1.0, (candidates > -math.inf).sum(-1), (mass_before < pp[:, None]).sum(-1))
        threshold = candidates.sort(-1, descending=True).values[rows, n_keep.clamp(min=1) - 1]
        out[start:start + _CHUNK] = torch.where(x >= threshold[:, None], logits[start:start + _CHUNK],
                                                torch.full_like(logits[start:start + _CHUNK], -math.inf))
    return out
