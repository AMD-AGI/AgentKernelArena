"""Inputs for apply_min_p: logits rows, heterogeneous per-request min_p values."""
import torch

_P_CHOICES = (0.02, 0.05, 0.1, 0.3, 0.5)
_CHUNK = 128
# Elements closer than this to a row's threshold are moved clear of it, so the fp32 kernel and the
# float64 reference cannot disagree about which side of the threshold an element is on.
_MARGIN = 1.0e-3


def make_inputs(case, inputs, generator, device):
    n_tokens = case["axes"]["n_tokens"]
    vocab = case["axes"]["vocab_size"]
    mapping = inputs["expanded_idx_mapping"].long()
    table = torch.tensor(_P_CHOICES, dtype=torch.float32, device=device)
    choice = torch.randint(0, len(_P_CHOICES), (n_tokens,), generator=generator, device=device)
    min_p = table[choice]
    ids = torch.arange(n_tokens, device=device)
    min_p = torch.where(ids % 16 == 2, torch.zeros_like(min_p), min_p)    # filter disabled
    min_p = torch.where(ids % 16 == 9, torch.ones_like(min_p), min_p)     # only the maximum survives
    logits = torch.empty(n_tokens, vocab, dtype=torch.float32, device=device)
    for start in range(0, n_tokens, _CHUNK):
        end = min(n_tokens, start + _CHUNK)
        x = torch.randn(end - start, vocab, generator=generator, device=device, dtype=torch.float32) * 2.0
        p = min_p[mapping[start:end]].double()
        active = ((p > 0) & (p < 1))[:, None]
        threshold = x.max(-1).values.double() + torch.log(p)
        near = ((x.double() - threshold[:, None]).abs() < _MARGIN) & active
        below = (threshold - 2.0 * _MARGIN).float()[:, None].expand_as(x)
        logits[start:end] = torch.where(near, below, x)
    return {"logits": logits, "min_p": min_p}
