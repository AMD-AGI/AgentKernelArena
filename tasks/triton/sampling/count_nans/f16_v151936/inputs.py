"""Logits with NaNs of varying density.

Most rows are clean (the production case: the count is a health metric), some carry a handful of
NaNs, some 5% and a few are entirely NaN. Every row also holds a few positive and negative
infinities, which must not be counted. Generated in row chunks to bound temporary memory.
"""
import torch

CHUNK_ROWS = 256
ROW_KINDS = (0.0, 0.0, 1.0e-4, 0.05, 1.0)  # NaN probability of each row kind
KIND_WEIGHTS = (0.4, 0.2, 0.2, 0.15, 0.05)


def make_inputs(case, inputs, generator, device):
    rows, vocab = case["axes"]["n_reqs"], case["axes"]["vocab_size"]
    logits = torch.empty(rows, vocab, dtype=torch.float16, device=device)
    probabilities = torch.tensor(ROW_KINDS, dtype=torch.float32, device=device)
    cumulative = torch.cumsum(torch.tensor(KIND_WEIGHTS, dtype=torch.float32, device=device), 0)
    for start in range(0, rows, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, rows)
        chunk = torch.randn(stop - start, vocab, generator=generator, device=device, dtype=torch.float32)
        pick = torch.rand(stop - start, generator=generator, device=device)
        kind = (pick[:, None] >= cumulative[None, :-1]).sum(-1)
        p = probabilities[kind][:, None]
        chunk[torch.rand(chunk.shape, generator=generator, device=device) < p] = float("nan")
        special = torch.rand(chunk.shape, generator=generator, device=device)
        chunk[special < 1.0e-5] = float("inf")
        chunk[(special >= 1.0e-5) & (special < 2.0e-5)] = float("-inf")
        logits[start:stop] = chunk
    return {"logits": logits}
