"""Inputs for token_ranks: logits rows and sampled ids covering the best, worst, tied and typical cases."""
import torch

_CHUNK = 256


def make_inputs(case, inputs, generator, device):
    n_rows = case["axes"]["n_rows"]
    vocab = case["axes"]["vocab_size"]
    logits = torch.empty(n_rows, vocab, dtype=torch.float32, device=device)
    ids = torch.randint(0, vocab, (n_rows,), generator=generator, device=device, dtype=torch.int64)
    for start in range(0, n_rows, _CHUNK):
        end = min(n_rows, start + _CHUNK)
        logits[start:end] = torch.randn(end - start, vocab, generator=generator, device=device,
                                        dtype=torch.float32) * 2.0
    rows = torch.arange(n_rows, device=device)
    # Row classes by index: the greedy token (rank 1), the least likely token (rank vocab_size, or the
    # tie count at the minimum), a token whose logit is duplicated elsewhere (exercises ">="), and
    # a typical sampled token.
    best = rows[rows % 8 == 0]
    worst = rows[rows % 8 == 1]
    tied = rows[rows % 8 == 2]
    first = rows[rows % 8 == 3]
    last = rows[rows % 8 == 4]
    ids[first] = 0
    ids[last] = vocab - 1
    if best.numel():
        ids[best] = logits[best].argmax(-1)
    if worst.numel():
        ids[worst] = logits[worst].argmin(-1)
    if tied.numel():
        partner = (ids[tied] + 1 + tied % 7) % vocab
        logits[tied, partner] = logits[tied, ids[tied]]
    return {"logits": logits, "sampled_token_ids": ids}
