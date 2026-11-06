"""Inputs for token_logprobs: token ids with boundary, duplicated and typical entries."""
import torch


def make_inputs(case, inputs, generator, device):
    n_rows = case["axes"]["n_rows"]
    n_ids = case["axes"]["n_ids"]
    vocab = case["axes"]["vocab_size"]
    logits = inputs["logits"]
    ids = torch.randint(0, vocab, (n_rows, n_ids), generator=generator, device=device, dtype=torch.int64)
    rows = torch.arange(n_rows, device=device)
    # Rows by index modulo 4: 0, the first and last vocabulary ids (the last lies in the partial tile
    # of the logits scan); 1, the row maximum and the row minimum; 2, a duplicated id; the rest, typical.
    r0 = rows[rows % 4 == 0]
    r1 = rows[rows % 4 == 1]
    r2 = rows[rows % 4 == 2]
    ids[r0, 0] = 0
    ids[r0, 1] = vocab - 1
    if r1.numel():
        ids[r1, 0] = logits[r1].argmax(-1)
        ids[r1, 1] = logits[r1].argmin(-1)
    ids[r2, 1] = ids[r2, 0]
    return {"token_ids": ids}
