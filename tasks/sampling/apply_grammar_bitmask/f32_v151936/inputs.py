"""Packed grammar bitmasks of a structured-output batch.

Three quarters of the logit rows belong to a request with a grammar; their rows are chosen at random
(the kernel receives the bitmask-row to logits-row mapping). Each bitmask row allows a random
fraction of the vocabulary, drawn per row from very restrictive to permissive grammars, and all 32
bits of every int32 word are used (a set bit means the token is allowed).
"""
import torch

ALLOWED_FRACTIONS = (0.002, 0.02, 0.3, 0.7, 0.98)
_CHUNK = 256


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_rows, n_masks, vocab, words = axes["n_tokens"], axes["n_masks"], axes["vocab_size"], axes["bitmask_words"]
    rows = torch.randperm(n_rows, generator=generator, device=device)[:n_masks]
    fractions = torch.tensor(ALLOWED_FRACTIONS, device=device)[
        torch.randint(0, len(ALLOWED_FRACTIONS), (n_masks,), generator=generator, device=device)]
    shifts = torch.arange(32, device=device, dtype=torch.int64)
    packed = torch.empty((n_masks, words), dtype=torch.int32, device=device)
    for begin in range(0, n_masks, _CHUNK):
        chunk = fractions[begin:begin + _CHUNK]
        draw = torch.rand((chunk.numel(), words * 32), generator=generator, device=device)
        bits = (draw < chunk[:, None]).reshape(chunk.numel(), words, 32).long()
        value = (bits << shifts).sum(-1)
        packed[begin:begin + _CHUNK] = torch.where(value >= 2 ** 31, value - 2 ** 32, value).to(torch.int32)
    return {"logits_indices": rows.to(torch.int32), "bitmask": packed}
