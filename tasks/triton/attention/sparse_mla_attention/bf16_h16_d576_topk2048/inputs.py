"""Top-k selection of a single causal prefill sequence for sparse MLA attention."""
import torch

_CHUNK = 64


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n, topk = int(axes["n_tokens"]), int(axes["topk"])
    width = min(topk, n)
    indices = torch.full((n, topk), -1, dtype=torch.int32, device=device)
    columns = torch.arange(n, device=device)
    for start in range(0, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        rows = torch.arange(start, stop, device=device)
        # Token t may select any of the positions 0..t (the indexer's causal candidates); later
        # positions get +inf and sort last. Rows with fewer than topk candidates are padded with -1.
        score = torch.rand((stop - start, n), generator=generator, device=device)
        score = score.masked_fill(columns[None, :] > rows[:, None], float("inf"))
        chosen = score.argsort(dim=1)[:, :width]
        valid = torch.arange(width, device=device)[None, :] <= rows[:, None]
        indices[start:stop, :width] = torch.where(valid, chosen, -1).to(torch.int32)
    cu_seqlens_q = torch.tensor([0, n], dtype=torch.int32, device=device)
    return {"topk_indices": indices, "cu_seqlens_q": cu_seqlens_q}
