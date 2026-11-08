"""Ragged index lists of a DeepSeek-V4 CSA layer during prefill of one prompt.

The pool is the prompt's compressed entries (compression ratio 4, one per four tokens: n_queries // 4 rows)
followed by one row per prompt token. Token i (position i) attends to
  - min((i + 1) // 4, topk) compressed entries: all of the (i + 1) // 4 available ones while there are at most
    topk, otherwise topk distinct ones drawn uniformly (the indexer's selection), in increasing order;
  - its sliding window: the rows of tokens i - min(i + 1, window) + 1 .. i.
The rows are packed back to back from offset 0, as the combine kernel of the production path writes them;
the unused tail of the capacity is -1.
"""
import torch

RATIO = 4


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n, topk, window = int(axes["n_queries"]), int(axes["topk"]), int(axes["window"])
    n_comp = n // RATIO
    pos = torch.arange(n, device=device)
    avail = (pos + 1) // RATIO
    topk_len = torch.minimum(avail, torch.full_like(avail, topk))
    swa_len = torch.minimum(pos + 1, torch.full_like(pos, window))
    width = topk + window
    table = torch.full((n, width), -1, dtype=torch.int64, device=device)
    column = torch.arange(width, device=device)[None, :]
    # Compressed part: everything available, or a uniform sample of topk entries.
    take_all = avail <= topk
    table[:, :topk] = torch.where(column[:, :topk] < topk_len[:, None], column[:, :topk].expand(n, topk), -1)
    heavy = (~take_all).nonzero().flatten()
    for lo in range(0, heavy.numel(), 2048):
        rows = heavy[lo:lo + 2048]
        draws = torch.rand(rows.numel(), n_comp, generator=generator, device=device)
        draws = torch.where(torch.arange(n_comp, device=device)[None, :] < avail[rows, None], draws,
                            torch.full_like(draws, -1.0))
        table[rows, :topk] = draws.topk(topk, dim=1).indices.sort(dim=1).values
    # Window part: rows n_comp + (i - swa_len + 1 .. i), right after the compressed entries of the row.
    offset = column[:, :window]
    swa = n_comp + (pos[:, None] - swa_len[:, None] + 1) + offset
    swa = torch.where(offset < swa_len[:, None], swa, -1)
    # Move the window entries to follow the row's compressed entries.
    dest = topk_len[:, None] + offset
    keep = offset < swa_len[:, None]
    row_ids = pos[:, None].expand(n, window)
    table[row_ids[keep], dest[keep]] = swa[keep]
    lengths = topk_len + swa_len
    mask = column < lengths[:, None]
    flat = torch.full((n * width,), -1, dtype=torch.int32, device=device)
    packed = table[mask].to(torch.int32)
    flat[:packed.numel()] = packed
    indptr = torch.zeros(n + 1, dtype=torch.int32, device=device)
    indptr[1:] = lengths.cumsum(0).to(torch.int32)
    return {"indices": flat, "indptr": indptr}
