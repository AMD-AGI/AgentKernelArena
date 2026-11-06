"""Ragged CSR top-k selection for the DeepSeek-V4 sparse prefill task."""
import torch

# Most rows are full (index_topk); every 16th row cycles through the lengths that straddle the
# kernel's 16-row key block, including an empty row.
_RAGGED = (0, 1, 15, 16, 17, 31, 32, 33, 127, 255, 511, 512)


def _row_lengths(n_queries, topk, device):
    row = torch.arange(n_queries, device=device)
    pattern = torch.tensor(_RAGGED, device=device, dtype=torch.int64).clamp(max=topk)
    return torch.where(row % 16 == 0, pattern[(row // 16) % len(_RAGGED)],
                       torch.full_like(row, topk))


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n, topk, num_kv = int(axes["n_queries"]), int(axes["topk"]), int(axes["num_kv"])
    lengths = _row_lengths(n, topk, device)
    # Each row selects topk positions of the pool (sorted, with replacement as the original harness
    # did); only the first lengths[i] of them are kept.
    slots = torch.randint(0, num_kv, (n, topk), generator=generator, device=device).sort(dim=1).values
    keep = torch.arange(topk, device=device)[None, :] < lengths[:, None]
    flat = torch.full((n * topk,), -1, dtype=torch.int32, device=device)
    packed = slots[keep].to(torch.int32)
    flat[:packed.numel()] = packed
    indptr = torch.zeros(n + 1, dtype=torch.int32, device=device)
    indptr[1:] = lengths.cumsum(0).to(torch.int32)
    return {"indices": flat, "indptr": indptr}
