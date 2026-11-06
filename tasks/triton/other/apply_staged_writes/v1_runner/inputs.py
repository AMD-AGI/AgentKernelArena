"""Staged writes of one scheduler step (vLLM v1 model runner).

Every write targets a distinct row (writes to one row would race) at a random column offset and has a
random length: the lengths of a batch average ``seg_len``, range from 0 to ``2 * seg_len`` (zero-length
segments occur) and always sum to ``n_writes * seg_len`` so the flat contents buffer is exactly full.
Every segment fits inside its row.
"""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n, seg, rows, width = axes["n_writes"], axes["seg_len"], axes["num_rows"], axes["row_size"]
    cap = min(width, 2 * seg)
    lens = torch.full((n,), seg, dtype=torch.int64, device=device)
    # Redistribute the length of random pairs: the pair sum (hence the total) is preserved.
    for _ in range(3):
        order = torch.randperm(n, generator=generator, device=device)
        a, b = order[0:n - n % 2:2], order[1:n - n % 2:2]
        total = lens[a] + lens[b]
        low = (total - cap).clamp(min=0)
        high = total.clamp(max=cap)
        new_a = low + (torch.rand(a.shape, generator=generator, device=device) * (high - low + 1).double()
                       ).floor().long().clamp(max=high - low)
        lens[a] = new_a
        lens[b] = total - new_a
    starts = (torch.rand((n,), generator=generator, device=device) * (width - lens + 1).double()).floor().long()
    starts = torch.minimum(starts, width - lens)
    row_ids = torch.randperm(rows, generator=generator, device=device)[:n]
    return {"write_indices": row_ids.to(torch.int32), "write_starts": starts.to(torch.int32),
            "write_cu_lens": torch.cumsum(lens, 0).to(torch.int32)}
