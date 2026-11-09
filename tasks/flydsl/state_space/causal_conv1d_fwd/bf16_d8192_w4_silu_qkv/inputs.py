"""Packed-sequence metadata of the causal conv1d (split q/k/v) task."""
import torch

CHUNK = 64          # tokens per kernel program (the kernel's fixed tile)
MAX_SLACK = CHUNK - 1


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_tokens, n_seqs, n_chunks, cache_lines = (axes["n_tokens"], axes["n_seqs"], axes["n_chunks"],
                                               axes["cache_lines"])
    # Random sequence lengths whose chunk counts add up to exactly n_chunks, the size of the schedule
    # arrays: split n_chunks into n_seqs positive chunk counts, then draw each length inside its count
    # (64 * (count - 1) + 1 .. 64 * count) so that the lengths add up to n_tokens.
    if n_seqs > 1:
        cuts = torch.randperm(n_chunks - 1, generator=generator, device=device)[: n_seqs - 1] + 1
        cuts = torch.cat([cuts.new_zeros(1), cuts.sort().values, cuts.new_full((1,), n_chunks)])
        counts = (cuts[1:] - cuts[:-1]).tolist()
    else:
        counts = [n_chunks]
    low = [CHUNK * (c - 1) + 1 for c in counts]
    slack = n_tokens - sum(low)
    if not 0 <= slack <= MAX_SLACK * n_seqs:
        raise ValueError("n_tokens is not reachable with n_chunks chunks")
    weights = torch.rand(n_seqs, generator=generator, device=device, dtype=torch.float64)
    extra = torch.floor(slack * weights / weights.sum()).long().clamp(max=MAX_SLACK).tolist()
    order = torch.randperm(n_seqs, generator=generator, device=device).tolist()
    remaining, i = slack - sum(extra), 0
    while remaining > 0:
        j = order[i % n_seqs]
        if extra[j] < MAX_SLACK:
            extra[j] += 1
            remaining -= 1
        i += 1
    lengths = torch.tensor([lo + e for lo, e in zip(low, extra)], dtype=torch.int64)
    bounds = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)]).to(torch.int32).to(device)
    # (sequence, chunk) pair of every kernel program, as the attention metadata of a serving step holds it.
    seq_ids = torch.repeat_interleave(torch.arange(n_seqs, dtype=torch.int64), torch.tensor(counts))
    first = torch.tensor(counts).cumsum(0) - torch.tensor(counts)
    chunk_ids = torch.arange(n_chunks, dtype=torch.int64) - torch.repeat_interleave(first, torch.tensor(counts))
    # Each sequence owns a distinct cache slot; the pool also holds slots no sequence touches.
    slots = torch.randperm(cache_lines, generator=generator, device=device)[:n_seqs].to(torch.int32)
    has_state = torch.rand(n_seqs, generator=generator, device=device) < 0.5
    return {"query_start_loc": bounds, "cache_indices": slots, "has_initial_state": has_state,
            "chunk_sequence_ids": seq_ids.to(torch.int32).to(device), "chunk_ids": chunk_ids.to(torch.int32).to(device)}
