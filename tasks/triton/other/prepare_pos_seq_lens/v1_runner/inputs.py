"""Scheduled batch of one model-runner step (vLLM v1).

Request b of the batch has query_lens[b] new tokens, a random persistent request slot (distinct slots)
and a random number of already computed tokens. Query lengths are random, average query_len, range up
to twice that and sum to num_reqs * query_len, so the flat token buffer is exactly full.
"""
import torch


def ragged_lengths(n, mean, generator, device, min_len=1, max_len=None):
    """n lengths averaging mean, each in [min_len, min(2 * mean, max_len)], summing exactly to n * mean."""
    cap = 2 * mean if max_len is None else min(2 * mean, max_len)
    lens = torch.full((n,), mean, dtype=torch.int64, device=device)
    # Redistribute the length of random pairs: the pair sum (hence the total) is preserved.
    for _ in range(3):
        order = torch.randperm(n, generator=generator, device=device)
        a, b = order[0:n - n % 2:2], order[1:n - n % 2:2]
        total = lens[a] + lens[b]
        low = (total - cap).clamp(min=min_len)
        high = (total - min_len).clamp(max=cap)
        span = (high - low).clamp(min=0)
        draw = torch.rand(a.shape, generator=generator, device=device, dtype=torch.float64)
        new_a = low + (draw * (span + 1).double()).floor().long().clamp(max=span)
        lens[a] = new_a
        lens[b] = total - new_a
    return lens


def rand_below(bound, generator, device):
    """Uniform integers in [0, bound) for an int64 tensor of bounds (bound >= 1)."""
    draw = torch.rand(bound.shape, generator=generator, device=device, dtype=torch.float64)
    return (draw * bound.double()).floor().long().clamp(max=bound - 1)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n, q, slots = axes["num_reqs"], axes["query_len"], axes["max_num_reqs"]
    lens = ragged_lengths(n, q, generator, device)
    start = torch.zeros(n + 1, dtype=torch.int64, device=device)
    start[1:] = torch.cumsum(lens, 0)
    slot_ids = torch.randperm(slots, generator=generator, device=device)[:n]
    return {"idx_mapping": slot_ids.to(torch.int32), "query_start_loc": start.to(torch.int32)}
