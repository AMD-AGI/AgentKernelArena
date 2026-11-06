"""Scheduled batch of one model-runner step (vLLM v1) made of prefill requests.

Request b of the batch uses a random distinct persistent slot and has a random number of computed
tokens; its chunk of query_lens[b] tokens either finishes the prefill exactly, leaves one or two
tokens, or leaves a random remainder of the prompt. Query lengths average
query_len, range up to twice that (bounded by the row length) and sum to num_reqs * query_len.
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
    n, q, slots, width = axes["num_reqs"], axes["query_len"], axes["max_num_reqs"], axes["max_model_len"]
    lens = ragged_lengths(n, q, generator, device, max_len=width)
    start = torch.zeros(n + 1, dtype=torch.int64, device=device)
    start[1:] = torch.cumsum(lens, 0)
    slot_ids = torch.randperm(slots, generator=generator, device=device)[:n]

    # Chunk [computed, computed + len) inside the row, then a remainder of the prompt.
    computed = rand_below(width - lens + 1, generator, device)
    room = width - computed - lens
    pick = torch.rand((n,), generator=generator, device=device)
    small = rand_below(torch.full((n,), 3, dtype=torch.int64, device=device), generator, device)
    extra = torch.where(pick < 0.25, torch.zeros_like(room),
                        torch.where(pick < 0.5, torch.minimum(small, room), rand_below(room + 1, generator, device)))
    prefill_len = computed + lens + extra

    # Slots that are not scheduled hold unrelated state.
    all_computed = rand_below(torch.full((slots,), width, dtype=torch.int64, device=device), generator, device)
    all_prefill = rand_below(torch.full((slots,), width, dtype=torch.int64, device=device), generator, device)
    all_computed[slot_ids] = computed
    all_prefill[slot_ids] = prefill_len
    return {"idx_mapping": slot_ids.to(torch.int32), "query_start_loc": start.to(torch.int32),
            "prefill_len": all_prefill.to(torch.int32), "num_computed_tokens": all_computed.to(torch.int32)}
