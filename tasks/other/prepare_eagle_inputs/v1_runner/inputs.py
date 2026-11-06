"""Target-model batch of one speculative-decoding step (vLLM v1 model runner).

Request b has a random distinct persistent slot and query_lens[b] target tokens (average query_len,
up to twice that, summing to num_reqs * query_len). Its positions continue from a random number of
computed tokens. Of the last tokens of a request up to three are rejected, but at least one token
always remains. About a quarter of the requests sampled no token (chunked prefill) and take the next
prefill token instead of the last sampled one.
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
    total = axes["total_tokens"]
    lens = ragged_lengths(n, q, generator, device)
    start = torch.zeros(n + 1, dtype=torch.int64, device=device)
    start[1:] = torch.cumsum(lens, 0)
    slot_ids = torch.randperm(slots, generator=generator, device=device)[:n]
    computed = rand_below(torch.full((n,), 4096, dtype=torch.int64, device=device), generator, device)
    request_of_token = torch.repeat_interleave(torch.arange(n, device=device), lens, output_size=total)
    positions = computed[request_of_token] + torch.arange(total, device=device) - start[:-1][request_of_token]
    rejected = rand_below(torch.clamp(lens, max=4), generator, device)
    sampled = (torch.rand((n,), generator=generator, device=device) >= 0.25).to(torch.int32)
    seq_lens = computed + lens + rand_below(torch.full((n,), 64, dtype=torch.int64, device=device), generator, device)
    return {"target_positions": positions, "idx_mapping": slot_ids.to(torch.int32),
            "num_sampled": sampled, "num_rejected": rejected.to(torch.int32),
            "query_start_loc": start.to(torch.int32), "seq_lens": seq_lens.to(torch.int32)}
