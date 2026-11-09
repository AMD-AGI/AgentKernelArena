"""Jagged metadata of an HSTU batch: sequence offsets, candidate counts and the length sort order.

A case fixes the number of sequences (batch), the packed token count (n_tokens) and the longest
sequence (the max_seq_len scalar). Every draw places one sequence of exactly max_seq_len tokens at a
random position and splits the remaining tokens over the other sequences with random lengths in
[1, max_seq_len]. Each sequence ends with 1 to 10 candidate tokens (the ranking candidates of the
production model), capped so that at least one history token remains after the context tokens;
sequences that are not longer than the context get none.
"""
import torch

MAX_CANDIDATES = 10


def _split(total, count, cap, weights):
    """count lengths in [1, cap] summing to total, roughly proportional to weights."""
    lengths = [1] * count
    remaining = total - count
    while remaining > 0:
        room = [i for i in range(count) if lengths[i] < cap]
        weight_sum = sum(weights[i] for i in room)
        given = 0
        for i in room:
            add = min(cap - lengths[i], int(remaining * weights[i] / weight_sum))
            lengths[i] += add
            given += add
        if given == 0:
            for i in sorted(room, key=lambda j: -weights[j]):
                if given == remaining:
                    break
                lengths[i] += 1
                given += 1
        remaining -= given
    return lengths


def make_inputs(case, inputs, generator, device):
    batch = case["axes"]["batch"]
    n_tokens = case["axes"]["n_tokens"]
    longest = case["scalars"]["max_seq_len"]
    context = case["scalars"].get("contextual_seq_len", 0)
    others = batch - 1
    if not 1 <= longest <= n_tokens or not others <= n_tokens - longest <= others * longest:
        raise ValueError(f"case {case['id']}: {batch} sequences of at most {longest} tokens "
                         f"cannot hold {n_tokens} tokens with the longest exactly {longest}")
    if others:
        weights = (0.25 + 0.75 * torch.rand(others, generator=generator, device=device)).tolist()
        lengths = _split(n_tokens - longest, others, longest, weights)
        position = int(torch.randint(0, batch, (1,), generator=generator, device=device).item())
        lengths.insert(position, longest)
    else:
        lengths = [longest]
    drawn = torch.randint(1, MAX_CANDIDATES + 1, (batch,), generator=generator, device=device).tolist()
    targets = [min(t, max(length - context - 1, 0)) for t, length in zip(drawn, lengths)]
    seq_len = torch.tensor(lengths, dtype=torch.int32, device=device)
    offsets = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(seq_len, dim=0).to(torch.int32)
    order = sorted(range(batch), key=lambda i: -lengths[i])
    return {
        "seq_offsets": offsets,
        "num_targets": torch.tensor(targets, dtype=torch.int32, device=device),
        "sort_by_length_indices": torch.tensor(order, dtype=torch.int64, device=device),
    }
