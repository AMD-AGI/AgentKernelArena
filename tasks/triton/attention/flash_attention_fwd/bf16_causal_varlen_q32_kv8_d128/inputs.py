"""Ragged sequence metadata for the packed causal prefill batch.

A case fixes the number of sequences (batch), the packed token count (n_tokens)
and the longest sequence (the max_input_len scalar). Every draw places one
sequence of exactly max_input_len tokens at a random position and splits the
remaining tokens over the other sequences with random lengths in
[1, max_input_len], so lengths, offsets and the longest sequence's position
change between draws while the launch grid does not.
"""
import torch


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
    longest = case["scalars"]["max_input_len"]
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
    seq_len = torch.tensor(lengths, dtype=torch.int32, device=device)
    start_loc = torch.zeros(batch, dtype=torch.int32, device=device)
    start_loc[1:] = torch.cumsum(seq_len[:-1], dim=0).to(torch.int32)
    return {"b_start_loc": start_loc, "b_seq_len": seq_len}
