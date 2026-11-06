"""Sequence lengths of one packed batch.

The lengths are random but constrained by the task axes: they sum to n_tokens, no length exceeds
max_len, and exactly one sequence (at a random position) has length max_len. Zero-length sequences
occur when n_tokens is small.
"""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    total, batch, max_len = axes["n_tokens"], axes["batch"], axes["max_len"]
    rest = total - max_len
    weights = torch.rand((batch - 1,), generator=generator, device=device, dtype=torch.float64)
    lengths = [min(max_len, int(rest * w / float(weights.sum()))) for w in weights.tolist()]
    deficit = rest - sum(lengths)
    while deficit > 0:
        for i in range(batch - 1):
            take = min(deficit, max_len - lengths[i])
            lengths[i] += take
            deficit -= take
            if deficit == 0:
                break
    lengths.append(max_len)
    order = torch.randperm(batch, generator=generator, device=device).tolist()
    shuffled = [lengths[i] for i in order]
    return {"lengths": torch.tensor(shuffled, dtype=torch.int32, device=device)}
