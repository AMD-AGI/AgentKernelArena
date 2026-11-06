"""Segment offsets for the jagged operand."""
import torch


def make_inputs(case, inputs, generator, device):
    """Prefix sums of per-sequence lengths that add up to ``total_rows``.

    Lengths are drawn proportionally to weights in [0.5, 1.5) and stay below the case's
    ``max_seq_len`` (twice the mean length); sequences may be empty only when there are fewer
    rows than sequences.
    """
    batch = case["axes"]["batch"]
    total = case["axes"]["total_rows"]
    max_seq_len = case["scalars"]["max_seq_len"]
    weights = 0.5 + torch.rand(batch, generator=generator, device=device, dtype=torch.float64)
    lengths = torch.floor(weights / weights.sum() * total).to(torch.int64)
    remainder = total - int(lengths.sum())
    lengths[:remainder] += 1
    if int(lengths.max()) > max_seq_len:
        raise ValueError("generated sequence length exceeds max_seq_len")
    offsets = torch.zeros(batch + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    return {"seq_offsets": offsets}
