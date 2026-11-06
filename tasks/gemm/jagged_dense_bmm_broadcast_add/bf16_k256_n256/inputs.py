"""Ragged group boundaries of one jagged batch.

``seq_offsets`` is the cumulative row count of the groups, as a recommendation model packs the
variable-length histories of a batch of users into one jagged tensor. The group lengths are random,
about one group in six is empty, one group holds exactly ``max_seq_len`` rows (the longest, as the
host knows it) and the lengths sum to ``total_rows``.
"""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    total_rows = axes["total_rows"]
    max_seq_len = axes["max_seq_len"]
    n_groups = axes["n_groups"]
    if case["scalars"]["max_seq_len"] != max_seq_len:
        raise ValueError("Each case's max_seq_len scalar must equal its max_seq_len axis")
    if not max_seq_len <= total_rows <= n_groups * max_seq_len:
        raise ValueError("total_rows must lie between max_seq_len and n_groups * max_seq_len")

    longest = int(torch.randint(0, n_groups, (1,), generator=generator, device=device))
    lengths = torch.zeros(n_groups, dtype=torch.int64, device=device)
    lengths[longest] = max_seq_len
    weights = torch.rand((n_groups,), generator=generator, device=device)
    weights[torch.rand((n_groups,), generator=generator, device=device) < 1.0 / 6] = 0.0
    weights[longest] = 0.0
    remaining = total_rows - max_seq_len
    # Spread the remaining rows in proportion to the weights without exceeding max_seq_len per group.
    while remaining > 0:
        room = max_seq_len - lengths
        active = (weights > 0) & (room > 0)
        if not bool(active.any()):
            # Every weighted group is full: reopen the empty groups that still have room.
            weights = torch.where((room > 0) & (torch.arange(n_groups, device=device) != longest),
                                  torch.ones_like(weights), weights)
            active = (weights > 0) & (room > 0)
        share = torch.where(active, weights, torch.zeros_like(weights))
        grant = torch.minimum((share / share.sum() * remaining).floor().to(torch.int64), room)
        if int(grant.sum()) == 0:
            # Fewer rows remain than groups: hand single rows to random active groups.
            order = torch.argsort(torch.rand((n_groups,), generator=generator, device=device)
                                  + (~active).to(torch.float32) * 2.0)
            picked = order[: min(remaining, int(active.sum()))]
            grant = torch.zeros_like(lengths)
            grant[picked] = 1
        lengths += grant
        remaining -= int(grant.sum())

    seq_offsets = torch.zeros(n_groups + 1, dtype=torch.int64, device=device)
    seq_offsets[1:] = lengths.cumsum(0)
    return {"seq_offsets": seq_offsets}
