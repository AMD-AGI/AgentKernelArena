"""Neighborhood indices for PAConv score assignment.

``knn_idx[b, n]`` lists the K neighbors of center n; neighbor 0 is the center itself (a valid point), the
others are random points, and about 3% of the non-center entries are out of range (negative or >= N0),
which the operator must treat as outside the neighborhood.
"""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, n_centers, n_points, k = axes["batch"], axes["n_centers"], axes["n_points"], axes["K"]
    idx = torch.randint(0, n_points, (batch, n_centers, k), generator=generator, device=device,
                        dtype=torch.int64)
    draw = torch.rand((batch, n_centers, k), generator=generator, device=device)
    invalid = torch.where(torch.rand((batch, n_centers, k), generator=generator, device=device) < 0.5,
                          -1 - idx % 7, n_points + idx % 7)
    idx = torch.where((draw < 0.03) & (torch.arange(k, device=device) > 0), invalid, idx)
    return {"knn_idx": idx}
