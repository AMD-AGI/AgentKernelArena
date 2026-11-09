"""Routing of an MoE layer: how many of the m rows go to each of the 64 experts.

The rows are tokens x top-k assignments already sorted by expert, so the operator only sees the group sizes. Expert
popularity is skewed (a lognormal gate-logit spread), and every row is drawn independently from it, so the loads are
uneven, a few experts hold several times the mean load, and at small m some experts receive no rows at all.
"""
import torch

POPULARITY_SPREAD = 0.7


def make_inputs(case, inputs, generator, device):
    m = case["axes"]["m"]
    groups = case["axes"]["num_groups"]
    logits = POPULARITY_SPREAD * torch.randn(groups, generator=generator, device=device, dtype=torch.float32)
    popularity = torch.softmax(logits, dim=0)
    rows = torch.multinomial(popularity, m, replacement=True, generator=generator)
    return {"group_sizes": torch.bincount(rows, minlength=groups).to(torch.int32)}
