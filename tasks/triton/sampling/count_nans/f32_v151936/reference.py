"""Exact reference of the per-row NaN count."""
import torch


def reference(*, logits):
    return torch.isnan(logits).sum(dim=-1).to(torch.int32)
