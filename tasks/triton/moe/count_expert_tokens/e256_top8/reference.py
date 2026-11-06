"""Exact per-expert assignment counts, used only for correctness."""
import torch


def reference(*, topk_ids):
    ids = topk_ids.reshape(-1).to(torch.int64)
    valid = ids[ids >= 0]
    return torch.bincount(valid, minlength=256).to(torch.int32)
