"""Learned position-add reference used only for correctness."""
import torch


def reference(*, x, pos_embedding):
    positions = torch.arange(x.shape[1], device=x.device)
    pos64 = pos_embedding.double().index_select(0, positions)
    out = torch.empty_like(x)
    # float64 sum, rounded to float32, computed in batch chunks to bound temporary memory.
    for start in range(0, x.shape[0], 64):
        out[start:start + 64] = (x[start:start + 64].double() + pos64).to(x.dtype)
    return out
