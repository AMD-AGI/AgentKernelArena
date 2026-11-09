"""Float64 ragged grouped GEMM reference used only for correctness."""
import torch


def reference(*, lhs, rhs, group_sizes):
    out = torch.zeros(lhs.shape[0], rhs.shape[2], dtype=torch.float64, device=lhs.device)
    start = 0
    for expert, rows in enumerate(group_sizes.tolist()):
        if rows:
            out[start:start + rows] = lhs[start:start + rows].double() @ rhs[expert].double()
        start += rows
    return out
