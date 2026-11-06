"""High-precision reference for the jagged x dense batched matmul with broadcast bias, used only for correctness."""
import torch


def reference(*, seq_offsets, jagged, dense, bias, max_seq_len):
    out = torch.zeros((jagged.shape[0], dense.shape[2]), dtype=torch.float64, device=jagged.device)
    offsets = seq_offsets.tolist()
    for group in range(dense.shape[0]):
        start, end = offsets[group], offsets[group + 1]
        if end > start:
            out[start:end] = jagged[start:end].double() @ dense[group].double() + bias[group].double()
    return out
