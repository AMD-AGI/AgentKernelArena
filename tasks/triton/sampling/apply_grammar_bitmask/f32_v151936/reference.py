"""Independent PyTorch reference for grammar-bitmask masking (exact: only -inf stores)."""
import torch

_CHUNK = 128


def reference(*, logits, logits_indices, bitmask):
    """Set logits[logits_indices[m], v] to -inf wherever bit (v % 32) of bitmask[m, v // 32] is 0."""
    out = logits.clone()
    vocab = out.shape[1]
    shifts = torch.arange(32, device=out.device, dtype=torch.int64)
    for begin in range(0, bitmask.shape[0], _CHUNK):
        words = bitmask[begin:begin + _CHUNK].long()
        allowed = ((words[:, :, None] >> shifts) & 1).reshape(words.shape[0], -1)[:, :vocab]
        rows = logits_indices[begin:begin + _CHUNK].long()
        out[rows] = torch.where(allowed == 0, torch.full_like(out[rows], float("-inf")), out[rows])
    return out
