"""float64 reference of vLLM pack_seq: sequences concatenated along the token axis are placed into
a padded [batch, max_len, hidden] tensor; positions beyond a sequence's length hold pad_value."""
import torch


def reference(*, x, lengths, pad_value):
    batch = lengths.numel()
    max_len = int(lengths.max().item())
    out = torch.full((batch, max_len, x.shape[1]), float(pad_value), dtype=torch.float64, device=x.device)
    start = 0
    for b, length in enumerate(lengths.tolist()):
        out[b, :length] = x[start:start + length].double()
        start += length
    return out
