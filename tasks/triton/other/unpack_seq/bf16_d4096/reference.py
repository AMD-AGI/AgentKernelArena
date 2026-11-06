"""float64 reference of vLLM unpack_seq: the first lengths[b] rows of every padded sequence are
concatenated along the token axis; padding rows are dropped."""
import torch


def reference(*, packed, lengths):
    parts = [packed[b, :length].double() for b, length in enumerate(lengths.tolist())]
    return torch.cat(parts, dim=0)
