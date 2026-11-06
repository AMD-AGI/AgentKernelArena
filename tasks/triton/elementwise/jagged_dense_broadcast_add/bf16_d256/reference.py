"""High-precision jagged + dense broadcast add reference used only for correctness."""
import torch


def reference(*, max_seq_len, seq_offsets, jagged, dense):
    lengths = seq_offsets[1:] - seq_offsets[:-1]
    batch_of_row = torch.repeat_interleave(torch.arange(dense.shape[0], device=jagged.device), lengths,
                                           output_size=jagged.shape[0])
    return jagged.double() + dense.double()[batch_of_row]
