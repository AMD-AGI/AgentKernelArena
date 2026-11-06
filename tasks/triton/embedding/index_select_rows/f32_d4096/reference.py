"""Row-gather reference used only for correctness."""
import torch


def reference(*, table, idx):
    # A gather copies values, so the float32 result is exact (and avoids a float64 copy of the output).
    return table.index_select(0, idx)
