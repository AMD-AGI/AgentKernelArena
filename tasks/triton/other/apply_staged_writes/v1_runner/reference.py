"""Exact integer reference of the staged-write applier."""
import torch


def reference(*, output, write_indices, write_starts, write_contents, write_cu_lens):
    result = output.clone()
    rows = write_indices.tolist()
    starts = write_starts.tolist()
    ends = write_cu_lens.tolist()
    begin = 0
    for row, start, end in zip(rows, starts, ends):
        if end < begin:
            raise ValueError("write_cu_lens must be non-decreasing")
        result[row, start:start + end - begin] = write_contents[begin:end]
        begin = end
    return result
