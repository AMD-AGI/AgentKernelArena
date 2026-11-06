"""Exact reference of the DeepSeek-V4 paged top-512 transform, used only for correctness.

Row ``b`` with valid length ``L = seq_lens[b]`` selects positions of ``scores[b, :L]``: all of
``0 .. L - 1`` in order when ``L <= 512``, otherwise the 512 positions holding the largest scores.
Each selected position ``p`` becomes the KV slot ``page_tables[b, p // page_size] * page_size +
p % page_size``; unused output slots are -1. Only the valid prefix is read. Selection only compares
scores, so widening them to float64 is exact; ties (absent from the task's inputs) resolve to the
lower position.
"""
import torch

TOPK = 512
ROW_CHUNK = 64


def reference(*, scores, seq_lens, page_tables, page_size, metadata):
    batch, width = scores.shape
    lengths = seq_lens.to(torch.int64)
    if bool(((lengths < 0) | (lengths > width)).any()):
        raise ValueError("seq_lens must lie in [0, width]")
    if bool(((lengths + page_size - 1) // page_size > page_tables.shape[1]).any()):
        raise ValueError("the page table does not cover a valid prefix")
    out = torch.full((batch, TOPK), -1, dtype=torch.int64, device=scores.device)
    positions = torch.arange(width, device=scores.device)
    prefix = positions[:TOPK]
    for start in range(0, batch, ROW_CHUNK):
        stop = min(start + ROW_CHUNK, batch)
        length = lengths[start:stop, None]
        valid = positions[None, :] < length
        masked = scores[start:stop].double().masked_fill(~valid, float("-inf"))
        largest = torch.sort(masked, dim=1, descending=True, stable=True).indices[:, :TOPK]
        selected = torch.where(length <= TOPK, prefix[None, :], largest)
        pages = page_tables[start:stop].to(torch.int64).gather(1, selected // page_size)
        slots = pages * page_size + selected % page_size
        out[start:stop] = torch.where(selected < length, slots, torch.full_like(slots, -1))
    return out
