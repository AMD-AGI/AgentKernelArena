from __future__ import annotations
from functools import partial as partial
import torch as torch

def _reference_output(scores, k):
    return torch.empty((scores.shape[0], k), dtype=torch.int32, device=scores.device)


def _validate_operands(scores, seq_lens, page_tables, page_size, metadata):
    """Check metadata only; never synchronize device contents during dispatch."""
    if (
        type(page_size) is not int
        or not 0 < page_size < 2**32
        or page_size & (page_size - 1)
    ):
        raise ValueError("page_size must be a uint32 power of two")
    tensors = [scores, seq_lens, metadata]
    if page_tables is not None:
        tensors.append(page_tables)
    if not all(
        isinstance(t, torch.Tensor)
        and t.layout == torch.strided
        and not t.is_conj()
        and not t.is_neg()
        for t in tensors
    ):
        raise ValueError("dense tensor inputs required")
    if scores.ndim != 2 or scores.dtype != torch.float32:
        raise ValueError("scores must be FP32 [batch, width]")
    batch, width = scores.shape
    if scores.stride(1) != 1 or scores.stride(0) % 4 or scores.storage_offset() % 4:
        raise ValueError("scores must have aligned rows and unit column stride")
    if seq_lens.shape != (batch,) or metadata.shape != (batch + 1, 2):
        raise ValueError("expected seq_lens [batch] and metadata [batch + 1, 2]")
    if not seq_lens.is_contiguous() or not metadata.is_contiguous():
        raise ValueError("lengths and metadata must be contiguous")
    if any(t.dtype != torch.int32 or t.device != scores.device for t in tensors[1:]):
        raise ValueError(
            "lengths, metadata and page table must be int32 on scores.device"
        )
    if page_tables is not None and (
        page_tables.ndim != 2
        or page_tables.shape[0] != batch
        or page_tables.stride(1) != 1
    ):
        raise ValueError("page_tables must be [batch, pages] with unit column stride")
    return batch, width


def _validate_call(
    scores,
    seq_lens,
    page_tables,
    out_page_indices,
    page_size,
    metadata,
    out_raw_indices=None,
):
    batch, width = _validate_operands(
        scores, seq_lens, page_tables, page_size, metadata
    )
    if not isinstance(out_page_indices, torch.Tensor) or out_page_indices.ndim != 2:
        raise ValueError("out_page_indices must be a [batch, K] tensor")
    if (
        type(out_page_indices.shape[1]) is not int
        or not 0 < out_page_indices.shape[1] <= 2048
    ):
        raise ValueError("output width must be in (0, 2048]")
    if out_raw_indices is not None and page_tables is None:
        raise ValueError("a second raw output requires a page table")
    tensors = [scores, seq_lens, metadata, page_tables]
    for destination in (out_page_indices, out_raw_indices):
        if destination is None:
            continue
        if (
            not isinstance(destination, torch.Tensor)
            or destination.layout != torch.strided
            or destination.shape != (batch, out_page_indices.shape[1])
            or destination.dtype != torch.int32
            or destination.device != scores.device
            or not destination.is_contiguous()
            or destination.is_conj()
            or destination.is_neg()
            or any(
                isinstance(t, torch.Tensor) and torch._C._overlaps(destination, t)
                for t in tensors
            )
        ):
            raise ValueError(
                "destinations must be disjoint contiguous int32 [batch, K] tensors"
            )
        tensors.append(destination)
    return batch, width


def topk_reference(
    scores,
    seq_lens,
    page_tables,
    out_page_indices,
    page_size,
    metadata,
    out_raw_indices=None,
):
    """Write selected positions/slots into the supplied buffers and return None."""
    batch, width = _validate_call(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        metadata,
        out_raw_indices,
    )
    out_page_indices.fill_(-1)
    if out_raw_indices is not None:
        out_raw_indices.fill_(-1)
    for row in range(batch):
        length = int(seq_lens[row])
        if not 0 <= length <= width:
            raise ValueError("sequence lengths must be in [0, width]")
        if (
            page_tables is not None
            and (length + page_size - 1) // page_size > page_tables.shape[1]
        ):
            raise ValueError("page table does not cover the valid sequence")
        values = scores[row, :length]
        if torch.isnan(values).any():
            raise ValueError("NaN scores are unsupported")
        if not length:
            continue
        indices = (
            torch.arange(length, device=scores.device)
            if length <= out_page_indices.shape[1]
            else torch.argsort(values, descending=True, stable=True)[
                : out_page_indices.shape[1]
            ]
        )
        if page_tables is None:
            out_page_indices[row, : indices.numel()] = indices.to(torch.int32)
        else:
            pages = page_tables[row, indices // page_size].to(torch.int64)
            mapped = pages * page_size + indices % page_size
            if (pages < 0).any() or (mapped > 2**31 - 1).any():
                raise ValueError("page mapping outside signed int32")
            out_page_indices[row, : indices.numel()] = mapped.to(torch.int32)
        if out_raw_indices is not None:
            out_raw_indices[row, : indices.numel()] = indices.to(torch.int32)


def _paged_reference(scores, seq_lens, metadata, page_size, page_tables, *, k):
    out_page_indices = _reference_output(scores, k)
    topk_reference(scores, seq_lens, page_tables, out_page_indices, page_size, metadata)
    return out_page_indices


_callable = partial(_paged_reference, **{'k': 512})


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
