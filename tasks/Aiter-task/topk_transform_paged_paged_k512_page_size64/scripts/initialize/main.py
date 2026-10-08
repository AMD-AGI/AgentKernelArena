from __future__ import annotations
import torch as torch

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


def initialize_topk_inputs(inputs, *, seed=0):
    """Initialize tie-free scores, valid lengths/table, and a conservative v2 plan."""
    scores, lengths, metadata = (
        inputs[name] for name in ("scores", "seq_lens", "metadata")
    )
    table = inputs.get("page_tables")
    page_size = inputs["page_size"]
    batch, width = _validate_operands(scores, lengths, table, page_size, metadata)
    if width > 2**24:
        raise ValueError("tie-free FP32 initialization requires width <= 2**24")
    generator = torch.Generator(device=scores.device).manual_seed(seed)
    for row in range(batch):
        # Bounded values avoid overflowing the backend's FP16 coarse histogram.
        scores[row].copy_(
            torch.randperm(width, device=scores.device, generator=generator).float()
            / max(width, 1)
        )
        if table is not None:
            table[row].copy_(
                torch.randperm(
                    table.shape[1], device=table.device, generator=generator
                ).to(torch.int32)
            )
    capacity = width if table is None else min(width, table.shape[1] * page_size)
    choices = [0, capacity // 4, capacity // 2, capacity]
    lengths.copy_(
        torch.tensor(
            [choices[(row + seed) % len(choices)] for row in range(batch)],
            dtype=torch.int32,
            device=lengths.device,
        )
    )
    # Match plan_topk_v2(lengths, static_threshold=INT32_MAX) without device reads.
    metadata.zero_()
    metadata[0, 0] = 2**31 - 1
    return inputs


_callable = initialize_topk_inputs


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
