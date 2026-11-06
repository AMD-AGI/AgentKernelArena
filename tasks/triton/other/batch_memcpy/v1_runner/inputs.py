"""Copy tables of a batched state copy.

The pools hold ``batch`` slots of ``copy_bytes`` bytes. Copy sizes vary around the slot size (pairs of
``copy_bytes -/+ d`` with an arbitrary byte count ``d``, so most sizes are not multiples of any block
size and exercise the masked tail of every copy); they always sum to the pool size. Source and
destination layouts pack the copies back to back in two independent random orders, so the copies
tile the destination pool exactly: every destination byte is written once and none is left to chance.
Offsets are in bytes from the pool base.
"""
import torch


def _packed_offsets(sizes, generator, device):
    order = torch.randperm(sizes.shape[0], generator=generator, device=device)
    ordered = sizes[order].to(torch.int64)
    starts = torch.cumsum(ordered, 0) - ordered
    offsets = torch.empty_like(starts)
    offsets[order] = starts
    return offsets


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, copy_bytes = axes["batch"], axes["copy_bytes"]
    half = batch // 2
    d = torch.randint(1, copy_bytes // 2, (half,), generator=generator, device=device, dtype=torch.int64)
    sizes = torch.full((batch,), copy_bytes, dtype=torch.int64, device=device)
    sizes[:half] = copy_bytes - d
    sizes[half:2 * half] = copy_bytes + d
    sizes = sizes[torch.randperm(batch, generator=generator, device=device)]
    return {
        "src_offsets": _packed_offsets(sizes, generator, device),
        "dst_offsets": _packed_offsets(sizes, generator, device),
        "sizes": sizes.to(torch.int32),
    }
