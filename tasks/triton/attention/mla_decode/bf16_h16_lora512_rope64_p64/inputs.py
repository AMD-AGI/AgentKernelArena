"""Paged-cache routing for decode: shuffled physical pages, ragged sequence lengths."""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, ctx = int(axes["batch"]), int(axes["ctx"])
    max_blocks, num_blocks = int(axes["max_blocks"]), int(axes["num_blocks"])
    # Every sequence owns max_blocks distinct physical pages, scattered over the whole cache the
    # way a block allocator hands them out after many requests have come and gone.
    pages = torch.randperm(num_blocks, generator=generator, device=device).to(torch.int32)
    block_tables = pages.reshape(batch, max_blocks).contiguous()
    # The longest sequence is exactly ctx; the others are between half of it and ctx tokens, so
    # partial last pages and short rows are exercised together with a full one.
    low = max(1, (ctx + 1) // 2)
    draw = torch.rand(batch, generator=generator, device=device, dtype=torch.float32)
    lengths = low + (draw * (ctx - low + 1)).to(torch.int64).clamp(max=ctx - low)
    lengths[0] = ctx
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
    return {"block_tables": block_tables, "seq_lens": lengths.to(torch.int32),
            "cu_seqlens_q": cu_seqlens_q}
