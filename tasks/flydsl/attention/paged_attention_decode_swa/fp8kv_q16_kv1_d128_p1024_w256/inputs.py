"""FP8 paged KV cache, block tables and context lengths of one sliding-window decode batch.

Each sequence keeps only the 1024-token pages that hold its last ``sliding_window + 1`` tokens (one
or two pages); the block-table entries of earlier pages point at the reserved null page 0, as a
serving framework does once those pages leave the window and are freed. Every sequence owns two
pages, dealt in random order (``num_blocks - 1 = 2 * num_seqs``); a sequence whose window lies in
one page leaves its second page unreferenced. The null page and the unreferenced pages are NaN,
so that any read of them reaching the result is visible. Referenced pages are fully finite:
the slots before the window hold older tokens and the slots after the last token hold stale data
from an earlier request, neither of which may affect the result.

Keys and values are standard-normal activations quantized to float8_e4m3fn with one per-tensor
scale each (absolute maximum / 448), the dequantization multipliers ``key_scale`` and
``value_scale``.
"""
import torch

_FP8_MAX = 448.0
_FP8_NAN = 0x7F


def _quantize(real):
    # In place, so the largest cases keep a single float32 copy of the cache.
    low, high = torch.aminmax(real)
    scale = torch.maximum(-low, high).clamp(min=1e-12) / _FP8_MAX
    codes = real.div_(scale).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return codes, scale.reshape(1).to(torch.float32)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    num_seqs = axes["num_seqs"]
    max_context_len = axes["max_context_len"]
    block_size = axes["block_size"]
    max_blocks = axes["max_blocks_per_seq"]
    num_blocks = axes["num_blocks"]
    num_kv_heads = axes["num_kv_heads"]
    head_size = axes["head_size"]
    x = axes["x"]
    window = case["scalars"]["sliding_window"]

    # Context lengths of a steady decode batch: the longest is max_context_len, the others between
    # half of it and all of it.
    low = (max_context_len + 1) // 2
    context_lengths = torch.randint(low, max_context_len + 1, (num_seqs,), generator=generator, device=device)
    longest = torch.randint(0, num_seqs, (1,), generator=generator, device=device)
    context_lengths[longest] = max_context_len

    # Pages holding the window [max(0, len - 1 - window), len - 1].
    first = (context_lengths - 1 - window).clamp(min=0) // block_size
    last = (context_lengths - 1) // block_size
    pages = (torch.randperm(num_blocks - 1, generator=generator, device=device) + 1).view(num_seqs, 2)
    block_tables = torch.zeros((num_seqs, max_blocks), dtype=torch.int64, device=device)
    block_tables.scatter_(1, first[:, None], pages[:, :1])
    block_tables.scatter_(1, last[:, None], pages[:, 1:])
    referenced = torch.zeros(num_blocks, dtype=torch.bool, device=device)
    referenced[block_tables.reshape(-1)] = True
    referenced[0] = False

    keys = torch.randn((num_blocks, num_kv_heads, head_size // x, block_size, x), generator=generator,
                       device=device, dtype=torch.float32)
    key_cache, key_scale = _quantize(keys)
    del keys
    values = torch.randn((num_blocks, num_kv_heads, head_size, block_size), generator=generator,
                         device=device, dtype=torch.float32)
    value_cache, value_scale = _quantize(values)
    del values
    key_cache.view(torch.uint8)[~referenced] = _FP8_NAN
    value_cache.view(torch.uint8)[~referenced] = _FP8_NAN
    return {"key_cache": key_cache, "value_cache": value_cache, "block_tables": block_tables.to(torch.int32),
            "context_lengths": context_lengths.to(torch.int32), "key_scale": key_scale,
            "value_scale": value_scale}
