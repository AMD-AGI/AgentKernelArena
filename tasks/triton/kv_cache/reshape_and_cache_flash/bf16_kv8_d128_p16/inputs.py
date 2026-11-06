"""Slot mapping of a paged-KV-cache write.

Tokens fill physical blocks in order, as a block allocator hands them out: token t goes to
slot ``block_ids[t // block_size] * block_size + t % block_size`` with the block ids a random
permutation of the pool, so slots are unique but blocks are scattered. The last 1/32 of the
tokens is graph padding (slot -1) and about 1% of the others are skipped slots (-1) too.
"""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_tokens, block_size, n_blocks = axes["n_tokens"], axes["block_size"], axes["n_blocks"]
    blocks = torch.randperm(n_blocks, generator=generator, device=device)
    t = torch.arange(n_tokens, device=device, dtype=torch.int64)
    slots = blocks[t // block_size].to(torch.int64) * block_size + t % block_size
    skipped = torch.rand(n_tokens, generator=generator, device=device) < 0.01
    skipped |= t >= n_tokens - n_tokens // 32
    return {"slot_mapping": torch.where(skipped, torch.full_like(slots, -1), slots)}
