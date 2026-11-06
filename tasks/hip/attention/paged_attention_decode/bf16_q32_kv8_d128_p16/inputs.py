"""Paged KV cache, block table and context lengths of one decode batch.

The longest sequence has max_seq_len tokens and the others between half of it and all of it: the
steady state of a batch whose requests generate as many tokens as their prompts hold (1024-token
prompts and outputs give contexts of 1024 to 2048). Physical page 0 is vLLM's reserved null page.
Pages 1..num_blocks-1 are dealt to the sequences in random order
(num_blocks - 1 = num_seqs * max_blocks_per_seq), so consecutive logical blocks of a sequence
live in scattered pages; table entries past a sequence's last page are 0. Every cache slot that holds no token of any sequence -- the tail of a partial last page,
pages past a sequence's last page and page 0 -- holds stale finite values 64 times larger than
the token values, as reused cache memory does in production. They must not reach the result.
"""
import torch

_STALE_SCALE = 64.0


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    num_seqs = axes["num_seqs"]
    max_seq_len = axes["max_seq_len"]
    block_size = axes["block_size"]
    max_blocks = axes["max_blocks_per_seq"]
    num_blocks = axes["num_blocks"]
    num_kv_heads = axes["num_kv_heads"]
    head_size = axes["head_size"]
    x = axes["x"]

    low = (max_seq_len + 1) // 2
    seq_lens = torch.randint(low, max_seq_len + 1, (num_seqs,), generator=generator, device=device)
    longest = torch.randint(0, num_seqs, (1,), generator=generator, device=device)
    seq_lens[longest] = max_seq_len

    pages = (torch.randperm(num_blocks - 1, generator=generator, device=device) + 1).view(num_seqs, max_blocks)
    logical = torch.arange(max_blocks, device=device)
    used = (seq_lens + block_size - 1) // block_size
    block_table = torch.where(logical[None, :] < used[:, None], pages, torch.zeros_like(pages))

    # Tokens held by every physical page; pages are unique, so the scatter is deterministic.
    tokens = (seq_lens[:, None] - logical[None, :] * block_size).clamp(0, block_size)
    filled = torch.zeros(num_blocks, dtype=torch.int64, device=device)
    filled[pages.reshape(-1)] = tokens.reshape(-1)
    live = torch.arange(block_size, device=device)[None, :] < filled[:, None]  # [num_blocks, block_size]

    def cache(shape, slot_mask):
        values = torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
        stale = torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16).mul_(_STALE_SCALE)
        return torch.where(slot_mask, values, stale)

    key_cache = cache((num_blocks, num_kv_heads, head_size // x, block_size, x), live[:, None, None, :, None])
    value_cache = cache((num_blocks, num_kv_heads, head_size, block_size), live[:, None, None, :])
    return {"key_cache": key_cache, "value_cache": value_cache,
            "block_table": block_table.to(torch.int32), "seq_lens": seq_lens.to(torch.int32)}
