"""Float64 paged decode attention over vLLM's ROCm KV-cache layout, used only for correctness."""
import torch

# Context tokens gathered per step (about 1 GB of float64 keys and values).
_CHUNK_TOKENS = 1 << 18


def reference(*, query, key_cache, value_cache, block_table, seq_lens, scale):
    num_blocks, num_kv_heads, head_outer, block_size, x = key_cache.shape
    head_size = head_outer * x
    num_seqs, num_query_heads, _ = query.shape
    group = num_query_heads // num_kv_heads
    # Token-major pages: [num_blocks, block_size, num_kv_heads, head_size].
    keys = key_cache.permute(0, 3, 1, 2, 4).reshape(num_blocks, block_size, num_kv_heads, head_size)
    values = value_cache.permute(0, 3, 1, 2)
    table = block_table.long()
    tokens = table.shape[1] * block_size
    valid = torch.arange(tokens, device=query.device)[None, :] < seq_lens.long()[:, None]
    out = torch.empty((num_seqs, num_query_heads, head_size), dtype=torch.float64, device=query.device)
    step = max(1, _CHUNK_TOKENS // tokens)
    for start in range(0, num_seqs, step):
        stop = min(num_seqs, start + step)
        mask = valid[start:stop]
        k = keys[table[start:stop]].reshape(stop - start, tokens, num_kv_heads, head_size).double()
        v = values[table[start:stop]].reshape(stop - start, tokens, num_kv_heads, head_size).double()
        # Slots past seq_len (NaN in the cache) contribute nothing.
        k = torch.where(mask[:, :, None, None], k, torch.zeros_like(k))
        v = torch.where(mask[:, :, None, None], v, torch.zeros_like(v))
        q = query[start:stop].double().view(stop - start, num_kv_heads, group, head_size)
        scores = torch.einsum("skgd,stkd->skgt", q, k) * scale
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[start:stop] = torch.einsum("skgt,stkd->skgd", probs, v).reshape(stop - start, num_query_heads,
                                                                             head_size)
    return out
