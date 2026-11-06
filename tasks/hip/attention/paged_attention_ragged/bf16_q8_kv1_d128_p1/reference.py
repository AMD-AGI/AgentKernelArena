"""Float64 decode attention over a token pool addressed by ragged (CSR) page lists, used only for correctness."""
import torch

# Context tokens gathered per step (about 0.5 GB of float64 keys and values).
_CHUNK_TOKENS = 1 << 18


def reference(*, query, key_cache, value_cache, kv_indptr, kv_indices, scale, max_context_len):
    num_seqs, num_query_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[1]
    group = num_query_heads // num_kv_heads
    indptr = kv_indptr.long()
    lengths = indptr[1:] - indptr[:-1]
    indices = kv_indices.long()
    out = torch.empty((num_seqs, num_query_heads, head_size), dtype=torch.float64, device=query.device)
    step = max(1, _CHUNK_TOKENS // max_context_len)
    for start in range(0, num_seqs, step):
        stop = min(num_seqs, start + step)
        length = lengths[start:stop]
        tokens = int(length.max())
        offsets = torch.arange(tokens, device=query.device)
        valid = offsets[None, :] < length[:, None]
        # Positions past a sequence's length read its first token's slot and are masked below.
        positions = torch.where(valid, indptr[start:stop, None] + offsets[None, :], indptr[start:stop, None])
        slots = indices[positions]  # [seqs, tokens]
        k = key_cache[slots].double()  # [seqs, tokens, kv_heads, head_size]
        v = value_cache[slots].double()
        q = query[start:stop].double().view(stop - start, num_kv_heads, group, head_size)
        scores = torch.einsum("skgd,stkd->skgt", q, k) * scale
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[start:stop] = torch.einsum("skgt,stkd->skgd", probs, v).reshape(stop - start, num_query_heads,
                                                                             head_size)
    return out
