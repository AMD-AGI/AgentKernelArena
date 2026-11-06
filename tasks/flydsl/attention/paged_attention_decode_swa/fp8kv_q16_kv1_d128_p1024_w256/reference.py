"""Float64 sliding-window paged decode attention over the dequantized FP8 cache, used only for correctness."""
import torch

# Sequences gathered per step (keys and values of their windows in float64).
_CHUNK_SEQS = 512


def reference(*, query, key_cache, value_cache, block_tables, context_lengths, key_scale, value_scale, scale,
              sliding_window):
    num_blocks, num_kv_heads, head_outer, block_size, x = key_cache.shape
    head_size = head_outer * x
    num_seqs, num_query_heads, _ = query.shape
    group = num_query_heads // num_kv_heads
    window = sliding_window + 1  # keys at distance 0..sliding_window from the decoded token
    key_bytes = key_cache.view(torch.uint8)
    value_bytes = value_cache.view(torch.uint8)
    k_scale = key_scale.double().reshape(())
    v_scale = value_scale.double().reshape(())
    lengths = context_lengths.long()
    out = torch.empty((num_seqs, num_query_heads, head_size), dtype=torch.float64, device=query.device)
    offsets = torch.arange(window, device=query.device)
    for start in range(0, num_seqs, _CHUNK_SEQS):
        stop = min(num_seqs, start + _CHUNK_SEQS)
        length = lengths[start:stop]
        first = (length - 1 - sliding_window).clamp(min=0)
        positions = first[:, None] + offsets[None, :]  # [seqs, window]
        valid = positions < length[:, None]
        positions = torch.where(valid, positions, first[:, None])
        pages = block_tables[start:stop].long().gather(1, positions // block_size)
        slots = positions % block_size
        # key_cache[page, kv_head, d // x, slot, d % x] -> [seqs, window, kv_head, head_size]
        k = key_bytes[pages, :, :, slots, :].view(torch.float8_e4m3fn).double()
        k = k.reshape(stop - start, window, num_kv_heads, head_size) * k_scale
        # value_cache[page, kv_head, d, slot] -> [seqs, window, kv_head, head_size]
        v = value_bytes[pages, :, :, slots].view(torch.float8_e4m3fn).double() * v_scale
        q = query[start:stop].double().view(stop - start, num_kv_heads, group, head_size)
        scores = torch.einsum("skgd,swkd->skgw", q, k) * scale
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[start:stop] = torch.einsum("skgw,swkd->skgd", probs, v).reshape(stop - start, num_query_heads,
                                                                             head_size)
    return out
