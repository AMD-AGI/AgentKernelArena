"""Float64 decode attention over the dequantized FP8 paged KV cache, used only for correctness.

The worklist inputs (partition_indptr, work_*, reduce_*) only schedule the computation and are not read here.
"""
import torch

# Context tokens gathered per step (about 0.5 GB of float64 keys and values).
_CHUNK_TOKENS = 1 << 18


def reference(*, query, key_cache, value_cache, key_scale, value_scale, context_lengths, kv_indptr,
              kv_page_indices, partition_indptr, work_indptr, work_info, reduce_indptr, reduce_final_map,
              reduce_partial_map, softmax_scale):
    num_blocks, num_kv_heads, _, block_size, _ = key_cache.shape
    num_seqs, num_query_heads, head_size = query.shape
    group = num_query_heads // num_kv_heads
    device = query.device
    pages_used = kv_indptr[1:].long() - kv_indptr[:-1].long()
    max_pages = max(1, int(pages_used.max()))
    tokens = max_pages * block_size
    out = torch.empty((num_seqs, num_query_heads, head_size), dtype=torch.float64, device=device)
    step = max(1, _CHUNK_TOKENS // tokens)
    for start in range(0, num_seqs, step):
        stop = min(num_seqs, start + step)
        count = stop - start
        # Page table of the chunk; entries past a sequence's last page are masked below.
        slot = kv_indptr[start:stop].long()[:, None] + torch.arange(max_pages, device=device)[None, :]
        slot = torch.minimum(slot, (kv_indptr[start + 1:stop + 1].long() - 1).clamp(min=0)[:, None])
        table = kv_page_indices.long()[slot].reshape(-1)
        # key [page, head, d // 16, slot, d % 16] -> [page, slot, head, d]
        k = key_cache[table].permute(0, 3, 1, 2, 4).reshape(count * max_pages, block_size, num_kv_heads, head_size)
        # value [page, head, slot // 16, d, slot % 16] -> [page, slot, head, d]
        v = value_cache[table].permute(0, 2, 4, 1, 3).reshape(count * max_pages, block_size, num_kv_heads, head_size)
        ks = key_scale[table].permute(0, 2, 1, 3)
        vs = value_scale[table].permute(0, 2, 1, 3)
        k = (k.double() * ks.double()).reshape(count, tokens, num_kv_heads, head_size)
        v = (v.double() * vs.double()).reshape(count, tokens, num_kv_heads, head_size)
        valid = torch.arange(tokens, device=device)[None, :] < context_lengths[start:stop].long()[:, None]
        q = query[start:stop].double().view(count, num_kv_heads, group, head_size)
        scores = torch.einsum("skgd,stkd->skgt", q, k) * float(softmax_scale)
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[start:stop] = torch.einsum("skgt,stkd->skgd", probs, v).reshape(count, num_query_heads, head_size)
    return out
