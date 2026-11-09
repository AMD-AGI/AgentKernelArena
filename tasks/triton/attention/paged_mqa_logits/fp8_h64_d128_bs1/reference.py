"""float64 reference for the FP8 paged MQA logits of the DeepSeek-V3.2 indexer (decode)."""
import torch

_KEY_CHUNK = 32768  # keys scored per step, bounding the temporaries


def reference(*, q, kv_cache, weights, context_lens, block_tables):
    batch, next_n, heads, head_size = q.shape
    num_blocks, block_size = kv_cache.shape[0], kv_cache.shape[1]
    max_model_len = block_tables.shape[1] * block_size
    rows = kv_cache.reshape(num_blocks, block_size * (head_size + 4))
    codes = rows[:, : block_size * head_size]
    scales = rows[:, block_size * head_size:]
    logits = torch.full((batch * next_n, max_model_len), float("-inf"), dtype=torch.float64, device=q.device)
    for b in range(batch):
        ctx = int(context_lens[b])
        blocks = block_tables[b, : (ctx + block_size - 1) // block_size].long()
        key = codes[blocks].contiguous().view(torch.float8_e4m3fn).reshape(-1, head_size)[:ctx].double()
        key_scale = scales[blocks].contiguous().view(torch.float32).reshape(-1)[:ctx].double()
        query = q[b].double().reshape(next_n * heads, head_size)
        head_weight = weights[b * next_n:(b + 1) * next_n].double()[:, :, None]
        for start in range(0, ctx, _KEY_CHUNK):
            stop = min(ctx, start + _KEY_CHUNK)
            score = (query @ key[start:stop].T).reshape(next_n, heads, stop - start)
            score = torch.relu(score * key_scale[None, None, start:stop]) * head_weight
            logits[b * next_n:(b + 1) * next_n, start:stop] = score.sum(dim=1)
        # Query n of the step sees keys 0 .. ctx - next_n + n: the later draft tokens are causally masked.
        cols = torch.arange(ctx, device=q.device)[None, :]
        limit = (ctx - next_n + torch.arange(next_n, device=q.device))[:, None]
        masked = logits[b * next_n:(b + 1) * next_n, :ctx]
        logits[b * next_n:(b + 1) * next_n, :ctx] = torch.where(cols <= limit, masked,
                                                                 torch.full_like(masked, float("-inf")))
    return logits
