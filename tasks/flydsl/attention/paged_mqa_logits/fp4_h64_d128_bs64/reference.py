"""float64 reference for the paged MXFP4 MQA logits of the DeepSeek-V4 indexer (decode)."""
import torch

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_KEY_CHUNK = 32768  # keys scored per step, bounding the temporaries


def _dequantize(packed, e8m0):
    """float64 values of packed e2m1 codes [..., D / 2] with E8M0 scales per 32 elements [..., D / 32]."""
    nibble = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(*packed.shape[:-1], -1)
    levels = torch.tensor(_E2M1, dtype=torch.float64, device=packed.device)
    value = levels[(nibble & 7).long()]
    value = torch.where((nibble & 8) != 0, -value, value)
    scale = torch.exp2(e8m0.double() - 127.0)
    value = value.reshape(*nibble.shape[:-1], -1, 32) * scale[..., None]
    return value.reshape(*nibble.shape)


def reference(*, q_fp4, q_scale, kv_cache, kv_scale, weights, context_lens, block_tables, cta_info, weight_scale):
    batch, next_n, heads, _ = q_fp4.shape
    block_size = kv_cache.shape[1]
    max_seq_len = (block_tables.shape[1] - 4) * block_size  # the table carries 4 guard pages
    query = _dequantize(q_fp4, q_scale)  # [batch, next_n, heads, head_size]
    logits = torch.full((batch * next_n, max_seq_len), float("-inf"), dtype=torch.float64, device=q_fp4.device)
    for b in range(batch):
        ctx = int(context_lens[b])
        blocks = block_tables[b, : (ctx + block_size - 1) // block_size].long()
        key = _dequantize(kv_cache[blocks], kv_scale[blocks]).reshape(-1, query.shape[-1])[:ctx]
        q_b = query[b].reshape(next_n * heads, -1)
        head_weight = weights[b * next_n:(b + 1) * next_n].double()[:, :, None]
        for start in range(0, ctx, _KEY_CHUNK):
            stop = min(ctx, start + _KEY_CHUNK)
            score = (q_b @ key[start:stop].T).reshape(next_n, heads, stop - start)
            logits[b * next_n:(b + 1) * next_n, start:stop] = (torch.relu(score) * head_weight).sum(dim=1) * weight_scale
        # Query n of the step sees keys 0 .. ctx - next_n + n: the later draft tokens are causally masked.
        cols = torch.arange(ctx, device=q_fp4.device)[None, :]
        limit = (ctx - next_n + torch.arange(next_n, device=q_fp4.device))[:, None]
        masked = logits[b * next_n:(b + 1) * next_n, :ctx]
        logits[b * next_n:(b + 1) * next_n, :ctx] = torch.where(cols <= limit, masked,
                                                                 torch.full_like(masked, float("-inf")))
    return logits
