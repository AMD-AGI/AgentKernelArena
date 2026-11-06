"""High-precision reference of the lightning-attention decode step, used only for correctness."""
import torch

BATCHES_PER_STEP = 16  # batch entries per step, to bound float64 temporaries


def reference(*, q, k, v, kv_cache, slope_rate, slot_idx):
    batch, heads, dim = q.shape
    out = torch.empty(batch, heads * dim, dtype=torch.float64, device=q.device)
    final = kv_cache.double()
    decay = torch.exp(-slope_rate.double())[None, :, None, None]
    for start in range(0, batch, BATCHES_PER_STEP):
        stop = min(start + BATCHES_PER_STEP, batch)
        slots = slot_idx[start:stop].long()
        kv = torch.einsum("bhd,bhe->bhde", k[start:stop].double(), v[start:stop].double())
        kv = kv + decay * kv_cache[slots].double()
        out[start:stop] = torch.einsum("bhd,bhde->bhe", q[start:stop].double(), kv).reshape(-1, heads * dim)
        final[slots] = kv
    return out, final
