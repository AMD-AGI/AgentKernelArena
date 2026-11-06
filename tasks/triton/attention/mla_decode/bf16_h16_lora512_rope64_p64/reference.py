"""Float64 absorbed-MLA decode over a paged latent cache, used only for correctness."""
import torch

KV_LORA_RANK = 512


def reference(*, q, kv_cache, block_tables, seq_lens, cu_seqlens_q, softmax_scale):
    batch, num_heads, head_size = q.shape
    page_size = kv_cache.shape[1]
    out = torch.empty(batch, num_heads, KV_LORA_RANK, dtype=torch.float64, device=q.device)
    for b in range(batch):
        length = int(seq_lens[b])
        pages = block_tables[b, : -(-length // page_size)].long()
        latent = kv_cache[pages].reshape(-1, head_size)[:length].double()
        query = q[int(cu_seqlens_q[b])].double()
        probabilities = torch.softmax((query @ latent.T) * softmax_scale, dim=-1)
        out[b] = probabilities @ latent[:, :KV_LORA_RANK]
    return out
