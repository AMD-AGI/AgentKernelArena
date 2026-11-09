"""Float64 absorbed-MLA decode over a token-granular latent cache, used only for correctness."""
import torch

KV_LORA_RANK = 512


def reference(*, q, kv_cache, block_tables, seq_lens, softmax_scale):
    batch, num_heads, _ = q.shape
    out = torch.empty(batch, num_heads, KV_LORA_RANK, dtype=torch.float64, device=q.device)
    for b in range(batch):
        length = int(seq_lens[b])
        latent = kv_cache[block_tables[b, :length].long()].double()
        probabilities = torch.softmax((q[b].double() @ latent.T) * softmax_scale, dim=-1)
        out[b] = probabilities @ latent[:, :KV_LORA_RANK]
    return out
