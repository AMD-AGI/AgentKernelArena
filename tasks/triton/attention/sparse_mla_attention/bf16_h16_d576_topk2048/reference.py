"""float64 sparse MLA attention over a latent KV cache, used only for correctness."""
import torch

_CHUNK = 32


def reference(*, q, kv, topk_indices, cu_seqlens_q, softmax_scale):
    n_tokens, num_heads, head_size = q.shape
    kv_lora_rank = head_size - 64
    rows = kv.reshape(-1, head_size).double()
    out = torch.zeros(n_tokens, num_heads, kv_lora_rank, dtype=torch.float32, device=q.device)
    for start in range(0, n_tokens, _CHUNK):
        stop = min(start + _CHUNK, n_tokens)
        index = topk_indices[start:stop].long()
        valid = (index >= 0) & (index < rows.shape[0])
        selected = rows[index.clamp(0, rows.shape[0] - 1)]                  # [c, topk, head_size]
        scores = torch.einsum("chd,ckd->chk", q[start:stop].double(), selected) * float(softmax_scale)
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        peak = scores.amax(dim=-1, keepdim=True)
        peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
        weights = torch.exp(scores - peak)
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1e-300)
        values = selected[..., :kv_lora_rank]
        out[start:stop] = (torch.einsum("chk,ckd->chd", weights, values) / denominator).float()
    return out
