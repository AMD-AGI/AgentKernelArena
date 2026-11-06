"""High-precision decode attention reference (float64), used only for correctness."""
import torch


def reference(*, q, k_buffer, v_buffer, kv_indptr, kv_indices, num_kv_splits, sm_scale):
    batch, n_heads, head_dim = q.shape
    n_kv_heads = k_buffer.shape[1]
    group = n_heads // n_kv_heads
    bounds = kv_indptr.tolist()
    out = torch.empty(batch, n_heads, head_dim, dtype=torch.float64, device=q.device)
    for b in range(batch):
        rows = kv_indices[bounds[b]:bounds[b + 1]]
        k = k_buffer[rows].double()
        v = v_buffer[rows].double()
        qb = q[b].double().reshape(n_kv_heads, group, head_dim)
        scores = torch.einsum("kgd,lkd->kgl", qb, k) * float(sm_scale)
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("kgl,lkd->kgd", probs, v).reshape(n_heads, head_dim)
    return out
