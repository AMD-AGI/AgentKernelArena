"""float64 reference for the FP8 MQA logits of the DeepSeek-V3.2 indexer."""
import torch


def reference(*, q, kv, kv_scales, weights, cu_seqlen_ks, cu_seqlen_ke):
    seq_len, num_heads, head_size = q.shape
    seq_len_kv = kv.shape[0]
    kv64 = kv.double()
    scales = kv_scales.double()
    cols = torch.arange(seq_len_kv, device=q.device)[None, :]
    logits = torch.empty(seq_len, seq_len_kv, dtype=torch.float64, device=q.device)
    rows_per_chunk = max(1, (1 << 24) // (num_heads * seq_len_kv))
    for start in range(0, seq_len, rows_per_chunk):
        stop = min(seq_len, start + rows_per_chunk)
        rows = stop - start
        q_chunk = q[start:stop].double().reshape(rows * num_heads, head_size)
        score = (q_chunk @ kv64.T).reshape(rows, num_heads, seq_len_kv)
        score = torch.relu(score * scales[None, None, :]) * weights[start:stop].double()[:, :, None]
        value = score.sum(dim=1)
        lo = cu_seqlen_ks[start:stop].long().clamp(min=0)[:, None]
        hi = cu_seqlen_ke[start:stop].long().clamp(max=seq_len_kv)[:, None]
        inside = (cols >= lo) & (cols < hi)
        logits[start:stop] = torch.where(inside, value, torch.full_like(value, float("-inf")))
    return logits
