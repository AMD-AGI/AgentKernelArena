"""High-precision paged-KV batch prefill attention with logits soft-capping, for correctness only."""
import torch

# Upper bound on the float64 score block (heads x rows x keys) held at once, in elements.
_SCORE_ELEMENTS = 1 << 27


def reference(*, q, k_cache, v_cache, cu_seqlens_q, kv_indptr, kv_page_indices, kv_last_page_lens,
              softmax_scale, logits_soft_cap):
    n_tokens, n_heads, head_size = q.shape
    _, page_size, n_kv_heads, _ = k_cache.shape
    group = n_heads // n_kv_heads
    out = torch.zeros(n_tokens, n_heads, head_size, dtype=torch.float64, device=q.device)
    q_starts = cu_seqlens_q.tolist()
    page_starts = kv_indptr.tolist()
    last_lens = kv_last_page_lens.tolist()
    scale = float(softmax_scale)
    cap = float(logits_soft_cap)
    for b in range(len(last_lens)):
        first, stop = q_starts[b], q_starts[b + 1]
        qo_len = stop - first
        pages = kv_page_indices[page_starts[b]:page_starts[b + 1]].long()
        kv_len = (pages.numel() - 1) * page_size + last_lens[b]
        # Dense keys/values of this sequence, gathered from its pages in page-list order.
        keys = k_cache[pages].reshape(-1, n_kv_heads, head_size)[:kv_len].double()
        values = v_cache[pages].reshape(-1, n_kv_heads, head_size)[:kv_len].double()
        keys = keys.repeat_interleave(group, dim=1).permute(1, 0, 2)      # [H, S, D]
        values = values.repeat_interleave(group, dim=1).permute(1, 0, 2)  # [H, S, D]
        queries = q[first:stop].double().permute(1, 0, 2)                # [H, Q, D]
        offset = kv_len - qo_len  # bottom-right aligned causal mask
        rows = max(1, _SCORE_ELEMENTS // (n_heads * kv_len))
        for r0 in range(0, qo_len, rows):
            r1 = min(qo_len, r0 + rows)
            visible = max(0, min(kv_len, offset + r1))  # the chunk's last row sees keys [0, offset + r1)
            scores = torch.matmul(queries[:, r0:r1], keys[:, :visible].transpose(1, 2)) * scale
            if cap > 0:
                scores = cap * torch.tanh(scores / cap)
            row = torch.arange(r0, r1, device=q.device)[:, None]
            col = torch.arange(visible, device=q.device)[None, :]
            scores = scores.masked_fill(col > offset + row, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[first + r0:first + r1] = torch.matmul(probs, values[:, :visible]).permute(1, 0, 2)
    return out
