"""High-precision paged prefix-prefill attention with ALiBi, used only for correctness."""
import torch

# Upper bound on the float64 score block (heads x rows x keys) held at once, in elements.
_SCORE_ELEMENTS = 1 << 28


def reference(*, q, k, v, k_cache, v_cache, b_loc, b_start_loc, b_seq_len, alibi_slopes, sm_scale):
    n_tokens, n_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]
    group = n_heads // n_kv_heads
    slopes = alibi_slopes.double()[:, None, None]
    out = torch.zeros(n_tokens, n_heads, head_dim, dtype=torch.float64, device=q.device)
    starts = b_start_loc.tolist()
    lengths = b_seq_len.tolist()
    for b in range(len(lengths)):
        first, stop = starts[b], starts[b + 1]
        q_len = stop - first
        ctx_len = lengths[b] - q_len
        pages = b_loc[b].long()
        # Dense context keys/values rebuilt from the paged caches through the block table.
        kc = k_cache[pages].permute(0, 3, 1, 2, 4).reshape(-1, n_kv_heads, head_dim)[:ctx_len]
        vc = v_cache[pages].permute(0, 3, 1, 2).reshape(-1, n_kv_heads, head_dim)[:ctx_len]
        keys = torch.cat([kc, k[first:stop]]).double().repeat_interleave(group, dim=1)
        values = torch.cat([vc, v[first:stop]]).double().repeat_interleave(group, dim=1)
        keys = keys.permute(1, 0, 2)      # [H, S, D]
        values = values.permute(1, 0, 2)
        queries = q[first:stop].double().permute(1, 0, 2)  # [H, Q, D]
        rows = max(1, _SCORE_ELEMENTS // (n_heads * (ctx_len + q_len)))
        for r0 in range(0, q_len, rows):
            r1 = min(q_len, r0 + rows)
            visible = ctx_len + r1
            scores = torch.matmul(queries[:, r0:r1], keys[:, :visible].transpose(1, 2)) * float(sm_scale)
            position = ctx_len + torch.arange(r0, r1, device=q.device)[:, None]
            key = torch.arange(visible, device=q.device)[None, :]
            scores = scores + slopes * (key - position).double()
            scores = scores.masked_fill(key > position, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[first + r0:first + r1] = torch.matmul(probs, values[:, :visible]).permute(1, 0, 2)
    return out
