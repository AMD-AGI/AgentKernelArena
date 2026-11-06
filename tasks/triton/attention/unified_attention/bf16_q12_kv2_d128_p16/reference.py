"""float64 reference of causal paged attention over a block-table KV cache."""
import torch

# Query rows scored together; bounds the float64 score matrix (heads x rows x keys).
_ROW_CHUNK = 256


def reference(*, q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table):
    num_tokens, num_q_heads, head_dim = q.shape
    block_size, num_kv_heads = k_cache.shape[1], k_cache.shape[2]
    group = num_q_heads // num_kv_heads
    scale = head_dim ** -0.5
    out = torch.zeros(q.shape, dtype=torch.float64, device=q.device)
    starts = cu_seqlens_q.tolist()
    lengths = seqused_k.tolist()
    for seq, kv_len in enumerate(lengths):
        q0, q1 = starts[seq], starts[seq + 1]
        q_len = q1 - q0
        blocks = block_table[seq, : (kv_len + block_size - 1) // block_size].long()
        keys = k_cache[blocks].reshape(-1, num_kv_heads, head_dim)[:kv_len].double()
        values = v_cache[blocks].reshape(-1, num_kv_heads, head_dim)[:kv_len].double()
        for r0 in range(0, q_len, _ROW_CHUNK):
            r1 = min(r0 + _ROW_CHUNK, q_len)
            # Query row i of this sequence sits at absolute position kv_len - q_len + i.
            pos = torch.arange(kv_len - q_len + r0, kv_len - q_len + r1, device=q.device)
            k_hi = kv_len - q_len + r1
            rows = q[q0 + r0 : q0 + r1].double().reshape(r1 - r0, num_kv_heads, group, head_dim)
            scores = torch.einsum("rhgd,khd->hgrk", rows, keys[:k_hi]) * scale
            visible = torch.arange(k_hi, device=q.device)[None, :] <= pos[:, None]
            scores = scores.masked_fill(~visible, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            result = torch.einsum("hgrk,khd->rhgd", probs, values[:k_hi])
            out[q0 + r0 : q0 + r1] = result.reshape(r1 - r0, num_q_heads, head_dim)
    return out
