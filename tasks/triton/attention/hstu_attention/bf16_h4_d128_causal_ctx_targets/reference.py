"""High-precision HSTU attention forward (float64), used only for correctness."""
import torch

QUERY_CHUNK = 512


def _positions(length, contextual_seq_len, n_targets, device):
    """Attention positions of the tokens of one sequence and the position bound of its non-targets."""
    ids = torch.arange(length, device=device)
    max_ids = length
    if contextual_seq_len > 0:
        ids = (ids - contextual_seq_len + 1).clamp(min=0)
        max_ids = max_ids - contextual_seq_len + 1
    max_ids = max_ids - n_targets
    ids = ids.clamp(max=max_ids)
    return ids, max_ids


def reference(*, q, k, v, seq_offsets, num_targets, sort_by_length_indices, max_seq_len,
              contextual_seq_len, alpha):
    n_tokens, n_heads, head_dim = q.shape
    out = torch.zeros(q.shape, dtype=torch.float64, device=q.device)
    offsets = seq_offsets.tolist()
    targets = num_targets.tolist()
    for b in range(len(offsets) - 1):
        start, end = offsets[b], offsets[b + 1]
        length = end - start
        keys = k[start:end].double()                       # [L, H, d]
        values = v[start:end].double()
        ids, max_ids = _positions(length, contextual_seq_len, targets[b], q.device)
        raw = torch.arange(length, device=q.device)
        for q0 in range(0, length, QUERY_CHUNK):
            q1 = min(q0 + QUERY_CHUNK, length)
            queries = q[start + q0:start + q1].double()
            scores = torch.einsum("ihd,jhd->hij", queries, keys) * alpha
            weights = torch.nn.functional.silu(scores) / max_seq_len
            row_ids = ids[q0:q1, None]
            valid = (raw[q0:q1, None] == raw[None, :]) | (row_ids - ids[None, :] > 0)
            if contextual_seq_len > 0:
                valid = valid | ((row_ids == 0) & (ids[None, :] < max_ids))
            weights = weights * valid
            out[start + q0:start + q1] = torch.einsum("hij,jhd->ihd", weights, values)
    return out
