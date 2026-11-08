"""Float64 reference of the fused split-K reduce, query/key-value RMSNorm, RoPE tail and sliding-window write."""
import torch


def _rms_norm(x, weight, eps):
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return y if weight is None else y * weight


def _rope_tail_interleaved(x, cos, sin, rope_dim):
    """Rotate the last rope_dim values of x [..., D] pairwise (x[2i], x[2i+1]); cos, sin [..., rope_dim // 2]."""
    nope, pe = x[..., :-rope_dim], x[..., -rope_dim:]
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    rotated = torch.stack((-pe[..., 1::2], pe[..., ::2]), dim=-1).flatten(-2)
    return torch.cat((nope, pe * cos + rotated * sin), dim=-1)


def reference(*, q, kv, kv_norm_weight, positions, cos, sin, write_indices, batch_id_per_token, state_slot_mapping,
              swa_kv, q_eps, kv_eps):
    _, n_tokens, width = q.shape
    head_dim = kv.shape[1]
    heads = width // head_dim
    rope_dim = 2 * cos.shape[-1]
    window = swa_kv.shape[1]
    cos_t = cos.double()[positions]
    sin_t = sin.double()[positions]

    q_norm = _rms_norm(q.double().sum(dim=0).reshape(n_tokens, heads, head_dim), None, float(q_eps))
    q_out = _rope_tail_interleaved(q_norm, cos_t[:, None, :], sin_t[:, None, :], rope_dim)

    kv_out = kv.double()
    swa = swa_kv.double()
    rows = write_indices[write_indices >= 0].long()
    done = _rope_tail_interleaved(_rms_norm(kv.double()[rows], kv_norm_weight.double(), float(kv_eps)),
                                  cos_t[rows], sin_t[rows], rope_dim)
    kv_out[rows] = done
    slots = state_slot_mapping[batch_id_per_token[rows].long()].long()
    swa[slots, positions[rows] % window] = done
    return {"q_out": q_out, "kv": kv_out, "swa_kv": swa}
