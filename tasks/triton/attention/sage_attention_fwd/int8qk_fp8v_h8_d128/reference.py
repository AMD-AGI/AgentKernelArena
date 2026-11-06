"""High-precision causal attention, used only for correctness of the quantized SageAttention."""
import torch

# Upper bound on the float64 score block (batch x heads x rows x keys) held at once, in elements.
_SCORE_ELEMENTS = 1 << 28


def reference(*, q, k, v, softmax_scale):
    batch, seqlen, n_heads, _ = q.shape
    queries = q.double().permute(0, 2, 1, 3)  # [B, H, S, D]
    keys = k.double().permute(0, 2, 1, 3)
    values = v.double().permute(0, 2, 1, 3)
    out = torch.empty_like(queries)
    rows = max(1, _SCORE_ELEMENTS // (batch * n_heads * seqlen))
    for r0 in range(0, seqlen, rows):
        r1 = min(seqlen, r0 + rows)
        scores = torch.matmul(queries[:, :, r0:r1], keys[:, :, :r1].transpose(-1, -2)) * float(softmax_scale)
        row = torch.arange(r0, r1, device=q.device)[:, None]
        col = torch.arange(r1, device=q.device)[None, :]
        scores = scores.masked_fill(col > row, float("-inf"))
        out[:, :, r0:r1] = torch.matmul(torch.softmax(scores, dim=-1), values[:, :, :r1])
    return out.permute(0, 2, 1, 3).contiguous()
