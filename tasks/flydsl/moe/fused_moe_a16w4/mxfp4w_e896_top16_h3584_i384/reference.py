"""float64 reference for the bf16 x MXFP4 Kimi-K3 routed MoE block, used only for correctness.

For token t and routed slot j (expert e = topk_ids[t, j]):
  g      = x[t] @ dequant(w1[e])^T                      (2I values: gate = g[:I], up = g[I:])
  act    = 4 * tanh(gate / 4) * sigmoid(gate) * 25 * tanh(up / 25)        (SiTUv2, beta 4, linear beta 25)
  out[t] = sum_j topk_weights[t, j] * (act @ dequant(w2[e])^T)
MXFP4 dequantization: e2m1 code (low nibble = even element) times 2^(scale byte - 127) per 32 elements.
SiTUv2 is the model's definition (vLLM SituAndMul, AITER situv2), without the +-7 input clamp of AITER's a16w4
stage-1 kernel; the task's inputs keep |gate| and |up| below 7, where the two agree.
Everything is evaluated in float64 (no intermediate rounding).
"""
import torch

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_GROUP = 32
_SITU_BETA = 4.0
_SITU_LINEAR_BETA = 25.0


def _dequant(codes, scales):
    """[N, K/2] packed e2m1 and [N, K/32] e8m0 -> float64 [N, K]."""
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=codes.device)
    low = lut[(codes & 0xF).long()]
    high = lut[(codes >> 4).long()]
    values = torch.stack((low, high), dim=-1).reshape(codes.shape[0], -1)
    scale = torch.exp2(scales.to(torch.float64) - 127.0).repeat_interleave(_GROUP, dim=-1)
    return values * scale


def reference(*, hidden_states, w1, w1_scale, w2, w2_scale, topk_weights, topk_ids):
    n_tokens, hidden = hidden_states.shape
    inter = w2.shape[-1] * 2
    x = hidden_states.to(torch.float64)
    out = torch.zeros(n_tokens, hidden, dtype=torch.float64, device=hidden_states.device)
    flat_ids = topk_ids.reshape(-1).long()
    flat_weights = topk_weights.reshape(-1).to(torch.float64)
    token_of = torch.arange(n_tokens, device=hidden_states.device).repeat_interleave(topk_ids.shape[1])
    order = torch.argsort(flat_ids, stable=True)
    experts, counts = torch.unique_consecutive(flat_ids[order], return_counts=True)
    start = 0
    for expert, count in zip(experts.tolist(), counts.tolist()):
        slots = order[start:start + count]
        start += count
        tokens = token_of[slots]
        g = x[tokens] @ _dequant(w1[expert], w1_scale[expert]).T
        gate, up = g[:, :inter], g[:, inter:]
        act = (_SITU_BETA * torch.tanh(gate / _SITU_BETA) * torch.sigmoid(gate)
               * _SITU_LINEAR_BETA * torch.tanh(up / _SITU_LINEAR_BETA))
        y = act @ _dequant(w2[expert], w2_scale[expert]).T
        out.index_add_(0, tokens, y * flat_weights[slots, None])
    return out
