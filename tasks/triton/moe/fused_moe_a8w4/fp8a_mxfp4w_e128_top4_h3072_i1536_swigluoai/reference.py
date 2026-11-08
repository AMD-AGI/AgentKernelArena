"""float64 reference for the static-FP8 x MXFP4 GPT-OSS MoE block with expert biases, used only for correctness.

For token t and routed slot j (expert e = topk_ids[t, j]):
  x_q    = float8_e4m3fn(x[t] / w1_input_scale)                  (round to nearest even; float32 division)
  g      = (x_q @ dequant(w1[e])^T) * w1_input_scale + w1_bias[e] (2I values: gate = g[:I], up = g[I:])
  gate   = min(gate, 7),  up = clamp(up, -7, 7)
  act    = gate * sigmoid(1.702 * gate) * (up + 1)
  a_q    = float8_e4m3fn(float32(act) / w2_input_scale)
  y      = (a_q @ dequant(w2[e])^T) * w2_input_scale + w2_bias[e]
  out[t] = sum_j topk_weights[t, j] * y
MXFP4 dequantization: e2m1 code (low nibble = even element) times 2^(scale byte - 127) per 32 elements.
Both FP8 quantizations model the operator's stored FP8 activations exactly; everything else is float64 with no
intermediate rounding (the products of FP8 codes and e2m1 values are exact in float64).
"""
import torch

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_GROUP = 32
_ALPHA = 1.702
_LIMIT = 7.0
_ROW_CHUNK = 4096  # rows of one expert evaluated at a time, to bound float64 temporaries


def _dequant(codes, scales):
    """[N, K/2] packed e2m1 and [N, K/32] e8m0 -> float64 [N, K]."""
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=codes.device)
    low = lut[(codes & 0xF).long()]
    high = lut[(codes >> 4).long()]
    values = torch.stack((low, high), dim=-1).reshape(codes.shape[0], -1)
    scale = torch.exp2(scales.to(torch.float64) - 127.0).repeat_interleave(_GROUP, dim=-1)
    return values * scale


def _static_fp8(x32, scale32):
    """float8_e4m3fn(x / scale) as float64 code values (the division is float32, as in the kernels)."""
    return (x32 / scale32).to(torch.float8_e4m3fn).to(torch.float64)


def reference(*, hidden_states, w1, w1_scale, w1_bias, w1_input_scale, w2, w2_scale, w2_bias, w2_input_scale,
              topk_weights, topk_ids):
    n_tokens, hidden = hidden_states.shape
    inter = w2.shape[-1] * 2
    s1 = w1_input_scale.reshape(())
    s2 = w2_input_scale.reshape(())
    x = hidden_states.to(torch.float32)
    out = torch.zeros(n_tokens, hidden, dtype=torch.float64, device=hidden_states.device)
    flat_ids = topk_ids.reshape(-1).long()
    flat_weights = topk_weights.reshape(-1).to(torch.float64)
    token_of = torch.arange(n_tokens, device=hidden_states.device).repeat_interleave(topk_ids.shape[1])
    order = torch.argsort(flat_ids, stable=True)
    experts, counts = torch.unique_consecutive(flat_ids[order], return_counts=True)
    start = 0
    for expert, count in zip(experts.tolist(), counts.tolist()):
        slots_of_expert = order[start:start + count]
        start += count
        w1_dq = _dequant(w1[expert], w1_scale[expert])
        w2_dq = _dequant(w2[expert], w2_scale[expert])
        for lo in range(0, count, _ROW_CHUNK):
            slots = slots_of_expert[lo:lo + _ROW_CHUNK]
            tokens = token_of[slots]
            g = (_static_fp8(x[tokens], s1) @ w1_dq.T) * s1.to(torch.float64) + w1_bias[expert].to(torch.float64)
            gate = g[:, :inter].clamp(max=_LIMIT)
            up = g[:, inter:].clamp(min=-_LIMIT, max=_LIMIT)
            act = gate * torch.sigmoid(_ALPHA * gate) * (up + 1.0)
            y = (_static_fp8(act.to(torch.float32), s2) @ w2_dq.T) * s2.to(torch.float64) \
                + w2_bias[expert].to(torch.float64)
            out.index_add_(0, tokens, y * flat_weights[slots, None])
    return out
