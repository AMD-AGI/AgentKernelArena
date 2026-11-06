"""Float64 reference for the MXFP4 (a4w4) routed MoE layer, used only for correctness.

For token t and routed slot j (expert e = topk_ids[t, j]):
  xq     = mxfp4(hidden_states[t])                                 (per 1x32 block along H)
  g      = xq @ dequant(w1[e])^T                                   (2I values: gate = g[:I], up = g[I:])
  act    = bfloat16(silu(gate) * up)
  out[t] = sum_j topk_weights[t, j] * (mxfp4(act) @ dequant(w2[e])^T)
mxfp4(v) is the operator's activation quantizer, AITER's production MXFP4 quantization (RoundUp scale mode,
kDefaultMxScaleRoundMode): per block of 32 values, the e8m0 scale is the smallest power of two 2^s with
max(amax, 1e-10) <= 6 * 2^s (s clamped to [-127, 127]), and v / 2^s rounds to the nearest e2m1 value with ties
to even. Products and sums run in float64; the quantizations and the bfloat16 rounding of the SiLU-and-mul
output are the operator's semantics and are reproduced exactly.
"""
import torch

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_GROUP = 32
_ROWS = 2048


def _dequant(codes, scales):
    """[N, K/2] packed e2m1 (low nibble first) and [N, K/32] e8m0 -> float64 [N, K]."""
    magnitude = torch.tensor(_E2M1, dtype=torch.float64, device=codes.device)
    lut = torch.cat((magnitude, -magnitude))
    low = lut[(codes & 0xF).long()]
    high = lut[(codes >> 4).long()]
    values = torch.stack((low, high), dim=-1).reshape(codes.shape[0], -1)
    scale = torch.exp2(scales.to(torch.float64) - 127.0).repeat_interleave(_GROUP, dim=-1)
    return values * scale


def _mxfp4_round_trip(values):
    """The operator's MXFP4 quantization of [..., K] (bfloat16 values), returned dequantized in float64."""
    blocks = values.to(torch.float64).unflatten(-1, (-1, _GROUP))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    # amax = m * 2^e with m in [0.5, 1): amax <= 6 * 2^s first holds at s = e - 3 when m <= 0.75, else e - 2.
    mantissa, exponent = torch.frexp(amax)
    exponent = (exponent - 3 + (mantissa > 0.75).to(exponent.dtype)).clamp(-127, 127).to(torch.float64)
    scaled = blocks * torch.exp2(-exponent)
    grid = torch.tensor(_E2M1, dtype=torch.float64, device=values.device)
    midpoints = (grid[1:] + grid[:-1]) / 2
    magnitude = scaled.abs()
    code = torch.bucketize(magnitude, midpoints)
    # A value on a midpoint rounds to the even code (round to nearest, ties to even).
    tie = (code < 7) & (magnitude == midpoints[code.clamp(max=6)]) & (code % 2 == 1)
    quantized = torch.sign(scaled) * grid[code + tie.to(code.dtype)]
    return (quantized * torch.exp2(exponent)).flatten(-2)


def reference(*, hidden_states, w1, w1_scale, w2, w2_scale, topk_weights, topk_ids):
    n_tokens, hidden = hidden_states.shape
    inter = w2.shape[-1] * 2
    # Row chunks bound the float64 temporaries of large batches; the result does not depend on them.
    x = torch.cat([_mxfp4_round_trip(rows) for rows in hidden_states.split(_ROWS)])
    out = torch.zeros(n_tokens, hidden, dtype=torch.float64, device=hidden_states.device)
    weights = topk_weights.to(torch.float64)
    for expert in torch.unique(topk_ids).tolist():
        tokens, slots = (topk_ids == expert).nonzero(as_tuple=True)
        w1_e = _dequant(w1[expert], w1_scale[expert])
        w2_e = _dequant(w2[expert], w2_scale[expert])
        for t, j in zip(tokens.split(_ROWS), slots.split(_ROWS)):
            g = x[t] @ w1_e.T
            gate, up = g[:, :inter], g[:, inter:]
            act = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
            y = _mxfp4_round_trip(act) @ w2_e.T
            out.index_add_(0, t, y * weights[t, j].unsqueeze(1))
    return out
