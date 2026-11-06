"""Float64 reference of the fused MXFP4 MoE (MXFP4 activations and weights, SiLU, weights in stage 2).

For each token t and slot k with expert e = topk_ids[t, k]:
    x      = mxfp4(hidden_states[t])                        per 1x32 block of the hidden axis
    gate   = x . dequant(w1[e, :I]),  up = x . dequant(w1[e, I:])
    h      = bfloat16(silu(gate) * up),  h_q = mxfp4(h)      per 1x32 block of the intermediate axis
    out[t] = sum_k topk_weights[t, k] * (h_q . dequant(w2[e]))
mxfp4() is AITER's dynamic MXFP4 quantization with its default (RoundUp) scale: the block scale is the
smallest power of two 2^s with amax <= 6 * 2^s (ceil_pow2(amax / 6); s = -127 for an all-zero block), and
every value divided by the scale rounds to the nearest e2m1 value (0, 0.5, 1, 1.5, 2, 3, 4, 6), ties to the
even code. Every product and sum is taken in float64; out is returned in float64.
"""
import torch

_GROUP = 32
_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Decision boundaries between neighbouring e2m1 magnitudes; a value on a boundary goes to the even code.
_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (False, True, False, True, False, True, False)
_ROWS = 2048


def _dequantize(packed, scales):
    """packed [..., k // 2] uint8 e2m1 pairs (even index in the low nibble), scales [..., k // 32] e8m0."""
    table = torch.tensor(_GRID + tuple(-v for v in _GRID), dtype=torch.float64, device=packed.device)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
    values = table[codes].unflatten(-1, (-1, _GROUP))
    return (values * torch.exp2(scales.double() - 127.0).unsqueeze(-1)).flatten(-2)


def _mxfp4(values):
    """Quantize-dequantize float64 values in 1x32 blocks of the last axis."""
    groups = values.unflatten(-1, (-1, _GROUP))
    amax = groups.abs().amax(dim=-1, keepdim=True)
    mantissa, exponent = torch.frexp(amax)  # amax = mantissa * 2^exponent, mantissa in [0.5, 1)
    # amax <= 6 * 2^(exponent - 3) iff mantissa <= 0.75; otherwise the next power of two is the scale.
    scale_exponent = exponent - 3 + (mantissa > 0.75).to(exponent.dtype)
    scale_exponent = torch.where(amax > 0, scale_exponent, torch.full_like(scale_exponent, -127))
    scale = torch.exp2(scale_exponent.clamp(-127, 127).double())
    magnitude = groups.abs() / scale
    code = torch.zeros(magnitude.shape, dtype=torch.long, device=values.device)
    for bound, tie_up in zip(_BOUNDS, _TIE_UP):
        code += ((magnitude >= bound) if tie_up else (magnitude > bound)).long()
    grid = torch.tensor(_GRID, dtype=torch.float64, device=values.device)
    return (torch.sign(groups) * grid[code] * scale).flatten(-2)


def reference(*, hidden_states, w1, w1_scale, w2, w2_scale, topk_weights, topk_ids):
    num_tokens, hidden = hidden_states.shape
    inter = w2.shape[-1] * 2
    top_k = topk_ids.shape[1]
    device = hidden_states.device
    x = torch.cat([_mxfp4(rows.double()) for rows in hidden_states.split(_ROWS)])
    out = torch.zeros(num_tokens, hidden, dtype=torch.float64, device=device)
    expert_of = topk_ids.reshape(-1).long()
    weight_of = topk_weights.reshape(-1).double()
    token_of = torch.arange(expert_of.numel(), device=device) // top_k
    for expert in torch.unique(expert_of).tolist():
        pairs = (expert_of == expert).nonzero().squeeze(1)
        w1_e = _dequantize(w1[expert], w1_scale[expert])
        w2_e = _dequantize(w2[expert], w2_scale[expert])
        for chunk in pairs.split(_ROWS):
            tokens = token_of[chunk]
            gate_up = x[tokens] @ w1_e.T
            gate, up = gate_up[:, :inter], gate_up[:, inter:]
            # A float32 kernel rounds its float32 activation to bfloat16; so does this float64 value.
            h = (gate * torch.sigmoid(gate) * up).to(torch.float32).to(torch.bfloat16).double()
            y = _mxfp4(h) @ w2_e.T
            out.index_add_(0, tokens, y * weight_of[chunk, None])
    return out
