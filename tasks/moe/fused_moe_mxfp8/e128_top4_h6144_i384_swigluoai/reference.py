"""High-precision MXFP8 routed-experts MoE reference used only for correctness.

Matrix products run in float64 over exactly dequantized MXFP8 operands. The two quantization steps and the
bfloat16 rounding of the GEMM1 output and of the activation are part of the operator's semantics and are
reproduced exactly (float32 scale math, round-to-nearest-even into float8_e4m3fn).
"""
import torch

_E4M3_MAX = 448.0


def _quantize_dequantize(x):
    """MXFP8 round trip of a bfloat16 matrix: values and per-1x32-block scale as the operator computes them."""
    shape = x.shape
    blocked = x.to(torch.float32).view(*shape[:-1], shape[-1] // 32, 32)
    amax = blocked.abs().amax(dim=-1).clamp(min=1e-30)
    biased = (torch.ceil(torch.log2(amax / _E4M3_MAX)) + 127.0).clamp(0.0, 254.0)
    descale = torch.exp2(biased - 127.0).unsqueeze(-1)
    q = (blocked / descale).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
    return (q.double() * descale.double()).view(shape)


def _dequantize(values, scales):
    factors = torch.exp2(scales.double() - 127.0).repeat_interleave(32, dim=-1)
    return values.double() * factors


def reference(*, hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, alpha, beta, limit):
    num_tokens, hidden = hidden_states.shape
    num_experts, gate_up, _ = w13.shape
    inter = gate_up // 2
    x = _quantize_dequantize(hidden_states)
    out = torch.zeros((num_tokens, hidden), dtype=torch.float64, device=hidden_states.device)
    weights = topk_weights.double()
    for expert in range(num_experts):
        token, slot = (topk_ids == expert).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        g = (x[token] @ _dequantize(w13[expert], w13_scale[expert]).t()).to(torch.bfloat16).double()
        gate = g[:, :inter].clamp(max=float(limit))
        up = g[:, inter:].clamp(-float(limit), float(limit))
        act = (gate * torch.sigmoid(float(alpha) * gate) * (up + float(beta))).to(torch.bfloat16)
        act = _quantize_dequantize(act)
        y = (act @ _dequantize(w2[expert], w2_scale[expert]).t()) * weights[token, slot].unsqueeze(1)
        out.index_add_(0, token, y)
    return out
