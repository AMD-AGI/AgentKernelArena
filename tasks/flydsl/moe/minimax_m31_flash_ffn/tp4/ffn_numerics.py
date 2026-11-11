# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Float64 numerics of the MiniMax-M3.1-Flash FFN launch on a TP4 group (reference.py and inputs.py).

Logical inputs (inputs.py): the rank-specific tensors are stacked on a leading TP axis; codes are
unshuffled and scales are raw E8M0 bytes. The operator, per step of S rows:

  o_proj   rank r: per-token FP8 of attn[r] (scale = fp32(amax * fp32(1/448)), x / scale clamped to
           +-448 and rounded to E4M3, RNE) times its rowwise-FP8 o_proj shard (E4M3 x per-row fp32
           scale): p_r = bf16(q . w * scale_x * scale_w)
  seam     a = bf16(sum_r p_r) (the attention all-reduce); h = a + res (the residual, written as
           res_out); Gemma RMSNorm xn = bf16(h * rsqrt(mean(h^2) + 1e-6) * (1 + norm_w))
  router   logits = xn . router_w (fp32 weight); top 4 of sigmoid(logits) + bias (the bias only
           selects); weights = the picks' sigmoids / their sum * 2.0
  experts  rank r, intermediate slice of 768: routed expert e (MXFP4: E2M1 codes, low nibble first,
           times 2^(e8m0 - 127) per 32 along K) and the shared expert (MXFP8: E4M3 times 2^(e8m0 -
           127) per 32): gate | up = xn . w13^T, SwiGLU-OAI mid = bf16(min(g, 7) * sigmoid(1.702 *
           min(g, 7)) * (clamp(u, -7, 7) + 1)), y = mid . w2^T; m_r = bf16(shared + sum_k weight_k
           y_k) (the rank's partial)
  out      sum_r m_r (the MoE all-reduce)

Every product, sum and nonlinearity is evaluated in float64 (float32 under the QA calibration probe);
the roundings listed are the operator's own (bf16 activations and partials, FP8 activation codes).
"""
import torch

HIDDEN = 6144
ATTN_K = 2048  # a TP4 rank's o_proj input: 16 heads of 128
INTER = 768  # a TP4 rank's expert intermediate (3072 / 4)
EXPERTS = 128
TOPK = 4
TP = 4
GROUP = 32  # MX scale group along K
EPS = 1e-6
ALPHA, LIMIT = 1.702, 7.0
ROUTED_SCALE = 2.0
FP8_MAX = 448.0
INV_FP8_MAX_F32 = 1.0 / 448.0  # rounded to fp32 where used, as the kernel's f32 constant
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _f64(x):
    return x.to(torch.float64)


def bf16(x):
    """Round to bfloat16 (RNE) and back."""
    return x.to(torch.bfloat16).to(x.dtype if x.dtype.is_floating_point else torch.float64)


def f32(x):
    return x.to(torch.float32).to(x.dtype)


def per_token_fp8(x):
    """AITER's per-token FP8 (per_token_quant_hip): (E4M3 values, per-row scale)."""
    x64 = _f64(x)
    amax = x64.abs().amax(-1, keepdim=True)
    inv = f32(torch.full_like(amax, INV_FP8_MAX_F32))
    scale = f32(amax * inv).clamp_min(torch.finfo(torch.float32).tiny)
    q = (x64 / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return _f64(q), scale


def oproj_partials(attn, wo, wos):
    """Every rank's bf16 o_proj partial [TP, S, HIDDEN]."""
    parts = []
    for r in range(attn.shape[0]):
        q, scale = per_token_fp8(attn[r])
        acc = q @ _f64(wo[r]).T
        parts.append(bf16(acc * scale * _f64(wos[r]).unsqueeze(0)))
    return torch.stack(parts)


def seam(attn, res, wo, wos, norm_w):
    """(h, xn): the residual after attention and the post-attention Gemma RMSNorm of it in bf16."""
    a = bf16(oproj_partials(attn, wo, wos).sum(0))
    h = a + _f64(res)
    rstd = torch.rsqrt(h.square().mean(-1, keepdim=True) + EPS)
    xn = bf16(h * rstd * (1.0 + _f64(norm_w)))
    return h, xn


def route(xn, router_w, bias):
    """(ids [S, 4] in selection order, weights [S, 4], logits [S, 128])."""
    logits = xn @ _f64(router_w).T
    scores = torch.sigmoid(logits)
    ids = (scores + _f64(bias)).topk(TOPK, dim=-1).indices
    picked = scores.gather(1, ids)
    return ids, picked / picked.sum(-1, keepdim=True) * ROUTED_SCALE, logits


def _group_scale(scale):
    """E8M0 bytes [..., K / 32] -> 2^(e - 127) per element [..., K]."""
    return torch.exp2(_f64(scale) - 127.0).repeat_interleave(GROUP, dim=-1)


def mxfp4_values(codes, scale):
    """MXFP4 codes [N, K / 2] (two E2M1 a byte, low nibble first) and E8M0 scales [N, K / 32] ->
    the values [N, K]."""
    table = torch.tensor(E2M1, dtype=torch.float64, device=codes.device)
    c = codes.to(torch.int64)
    nib = torch.stack([c & 0xF, c >> 4], dim=-1).reshape(codes.shape[0], -1)
    return table[nib] * _group_scale(scale)


def mxfp8_values(q, scale):
    """MXFP8: E4M3 [N, K] and E8M0 scales [N, K / 32] -> the values [N, K]."""
    return _f64(q) * _group_scale(scale)


def swiglu_oai(gate_up):
    """SwiGLU-OAI of [.., 2 * INTER] (gate, then up), alpha 1.702, limit 7, rounded to bf16."""
    g = gate_up[..., :INTER].clamp(max=LIMIT)
    u = gate_up[..., INTER:].clamp(-LIMIT, LIMIT)
    return bf16(g * torch.sigmoid(ALPHA * g) * (u + 1.0))


def moe_partials(xn, ids, weights, w13, s13, w2, s2, w13s, s13s, w2s, s2s):
    """Every rank's bf16 MoE partial [TP, S, HIDDEN]: the shared expert plus the routed picks."""
    parts = []
    for r in range(w13.shape[0]):
        acc = swiglu_oai(xn @ mxfp8_values(w13s[r], s13s[r]).T) @ mxfp8_values(w2s[r], s2s[r]).T
        for e in torch.unique(ids).tolist():
            rows, ks = (ids == e).nonzero(as_tuple=True)
            mid = swiglu_oai(xn[rows] @ mxfp4_values(w13[r, e], s13[r, e]).T)
            y = mid @ mxfp4_values(w2[r, e], s2[r, e]).T
            acc = acc.index_add(0, rows, y * weights[rows, ks].unsqueeze(-1))
        parts.append(bf16(acc))
    return torch.stack(parts)


def ffn(inp):
    """{"out", "res_out"} of the logical inputs ``inp`` (see the module docstring)."""
    h, xn = seam(inp["attn"], inp["res"], inp["wo"], inp["wos"], inp["norm_w"])
    ids, weights, _ = route(xn, inp["router_w"], inp["bias"])
    parts = moe_partials(xn, ids, weights, inp["w13"], inp["s13"], inp["w2"], inp["s2"], inp["w13s"],
                         inp["s13s"], inp["w2s"], inp["s2s"])
    return {"out": parts.sum(0), "res_out": h}
