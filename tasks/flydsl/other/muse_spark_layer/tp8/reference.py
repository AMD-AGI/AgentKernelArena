# SPDX-License-Identifier: Apache-2.0
"""Float64 reference of one Muse Spark MXFP4 decoder layer step at TP8 (logical, all heads and experts).

Composed from the per-launch references of sgl-amd-mono's tests (test/kernels/test_muse_spark_{kattn,k2,k3,k1}.py
and the 8-rank test_muse_spark_k3_tp8.py), computed in float64 except where the operator defines a quantization
or a materialized bfloat16 tensor:

  attention   per request and KV head (query head h reads KV head h // 8): s = 0.125 q.k over the request's
              seq_lens + 8 keys, causal for the step's 8 rows (row j sees keys < seq_lens + j + 1), softmax,
              o = P V; o is a bfloat16 tensor (K2 reads it rounded)
  K2          og = bf16(o rstd(o) sigmoid(g)) per head (eps 1e-5); attn = bf16(og w_o^T) (the all-reduced o_proj);
              nb = bf16(attn rstd(attn)) (eps 1e-8); new = alpha res + beta nb; x = bf16(new rstd(new) pre_w)
              (eps 1e-5); sigmoid(0.17677669529663687 x (router_hi + router_lo)^T) + bias, top-8 (ties to the
              lower expert);
              scores = picked sigmoids / (their sum + 1e-15); pooled = bf16(x w_pool^T);
              xin = pooled rstd(pooled) xn_w (eps 1e-5), a bfloat16 tensor
  K3          per (token, slot): MXFP4(xin) (aiter's _mxfp4_quant_op rule) against the E2M1 x E8M0 gate | up rows,
              mid = MXFP4(bf16(silu(gate) up)), y = bf16(mid w2^T), v = bf16(y pen_w),
              out += score v / sqrt(mean(v^2) + 1e-8); part = bf16(out); branch = bf16(part w_unpool^T);
              post = bf16(branch pfw); nb2 = bf16(post rstd(post)) (eps 1e-8); res = alpha2 new + beta2 nb2;
              attn_out = res rstd(res) npw (eps 1e-5), a bfloat16 tensor
  next K1     y = bf16(attn_out w_qkvg1^T); q = rope(bf16(bf16(y_q rstd) 5.4730064863875)),
              k = rope(bf16(y_k rstd)) (per head, eps 1e-5), v = y_v, gate = y_g; interleaved RoPE on
              bf16 cos / sin with each product rounded; the sliding cache rows swa_map[slots] take
              fp8(bf16(k)) and fp8(v) (E4M3, round to nearest even, saturating at 448)

Returned bfloat16 / fp8 outputs are the float64 values before their final rounding (for the fp8 caches: the
written rows' bf16(k) and v clamped to +-448; the other rows their fp8 contents).
"""
import torch

HIDDEN = 8192
HEAD_DIM = 64
Q_HEADS = 128
KV_HEADS = 16
GQA = Q_HEADS // KV_HEADS
Q_LEN = 8
EXPERTS = 256
TOP_K = 8
MOE_HIDDEN = 4096
INTER = 4096
PAGE = 16
SOFTMAX_SCALE = 0.125
RMS_EPS = 1e-5
POST_EPS = 1e-8
ROUTE_EPS = 1e-15
ROUTE_SCALE = 1.0
ROUTER_MULT = 0.17677669529663687
QK_SCALE = 5.4730064863875
FP8_MAX = 448.0
K_ROW0 = Q_HEADS * HEAD_DIM
V_ROW0 = K_ROW0 + KV_HEADS * HEAD_DIM
G_ROW0 = V_ROW0 + KV_HEADS * HEAD_DIM
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _bf(t):
    """Round to bfloat16 (nearest even) and back to float64."""
    return t.to(torch.bfloat16).to(torch.float64)


def _rstd(t, eps):
    return torch.rsqrt(t.square().mean(-1, keepdim=True) + eps)


def _mx_dequant(x):
    """x (bfloat16 values as float64, [..., N]) through aiter's MXFP4 rule and back: 32-value blocks, the E8M0
    scale from the block max rounded at mantissa bit 21 (unbiased exponent floor(log2) - 2, at least -127),
    E2M1 codes of x * 2^-unbiased rounded half to even and saturating at 6 (the fp32 bit arithmetic of the
    rule, exact on bfloat16 inputs)."""
    shape = x.shape
    xb = x.to(torch.float32).reshape(*shape[:-1], -1, 32)
    amax = xb.abs().amax(-1, keepdim=True)
    bits = (amax.view(torch.int32) + 0x200000) & -0x800000
    unb = (((bits >> 23) & 0xFF) - 129).clamp(min=-127)
    qx = xb * torch.ldexp(torch.ones_like(amax), -unb)
    a = qx.abs()
    ab = a.view(torch.int32)
    den = ((a + 4194304.0).view(torch.int32) - (149 << 23)) & 0xFF
    nrm = ((ab + ((-126 << 23) + 0x1FFFFF) + ((ab >> 22) & 1)) >> 22) & 0xFF
    code = torch.where(a >= 6.0, torch.full_like(den, 7), torch.where(a < 1.0, den, nrm))
    mag = torch.tensor(E2M1, dtype=torch.float64, device=x.device)[code.long()]
    val = torch.where(qx < 0, -mag, mag)
    return (val * torch.ldexp(torch.ones_like(amax, dtype=torch.float64), unb.to(torch.float64))).reshape(shape)


def _dequant_e2m1(packed, scales):
    """Quark MXFP4 [rows, k / 2] codes (low nibble = even k) with E8M0 scales [rows, k / 32] -> float64."""
    rows = packed.shape[0]
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(rows, -1).long()
    lut = torch.tensor(E2M1, dtype=torch.float32, device=packed.device)
    mag = lut[codes & 7]
    val = torch.where((codes & 8) != 0, -mag, mag)
    scale = torch.ldexp(torch.ones(scales.shape, dtype=torch.float32, device=packed.device),
                        scales.to(torch.int32) - 127)
    return (val.reshape(rows, -1, 32) * scale[..., None]).reshape(rows, -1).to(torch.float64)


def _attention(q, k_cache, v_cache, page_table, seq_lens):
    rows = q.shape[0]
    R = rows // Q_LEN
    q64 = q.to(torch.float64).reshape(rows, Q_HEADS, HEAD_DIM)
    o = torch.zeros((rows, Q_HEADS, HEAD_DIM), dtype=torch.float64, device=q.device)
    for r in range(R):
        n = int(seq_lens[r]) + Q_LEN
        pages = page_table[r, :-(-n // PAGE)].long()
        slots = (pages[:, None] * PAGE + torch.arange(PAGE, device=q.device)[None, :]).reshape(-1)[:n]
        visible = (torch.arange(n, device=q.device)[None, :]
                   <= (n - Q_LEN + torch.arange(Q_LEN, device=q.device))[:, None])  # [8 rows, n keys]
        for h in range(KV_HEADS):
            K = k_cache[slots, h].to(torch.float64)
            V = v_cache[slots, h].to(torch.float64)
            Q = q64[r * Q_LEN:(r + 1) * Q_LEN, h * GQA:(h + 1) * GQA]  # [8 rows, 8 heads, 64]
            s = SOFTMAX_SCALE * torch.einsum("thd,nd->thn", Q, K)
            s = s.masked_fill(~visible[:, None, :], float("-inf"))
            o[r * Q_LEN:(r + 1) * Q_LEN, h * GQA:(h + 1) * GQA] = torch.einsum("thn,nd->thd", torch.softmax(s, -1), V)
    return o.reshape(rows, HIDDEN)


def _rope(value, cos_sin, positions):
    """Interleaved RoPE on (2i, 2i + 1) with bf16 cos / sin, each product rounded; the sum unrounded."""
    cs = cos_sin[positions].to(torch.float64)  # [rows, 64]: cos 32 | sin 32
    d = torch.arange(HEAD_DIM, device=value.device)
    cos = _bf(cs[:, None, d // 2])
    sin = _bf(cs[:, None, HEAD_DIM // 2 + d // 2])
    other = value.reshape(*value.shape[:2], HEAD_DIM // 2, 2).flip(-1).reshape(value.shape)
    rotated = torch.where(d % 2 == 0, -other, other)
    return _bf(value * cos) + _bf(rotated * sin)


def reference(*, q, g, k_cache, v_cache, page_table, seq_lens, res, w_o, alpha, beta, pre_w, router_hi, router_lo,
              expert_bias, w_pool, xn_w, w13, s13, w2, s2, pen_w, pfw, alpha2, beta2, npw, w_unpool, w_qkvg1,
              cos_sin, positions, slots, swa_map, k_cache1, v_cache1, guard):
    del guard  # the launch's epoch / deadline words: no effect on the result
    f64 = torch.float64
    rows = q.shape[0]
    # ---- the attention
    o = _attention(q, k_cache, v_cache, page_table, seq_lens)

    # ---- K2: o-norm-gate, o_proj, the attention boundary, the router, the pooler
    ob = _bf(o).reshape(rows, Q_HEADS, HEAD_DIM)
    gate = torch.sigmoid(g.to(f64).reshape(rows, Q_HEADS, HEAD_DIM))
    og = _bf(ob * _rstd(ob, RMS_EPS) * gate).reshape(rows, HIDDEN)
    attn = _bf(og @ w_o.to(f64).t())
    nb = _bf(attn * _rstd(attn, POST_EPS))
    new = alpha.to(f64) * res.to(f64) + beta.to(f64) * nb
    x = _bf(new * _rstd(new, RMS_EPS) * pre_w.to(f64))
    router = router_hi.to(f64) + router_lo.to(f64)  # [EXPERTS, HIDDEN]: the fp32 router's bf16 hi + lo split
    sig = torch.sigmoid((x @ router.t()) * ROUTER_MULT)
    biased = sig + expert_bias.to(f64)
    ids = torch.argsort(biased, dim=-1, descending=True, stable=True)[:, :TOP_K]
    picked = sig.gather(1, ids)
    scores = picked / (picked.sum(-1, keepdim=True) + ROUTE_EPS) * ROUTE_SCALE
    pooled = _bf(x @ w_pool.to(f64).t())
    xin = pooled * _rstd(pooled, RMS_EPS) * xn_w.to(f64)
    xin_b = _bf(xin)

    # ---- K3: the experts (all 256, every pick), the finalize, the unpooler, the FFN boundary
    out = torch.zeros((rows, MOE_HIDDEN), dtype=f64, device=q.device)
    pen = pen_w.to(f64)
    for e in torch.unique(ids).tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        gu = _mx_dequant(xin_b[tok]) @ _dequant_e2m1(w13[e], s13[e]).t()
        gt, up = gu[:, :INTER], gu[:, INTER:]
        mid = _mx_dequant(_bf(gt * torch.sigmoid(gt) * up))
        y = _bf(mid @ _dequant_e2m1(w2[e], s2[e]).t())
        v = _bf(y * pen)
        out.index_add_(0, tok, (scores[tok, slot] / torch.sqrt(v.square().mean(-1) + POST_EPS))[:, None] * v)
    part = _bf(out)
    branch = _bf(part @ w_unpool.to(f64).t())
    post = _bf(branch * pfw.to(f64))
    nb2 = _bf(post * _rstd(post, POST_EPS))
    res_out = alpha2.to(f64) * new + beta2.to(f64) * nb2
    attn_out = res_out * _rstd(res_out, RMS_EPS) * npw.to(f64)

    # ---- the next layer's K1 (a sliding layer with RoPE)
    y1 = _bf(attn_out) @ w_qkvg1.to(f64).t()
    yb = _bf(y1)
    qh = yb[:, :K_ROW0].reshape(rows, Q_HEADS, HEAD_DIM)
    kh = yb[:, K_ROW0:V_ROW0].reshape(rows, KV_HEADS, HEAD_DIM)
    q1 = _rope(_bf(_bf(qh * _rstd(qh, RMS_EPS)) * QK_SCALE), cos_sin, positions).reshape(rows, -1)
    k1 = _rope(_bf(kh * _rstd(kh, RMS_EPS)), cos_sin, positions)
    v1 = y1[:, V_ROW0:G_ROW0].reshape(rows, KV_HEADS, HEAD_DIM)
    g1 = y1[:, G_ROW0:]
    at = swa_map[slots]
    k_cache1_out = k_cache1.to(f64)
    v_cache1_out = v_cache1.to(f64)
    # the written rows as the values the kernel rounds to fp8 (the comparison adds one fp8 rounding);
    # every other row keeps its fp8 contents exactly
    k_cache1_out[at] = _bf(k1).clamp(-FP8_MAX, FP8_MAX)
    v_cache1_out[at] = v1.clamp(-FP8_MAX, FP8_MAX)

    return {
        "o": o, "ids": ids.to(torch.int32), "scores": scores, "xin": xin, "attn_out": attn_out,
        "k_out": k1.reshape(rows, -1), "v_out": v1.reshape(rows, -1), "q": q1, "g": g1, "res": res_out,
        "k_cache1": k_cache1_out, "v_cache1": v_cache1_out,
    }
