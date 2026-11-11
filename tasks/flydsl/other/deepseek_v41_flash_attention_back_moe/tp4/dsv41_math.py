# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Float64 definition of DeepSeek-V4.1-Flash's two decode launches K1 and K2 on a TP4 group, as the
sgl-amd-mono monokernels (sgl_amd_mono/deepseek_v41/gfx950, 52486703) define them.

Shared by ``reference.py`` (the correctness oracle) and ``inputs.py`` (whose generator plants the
margins of the launch's discrete decisions on the same math). It imports only torch.

Values are float64 on dequantized operands. Where the kernels store a narrow format the value is
rounded the same way: bf16 (round to nearest even, by way of float32), E4M3 (clamped to +-448,
round to nearest even) with E8M0 scales by the rule of the producing stage, computed in float32
exactly as the kernels compute it:

* ``vllm``: code = ceil_exp(RN_f32(max(amax, FLT_MIN) / 448)) clamped to 0..254 (vLLM's ROCm
  mxfp8_e4m3_quantize; K1's MXFP8 x and q_norm output, K2's wo_a output, the shared expert);
* ``aiter``: code = ceil_exp(RN_f32(max(amax, 1e-10) * f32(1 / 448))) clamped to 1..254 (the routed
  experts' x and intermediate, aiter's fused_mx_quant_moe_sort);
* ``record``: code = ceil_exp(RN_f32(max(amax, 1e-4) / 448)), no clamp (the fp8_ds_mla KV record's
  64-value NoPE blocks);

``ceil_exp(q)`` is the biased exponent of the smallest power of two >= q; the codes are
E4M3(clamp(x * 2^(127 - code), +-448)) and decode to code * 2^(code - 127).

mHC seam (one sublayer boundary; K1 runs the attention's, K2 the FFN's), per row:
    R'[o] = bf16(post[o] pend + sum_h comb[h][o] R[h])            (four streams o = 0..3)
    x = bf16(sum_h pre[h] R'[h])                                   (``pre``: the previous gate's)
    m = fn . R' (fn [24, 4 * 5120], column o * 5120 + c), v = m / sqrt(mean(R'^2) + 1e-20)
    pre' = sigmoid(v[0:4] s0 + b[0:4]) + 1e-6, post' = 2 sigmoid(v[4:8] s1 + b[4:8]),
    comb'[r][c] = Sinkhorn(v[8 + 4 r + c] s2 + b[8 + 4 r + c]): a softmax over c plus 1e-6, a
        division by (column sum + 1e-6), then 19 (row, column) normalizations, each by (sum + 1e-6)
    y = bf16(x / sqrt(mean(x^2) + 1e-20) * norm_w)
K1 front (TP-replicated wqkv_a, rank-sharded wq_b):
    x8 = MXFP8_vllm(y); qkv = bf16(x8 W_qkv^T) (FP8 weights with 32 x 32 E8M0 blocks);
    q_a = qkv[:1280]: qn = bf16(q_a / rms(q_a) * q_norm), q8 = MXFP8_vllm(qn), q = bf16(q8 W_qb^T),
        per head GPT-J RoPE of dims 448..511 (pair j: cos / sin of table column j / 32 + j), bf16;
    kv_a = qkv[1280:]: kn = kv_a / rms(kv_a) * kv_norm, GPT-J RoPE of dims 448..511, then the V4.1
        fp8_ds_mla record at the row's slot: dims 0..447 as E4M3 in 7 blocks of 64 (``record``
        rule), dims 448..511 as bf16, the 7 scale codes and a zero byte in the page's scale area;
    the K2 key table: kt[t][k] = topk[k] | 1 << 31 (k < ntopk), the window slots, then -1.
K2 back and FFN:
    per row t and head h: s_j = q . k_j / sqrt(512) over the row's keys (dequantized records;
        K = V, all 512 dims), o = sum_j p_j k_j with p = softmax over (s, sink_h) (the sink takes
        mass, no value); XO = bf16(inverse GPT-J RoPE of dims 448.. of bf16(o));
    wo_a per group g (8 heads, 4096 columns): y = bf16(XO_g W_oa,g^T); x8b = MXFP8_vllm(y);
    rank r's partial = bf16(x8b_r W_ob[:, r cols]^T); a = bf16(sum over ranks of the partials);
    FFN seam on (R = K1's residual, pend = a) -> res_out, pre_out, post_out, comb_out, h = y;
    router: logits = h W_g^T, score = sqrt(softplus(logits)) (logit itself past 20), the 6 largest
        score + bias are routed, weight = score * 1.5 / sum of the 6 scores;
    rank r: routed experts on its 576 of 2304 intermediate columns (x MXFP8_aiter, gate | up MXFP4,
        v = bf16(min(g, 10) sigmoid(min(g, 10)) clamp(u, -10, 10)), MXFP8_aiter(v), down MXFP4,
        c_k = bf16(y_k w_k), routed = bf16(routed + c_k) in top-k order), the shared expert on its
        576 columns (x MXFP8_vllm, FP8 block weights, g = bf16(min(bf16 gate, 10)),
        bf16(bf16(silu(g)) clamp(bf16 up, -10, 10)), MXFP8_vllm, down, bf16), partial =
        bf16(routed + shared); out = bf16(sum over ranks of the partials).
"""
import math

import torch

TP = 4
HIDDEN = 5120
HC = 4
MIX = HC * (HC + 2)
K_MIX = HC * HIDDEN
RMS_EPS = 1e-20
HC_EPS = 1e-6
SINKHORN = 20
POST_MULT = 2.0
Q_RANK = 1280
KV_DIM = 512
QKV_ROWS = Q_RANK + KV_DIM
HEAD_DIM = 512
NOPE = 448
ROPE = 64
HALF = ROPE // 2
N_HEADS = 64
O_GROUPS = 8
O_RANK = 1024
GROUP_K = N_HEADS // O_GROUPS * HEAD_DIM  # wo_a's K a group: its 8 heads
WINDOW = 128
TOPK_KEYS = 512
KEYS = TOPK_KEYS + WINDOW
DATA = 576  # a record's data bytes: 448 E4M3, then 64 bf16
SCALE_BYTES = 8  # a record's scale bytes: 7 E8M0 codes and a zero pad
RECORD = DATA + SCALE_BYTES
EXPERTS = 384
TOP_K = 6
ROUTE_SCALE = 1.5
LIMIT = 10.0
INTER = 2304  # a routed expert's and the shared expert's intermediate width
INTER_RANK = INTER // TP  # 576
INTER_PAD = 640  # a rank's routed intermediate, padded to a multiple of 128 with zero weights
FP8_MAX = 448.0
MX = 32
FLT_MIN = 1.1754943508222875e-38
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def page_bytes(block: int) -> int:
    """A V4.1 cache page: ``block`` 576-byte data rows, then ``block`` 8-byte scale rows, padded to 576 B."""
    return -(-block * RECORD // DATA) * DATA


# ------------------------------------------------------------------------------- rounding
def bf16(x):
    """bf16 rounding (nearest even, by way of float32), as float64."""
    return x.to(torch.float32).to(torch.bfloat16).to(torch.float64)


def e8m0(codes):
    """E8M0 codes -> float64 powers of two."""
    return torch.exp2(codes.to(torch.float64) - 127.0)


def _ceil_exp(q32):
    """Biased exponent of the smallest power of two >= q (float32, q > 0)."""
    bits = q32.contiguous().view(torch.int32)
    return ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)


def _pow2_f32(e):
    """2^(e - 127) as float32 for an int32 code tensor in 1..254."""
    return (e.to(torch.int32) << 23).view(torch.float32)


def mx_codes(amax32, rule):
    """The E8M0 code of each block's float32 amax under ``rule`` (vllm, aiter or record)."""
    one = torch.ones((), dtype=torch.float32, device=amax32.device)
    if rule == "vllm":
        q = amax32.clamp_min(FLT_MIN) / (one * FP8_MAX)
        return _ceil_exp(q).clamp(0, 254)
    if rule == "aiter":
        q = amax32.clamp_min(1e-10) * (one / FP8_MAX)
        return _ceil_exp(q).clamp(1, 254)
    if rule == "record":
        q = amax32.clamp_min(float(torch.tensor(1e-4, dtype=torch.float32))) / (one * FP8_MAX)
        return _ceil_exp(q)
    raise ValueError(rule)


def mx_quant(x, rule, block=MX):
    """Block quantization over the last axis: (E4M3 codes, int32 scale codes, float64 dequantized)."""
    x32 = x.to(torch.float32)
    g = x32.reshape(*x32.shape[:-1], x32.shape[-1] // block, block)
    code = mx_codes(g.abs().amax(dim=-1), rule)
    mul = _pow2_f32(254 - code).unsqueeze(-1)  # 2^(127 - code), exact
    q = (g * mul).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    deq = q.to(torch.float64) * e8m0(code).unsqueeze(-1)
    return q.reshape(x.shape), code, deq.reshape(x.shape)


def fp8_block_dequant(w, s, block=MX):
    """E4M3 [N, K] with E8M0 [N / 32, K / 32] block codes -> float64 [N, K]."""
    n, k = w.shape
    blocks = w.to(torch.float64).view(n // block, block, k // block, block)
    return (blocks * e8m0(s)[:, None, :, None]).view(n, k)


def mxfp4_dequant(codes, scales):
    """MXFP4 [..., N, K / 2] packed E2M1 (low nibble = even element), E8M0 [..., N, K / 32] -> float64."""
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=codes.device)
    v = torch.stack((lut[(codes & 0xF).long()], lut[(codes >> 4).long()]), dim=-1).flatten(-2)
    return v * e8m0(scales).repeat_interleave(MX, dim=-1)


def rope_rotate(v, cs, inverse=False):
    """GPT-J RoPE of the last 64 dims of v [..., 512] (pairs 448 + 2 j, 449 + 2 j; cs [..., 64]:
    32 cos, then 32 sin, broadcast over v's head axis by the caller)."""
    c, s = cs[..., :HALF], cs[..., HALF:]
    e, o = v[..., NOPE::2], v[..., NOPE + 1::2]
    if inverse:
        e2, o2 = e * c + o * s, o * c - e * s
    else:
        e2, o2 = e * c - o * s, e * s + o * c
    out = v.clone()
    out[..., NOPE::2], out[..., NOPE + 1::2] = e2, o2
    return out


# ------------------------------------------------------------------------------- mHC seam
def seam(res, pend, post, comb, pre, hc_fn, hc_scale, hc_base, norm_w):
    """One mHC boundary for S rows: res [S, 4, 5120], pend [S, 5120], post / pre [S, 4], comb [S, 4, 4]
    -> dict of R' (``res_out``), the layer input ``x``, the normed ``y`` and the gate's
    ``pre_out``, ``post_out``, ``comb_out``; also the pre-rounding values for margin checks."""
    f = torch.float64
    res, pend, post, comb, pre = (t.to(f) for t in (res, pend, post, comb, pre))
    r_raw = post[:, :, None] * pend[:, None, :] + torch.einsum("sho,shc->soc", comb, res)
    r_new = bf16(r_raw)
    x_raw = (pre[:, :, None] * r_new).sum(1)
    x = bf16(x_raw)
    flat = r_new.reshape(r_new.shape[0], K_MIX)
    v = (flat @ hc_fn.to(f).T) / torch.sqrt(flat.square().mean(-1, keepdim=True) + RMS_EPS)
    sc, base = hc_scale.to(f), hc_base.to(f)
    pre_out = torch.sigmoid(v[:, :HC] * sc[0] + base[:HC]) + HC_EPS
    post_out = POST_MULT * torch.sigmoid(v[:, HC:2 * HC] * sc[1] + base[HC:2 * HC])
    cm = (v[:, 2 * HC:] * sc[2] + base[2 * HC:]).view(-1, HC, HC)
    cm = torch.softmax(cm, dim=-1) + HC_EPS
    cm = cm / (cm.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(SINKHORN - 1):
        cm = cm / (cm.sum(-1, keepdim=True) + HC_EPS)
        cm = cm / (cm.sum(-2, keepdim=True) + HC_EPS)
    rstd = 1.0 / torch.sqrt(x.square().mean(-1, keepdim=True) + RMS_EPS)
    y_raw = x * rstd * norm_w.to(f)
    return {"res_out": r_new, "x": x, "y": bf16(y_raw), "pre_out": pre_out, "post_out": post_out,
            "comb_out": cm, "r_raw": r_raw, "x_raw": x_raw, "y_raw": y_raw}


# ------------------------------------------------------------------------------- KV records
def encode_records(k):
    """Rotated keys [n, 512] (float64) -> fp8_ds_mla records: (data uint8 [n, 576], scales uint8 [n, 8])
    with the ``record`` rule on the NoPE dims and bf16 RoPE dims (what K1's KV store writes)."""
    codes, sc, _ = mx_quant(k[:, :NOPE], "record", block=64)
    rope = k[:, NOPE:].to(torch.float32).to(torch.bfloat16)
    data = torch.cat([codes.view(torch.uint8), rope.contiguous().view(torch.uint8)], dim=1)
    scales = torch.cat([sc.to(torch.uint8), torch.zeros_like(sc[:, :1], dtype=torch.uint8)], dim=1)
    return data, scales


def record_bytes(cache, slots, block):
    """(data [n, 576], scales [n, 8]) uint8 of records ``slots`` of a paged cache [pages, page bytes]."""
    stride = cache.shape[1]
    flat = cache.reshape(-1)
    slots = slots.long()
    base = (slots // block) * stride
    off = slots % block
    ar = torch.arange(DATA, device=cache.device)
    data = flat[(base + off * DATA)[:, None] + ar]
    sc = flat[(base + block * DATA + off * SCALE_BYTES)[:, None] + torch.arange(SCALE_BYTES, device=cache.device)]
    return data, sc


def write_records(cache, slots, data, scales, block):
    """Write records into a paged cache [pages, page bytes] (in place)."""
    stride = cache.shape[1]
    flat = cache.view(-1)
    slots = slots.long()
    base = (slots // block) * stride
    off = slots % block
    flat[(base + off * DATA)[:, None] + torch.arange(DATA, device=cache.device)] = data
    flat[(base + block * DATA + off * SCALE_BYTES)[:, None] + torch.arange(SCALE_BYTES, device=cache.device)] = scales


def decode_records(data, scales):
    """Records -> float64 keys [n, 512] (K = V of the attention)."""
    nope = data[:, :NOPE].contiguous().view(torch.float8_e4m3fn).to(torch.float64)
    nope = nope * e8m0(scales[:, :NOPE // 64]).repeat_interleave(64, dim=1)
    rope = data[:, NOPE:].contiguous().view(torch.bfloat16).to(torch.float64)
    return torch.cat([nope, rope], dim=1)


# ------------------------------------------------------------------------------- K1 front
def k1_front(y, w):
    """K1's attention front on the seam output y [S, 5120] (bf16 values); ``w`` holds wqkv, wqkv_s,
    q_norm, kv_norm, wqb, wqb_s (logical, every rank's heads), cos_sin and pos.
    -> q [S, 64, 512] (bf16 values; ``q_raw`` before its final bf16 rounding), the rotated kn [S, 512] and
    the intermediates."""
    f = torch.float64
    _, _, x8 = mx_quant(y, "vllm")
    qkv = bf16(x8 @ fp8_block_dequant(w["wqkv"], w["wqkv_s"]).T)
    q_a, kv_a = qkv[:, :Q_RANK], qkv[:, Q_RANK:]
    qn = bf16(q_a / torch.sqrt(q_a.square().mean(-1, keepdim=True) + RMS_EPS) * w["q_norm"].to(f))
    _, _, q8 = mx_quant(qn, "vllm")
    rows = w["wqb"].shape[0] // TP  # a rank's rows at a time, to bound the float64 copy
    qb = torch.cat([bf16(q8 @ fp8_block_dequant(w["wqb"][r * rows:(r + 1) * rows],
                                                  w["wqb_s"][r * rows // MX:(r + 1) * rows // MX]).T)
                    for r in range(TP)], dim=1).view(y.shape[0], -1, HEAD_DIM)
    cs = w["cos_sin"][w["pos"].long()].to(f)
    q_raw = rope_rotate(qb, cs[:, None, :])
    rk = 1.0 / torch.sqrt(kv_a.square().mean(-1, keepdim=True) + RMS_EPS)
    kn = rope_rotate(kv_a * rk * w["kv_norm"].to(f), cs)
    return {"q": bf16(q_raw), "q_raw": q_raw, "kn": kn, "kv_a": kv_a, "x8": x8, "qkv": qkv}


def key_table(swa_idx, swa_lens, topk, topk_lens, slot):
    """K1's key table for K2 (front.stage_kt): kt [S, 640] int32, klen [2 S] int32."""
    s = swa_idx.shape[0]
    kt = torch.full((s, KEYS), -1, dtype=torch.int64, device=swa_idx.device)
    klen = torch.zeros(2 * s, dtype=torch.int64, device=swa_idx.device)
    for t in range(s):
        live = int(slot[t]) >= 0
        nswa = min(int(swa_lens[t]), WINDOW) if live else 0
        ntopk = min(int(topk_lens[t]), TOPK_KEYS) if live else 0
        top = topk[t, :ntopk].long()
        kt[t, :ntopk] = torch.where(top >= 0, top | (1 << 31), torch.full_like(top, -1))
        win = swa_idx[t, :nswa].long()
        kt[t, ntopk:ntopk + nswa] = torch.where(win >= 0, win, torch.full_like(win, -1))
        klen[t], klen[s + t] = ntopk + nswa, ntopk
    # slot | 1 << 31 as an int32 bit pattern
    kt = torch.where(kt >= (1 << 31), kt - (1 << 32), kt)
    return kt.to(torch.int32), klen.to(torch.int32)


# ------------------------------------------------------------------------------- K2 back
def row_keys(kt, klen, swa, comp, swa_block, comp_block, t):
    """Row t's keys from the key table (dequantized), in table order; an absent key (-1) is none."""
    n = int(klen[t])
    ent = kt[t, :n].long() & 0xFFFFFFFF
    valid = ent != 0xFFFFFFFF
    is_c = (ent >> 31) & 1 == 1
    slot = ent & 0x7FFFFFFF
    keys = []
    for mask, cache, block in ((valid & is_c, comp, comp_block), (valid & ~is_c, swa, swa_block)):
        if bool(mask.any()):
            keys.append(decode_records(*record_bytes(cache, slot[mask], block)))
    if not keys:
        return torch.zeros(0, HEAD_DIM, dtype=torch.float64, device=kt.device)
    return torch.cat(keys)


def attention_xo(q, kt, klen, swa, comp, swa_block, comp_block, sink, pos, cos_sin):
    """XO [S, 64 * 512] (bf16 values): sparse MQA with the sink, bf16 o, inverse RoPE, bf16."""
    f = torch.float64
    s = q.shape[0]
    o = torch.zeros(s, N_HEADS, HEAD_DIM, dtype=f, device=q.device)
    for t in range(s):
        k = row_keys(kt, klen, swa, comp, swa_block, comp_block, t)
        if k.shape[0] == 0:
            continue
        sc = (q[t].to(f) @ k.T) / math.sqrt(HEAD_DIM)  # [64, n]
        m = torch.maximum(sc.amax(dim=1), sink.to(f))
        p = torch.exp(sc - m[:, None])
        o[t] = (p @ k) / (p.sum(dim=1) + torch.exp(sink.to(f) - m))[:, None]
    cs = cos_sin[pos.long()].to(f)
    # the inverse rotation (e' = e cos + o sin, o' = o cos - e sin): back.stage_combine and check_k2.attention_xo
    return bf16(rope_rotate(bf16(o), cs[:, None, :], inverse=True)).reshape(s, N_HEADS * HEAD_DIM)


def wo_partials(xo, woa, woa_s, wob, wob_s):
    """XO -> each rank's bf16 wo_b partial [TP, S, 5120] (wo_a per group, MXFP8_vllm, wo_b)."""
    s = xo.shape[0]
    y = torch.empty(s, O_GROUPS * O_RANK, dtype=torch.float64, device=xo.device)
    for g in range(O_GROUPS):
        wa = fp8_block_dequant(woa[g * O_RANK:(g + 1) * O_RANK], woa_s[g * O_RANK // MX:(g + 1) * O_RANK // MX])
        y[:, g * O_RANK:(g + 1) * O_RANK] = bf16(xo[:, g * GROUP_K:(g + 1) * GROUP_K] @ wa.T)
    _, _, x8b = mx_quant(y, "vllm")
    wb = fp8_block_dequant(wob, wob_s)
    cols = O_GROUPS // TP * O_RANK
    return torch.stack([bf16(x8b[:, r * cols:(r + 1) * cols] @ wb[:, r * cols:(r + 1) * cols].T) for r in range(TP)])


def rank_sum(partials):
    """bf16 of the sum of the ranks' bf16 partials (the kernels sum them in rank order in float32)."""
    return bf16(partials.sum(0))


def k2_ffn_input(inp):
    """K2 up to the router's input: XO, the reduced attention output a, the FFN seam."""
    xo = attention_xo(inp["q"], inp["kt"], inp["klen"], inp["swa"], inp["comp"], inp["swa_block"],
                      inp["comp_block"], inp["sink"], inp["pos"], inp["cos_sin"])
    a = rank_sum(wo_partials(xo, inp["woa"], inp["woa_s"], inp["wob"], inp["wob_s"]))
    sm = seam(inp["res_in"], a, inp["post_in"], inp["comb_in"], inp["pre_in"], inp["hc_fn"], inp["hc_scale"],
              inp["hc_base"], inp["ffn_norm"])
    return {"xo": xo, "a": a, **sm}


def route(h, gate_w, bias):
    """(ids [S, 6] in top-k order, weights [S, 6], scores [S, 384], selection keys [S, 384])."""
    logits = h.to(torch.float64) @ gate_w.to(torch.float64).T
    score = torch.where(logits > 20.0, logits, torch.nn.functional.softplus(logits)).sqrt()
    key = score + bias.to(torch.float64)
    order = torch.sort(key, dim=-1, descending=True, stable=True).indices
    ids = order[:, :TOP_K]
    sel = score.gather(1, ids)
    return ids, sel * (ROUTE_SCALE / sel.sum(-1, keepdim=True).clamp_min(1e-20)), score, key


def moe_out(h, inp):
    """The all-reduced MoE output [S, 5120] of the FFN input h (bf16 values): the sum of the ranks' bf16
    partials before its final bf16 rounding (the output's own rounding is the comparison's)."""
    f = torch.float64
    s = h.shape[0]
    ids, wts, _, _ = route(h, inp["gate_w"], inp["bias"])
    _, _, xr = mx_quant(h, "aiter")
    _, _, xs = mx_quant(h, "vllm")
    contrib = torch.zeros(TP, s, TOP_K, HIDDEN, dtype=f, device=h.device)
    for e in ids.unique().tolist():
        rows, ks = (ids == e).nonzero(as_tuple=True)
        w13 = mxfp4_dequant(inp["w13"][e], inp["w13_s"][e])  # [4608, 5120]
        w2 = mxfp4_dequant(inp["w2"][e], inp["w2_s"][e])  # [5120, 2304]
        hg = xr[rows] @ w13.T
        for r in range(TP):
            cols = slice(r * INTER_RANK, (r + 1) * INTER_RANK)
            g = hg[:, cols].clamp(max=LIMIT)
            u = hg[:, INTER + r * INTER_RANK:INTER + (r + 1) * INTER_RANK].clamp(-LIMIT, LIMIT)
            _, _, mid = mx_quant(bf16(g * torch.sigmoid(g) * u), "aiter")
            y = mid @ w2[:, cols].T
            contrib[r, rows, ks] = bf16(y * wts[rows, ks, None])
        del w13, w2, hg
    sg = fp8_block_dequant(inp["sgu"], inp["sgu_s"])  # [4608, 5120]: gate rows, then up rows
    sw2 = fp8_block_dequant(inp["sw2"], inp["sw2_s"])  # [5120, 2304]
    hs = xs @ sg.T
    partials = torch.empty(TP, s, HIDDEN, dtype=f, device=h.device)
    for r in range(TP):
        cols = slice(r * INTER_RANK, (r + 1) * INTER_RANK)
        g = bf16(bf16(hs[:, cols]).clamp(max=LIMIT))
        u = bf16(hs[:, INTER + r * INTER_RANK:INTER + (r + 1) * INTER_RANK]).clamp(-LIMIT, LIMIT)
        _, _, smq = mx_quant(bf16(bf16(g * torch.sigmoid(g)) * u), "vllm")
        shared = bf16(smq @ sw2[:, cols].T)
        routed = torch.zeros(s, HIDDEN, dtype=f, device=h.device)
        for k in range(TOP_K):
            routed = bf16(routed + contrib[r, :, k])
        partials[r] = bf16(routed + shared)
    return partials.sum(0)
