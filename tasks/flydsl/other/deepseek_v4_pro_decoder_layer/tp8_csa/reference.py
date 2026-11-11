# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Float64 reference of one DeepSeek-V4-Pro CSA decoder layer (decode rows), used only for correctness.

It computes the layer as sgl-amd-mono's ``dsv4_kernel`` (52486703) defines it, logically (all 128
heads, all 16 o_a groups, the full 3072-wide experts; the TP all-reduces are exact sums), keeping the
kernel's quantization and bf16 rounding points and nothing else. Notation: bf16(x) rounds to bfloat16
(nearest even); fp8q(x, B) is the activation quantization of the kernel's FP8 GEMMs over 1 x B blocks
(scale = pow2_ceil(amax / 448), codes = e4m3 RNE of clamp(x / scale, +-448), value = codes * scale);
mx8(x) = fp8q(x, 32); fp4q(x) the 1 x 32 e2m1 quantization with scale pow2_ceil(amax / 6).

  1. mHC pre (attention side): over x = hidden[s] flattened (4 x 7168): m = (fn @ x) * rsqrt(mean(x^2)
     + 1e-6); pre = sigmoid(m[0:4] * sc0 + b[0:4]) + 1e-6, post = 2 sigmoid(m[4:8] * sc1 + b[4:8]),
     comb = Sinkhorn(m[8:24] * sc2 + b[8:24] as [j, k]): softmax over k, + 1e-6, column-normalize
     (sum over j, + 1e-6), then 19 x (row-normalize, column-normalize), each sum + 1e-6.
     xin = bf16(sum_j pre[j] hidden[s, j]).
  2. Input RMSNorm xa = bf16(xin * rsqrt(mean(xin^2) + 1e-6) * g_in). q_a | kv = bf16(W_qkv_a @
     fp8q(xa, 128)); the compressors' rows c_kv | c_gate | i_kv | i_gate = W_qkv_c @ xa (bf16 in the
     state).
  3. KV row: nv = kv * rsqrt(mean(kv^2) + 1e-6) * g_kv; the RoPE tail (dims 448..511, interleaved
     pairs, position p) rotated; the NoPE dims FP8 per 64-group with scale pow2_ceil(max(amax,
     448 * 2^-126) / 448). It is written to the SWA ring row dest_rows[0, s] (codes, e8m0 bytes,
     bf16 RoPE plane), and the compressor states get row p % 8 = [c_kv | c_gate] / [i_kv | i_gate].
  4. Query: qn = q_a * rsqrt(mean(q_a^2) + 1e-6) * g_q; q = bf16(W_q_b @ fp8q(qn, 128)) per head
     RMS-normalized (no weight), RoPE tail rotated, NoPE dims FP8 round-tripped as the KV row; bf16.
  5. Indexer (CSA): iq = bf16(W_i_q_b @ fp8q(qn, 128)) [64 heads x 128]; channels 64..127 rotated
     (bf16), Hadamard H128 / sqrt(128) (bf16), fp4q. Head weights w_h = bf16(sum_k xin_k rstd g_in,k
     i_w[h, k]) * 128^-0.5 * 64^-0.5. Live compressed entries are c < (p + 1) // 4; with more than 1024
     the attention takes the top 1024 by score_c = sum_h relu(iq_h . key_c) w_h (keys: the FP4 pool,
     ties to the lower index), otherwise all of them.
  6. Sparse attention over the 128-token window (the step's own row from step 3) and the selected
     compressed rows (K = V = the stored row, decoded exactly), softmax scale 512^-0.5 with a per-head
     sink in the denominator: o = sum_j e^(l_j - M) v_j / (sum_j e^(l_j - M) + e^(sink - M)).
     o = bf16(o), its RoPE tail de-rotated, then fp8q(., 128).
  7. o_a (16 groups of 8 heads): o_lora = bf16(W_o_a[g] @ o[heads of g]); attn = bf16(W_o_b @
     fp8q(o_lora, 128)) (the exact TP sum). a[k] = bf16(post[k] attn + sum_j comb[j, k] hidden[j]).
  8. mHC pre (FFN side) on a: x = bf16(sum_j pre_f[j] a[j]); xn = bf16(x * rsqrt(mean(x^2) + 1e-6)
     * g_post). Router: score = sqrt(softplus(bf16(w_r @ xn))); the top 6 of score + bias (ties to the
     lower id), weights score / sum * 2.5.
  9. Experts on xn: routed (MXFP4 weights on mx8(xn)) and shared (FP8 on fp8q(xn, 128)); mid =
     SwiGLU with gate min(g, 10) and up clamp(u, +-10); m = bf16(mid); the down input is mx8(m) for
     the shared expert and picks 1..5, fp8q(m, 128) for the 6th pick (the kernel's slot 6). ffn =
     bf16(sum_k w_k W_dn[e_k] @ m_k + W_sdn @ m_sh).
 10. out[k] = post_f[k] ffn + sum_j comb_f[j, k] a[j] (unrounded).

Mutable outputs are returned before the kernel's own rounding: the NoPE codes of the written KV row as
clamp(nv / scale, +-448) (kv_cache is compared as e4m3 values; its e8m0 bytes are exact), the RoPE
plane unrounded, the compressor state rows as the unrounded projections.
"""
import math

import torch

HIDDEN = 7168
HC = 4
Q_LORA = 1536
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = 448
HEADS = 128
O_GROUPS = 16
O_LORA = 1024
N_EXPERTS = 384
TOP_K = 6
INTER = 3072
IH = 64
IHD = 128
WINDOW = 128
CR = 4
C_RING = 8
K_PB = 64
INDEX_TOPK = 1024
N_COMP = (1 << 20) // CR  # the build's compressed slots (max_seq 2^20)
EPS = 1e-6
HC_EPS = 1e-6
SINKHORN_ITERS = 20
ROUTE_SCALE = 2.5
SWIGLU_LIMIT = 10.0
FP8_MAX = 448.0
FP4_MAX = 6.0
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
SCORE_CHUNK = 1 << 16


def bf16(x):
    return x.to(torch.bfloat16).to(torch.float64)


def _pow2_ceil(x):
    mant, exp = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), torch.where(mant == 0.5, exp - 1, exp))


def _fp8_codes(x):
    """e4m3 RNE of x (already within +-448), as float64."""
    return x.to(torch.float32).to(torch.float8_e4m3fn).to(torch.float64)


def fp8q(x, block, floor=0.0):
    """The kernel's FP8 activation quantization over 1 x ``block`` blocks of the last axis, dequantized:
    (value, pre-rounding codes, scale per block)."""
    shape = x.shape
    xb = x.reshape(*shape[:-1], shape[-1] // block, block)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(floor)
    scale = torch.where(amax > 0, _pow2_ceil(amax / FP8_MAX), torch.ones_like(amax))
    q = (xb / scale).clamp(-FP8_MAX, FP8_MAX)
    return (_fp8_codes(q) * scale).reshape(shape), q.reshape(shape), scale.squeeze(-1)


def kv_quant(x):
    """The KV row's / query's NoPE FP8 round trip per 64-group (scale floor 448 * 2^-126)."""
    return fp8q(x, 64, floor=FP8_MAX * 2.0 ** -126)


def _e2m1_rne(q):
    m = q.abs()
    idx = ((m > 0.25).long() + (m >= 0.75).long() + (m > 1.25).long() + (m >= 1.75).long()
           + (m > 2.5).long() + (m >= 3.5).long() + (m > 5.0).long())
    grid = torch.tensor(_E2M1[:8], dtype=torch.float64, device=q.device)
    return torch.copysign(grid[idx], q)


def fp4q(x):
    """1 x 32 e2m1 quantization with scale pow2_ceil(max(amax, 6 * 2^-126) / 6), dequantized."""
    shape = x.shape
    xb = x.reshape(*shape[:-1], shape[-1] // 32, 32)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(FP4_MAX * 2.0 ** -126)
    scale = _pow2_ceil(amax / FP4_MAX)
    return (_e2m1_rne((xb / scale).clamp(-FP4_MAX, FP4_MAX)) * scale).reshape(shape)


def rms(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)


def rope(x, cos, sin, inverse=False):
    """Interleaved rotation of the pairs (2i, 2i + 1) of x [..., 2n] by cos / sin [..., n]."""
    xe, xo = x[..., 0::2], x[..., 1::2]
    if inverse:
        sin = -sin
    out = torch.empty_like(x)
    out[..., 0::2] = xe * cos - xo * sin
    out[..., 1::2] = xo * cos + xe * sin
    return out


def _hadamard(n, device):
    h = torch.ones(1, 1, dtype=torch.float64, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def fp8_matrix(codes, scales):
    """FP8 codes [N, K] with 128 x 128 block scales [N / 128, K / 128] -> float64 [N, K]."""
    s = scales.double().repeat_interleave(128, 0).repeat_interleave(128, 1)
    return codes.double() * s


def mxfp4_matrix(codes, scales):
    """Packed e2m1 [N, K / 2] (low nibble = even k) and e8m0 [N, K / 32] -> float64 [N, K]."""
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=codes.device)
    v = torch.stack((lut[(codes & 0xF).long()], lut[(codes >> 4).long()]), -1).reshape(codes.shape[0], -1)
    return v * torch.exp2(scales.double() - 127.0).repeat_interleave(32, -1)


def hc_coefficients(x, fn, sb):
    """mHC pre / post / comb of x [S, 4, H] with the mixer fn [24, 4H] and sb = [3 scales | 24 bases]."""
    xf = x.reshape(x.shape[0], -1)
    sb = sb.double()
    m = (xf @ fn.double().T) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    pre = torch.sigmoid(m[:, :HC] * sb[0] + sb[3:3 + HC]) + HC_EPS
    post = 2.0 * torch.sigmoid(m[:, HC:2 * HC] * sb[1] + sb[3 + HC:3 + 2 * HC])
    cb = (m[:, 2 * HC:] * sb[2] + sb[3 + 2 * HC:]).reshape(-1, HC, HC)  # [s, j, k]
    c = torch.softmax(cb, dim=-1) + HC_EPS
    c = c / (c.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(SINKHORN_ITERS - 1):
        c = c / (c.sum(-1, keepdim=True) + HC_EPS)
        c = c / (c.sum(-2, keepdim=True) + HC_EPS)
    return pre, post, c


def decode_kv_rows(kv_cache, kv_rope, rows):
    """KV rows (K = V) [n, 512] of the unified pool, decoded as the kernel reads them."""
    raw = kv_cache.view(torch.uint8)[rows]
    codes = raw[:, :NOPE_DIM].contiguous().view(torch.float8_e4m3fn).to(torch.float64)
    groups = torch.arange(NOPE_DIM // 64, device=raw.device)
    e8 = raw[:, NOPE_DIM + 4 * (groups // 2) + 2 * (groups % 2)].to(torch.float64)
    nope = codes * torch.exp2(e8 - 127.0).repeat_interleave(64, -1)
    return torch.cat([nope, kv_rope[rows].to(torch.float64)], -1)


def decode_index_keys(i_cache, i_cache_s, pages, entries):
    """FP4 index keys [n, 128] of entries ``entries`` living in pages ``pages``."""
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=i_cache.device)
    codes = i_cache.view(-1, 4, K_PB, 16)[pages, :, entries]  # [n, 4, 16]
    vals = torch.stack((lut[(codes & 0xF).long()], lut[(codes >> 4).long()]), -1).reshape(-1, 4, 32)
    slot = (entries % 16) * 4 + entries // 16
    e8 = i_cache_s.view(-1, 4, K_PB)[pages, :, slot].to(torch.float64)  # [n, 4]
    return (vals * torch.exp2(e8 - 127.0)[..., None]).reshape(-1, IHD)


def _select_entries(p, iq, wp, i_cache, i_cache_s, block_table):
    """Compressed entries the attention gathers for a row at position p."""
    n_live = min((p + 1) // CR, N_COMP)
    entries = torch.arange(n_live, device=iq.device)
    if n_live <= INDEX_TOPK:
        return entries
    scores = torch.empty(n_live, dtype=torch.float64, device=iq.device)
    for c0 in range(0, n_live, SCORE_CHUNK):
        e = entries[c0:c0 + SCORE_CHUNK]
        keys = decode_index_keys(i_cache, i_cache_s, block_table[e // K_PB].long(), e % K_PB)
        scores[c0:c0 + SCORE_CHUNK] = (torch.relu(keys @ iq.T) * wp).sum(-1)
    # descending score, ties to the lower index (a stable sort of the negated scores)
    order = torch.sort(-scores, stable=True).indices
    return order[:INDEX_TOPK].sort().values


def _attend(q, keys, sink):
    """Softmax attention of q [H, 512] over keys [n, 512] (K = V) with a per-head sink [H]."""
    logits = (q @ keys.T) * HEAD_DIM ** -0.5
    m = torch.maximum(logits.amax(-1), sink)
    p = torch.exp(logits - m[:, None])
    return (p @ keys) / (p.sum(-1) + torch.exp(sink - m))[:, None]


def swiglu(g, u):
    g = g.clamp(max=SWIGLU_LIMIT)
    u = u.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    return g * torch.sigmoid(g) * u


def reference(*, hidden, positions, kv_cache, kv_rope, dest_rows, indices, rope_cos, rope_sin, g_in, g_q,
              g_kv, g_post, attn_sink, ape, g_ckv, c_state, i_ape, g_ickv, i_state, i_cache, hc_attn_fn,
              hc_attn_sb, hc_ffn_fn, hc_ffn_sb, w_qkv_a, s_qkv_a, w_qkv_c, w_q_b, s_q_b, w_i_q_b, s_i_q_b,
              i_w, w_o_a, s_o_a, w_o_b, s_o_b, w_r, bias, w_ug, s_ug, w_dn, s_dn, w_sug, s_sug, w_sdn, s_sdn,
              epoch, state_slots, block_tables, i_cache_s):
    dev = hidden.device
    S = hidden.shape[0]
    pos = positions.long()
    if bool(((pos + 1) % CR == 0).any()):
        raise ValueError("a row ends a compressor window: the task's workloads exclude compressor boundaries")
    h = hidden.double()
    cos = rope_cos.double()[pos]
    sin = rope_sin.double()[pos]

    # 1-2. mHC pre, input RMSNorm, the fused q_a / kv and compressor projections
    pre_a, post_a, comb_a = hc_coefficients(h, hc_attn_fn, hc_attn_sb)
    xin = bf16((pre_a[:, :, None] * h).sum(1))
    rstd = torch.rsqrt(xin.square().mean(-1, keepdim=True) + EPS)
    xa = bf16(xin * rstd * g_in.double())
    qkv = bf16(fp8q(xa, 128)[0] @ fp8_matrix(w_qkv_a, s_qkv_a).T)
    q_a, kv_a = qkv[:, :Q_LORA], qkv[:, Q_LORA:]
    comp = xa @ w_qkv_c.double().T  # [S, 2560]: c_kv | c_gate | i_kv | i_gate

    # 3. the KV row and the compressor state rows of this step
    nv = rms(kv_a) * g_kv.double()
    rot = rope(nv[:, NOPE_DIM:], cos, sin)
    dq, codes, scale = kv_quant(nv[:, :NOPE_DIM])
    kv_row = torch.cat([dq, bf16(rot)], -1)  # what the attention reads for the step's own token
    e8 = (torch.log2(scale).round() + 127).to(torch.uint8)
    pad = torch.cat([e8, torch.zeros_like(e8[:, :1])], -1).repeat_interleave(2, -1)  # [S, 16]
    kv_cache_out = kv_cache.to(torch.float32)
    kv_rope_out = kv_rope.to(torch.float32)
    swa = dest_rows[0].long()
    kv_cache_out[swa, :NOPE_DIM] = codes.float()
    kv_cache_out[swa, NOPE_DIM:NOPE_DIM + 16] = pad.view(torch.float8_e4m3fn).float()
    kv_rope_out[swa] = rot.float()
    slot = state_slots.long()
    ring = pos % C_RING
    cw, iw = 2 * HEAD_DIM, 2 * IHD
    c_state_out = c_state.clone()
    i_state_out = i_state.clone()
    c_state_out[slot, ring] = comp[:, :2 * cw].float()
    i_state_out[slot, ring] = comp[:, 2 * cw:].float()

    # 4. the query
    qa8 = fp8q(rms(q_a) * g_q.double(), 128)[0]
    q = rms(bf16(qa8 @ fp8_matrix(w_q_b, s_q_b).T).view(S, HEADS, HEAD_DIM))
    q_rot = rope(q[..., NOPE_DIM:], cos[:, None], sin[:, None])
    q = bf16(torch.cat([kv_quant(q[..., :NOPE_DIM])[0], q_rot], -1))

    # 5. the lightning indexer's query and head weights
    iq = bf16(qa8 @ fp8_matrix(w_i_q_b, s_i_q_b).T).view(S, IH, IHD)
    iq = torch.cat([iq[..., :IHD // 2], bf16(rope(iq[..., IHD // 2:], cos[:, None], sin[:, None]))], -1)
    iq = fp4q(bf16(iq @ _hadamard(IHD, dev)))
    wp = bf16((xin * rstd * g_in.double()) @ i_w.double().T) * IHD ** -0.5 * IH ** -0.5

    # 6. sparse attention with the sink, merge, inverse RoPE and wo_a's FP8 input
    o = torch.empty(S, HEADS, HEAD_DIM, dtype=torch.float64, device=dev)
    for s in range(S):
        win = indices[s, :WINDOW].long()
        live = win >= 0
        own = win == swa[s]
        old = win[live & ~own]
        keys = [decode_kv_rows(kv_cache, kv_rope, old), kv_row[s:s + 1]]
        sel = _select_entries(int(pos[s]), iq[s], wp[s], i_cache, i_cache_s, block_tables[s])
        if sel.numel():
            comp_rows = dest_rows[1, s].long() + block_tables[s][sel // K_PB].long() * K_PB + sel % K_PB
            keys.append(decode_kv_rows(kv_cache, kv_rope, comp_rows))
        o[s] = _attend(q[s], torch.cat(keys), attn_sink.double())
    o = bf16(o)
    o = torch.cat([o[..., :NOPE_DIM], rope(o[..., NOPE_DIM:], cos[:, None], sin[:, None], inverse=True)], -1)
    o = fp8q(o, 128)[0].reshape(S, O_GROUPS, -1)

    # 7. grouped o_a, o_b (the exact TP sum) and the attention-side mHC post
    w_oa = fp8_matrix(w_o_a, s_o_a).view(O_GROUPS, O_LORA, -1)
    o_lora = bf16(torch.einsum("sgk,gnk->sgn", o, w_oa)).reshape(S, O_GROUPS * O_LORA)
    attn = bf16(fp8q(o_lora, 128)[0] @ fp8_matrix(w_o_b, s_o_b).T)
    a = bf16(post_a[:, :, None] * attn[:, None] + torch.einsum("sjk,sjh->skh", comb_a, h))

    # 8. FFN-side mHC pre, post-attention RMSNorm, router
    pre_f, post_f, comb_f = hc_coefficients(a, hc_ffn_fn, hc_ffn_sb)
    x = bf16((pre_f[:, :, None] * a).sum(1))
    xn = bf16(rms(x) * g_post.double())
    logit = bf16(xn @ w_r.double().T)
    score = torch.sqrt(logit.clamp_min(0.0) + torch.log1p(torch.exp(-logit.abs())))
    keyed = score + bias.double()
    # descending key, ties to the lower id
    picks = torch.stack([torch.sort(-keyed[s], stable=True).indices[:TOP_K] for s in range(S)])
    raw = score.gather(1, picks)
    weights = raw / raw.sum(-1, keepdim=True).clamp_min(1e-20) * ROUTE_SCALE

    # 9. experts
    x_mx = fp8q(xn, 32)[0]
    x_sh = fp8q(xn, 128)[0]
    ffn = torch.zeros(S, HIDDEN, dtype=torch.float64, device=dev)
    for e in torch.unique(picks).tolist():
        w13 = mxfp4_matrix(w_ug[e], s_ug[e])
        w2 = mxfp4_matrix(w_dn[e], s_dn[e])
        for s, k in (picks == e).nonzero().tolist():
            hgu = w13 @ x_mx[s]
            m = bf16(swiglu(hgu[:INTER], hgu[INTER:]))
            m = fp8q(m, 128)[0] if k == TOP_K - 1 else fp8q(m, 32)[0]
            ffn[s] += weights[s, k] * (w2 @ m)
        del w13, w2
    w_sh = fp8_matrix(w_sug, s_sug)
    hgu = x_sh @ w_sh.T
    m = fp8q(bf16(swiglu(hgu[:, :INTER], hgu[:, INTER:])), 32)[0]
    ffn = bf16(ffn + m @ fp8_matrix(w_sdn, s_sdn).T)

    # 10. the FFN-side mHC post
    out = post_f[:, :, None] * ffn[:, None] + torch.einsum("sjk,sjh->skh", comb_f, a)
    return {"out": out, "kv_cache": kv_cache_out, "kv_rope": kv_rope_out, "c_state": c_state_out,
            "i_state": i_state_out}
