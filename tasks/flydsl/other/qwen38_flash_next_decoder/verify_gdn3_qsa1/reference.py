# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""float64 reference of one Qwen3.8-Flash-Next decoder launch: layers gdn, gdn, gdn, qsa of a NEXTN
verify step (R requests x 4 tokens; row t = request t // 4, draft step t % 4), used only for
correctness.

Every layer l: H1 = H + attn(mix_a(H)) * gate_a(H); H2 = H1 + moe(mix_m(H1)) * gate_m(H1), on the 4
hyper-connection residual branches H[t, b, :] (b < 4, 2560 each).

* mix(H): rstd[t, b] = (mean(H[t, b]^2) + 1e-6)^-1/2, Hn = H rstd (1 + norm), x = mean_b
  sigmoid(W_up silu(W_down Hn / 4))_b * Hn_b, gate[t, b] = 2 sigmoid(W_inj[b] . Hn / 4).
* W8A8 projections (in_proj, in_proj_ba, out_proj, qkv_proj, index_qk_proj, o_proj): per-token
  activation scale a = max|x| / 448, codes float8_e4m3fn(x / a) (round to nearest even), out =
  (codes @ W_e4m3^T) * w_scale[n] * a.
* GDN: q|k|v = silu(causal depthwise conv1d (4 taps) over the committed conv state (3) and the
  request's tokens); q, k l2-normalized (eps 1e-6 inside the sqrt), q scaled by 128^-1/2; beta =
  sigmoid(b), decay = exp(A * softplus(a + dt_bias)) (softplus linear above 20); per value head h
  (key head h // 3), state S[v, k] from the request's committed SSM state, token by token: vnew =
  beta (v - decay S k), o = decay S q + vnew (k . q), S = decay S + vnew k^T (written after every
  token); og = o rmsnorm(o) w_norm sigmoid(z); out = W8A8 out_proj(og).
* QSA: q / k / index q / index k Gemma RMS norms ((1 + w), eps 1e-6), NeoX RoPE on dims 0..63 from the
  bf16 cos / sin table; the step's raw index keys enter the request's 8-slot ring (slot pos % 8,
  with their positions); the group of 4 positions the step completes is compressed (mean of the raw
  keys, index-k norm, RoPE at the group's first position); index logits of every complete block b of
  row t: sum over the 4 index heads of relu(q_i . key_b) / sqrt(128); the exact top-512 blocks (all
  of them when there are fewer) plus the row's incomplete block positions are attended (softmax of
  q . k / 16, causal inside the request, GQA 12 query heads a KV head), times sigmoid(gate); out =
  W8A8 o_proj.
* MoE: logits = x @ router^T; top-10 of the 512 routed logits with softmax weights over the 10, plus
  the shared expert (index 512) weighted sigmoid(its gate logit); expert(x) = (silu(x G^T) * (x U^T))
  D^T with MXFP4 weights (e2m1 x 2^(e8m0 - 127) per 32).

Outputs: h_out, its sum-of-squares partials [t, b, 160] (16 consecutive elements each), the GDN
verify scratch (SSM state after every token, conv window after every token as [R, 10240, 6]: the
committed state's last 2 and the 4 conv inputs), the new K / V rows, the completed compressed index
key, and the index-key ring with its positions (mutable).
"""
import math

import torch

HIDDEN, HC, HC_DIM, LOWRANK = 2560, 4, 10240, 320
L = 4
EPS = 1e-6
N_EXP, TOPK, INTER, INTER_PAD = 512, 10, 640, 768
GDN_HK, GDN_HV, GDN_DK = 16, 48, 128
RING = 8
TOPB = 512
ROPE_DIM = 64
EXPERT_CHUNK = 8
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _silu(x):
    return x * torch.sigmoid(x)


def _w8a8(x, w, scale):
    """W8A8 FP8 GEMM with per-token dynamic activation quantization (AITER semantics)."""
    a = x.abs().amax(-1).clamp_min(1e-12) / 448.0
    codes = (x / a[:, None]).float().to(torch.float8_e4m3fn).double()
    return (codes @ w.double().T) * scale.double() * a[:, None]


def _mix(H, norm, down, up):
    """Hyper-connection mixer: (block input x [S, 2560], combine gates [S, 4])."""
    S = H.shape[0]
    rstd = torch.rsqrt(H.square().mean(-1) + EPS)
    Hn = H * rstd[..., None] * (1.0 + norm.double().view(HC, HIDDEN))
    hf = Hn.reshape(S, HC_DIM)
    down = down.double()
    d = hf @ down[:LOWRANK].T / HC
    inj = hf @ down[LOWRANK:LOWRANK + HC].T / HC
    u = (_silu(d) @ up.double().T).view(S, HC, HIDDEN)
    x = (torch.sigmoid(u) * Hn).mean(1)
    return x, 2.0 * torch.sigmoid(inj)


def _gemma(x, w):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * (1.0 + w.double())


def _rope(x, cos, sin):
    """NeoX RoPE of dims 0..63 (pairs i, i + 32); cos / sin [..., 32] broadcast over heads."""
    x = x.clone()
    a, b = x[..., :32].clone(), x[..., 32:ROPE_DIM].clone()
    x[..., :32] = a * cos - b * sin
    x[..., 32:ROPE_DIM] = b * cos + a * sin
    return x


def _softplus(x):
    return torch.where(x > 20.0, x, torch.log1p(torch.exp(torch.clamp(x, max=20.0))))


def _gdn(x, g, p, req_ms, R):
    """One GDN layer -> (y [S, 2560], states [R, 4, 48, 128, 128], conv windows [R, 10240, 6])."""
    S = x.shape[0]
    ip = _w8a8(x, p["gdn_in_proj"][g], p["gdn_in_proj_scale"][g])
    ba = _w8a8(x, p["gdn_ba_proj"][g][:2 * GDN_HV], p["gdn_ba_proj_scale"][g][:2 * GDN_HV])
    mixed, z = ip[:, :HC_DIM], ip[:, HC_DIM:].view(S, GDN_HV, GDN_DK)
    b, a = ba[:, :GDN_HV], ba[:, GDN_HV:]
    cw = p["gdn_conv_weight"][g].double()  # [10240, 4]
    conv_state = p["gdn_conv_state"][g]
    ssm_state = p["gdn_ssm_state"][g]
    act = torch.empty(S, HC_DIM, dtype=torch.float64, device=x.device)
    windows = torch.empty(R, HC_DIM, L + 2, dtype=torch.float64, device=x.device)
    for r in range(R):
        win = torch.cat([conv_state[req_ms[r]].double().T, mixed[r * L:(r + 1) * L]])  # [7, 10240]
        for j in range(L):
            act[r * L + j] = _silu((win[j:j + 4] * cw.T).sum(0))
        windows[r] = win[1:].T
    q = act[:, :2048].view(S, GDN_HK, GDN_DK)
    k = act[:, 2048:4096].view(S, GDN_HK, GDN_DK)
    v = act[:, 4096:].view(S, GDN_HV, GDN_DK)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * GDN_DK ** -0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    kh = torch.arange(GDN_HV, device=x.device) // (GDN_HV // GDN_HK)
    q, k = q[:, kh], k[:, kh]  # [S, 48, 128]
    beta = torch.sigmoid(b)
    decay = torch.exp(p["gdn_neg_a"][g].double() * _softplus(a + p["gdn_dt_bias"][g].double()))
    o = torch.empty(S, GDN_HV, GDN_DK, dtype=torch.float64, device=x.device)
    states = torch.empty(R, L, GDN_HV, GDN_DK, GDN_DK, dtype=torch.float64, device=x.device)
    for r in range(R):
        st = ssm_state[req_ms[r]].double()  # [48, v, k]
        for j in range(L):
            t = r * L + j
            gt = decay[t][:, None]
            sk = torch.einsum("hvk,hk->hv", st, k[t])
            sq = torch.einsum("hvk,hk->hv", st, q[t])
            vnew = (v[t] - gt * sk) * beta[t][:, None]
            o[t] = gt * sq + vnew * (k[t] * q[t]).sum(-1, keepdim=True)
            st = gt[..., None] * st + vnew[:, :, None] * k[t][:, None, :]
            states[r, j] = st
    og = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + EPS) * p["gdn_norm_weight"][g].double()
    og = (og * torch.sigmoid(z)).reshape(S, GDN_HV * GDN_DK)
    return _w8a8(og, p["gdn_out_proj"][g], p["gdn_out_proj_scale"][g]), states, windows


def _qsa(x, p, req, R):
    """The QSA layer -> (y, k_new [S, 2, 256], v_new, index_key_new [R, 128], ring, ring positions)."""
    S = x.shape[0]
    dev = x.device
    qp = torch.cat([_w8a8(x, p["qsa_qkv_proj"], p["qsa_qkv_proj_scale"]),
                    _w8a8(x, p["qsa_index_proj"], p["qsa_index_proj_scale"])], -1)
    heads = qp[:, :24 * 512].view(S, 24, 512)
    q, gate = heads[..., :256], heads[..., 256:]
    k = qp[:, 12288:12800].view(S, 2, 256)
    v = qp[:, 12800:13312].view(S, 2, 256)
    iq = qp[:, 13312:13824].view(S, 4, 128)
    ik = qp[:, 13824:13952]
    table = p["rope_cos_sin"].double()
    pos = p["positions"].long()
    cos, sin = table[pos, :32], table[pos, 32:ROPE_DIM]  # [S, 32]
    qr = _rope(_gemma(q, p["qsa_q_norm"]), cos[:, None], sin[:, None])
    kr = _rope(_gemma(k, p["qsa_k_norm"]), cos[:, None], sin[:, None])
    iqr = _rope(_gemma(iq, p["qsa_index_q_norm"]), cos[:, None], sin[:, None])
    ring_in = p["index_ring"].double()
    rpos_in = p["index_ring_positions"]
    ring = ring_in.clone()
    rpos = rpos_in.clone()
    rtt = p["req_to_token"]
    k_cache, v_cache, keys = p["k_cache"], p["v_cache"], p["index_keys"]
    seq = p["seq_lens"].long().tolist()
    comp_new = torch.empty(R, 128, dtype=torch.float64, device=dev)
    og = torch.empty(S, 24, 256, dtype=torch.float64, device=dev)
    for r in range(R):
        c, slot = seq[r], int(req[r])
        for j in range(L):
            ring[slot * RING + (c + j) % RING] = ik[r * L + j]
            rpos[slot * RING + (c + j) % RING] = pos[r * L + j]
        # the group of 4 positions the step completes: members from the step or the input ring
        pc = next(c + j for j in range(L) if (c + j + 1) % 4 == 0)
        ps = pc - 3
        members = [ik[r * L + m - c] if m >= c else ring_in[slot * RING + m % RING] for m in range(ps, pc + 1)]
        p_ps = int(pos[r * L + ps - c]) if ps >= c else int(rpos_in[slot * RING + ps % RING, 0])
        comp = _gemma(torch.stack(members).mean(0), p["qsa_index_k_norm"])
        comp_new[r] = _rope(comp, table[p_ps, :32], table[p_ps, 32:ROPE_DIM])
        gfirst = c // 4
        trow = rtt[slot].long()
        old_keys = keys[trow[torch.arange(gfirst, device=dev) * 4] // 4].double()  # [gfirst, 128]
        all_keys = torch.cat([old_keys, comp_new[r:r + 1]])
        for j in range(L):
            t = r * L + j
            pt = c + j
            nb = (pt + 1) // 4
            if nb > TOPB:
                logits = torch.relu(iqr[t] @ all_keys[:nb].T).sum(0) / math.sqrt(128)
                blocks = torch.topk(logits, TOPB).indices
            else:
                blocks = torch.arange(nb, device=dev)
            toks = (blocks[:, None] * 4 + torch.arange(4, device=dev)).reshape(-1)
            toks = torch.cat([toks, torch.arange(nb * 4, pt + 1, device=dev)])
            # context tokens from the caches, the step's own tokens (positions >= c) from this layer
            fresh = toks >= c
            loc = trow[torch.where(fresh, torch.zeros_like(toks), toks)]
            kk, vv = k_cache[loc].double(), v_cache[loc].double()  # [n, 2, 256]
            kk[fresh] = kr[r * L + toks[fresh] - c]
            vv[fresh] = v[r * L + toks[fresh] - c]
            kvh = torch.arange(24, device=dev) // 12
            scores = torch.einsum("hd,nhd->hn", qr[t], kk[:, kvh]) / 16.0
            probs = torch.softmax(scores, -1)
            og[t] = torch.einsum("hn,nhd->hd", probs, vv[:, kvh]) * torch.sigmoid(gate[t])
    y = _w8a8(og.reshape(S, 24 * 256), p["qsa_o_proj"], p["qsa_o_proj_scale"])
    return y, kr, v, comp_new, ring, rpos


def _mxfp4(codes, scales, k):
    """Packed e2m1 codes [..., N, >= k / 2] and e8m0 bytes [..., N, >= k / 32] -> float64 [..., N, k]."""
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=codes.device)
    c = codes[..., :k // 2].long()
    vals = torch.stack((lut[c & 0xF], lut[c >> 4]), -1).flatten(-2)
    s = torch.exp2(scales[..., :k // 32].double() - 127.0).repeat_interleave(32, -1)
    return vals * s


def _moe(x, li, p):
    S = x.shape[0]
    logits = x @ p["router"][li][:N_EXP + 1].double().T
    top, ids = torch.topk(logits[:, :N_EXP], TOPK, -1)
    wts = torch.cat([torch.softmax(top, -1), torch.sigmoid(logits[:, N_EXP:])], -1)
    ids = torch.cat([ids, torch.full_like(ids[:, :1], N_EXP)], -1)
    y = torch.zeros(S, HIDDEN, dtype=torch.float64, device=x.device)
    w13, s13, w2, s2 = (p[n][li] for n in ("moe_w13", "moe_w13_scale", "moe_w2", "moe_w2_scale"))
    experts = ids.unique()
    for c0 in range(0, experts.numel(), EXPERT_CHUNK):
        chunk = experts[c0:c0 + EXPERT_CHUNK]
        gate = _mxfp4(w13[chunk, :INTER], s13[chunk, :INTER], HIDDEN)
        up = _mxfp4(w13[chunk, INTER_PAD:INTER_PAD + INTER], s13[chunk, INTER_PAD:INTER_PAD + INTER], HIDDEN)
        down = _mxfp4(w2[chunk], s2[chunk], INTER)
        for i, e in enumerate(chunk.tolist()):
            rows, slot = torch.where(ids == e)
            xe = x[rows]
            mid = _silu(xe @ gate[i].T) * (xe @ up[i].T)
            y.index_add_(0, rows, wts[rows, slot][:, None] * (mid @ down[i].T))
    return y


def reference(**p):
    h_in = p["h_in"]
    S = h_in.shape[0]
    R = S // L
    req = p["req_pool_indices"].long()
    req_ms = p["mamba_slot_map"].long()[req].tolist()
    H = h_in.double().view(S, HC, HIDDEN)
    ssm, conv = [], []
    for li in range(4):
        x, gate = _mix(H, p["hc_attn_norm"][li], p["hc_attn_down"][li], p["hc_attn_up"][li])
        if li < 3:
            y, st, win = _gdn(x, li, p, req_ms, R)
            ssm.append(st)
            conv.append(win)
        else:
            y, k_new, v_new, key_new, ring, rpos = _qsa(x, p, req, R)
        H = H + y[:, None, :] * gate[:, :, None]
        x, gate = _mix(H, p["hc_mlp_norm"][li], p["hc_mlp_down"][li], p["hc_mlp_up"][li])
        H = H + _moe(x, li, p)[:, None, :] * gate[:, :, None]
    return {
        "h_out": H.reshape(S, HC_DIM),
        "h_out_ssq": H.view(S, HC, HIDDEN // 16, 16).square().sum(-1),
        "gdn_intermediate_ssm": torch.stack(ssm),
        "gdn_intermediate_conv": torch.stack(conv),
        "k_new": k_new,
        "v_new": v_new,
        "index_key_new": key_new,
        "index_ring": ring,
        "index_ring_positions": rpos,
    }
