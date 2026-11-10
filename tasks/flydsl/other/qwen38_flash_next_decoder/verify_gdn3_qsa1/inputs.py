# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Inputs of one Qwen3.8-Flash-Next decoder launch (3 GDN + 1 QSA layers) of a NEXTN verify step,
and the kernel-facing layout of its weights.

Random weights in SGLang's post-load gfx950 layouts (as sgl-amd-mono's
test/kernels/test_qwen38_flash_next_launches.py draws them, no checkpoint), the caches and states of R
requests at their contexts, and planted structure that keeps the launch's two discrete decisions
unambiguous between the float64 reference and any correct kernel:

* Router top-10 of 512. The residual streams carry 10 large "routing code" channels per row (in all
  four branches), drawn from a set of 512 code channels; each layer's router row of expert e has one
  large weight on code channel pi_layer(e) and no weight on the other code channels. A row's routed
  experts are then the 10 whose code channels it carries: their logits sit near 5, every other
  logit is a few tenths, a margin far beyond any rounding difference. The codes differ per row, so
  the 32 rows of an 8-request step route to about 200-300 distinct experts as in production.
* QSA top-512 block selection. Channel d0 of the streams is a large shared ("massive") activation;
  index-q heads 1 and 3 are the negations of heads 0 and 2 (so a block's logit is
  (|q0 . k| + |q2 . k|) / sqrt(128)), and the index-q weight of channel d0 is a known direction in
  the 64 non-rotated dims. Each request's compressed keys are planted relative to that direction:
  512 winners (cosine 0.75..0.95 to it) and losers (|cosine| <= 0.3), all of equal norm; the
  in-batch block's key, computed by the launch from the index-k projection (aligned with a direction
  orthogonal to the index queries), is a loser too. With 512 or fewer complete blocks every block is
  selected.

Logical inputs (what reference.py reads) have the kernel-facing shapes; only the storage order of the
packed weights differs (to_kernel_layout):
* FP8 projections [N, K] (E4M3, per-output-channel f32 scales): AITER shuffle_weight(w, (16, 16)).
* bf16 mixer / router weights [N, K]: sgl-amd-mono packing.pack_bf16 (16-row x 64-K MFMA tiles).
  hc_*_down holds the mixer down rows (0..319), the 4 combine-gate (block_inject) rows (320..323) and
  12 zero rows, as packing.pack_hc_down builds it; router holds the 512 gate rows, the shared-expert
  gate (row 512) and 15 zero rows (packing.pack_router).
* MXFP4 experts (513 = 512 routed + the shared expert at index 512): FusedMoE w13 [E, 1536, 1280]
  (gate rows 0..639, up rows 768..1407, zero rows elsewhere; e2m1 codes two per byte along K, low
  nibble first), w2 [E, 2560, 384] (K bytes 320..383 zero), e8m0 scales [E, 1536, 80] / [E, 2560, 24]
  (pad rows / groups 127); kernel-facing: shuffle_weight(w, (16, 16)) and e8m0_shuffle over the
  flattened [E * rows, groups] scales, as SGLang's AITER MoE keeps them.
"""
import math

import torch

HIDDEN, HC, HC_DIM, LOWRANK = 2560, 4, 10240, 320
L = 4
N_EXP, TOPK, INTER, INTER_PAD = 512, 10, 640, 768
GDN_HV, GDN_DK = 48, 128
N_GDN = 3
N_LAYERS = 4
RING = 8
NPART = HIDDEN // 16
QP_QKV = 24 * 512 + 512 + 512  # qkv_proj rows: [q_h | gate_h] x 24, k (2 x 256), v (2 x 256)
ROPE_THETA = 1.0e7
ROPE_DIM = 64
KV_PAGE = 64

# planted structure
MASSIVE = 24.0  # channel d0 of every stream
CODE = 16.0  # a row's 10 routing-code channels
ROUTER_CODE_WEIGHT = 1.0
INDEX_COLUMN = 24.0  # index q / k weight of channel d0 (FP8 units, per-channel scale K^-1/2)
WIN_COS = (0.75, 0.95)
LOSE_COS = 0.3
N_WINNERS = 512


def _randn(shape, g, device, std=1.0, mean=0.0):
    x = torch.randn(shape, generator=g, device=device, dtype=torch.float32)
    if std != 1.0:
        x.mul_(std)
    if mean:
        x.add_(mean)
    return x


def _uniform(shape, g, device, lo, hi):
    return torch.rand(shape, generator=g, device=device, dtype=torch.float32).mul_(hi - lo).add_(lo)


def _bf16_randn(shape, g, device, std=1.0, chunk=1 << 26):
    """Normal bf16 values drawn in chunks (bounded float32 temporaries for the KV caches)."""
    out = torch.empty(shape, dtype=torch.bfloat16, device=device)
    flat = out.view(-1)
    for s in range(0, flat.numel(), chunk):
        n = min(chunk, flat.numel() - s)
        flat[s:s + n] = _randn((n,), g, device, std).to(torch.bfloat16)
    return out


def _fp8_proj(rows, k, g, device, out_scale=1.0):
    """W8A8 projection: N(0, 1) E4M3 codes, per-channel f32 scale out_scale K^-1/2 e^(0.2 n)."""
    w = _randn((rows, k), g, device).to(torch.float8_e4m3fn)
    s = (out_scale / math.sqrt(k)) * torch.exp(0.2 * _randn((rows,), g, device))
    return w, s


def _unit(v):
    return v / v.norm(dim=-1, keepdim=True)


def _orthogonal(v, basis):
    """Rows of v with their components along the orthonormal rows of basis removed."""
    return v - (v @ basis.T) @ basis


def _mxfp4_experts(g, device):
    """One layer's FusedMoE w13 / w2 (random e2m1 codes) and e8m0 scales, padded as SGLang's AITER MoE."""
    E = N_EXP + 1
    u8 = dict(dtype=torch.uint8, generator=g, device=device)
    w13 = torch.randint(0, 256, (E, 2 * INTER_PAD, HIDDEN // 2), **u8)
    w13.view(E, 2, INTER_PAD, -1)[:, :, INTER:] = 0
    s13 = torch.randint(120, 122, (E, 2 * INTER_PAD, HIDDEN // 32), **u8)
    s13.view(E, 2, INTER_PAD, -1)[:, :, INTER:] = 127
    w2 = torch.randint(0, 256, (E, HIDDEN, INTER_PAD // 2), **u8)
    w2[..., INTER // 2:] = 0
    s2 = torch.randint(118, 120, (E, HIDDEN, 24), **u8)
    s2[..., INTER // 32:] = 127
    return w13, s13, w2, s2


def _rope_table(npos, device):
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, ROPE_DIM, 2, device=device, dtype=torch.float64) / ROPE_DIM))
    f = torch.outer(torch.arange(npos, device=device, dtype=torch.float64), inv)
    return torch.cat([f.cos(), f.sin()], -1).to(torch.bfloat16).contiguous()


def _gemma_dir(w_col, norm_w):
    """The direction of rmsnorm(x w_col) * (1 + norm_w) in the non-rotated dims 64..127 (x > 0)."""
    return _unit((w_col.double() * (1.0 + norm_w.double()))[..., ROPE_DIM:])


def make_inputs(case, inputs, generator, device):
    ax = case["axes"]
    R, ctx = ax["num_requests"], ax["context_len"]
    S = R * L
    req_slots, mamba_slots, rtt_stride = ax["req_slots"], ax["mamba_slots"], ax["rtt_stride"]
    kv_slots = ax["kv_slots"]
    g = generator
    out = {}

    # ---- requests: contexts (request 0 at ctx, the others in [ctx / 2, ctx]), slots, KV pages
    ctxs = [ctx] + [int(v) for v in torch.randint((ctx + 1) // 2, ctx + 1, (R - 1,), generator=g,
                                                    device=device).tolist()]
    seq_lens = torch.tensor(ctxs, dtype=torch.int64, device=device)
    req = torch.randperm(req_slots, generator=g, device=device)[:R]
    mamba_map = (torch.randperm(mamba_slots - 1, generator=g, device=device) + 1)[:req_slots]
    positions = (seq_lens[:, None] + torch.arange(L, device=device)).reshape(-1)
    n_pages = kv_slots // KV_PAGE
    pages = torch.randperm(n_pages - 1, generator=g, device=device) + 1  # page 0: SGLang's dummy slot
    rtt = torch.zeros(req_slots, rtt_stride, dtype=torch.int32, device=device)
    p0 = 0
    for r, c in enumerate(ctxs):
        npg = -(-(c + L) // KV_PAGE)
        tok = torch.arange(npg * KV_PAGE, device=device)
        rtt[req[r], :npg * KV_PAGE] = (pages[p0:p0 + npg].repeat_interleave(KV_PAGE) * KV_PAGE
                                       + tok % KV_PAGE).to(torch.int32)
        p0 += npg
    out.update(seq_lens=seq_lens, positions=positions, req_pool_indices=req.to(torch.int64),
               mamba_slot_map=mamba_map.to(torch.int32).contiguous(), req_to_token=rtt,
               rope_cos_sin=_rope_table(ax["rope_positions"], device))

    # ---- planted channels: d0 (massive, shared) and 512 routing-code channels
    chan = torch.randperm(HIDDEN, generator=g, device=device)
    d0, code = int(chan[0]), chan[1:1 + N_EXP]
    pi = torch.stack([torch.randperm(N_EXP, generator=g, device=device) for _ in range(N_LAYERS)])
    h = _randn((S, HC, HIDDEN), g, device)
    h[:, :, d0] = MASSIVE * (1 + 0.05 * _randn((S, HC), g, device))
    for t in range(S):
        sel = torch.randperm(N_EXP, generator=g, device=device)[:TOPK]
        h[t][:, code[sel]] = CODE * (1 + 0.05 * _randn((HC, TOPK), g, device))
    h_in = h.reshape(S, HC_DIM).to(torch.bfloat16)
    out["h_in"] = h_in
    out["h_in_ssq"] = h_in.float().view(S, HC, NPART, 16).square().sum(-1).contiguous()

    # ---- hyper-connection mixers (attention side, MoE side) of the 4 layers
    for pre in ("hc_attn", "hc_mlp"):
        norm = _randn((N_LAYERS, HC_DIM), g, device, 0.02)
        down = torch.zeros(N_LAYERS, LOWRANK + 16, HC_DIM, device=device)
        down[:, :LOWRANK] = _randn((N_LAYERS, LOWRANK, HC_DIM), g, device, 0.01)
        down[:, LOWRANK:LOWRANK + HC] = _randn((N_LAYERS, HC, HC_DIM), g, device, 0.015)
        up = _randn((N_LAYERS, HC_DIM, LOWRANK), g, device, 0.08)
        out[f"{pre}_norm"] = norm.to(torch.bfloat16)
        out[f"{pre}_down"] = down.to(torch.bfloat16)
        out[f"{pre}_up"] = up.to(torch.bfloat16)

    # ---- routers: expert e reads code channel pi[layer][e]; no other code-channel or d0 weight
    router = torch.zeros(N_LAYERS, N_EXP + 16, HIDDEN, device=device)
    router[:, :N_EXP + 1] = _randn((N_LAYERS, N_EXP + 1, HIDDEN), g, device, HIDDEN ** -0.5)
    router[:, :, code] = 0
    router[:, :, d0] = 0
    router[:, N_EXP + 1:] = 0
    for li in range(N_LAYERS):
        router[li, torch.arange(N_EXP, device=device), code[pi[li]]] = ROUTER_CODE_WEIGHT
    out["router"] = router.to(torch.bfloat16)

    # ---- MXFP4 experts of the 4 layers
    w13 = torch.empty(N_LAYERS, N_EXP + 1, 2 * INTER_PAD, HIDDEN // 2, dtype=torch.uint8, device=device)
    s13 = torch.empty(N_LAYERS, N_EXP + 1, 2 * INTER_PAD, HIDDEN // 32, dtype=torch.uint8, device=device)
    w2 = torch.empty(N_LAYERS, N_EXP + 1, HIDDEN, INTER_PAD // 2, dtype=torch.uint8, device=device)
    s2 = torch.empty(N_LAYERS, N_EXP + 1, HIDDEN, 24, dtype=torch.uint8, device=device)
    for li in range(N_LAYERS):
        w13[li], s13[li], w2[li], s2[li] = _mxfp4_experts(g, device)
    out.update(moe_w13=w13, moe_w13_scale=s13, moe_w2=w2, moe_w2_scale=s2)

    # ---- GDN layers 0-2
    ip, ip_s, ba, ba_s, op, op_s = [], [], [], [], [], []
    for _ in range(N_GDN):
        w, s = _fp8_proj(16384, HIDDEN, g, device)
        ip.append(w), ip_s.append(s)
        w, s = _fp8_proj(2 * GDN_HV, HIDDEN, g, device)
        ba.append(torch.cat([w, torch.zeros(128 - 2 * GDN_HV, HIDDEN, device=device).to(w.dtype)]))
        ba_s.append(torch.cat([s, torch.ones(128 - 2 * GDN_HV, device=device)]))
        w, s = _fp8_proj(HIDDEN, GDN_HV * GDN_DK, g, device, 0.3)
        op.append(w), op_s.append(s)
    out.update(gdn_in_proj=torch.stack(ip), gdn_in_proj_scale=torch.stack(ip_s),
               gdn_ba_proj=torch.stack(ba), gdn_ba_proj_scale=torch.stack(ba_s),
               gdn_out_proj=torch.stack(op), gdn_out_proj_scale=torch.stack(op_s))
    out["gdn_conv_weight"] = _randn((N_GDN, 10240, 4), g, device, 0.5).to(torch.bfloat16)
    out["gdn_neg_a"] = -torch.exp(_randn((N_GDN, GDN_HV), g, device).abs())
    out["gdn_dt_bias"] = _randn((N_GDN, GDN_HV), g, device, 0.5, -3.0)
    out["gdn_norm_weight"] = _randn((N_GDN, GDN_DK), g, device, 0.1, 1.0)
    out["gdn_conv_state"] = _bf16_randn((N_GDN, mamba_slots, 10240, 3), g, device, 0.5)
    out["gdn_ssm_state"] = _randn((N_GDN, mamba_slots, GDN_HV, GDN_DK, GDN_DK), g, device, 0.1)

    # ---- QSA layer 3: projections, norms
    w, s = _fp8_proj(QP_QKV, HIDDEN, g, device)
    out["qsa_qkv_proj"], out["qsa_qkv_proj_scale"] = w, s
    # index_qk_proj: index q heads 0..3 (rows 0..511) and the index k (rows 512..639); heads 1 / 3 are
    # heads 0 / 2 negated (codes and per-channel scales), so relu(q1 . k) + relu(q0 . k) = |q0 . k|
    w, s = _fp8_proj(640, HIDDEN, g, device)
    s[128:256] = s[0:128]
    s[384:512] = s[256:384]
    wi = w.float()
    wi[:, code] = 0  # the index rows ignore the routing codes
    q_norm_i = _randn((128,), g, device, 0.1).to(torch.bfloat16)
    k_norm_i = _randn((128,), g, device, 0.1).to(torch.bfloat16)
    col = torch.zeros(5, 128, device=device)
    for hd in (0, 2):
        col[hd, ROPE_DIM:] = INDEX_COLUMN * _randn((ROPE_DIM,), g, device)
    # the index queries' direction: raw q ~ x[d0] * scale * code of channel d0 (x[d0] > 0), then the norm
    qdir = [_gemma_dir(col[hd].to(torch.float8_e4m3fn).float() * s[hd * 128:(hd + 1) * 128], q_norm_i.float())
            for hd in (0, 2)]
    q0d, q2d = qdir
    basis = torch.linalg.qr(torch.stack([q0d, q2d]).T)[0].T  # orthonormal [2, 64]
    # the index k direction: after its scales and the index-k norm weight it is orthogonal to both queries
    vk = _unit(_orthogonal(_randn((1, ROPE_DIM), g, device).double(), basis)[0])
    vk_raw = _unit(vk / (1.0 + k_norm_i.double()[ROPE_DIM:]))  # direction of the raw (pre-norm) key
    vk_code = _unit(vk_raw / s[512 + ROPE_DIM:].double())
    col[4, ROPE_DIM:] = (INDEX_COLUMN * math.sqrt(ROPE_DIM) * vk_code).float()
    for hd in range(5):
        wi[hd * 128:(hd + 1) * 128, d0] = col[hd]
    wi[128:256] = -wi[0:128]
    wi[384:512] = -wi[256:384]
    out["qsa_index_proj"], out["qsa_index_proj_scale"] = wi.to(torch.float8_e4m3fn), s
    w, s = _fp8_proj(HIDDEN, 24 * 256, g, device, 0.3)
    out["qsa_o_proj"], out["qsa_o_proj_scale"] = w, s
    out["qsa_q_norm"] = _randn((256,), g, device, 0.1).to(torch.bfloat16)
    out["qsa_k_norm"] = _randn((256,), g, device, 0.1).to(torch.bfloat16)
    out["qsa_index_q_norm"] = q_norm_i
    out["qsa_index_k_norm"] = k_norm_i

    # ---- KV cache, compressed index keys (planted), index-key ring of the requests
    out["k_cache"] = _bf16_randn((kv_slots, 2, 256), g, device)
    out["v_cache"] = _bf16_randn((kv_slots, 2, 256), g, device, 0.6)
    win_dir = _unit(q0d + q2d)  # [64]
    keys = _planted_keys(kv_slots // 4, None, basis, win_dir, g, device)
    for r, c in enumerate(ctxs):
        n_old = c // 4
        if n_old == 0:
            continue
        slots = rtt[req[r], torch.arange(n_old, device=device) * 4].long() // 4
        win = torch.zeros(n_old, dtype=torch.bool, device=device)
        if n_old <= N_WINNERS:
            win[:] = True
        else:
            win[torch.randperm(n_old, generator=g, device=device)[:N_WINNERS]] = True
        keys[slots] = _planted_keys(n_old, win, basis, win_dir, g, device)
    out["index_keys"] = keys.to(torch.bfloat16)
    # raw (un-normalized) index keys of each request's last RING context positions, as the launch's
    # index-k projection makes them: about 30 v_k (dims 64..127) plus noise of 0.4 a dim
    ring = 0.4 * _randn((req_slots * RING, 128), g, device)
    ring_pos = torch.randint(0, 1 << 20, (req_slots * RING, 3), generator=g, device=device)
    vdir = vk_raw.float()
    for r, c in enumerate(ctxs):
        for p in range(max(0, c - RING), c):
            row = int(req[r]) * RING + p % RING
            ring[row, ROPE_DIM:] += 30.0 * vdir * (1 + 0.1 * float(_randn((1,), g, device)))
            ring_pos[row] = p
        for p in range(c - RING, min(c, 0)):
            ring_pos[int(req[r]) * RING + p % RING] = 0
    out["index_ring"] = ring.to(torch.bfloat16)
    out["index_ring_positions"] = ring_pos.to(torch.int64)
    return out


def _planted_keys(n, win, basis, win_dir, g, device):
    """n compressed index keys of norm sqrt(128): cosine to win_dir (dims 64..127) in WIN_COS for
    winners, at most LOSE_COS for the others (win None: all losers); the rest of each key is a random
    direction orthogonal to the index queries' planted directions."""
    rest = _randn((n, 128), g, device).double()
    rest[:, ROPE_DIM:] = _orthogonal(rest[:, ROPE_DIM:], basis)
    rest = _unit(rest)
    cos = _uniform((n,), g, device, -LOSE_COS, LOSE_COS).double()
    if win is not None:
        cos = torch.where(win, _uniform((n,), g, device, *WIN_COS).double(), cos)
    k = rest * torch.sqrt(1 - cos ** 2)[:, None]
    k[:, ROPE_DIM:] += cos[:, None] * win_dir
    return (k * math.sqrt(128)).float()


# ---------------------------------------------------------------- kernel-facing layout
def _shuffle16(w):
    """aiter shuffle_weight(w, (16, 16)) of a 1-byte [..., N, K] tensor."""
    *lead, n, k = w.shape
    x = w.view(torch.uint8).reshape(*lead, n // 16, 16, k // 32, 2, 16)
    nd = len(lead)
    perm = list(range(nd)) + [nd, nd + 2, nd + 3, nd + 1, nd + 4]
    return x.permute(*perm).contiguous().view(*lead, n, k).view(w.dtype)


def _e8m0_shuffle(s):
    """aiter e8m0_shuffle of [..., E, rows, groups] scales over the flattened [E * rows, groups]."""
    *lead, e, rows, groups = s.shape
    x = s.reshape(*lead, e * rows // 32, 2, 16, groups // 8, 2, 4)
    nd = len(lead)
    perm = list(range(nd)) + [nd, nd + 3, nd + 5, nd + 2, nd + 4, nd + 1]
    return x.permute(*perm).contiguous().view(*lead, e, rows, groups)


def _pack_bf16(w):
    """sgl-amd-mono packing.pack_bf16 of [..., N, K] (16-row x 64-K MFMA tiles), same shape."""
    *lead, n, k = w.shape
    x = w.reshape(*lead, n // 16, 16, k // 64, 2, 4, 8)
    nd = len(lead)
    perm = list(range(nd)) + [nd, nd + 2, nd + 3, nd + 4, nd + 1, nd + 5]
    return x.permute(*perm).contiguous().view(*lead, n, k)


def to_kernel_layout(case, inputs):
    inputs = dict(inputs)
    for name in ("hc_attn_down", "hc_attn_up", "hc_mlp_down", "hc_mlp_up", "router"):
        inputs[name] = _pack_bf16(inputs[name])
    for name in ("gdn_in_proj", "gdn_ba_proj", "gdn_out_proj", "qsa_qkv_proj", "qsa_index_proj",
                 "qsa_o_proj", "moe_w13", "moe_w2"):
        inputs[name] = _shuffle16(inputs[name])
    for name in ("moe_w13_scale", "moe_w2_scale"):
        inputs[name] = _e8m0_shuffle(inputs[name])
    return inputs
