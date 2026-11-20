"""Float64 reference of one Qwen3.8-Flash-Next verify launch: layers GDN, GDN, GDN, QSA.

The math is SGLang's Qwen4-Exp decoder layer (python/sglang/srt/models/qwen4_exp.py and the
modules it uses) on the dequantized weights, without the kernels' intermediate roundings or
activation quantization.
"""
import math

import torch
import torch.nn.functional as F

HIDDEN = 2560
HC = 4
EPS = 1e-6
GDN_HV = 48
GDN_DK = 128
GDN_KEY_DIM = 2048
GDN_CONV_DIM = 10240
EXPERTS = 512
TOPK = 10
INTER = 640
QSA_HEADS = 24
QSA_KV_HEADS = 2
QSA_HEAD_DIM = 256
INDEX_HEADS = 4
INDEX_DIM = 128
ROTARY = 64
TOKENS = 4
RING = 8
BLOCK = 4
BLOCK_TOPK = 512
KINDS = ("gdn", "gdn", "gdn", "qsa")
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _fp8(codes, scale):
    return codes.double() * scale.double()[..., None]


def _mxfp4(codes, scales):
    """MXFP4 [N, K / 2] codes (low nibble first) and E8M0 [N, K / 32] scales -> float64 [N, K]."""
    lut = torch.tensor(E2M1, dtype=torch.float64, device=codes.device)
    nib = torch.stack([codes & 0xF, codes >> 4], dim=-1).flatten(-2).long()
    vals = lut[nib].unflatten(-1, (-1, 32))
    return (vals * torch.exp2(scales.double() - 127.0)[..., None]).flatten(-2)


def _gemma(x, w):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * (1.0 + w.double())


def _rope_axis():
    """MRoPE axis of each of the 32 rotary pairs (interleaved sections 11 / 11 / 10)."""
    return [1 if i % 3 == 1 else (2 if i % 3 == 2 and i < 30 else 0) for i in range(ROTARY // 2)]


def _rope(x, pos3, cos_sin):
    """Neox rotary embedding of the first 64 dims of x [..., D] at MRoPE positions pos3 [3]."""
    axis = torch.tensor(_rope_axis(), device=x.device)
    p = pos3.long()[axis]
    i = torch.arange(ROTARY // 2, device=x.device)
    cos = cos_sin[p, i].double()
    sin = cos_sin[p, i + ROTARY // 2].double()
    x1, x2 = x[..., :ROTARY // 2], x[..., ROTARY // 2:ROTARY]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., ROTARY:]], dim=-1)


def _mix(h, norm, down, up, inj):
    """Gated hyper-connection read of the 4 streams h [S, 4, 2560]: the block input and the
    combine gates."""
    hn = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + EPS) * (1.0 + norm.double().view(HC, HIDDEN))
    flat = hn.flatten(1)
    d = F.silu(flat @ down.double().t() / HC)
    u = torch.sigmoid(d @ up.double().t()).view_as(hn)
    x = (u * hn).mean(1)
    gates = 2.0 * torch.sigmoid(flat @ inj.double().t() / HC)
    return x, gates


def _moe(x, router, w13, s13, w2, s2):
    logits = x @ router.double().t()
    prob = torch.softmax(logits[:, :EXPERTS], -1)
    wts, ids = prob.topk(TOPK, -1)
    wts = torch.cat([wts / wts.sum(-1, keepdim=True), torch.sigmoid(logits[:, EXPERTS:])], 1)
    ids = torch.cat([ids, torch.full_like(ids[:, :1], EXPERTS)], 1)
    y = torch.zeros_like(x)
    for e in ids.unique().tolist():
        rows, k = (ids == e).nonzero(as_tuple=True)
        gu = _mxfp4(w13[e], s13[e])
        dn = _mxfp4(w2[e], s2[e])
        xe = x[rows]
        mid = F.silu(xe @ gu[:INTER].t()) * (xe @ gu[INTER:].t())
        y.index_add_(0, rows, wts[rows, k, None] * (mid @ dn.t()))
    return y


def _gdn(x, layer, ctx):
    p = ctx
    R = p["R"]
    ip = x @ _fp8(p["gdn_in_proj"][layer], p["gdn_in_proj_scale"][layer]).t()
    ba = x @ _fp8(p["gdn_ba_proj"][layer], p["gdn_ba_scale"][layer]).t()
    mixed, z = ip[:, :GDN_CONV_DIM], ip[:, GDN_CONV_DIM:]
    cw = p["gdn_conv_w"][layer].double()
    neg_a = p["gdn_neg_a"][layer].double()
    dt_bias = p["gdn_dt_bias"][layer].double()
    norm_w = p["gdn_norm_w"][layer].double()
    y = torch.empty(x.shape[0], GDN_HV * GDN_DK, dtype=torch.float64, device=x.device)
    for r in range(R):
        slot = int(p["mamba_map"][int(p["req_pool_indices"][r])])
        win = torch.cat([p["gdn_conv_state"][layer, slot].double(), mixed[r * TOKENS:(r + 1) * TOKENS].t()], 1)
        p["iconv"][layer, r] = win[:, 1:]
        state = p["gdn_ssm_state"][layer, slot].double()
        for j in range(TOKENS):
            t = r * TOKENS + j
            c = F.silu((win[:, j:j + 4] * cw).sum(1))
            q = c[:GDN_KEY_DIM].view(-1, GDN_DK)
            k = c[GDN_KEY_DIM:2 * GDN_KEY_DIM].view(-1, GDN_DK)
            v = c[2 * GDN_KEY_DIM:].view(GDN_HV, GDN_DK)
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * GDN_DK ** -0.5
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            q = q.repeat_interleave(GDN_HV // q.shape[0], 0)
            k = k.repeat_interleave(GDN_HV // k.shape[0], 0)
            beta = torch.sigmoid(ba[t, :GDN_HV])
            decay = torch.exp(neg_a * F.softplus(ba[t, GDN_HV:] + dt_bias))
            state = state * decay[:, None, None]
            u = beta[:, None] * (v - torch.einsum("hvk,hk->hv", state, k))
            state = state + u[:, :, None] * k[:, None, :]
            p["issm"][layer, r, j] = state
            o = torch.einsum("hvk,hk->hv", state, q)
            o = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + EPS) * norm_w
            y[t] = (o * torch.sigmoid(z[t].view(GDN_HV, GDN_DK))).flatten()
    return y @ _fp8(p["gdn_out_proj"][layer], p["gdn_out_scale"][layer]).t()


def _qsa(x, ctx):
    p = ctx
    R, cos_sin, pos = p["R"], p["cos_sin"], p["positions"]
    S = x.shape[0]
    qkv = x @ _fp8(p["qsa_qkv_proj"], p["qsa_qkv_scale"]).t()
    qg = qkv[:, :QSA_HEADS * 2 * QSA_HEAD_DIM].view(S, QSA_HEADS, 2 * QSA_HEAD_DIM)
    q, gate = qg[..., :QSA_HEAD_DIM], qg[..., QSA_HEAD_DIM:]
    kv0 = QSA_HEADS * 2 * QSA_HEAD_DIM
    k = qkv[:, kv0:kv0 + QSA_KV_HEADS * QSA_HEAD_DIM].view(S, QSA_KV_HEADS, QSA_HEAD_DIM)
    v = qkv[:, kv0 + QSA_KV_HEADS * QSA_HEAD_DIM:].view(S, QSA_KV_HEADS, QSA_HEAD_DIM)
    idx = x @ _fp8(p["qsa_index_proj"], p["qsa_index_scale"]).t()
    iq = idx[:, :INDEX_HEADS * INDEX_DIM].view(S, INDEX_HEADS, INDEX_DIM)
    ik = idx[:, INDEX_HEADS * INDEX_DIM:]
    q, k, iq = _gemma(q, p["qsa_q_norm"]), _gemma(k, p["qsa_k_norm"]), _gemma(iq, p["qsa_iq_norm"])
    for t in range(S):
        q[t] = _rope(q[t], pos[:, t], cos_sin)
        k[t] = _rope(k[t], pos[:, t], cos_sin)
        iq[t] = _rope(iq[t], pos[:, t], cos_sin)
    kc, vc, ring, ring_pos, comp = p["k_cache"], p["v_cache"], p["ring"], p["ring_pos"], p["comp"]
    loc = p["out_cache_loc"].long()
    kc[loc] = k
    vc[loc] = v
    out = torch.empty(S, QSA_HEADS * QSA_HEAD_DIM, dtype=torch.float64, device=x.device)
    rtt = p["req_to_token"]
    for r in range(R):
        req, c = int(p["req_pool_indices"][r]), int(p["seq_lens"][r])
        rows = range(r * TOKENS, (r + 1) * TOKENS)
        new = {}
        for j, t in enumerate(rows):
            pj = c + j
            ring[req * RING + pj % RING, 0] = ik[t]
            ring_pos[req * RING + pj % RING] = pos[:, t]
            if (pj + 1) % BLOCK == 0:
                start = pj - BLOCK + 1
                members = torch.stack([ring[req * RING + m % RING, 0] for m in range(start, pj + 1)])
                key = _gemma(members.mean(0), p["qsa_ik_norm"])
                key = _rope(key, ring_pos[req * RING + start % RING], cos_sin)
                new[start // BLOCK] = key
                comp[int(rtt[req, start]) // BLOCK, 0] = key
        old_blocks = c // BLOCK
        cslots = rtt[req, :old_blocks * BLOCK:BLOCK].long() // BLOCK
        old_keys = comp[cslots, 0] if old_blocks else comp.new_zeros(0, INDEX_DIM)
        for j, t in enumerate(rows):
            pj = c + j
            nvis = (pj + 1) // BLOCK
            keys = torch.cat([old_keys] + [new[b][None] for b in range(old_blocks, nvis)])[:nvis]
            logits = torch.relu(iq[t] @ keys.t()).sum(0) / math.sqrt(INDEX_DIM)
            blocks = torch.arange(nvis, device=x.device) if nvis <= BLOCK_TOPK else logits.topk(BLOCK_TOPK).indices
            toks = (blocks[:, None] * BLOCK + torch.arange(BLOCK, device=x.device)).flatten()
            toks = torch.cat([toks, torch.arange(nvis * BLOCK, pj + 1, device=x.device)])
            slots = rtt[req, toks].long()
            kk, vv = kc[slots], vc[slots]
            for h in range(QSA_HEADS):
                kvh = h // (QSA_HEADS // QSA_KV_HEADS)
                prob = torch.softmax(kk[:, kvh] @ q[t, h] / math.sqrt(QSA_HEAD_DIM), 0)
                o = prob @ vv[:, kvh]
                out[t, h * QSA_HEAD_DIM:(h + 1) * QSA_HEAD_DIM] = o * torch.sigmoid(gate[t, h])
    return out @ _fp8(p["qsa_o_proj"], p["qsa_o_scale"]).t()


def reference(**inputs):
    p = dict(inputs)
    h = inputs["h_in"].double().view(-1, HC, HIDDEN)
    p["R"] = inputs["req_pool_indices"].numel()
    p["iconv"] = inputs["gdn_iconv"].double()
    p["issm"] = inputs["gdn_issm"].double()
    p["k_cache"] = inputs["k_cache"].float().double()
    p["v_cache"] = inputs["v_cache"].float().double()
    p["ring"] = inputs["index_ring"].double()
    p["ring_pos"] = inputs["index_ring_pos"].clone()
    p["comp"] = inputs["index_comp"].double()
    gdn_layer = 0
    for i, kind in enumerate(KINDS):
        x, gates = _mix(h, inputs["hc_norm"][i, 0], inputs["hc_down"][i, 0], inputs["hc_up"][i, 0],
                        inputs["hc_inj"][i, 0])
        if kind == "gdn":
            a = _gdn(x, gdn_layer, p)
            gdn_layer += 1
        else:
            a = _qsa(x, p)
        h = h + gates[:, :, None] * a[:, None, :]
        x, gates = _mix(h, inputs["hc_norm"][i, 1], inputs["hc_down"][i, 1], inputs["hc_up"][i, 1],
                        inputs["hc_inj"][i, 1])
        y = _moe(x, inputs["router"][i], inputs["moe_w13"][i], inputs["moe_w13_scale"][i],
                 inputs["moe_w2"][i], inputs["moe_w2_scale"][i])
        h = h + gates[:, :, None] * y[:, None, :]
    h_out = h.flatten(1)
    h_out_ssq = h.view(h.shape[0], HC, HIDDEN // 16, 16).square().sum(-1)
    return (h_out, h_out_ssq, p["issm"].float(), p["iconv"].float(), p["k_cache"].float(),
            p["v_cache"].float(), p["ring"].float(), p["ring_pos"], p["comp"].float())
