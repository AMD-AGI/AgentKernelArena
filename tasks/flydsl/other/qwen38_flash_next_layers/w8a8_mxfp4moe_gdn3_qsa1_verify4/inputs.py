"""Inputs of one Qwen3.8-Flash-Next verify launch (3 Gated DeltaNet layers, 1 QSA layer).

``make_inputs`` draws every logical input of a case: the requests' KV pages, request and
state slots, pools and weights. ``to_kernel_layout`` maps the logical weights to the layouts
SGLang keeps on gfx950 after loading (AITER's (16, 16) weight shuffle, E8M0 scale shuffle,
the intermediate dimension padded from 640 to 768, bf16 matrices in 16 x 64 MFMA tiles).

The residual streams carry one direction shared by every row and one direction per row, and
the router and index-query weights are aligned with them, so that each row's ten routed
experts and each request's 512 selected index blocks lead the rest by a wide margin. See
README.md for why the discrete selections need that margin.
"""
import math

import torch

HIDDEN = 2560
HC = 4
HC_DIM = HC * HIDDEN
HC_LOWRANK = 320
GDN_HV = 48
GDN_DK = 128
GDN_KEY_DIM = 16 * GDN_DK
GDN_VAL_DIM = GDN_HV * GDN_DK
GDN_CONV_DIM = 2 * GDN_KEY_DIM + GDN_VAL_DIM
GDN_QKVZ = GDN_CONV_DIM + GDN_VAL_DIM
EXPERTS = 512
TOPK = 10
INTER = 640
INTER_PAD = 768
QSA_HEADS = 24
QSA_HEAD_DIM = 256
QKV_ROWS = QSA_HEADS * 2 * QSA_HEAD_DIM + 2 * 2 * QSA_HEAD_DIM
INDEX_HEADS = 4
INDEX_DIM = 128
INDEX_ROWS = (INDEX_HEADS + 1) * INDEX_DIM
TOKENS = 4
RING = 8
PAGE = 64
BLOCK = 4
BLOCK_TOPK = 512
ROPE_THETA = 1.0e7
ROTARY = 64


def _normal(shape, std, generator, device, mean=0.0):
    return torch.randn(shape, generator=generator, device=device).mul_(std).add_(mean)


def _fp8(weight):
    """Per-output-channel FP8 E4M3 codes and float32 scales of a float weight [..., N, K]."""
    scale = weight.abs().amax(-1).clamp_min(1e-12) / 448.0
    return (weight / scale[..., None]).to(torch.float8_e4m3fn), scale.float()


def _random_fp8(shape, generator, device):
    """A per-channel FP8 projection at unit output scale (weights N(0, 1 / K))."""
    return _fp8(_normal(shape, shape[-1] ** -0.5, generator, device))


def _mxfp4(shape, generator, device):
    """Random MXFP4 codes [..., N, K / 2] and E8M0 scales [..., N, K / 32] of 2^-8 .. 2^-5."""
    *lead, n, k = shape
    codes = torch.randint(0, 256, (*lead, n, k // 2), generator=generator, device=device,
                          dtype=torch.uint8)
    scales = torch.randint(119, 123, (*lead, n, k // 32), generator=generator, device=device,
                           dtype=torch.uint8)
    return codes, scales


def _orthonormal(n, k, generator, device):
    """k orthonormal columns of R^n."""
    q, _ = torch.linalg.qr(torch.randn(n, k, generator=generator, device=device, dtype=torch.float64))
    return q.float()


def _contexts(requests, ctx, generator, device):
    """Context tokens of each request: the first has ``ctx``, the others ctx / 2 .. ctx. Contexts
    of 2045 .. 2047 tokens are moved to 2044, so a request either never has more than 512 complete
    blocks in its verify window or has at least 512 before it."""
    lo = max(16, ctx // 2)
    rest = torch.randint(lo, ctx + 1, (requests - 1,), generator=generator, device=device)
    out = [ctx] + rest.tolist()
    return [2044 if 2045 <= c <= 2047 else c for c in out]


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    R, ctx = axes["requests"], axes["ctx"]
    S = R * TOKENS
    g = generator
    contexts = _contexts(R, ctx, g, device)

    # request slots, state slots and 64-token KV pages, scattered over their pools
    req_slots = torch.randperm(axes["req_slots"], generator=g, device=device)[:R]
    state_slots = 1 + torch.randperm(axes["mamba_slots"] - 1, generator=g, device=device)[:R]
    pages = 1 + torch.randperm(axes["kv_slots"] // PAGE - 1, generator=g, device=device)
    req_to_token = torch.zeros(axes["req_slots"], axes["rtt_stride"], dtype=torch.int32, device=device)
    mamba_map = torch.zeros(axes["req_slots"], dtype=torch.int32, device=device)
    loc = torch.empty(S, dtype=torch.int64, device=device)
    positions = torch.empty(3, S, dtype=torch.int64, device=device)
    used = 0
    for r, c in enumerate(contexts):
        n = c + TOKENS
        npg = (n + PAGE - 1) // PAGE
        own = pages[used:used + npg]
        used += npg
        slots = (own[:, None] * PAGE + torch.arange(PAGE, device=device)).flatten()[:n]
        req_to_token[req_slots[r], :n] = slots.to(torch.int32)
        mamba_map[req_slots[r]] = state_slots[r].to(torch.int32)
        loc[r * TOKENS:(r + 1) * TOKENS] = slots[c:]
        positions[:, r * TOKENS:(r + 1) * TOKENS] = torch.arange(c, n, device=device)
    seq_lens = torch.tensor(contexts, dtype=torch.int64, device=device)

    # residual streams: a direction every row shares, one per row, and noise
    dirs = _orthonormal(HIDDEN, S + 1, g, device)
    shared, own_dir = dirs[:, 0], dirs[:, 1:].t()
    amp = 0.6 * math.sqrt(HIDDEN)
    h = _normal((S, HC, HIDDEN), 1.0, g, device)
    h += amp * (shared + own_dir)[:, None, :]
    h_in = h.reshape(S, HC_DIM).bfloat16()
    h_in_ssq = h_in.float().view(S, HC, HIDDEN // 16, 16).square().sum(-1)

    # hyper-connection mixers [layer, attn | mlp]
    hc_norm = _normal((4, 2, HC_DIM), 0.1, g, device).bfloat16()
    hc_down = _normal((4, 2, HC_LOWRANK, HC_DIM), HC_DIM ** -0.5, g, device).bfloat16()
    hc_up = _normal((4, 2, HC_DIM, HC_LOWRANK), HC_LOWRANK ** -0.5, g, device).bfloat16()
    hc_inj = _normal((4, 2, HC, HC_DIM), HC_DIM ** -0.5, g, device).bfloat16()

    # routers: each row's own direction leads ten random experts of every layer
    router = _normal((4, EXPERTS + 1, HIDDEN), 0.004, g, device)
    for layer in range(4):
        picks = torch.rand(S, EXPERTS, generator=g, device=device).argsort(-1)[:, :TOPK]
        for t in range(S):
            router[layer, picks[t]] += 0.5 * own_dir[t]
    router[:, EXPERTS] = _normal((4, HIDDEN), HIDDEN ** -0.5, g, device)
    router = router.bfloat16()

    w13, w13_scale = _mxfp4((4, EXPERTS + 1, 2 * INTER, HIDDEN), g, device)
    w2, w2_scale = _mxfp4((4, EXPERTS + 1, HIDDEN, INTER), g, device)

    gdn_in_proj, gdn_in_proj_scale = _random_fp8((3, GDN_QKVZ, HIDDEN), g, device)
    gdn_ba_proj, gdn_ba_scale = _random_fp8((3, 2 * GDN_HV, HIDDEN), g, device)
    gdn_out_proj, gdn_out_scale = _random_fp8((3, HIDDEN, GDN_VAL_DIM), g, device)
    gdn_neg_a = -torch.exp(_normal((3, GDN_HV), 1.0, g, device).abs_())

    qsa_qkv_proj, qsa_qkv_scale = _random_fp8((QKV_ROWS, HIDDEN), g, device)
    # index query rows: the non-rotated half of every head follows the shared direction
    index = _normal((INDEX_ROWS, HIDDEN), HIDDEN ** -0.5, g, device)
    key_dir = _orthonormal(INDEX_DIM - ROTARY, 1, g, device)[:, 0]
    for head in range(INDEX_HEADS):
        rows = slice(head * INDEX_DIM + ROTARY, (head + 1) * INDEX_DIM)
        index[rows] += key_dir[:, None] * shared[None, :]
    qsa_index_proj, qsa_index_scale = _fp8(index)
    # Random keys spread attention over ~2048 tokens, so its output is ~30x smaller than a
    # trained model's; o_proj carries that factor so the attention reaches the streams.
    qsa_o_proj, qsa_o_scale = _random_fp8((HIDDEN, QSA_HEADS * QSA_HEAD_DIM), g, device)
    qsa_o_scale *= 32.0

    # pools: KV rows, index-key ring, compressed index keys (the requests' complete blocks get
    # exactly 512 keys along the index direction when they have more than 512 blocks)
    kv_slots = axes["kv_slots"]
    k_cache = _normal((kv_slots, 2, QSA_HEAD_DIM), 1.0, g, device).bfloat16()
    v_cache = _normal((kv_slots, 2, QSA_HEAD_DIM), 1.0, g, device).bfloat16()
    index_ring = _normal((axes["req_slots"] * RING, 1, INDEX_DIM), 0.5, g, device).bfloat16()
    index_ring_pos = torch.zeros(axes["req_slots"] * RING, 3, dtype=torch.int64, device=device)
    comp = _normal((kv_slots // BLOCK, 1, INDEX_DIM), 1.0, g, device)
    for r, c in enumerate(contexts):
        for p in range(max(0, c - RING), c):
            index_ring_pos[req_slots[r] * RING + p % RING] = p
        blocks = c // BLOCK
        cslot = req_to_token[req_slots[r], :blocks * BLOCK:BLOCK].long() // BLOCK
        sign = -torch.ones(blocks, device=device)
        if blocks >= BLOCK_TOPK:
            hot = torch.randperm(blocks, generator=g, device=device)[:BLOCK_TOPK]
            sign[hot] = 1.0
        else:
            sign[:] = 1.0
        tail = sign[:, None] * math.sqrt(INDEX_DIM - ROTARY) * key_dir[None, :]
        tail = tail + _normal((blocks, INDEX_DIM - ROTARY), 0.2, g, device)
        comp[cslot, 0, ROTARY:] = tail
    index_comp = comp.bfloat16()

    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, ROTARY, 2, device=device).float() / ROTARY))
    freqs = torch.outer(torch.arange(axes["rope_rows"], device=device).float(), inv)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], -1).bfloat16()

    return {
        "h_in": h_in, "h_in_ssq": h_in_ssq, "positions": positions, "out_cache_loc": loc,
        "req_pool_indices": req_slots.long(), "seq_lens": seq_lens, "req_to_token": req_to_token,
        "mamba_map": mamba_map, "cos_sin": cos_sin, "k_cache": k_cache, "v_cache": v_cache,
        "index_ring": index_ring, "index_ring_pos": index_ring_pos, "index_comp": index_comp,
        "hc_norm": hc_norm, "hc_down": hc_down, "hc_up": hc_up, "hc_inj": hc_inj, "router": router,
        "moe_w13": w13, "moe_w13_scale": w13_scale, "moe_w2": w2, "moe_w2_scale": w2_scale,
        "gdn_in_proj": gdn_in_proj, "gdn_in_proj_scale": gdn_in_proj_scale,
        "gdn_ba_proj": gdn_ba_proj, "gdn_ba_scale": gdn_ba_scale,
        "gdn_out_proj": gdn_out_proj, "gdn_out_scale": gdn_out_scale, "gdn_neg_a": gdn_neg_a,
        "qsa_qkv_proj": qsa_qkv_proj, "qsa_qkv_scale": qsa_qkv_scale,
        "qsa_index_proj": qsa_index_proj, "qsa_index_scale": qsa_index_scale,
        "qsa_o_proj": qsa_o_proj, "qsa_o_scale": qsa_o_scale,
    }


# ---------------------------------------------------------------- kernel layouts

def _shuffle16(w):
    """AITER shuffle_weight(w, (16, 16)) of 1-byte elements: [..., N, K] in 16 x 32-byte tiles."""
    *lead, n, k = w.shape
    raw = w.contiguous().view(torch.uint8).reshape(-1, n // 16, 16, k // 32, 2, 16)
    return raw.permute(0, 1, 3, 4, 2, 5).contiguous().view(w.dtype).reshape(*lead, n, k)


def _e8m0_shuffle(s):
    """AITER e8m0_shuffle of [..., N, G] E8M0 scales (N % 32 == 0, G % 8 == 0)."""
    *lead, n, gr = s.shape
    raw = s.contiguous().reshape(-1, 2, 16, gr // 8, 2, 4)
    return raw.permute(0, 3, 5, 2, 4, 1).contiguous().reshape(*lead, n, gr)


def _pack_bf16(w):
    """bf16 [..., N, K] in the kernel's 16-row x 64-K MFMA tiles (same shape, permuted)."""
    *lead, n, k = w.shape
    t = w.contiguous().view(torch.int16).reshape(-1, n // 16, 16, k // 64, 2, 4, 8)
    return t.permute(0, 1, 3, 4, 5, 2, 6).contiguous().view(torch.bfloat16).reshape(*lead, n, k)


def _pad_rows(w, rows, value=0):
    out = torch.full((*w.shape[:-2], rows, w.shape[-1]), value, dtype=w.dtype, device=w.device)
    out[..., :w.shape[-2], :] = w
    return out


def _pad_cols(w, cols, value=0):
    out = torch.full((*w.shape[:-1], cols), value, dtype=w.dtype, device=w.device)
    out[..., :w.shape[-1]] = w
    return out


def to_kernel_layout(case, inputs):
    out = dict(inputs)
    # mixers: the down matrix carries the combine-gate rows in one extra 16-row tile
    down = torch.cat([inputs["hc_down"], _pad_rows(inputs["hc_inj"], 16)], dim=-2)
    out["hc_down"] = _pack_bf16(down)
    out["hc_up"] = _pack_bf16(inputs["hc_up"])
    out["router"] = _pack_bf16(_pad_rows(inputs["router"], EXPERTS + 16))
    # MXFP4 experts: gate rows 0..639 and up rows 768..1407 of a 1536-row w13, w2 K padded to 768
    w13 = inputs["moe_w13"]
    gu = torch.zeros(*w13.shape[:-2], 2 * INTER_PAD, w13.shape[-1], dtype=torch.uint8, device=w13.device)
    gu[..., :INTER, :] = w13[..., :INTER, :]
    gu[..., INTER_PAD:INTER_PAD + INTER, :] = w13[..., INTER:, :]
    s13 = inputs["moe_w13_scale"]
    gus = torch.full((*s13.shape[:-2], 2 * INTER_PAD, s13.shape[-1]), 127, dtype=torch.uint8,
                     device=s13.device)
    gus[..., :INTER, :] = s13[..., :INTER, :]
    gus[..., INTER_PAD:INTER_PAD + INTER, :] = s13[..., INTER:, :]
    out["moe_w13"] = _shuffle16(gu)
    out["moe_w13_scale"] = _e8m0_shuffle(gus)
    out["moe_w2"] = _shuffle16(_pad_cols(inputs["moe_w2"], INTER_PAD // 2))
    out["moe_w2_scale"] = _e8m0_shuffle(_pad_cols(inputs["moe_w2_scale"], INTER_PAD // 32, 127))
    # FP8 projections; in_proj_ba is padded to 128 rows (scale 1) as SGLang pads it for AITER
    for name in ("gdn_in_proj", "gdn_out_proj", "qsa_qkv_proj", "qsa_index_proj", "qsa_o_proj"):
        out[name] = _shuffle16(inputs[name])
    out["gdn_ba_proj"] = _shuffle16(_pad_rows(inputs["gdn_ba_proj"].view(torch.uint8), 128)
                                    .view(torch.float8_e4m3fn))
    out["gdn_ba_scale"] = _pad_cols(inputs["gdn_ba_scale"], 128, 1.0)
    return out
