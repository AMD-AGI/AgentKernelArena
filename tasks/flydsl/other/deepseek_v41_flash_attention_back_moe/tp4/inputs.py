# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Logical inputs of one DeepSeek-V4.1-Flash K2 launch (decode step) on a TP4 group, each rank's shard,
the kernel-facing weight layouts and each rank's expected outputs.

Logical inputs (what make_inputs returns and reference.py reads), every tensor as the whole model holds it:
  q [S, 64, 512] bf16: K1's rotated query, every head (rank r computes heads 16 r .. 16 r + 15);
  swa [swa_pages, 37440] and comp [comp_pages, 74880] uint8: the layer's paged V4.1 caches (SWA pages
    of 64 records, compressed pages of 128): per page the 576-byte records (448 E4M3 NoPE codes, 64 bf16
    RoPE values), then an 8-byte scale row per record (7 E8M0 codes, a zero), padded to 576 bytes;
  kt [S, 640], klen [2 S] int32: K1's key table (compressed slot | 1 << 31 for the top-k rows, then the
    window's slots, -1 past kv_len; klen = kv_len per row, then ntopk per row);
  pos [S] int64, sink [64] f32 (the attention sinks), cos_sin [1048576, 64] f32 (32 cos, then 32 sin);
  woa [8192, 4096] E4M3 + woa_s [256, 128] (wo_a of the 8 groups, 32 x 32 E8M0 blocks), wob [5120, 8192]
    E4M3 + wob_s [160, 256]; res_in [S, 4, 5120] bf16, post_in / pre_in [S, 4], comb_in [S, 4, 4] f32 (K1's
    seam outputs); hc_fn [24, 20480], hc_scale [3], hc_base [24] f32 and ffn_norm [5120] bf16 (the FFN mHC);
  gate_w [384, 5120] bf16, bias [384] f32; w13 [384, 4608, 2560] uint8 + w13_s [384, 4608, 160]
    (MXFP4 gate rows 0 .. 2303, up rows 2304 .. 4607, codes two per byte along the hidden axis, low nibble
    first, one E8M0 byte per 32), w2 [384, 5120, 1152] + w2_s [384, 5120, 72] (down); sgu [4608, 5120] E4M3 +
    sgu_s [144, 160] (the shared expert's gate then up rows), sw2 [5120, 2304] + sw2_s [160, 72].

shard_inputs gives rank r its heads (q, sink), its wo_a groups 2 r, 2 r + 1 (woa rows, wob columns 2048 r ..),
its routed intermediate slice (gate and up rows 576 r .., each padded to 640 rows with zero weights; w2's
columns 576 r .., padded to 640 with zero weights; scale columns padded to 24 with 0x7F) and its shared
expert slice; to_kernel_layout lays the MXFP4 experts out as SGLang's gfx950 loader does (aiter
shuffle_weight / shuffle_scale with the gate / up interleave, SGLANG_USE_AITER_MOE_GU_ITLV); everything else
is row-major and replicated.

Step model (decode, a row a request): row 0 has the case's context ctx, the others a context drawn in
[ctx / 2, ctx]; a row at context c has position c - 1, attends to its window (the last min(c, 128)
positions, its own record included: K1 wrote it) and to the top min(512, c // 2) rows of the layer's
compressed cache (compress ratio 2), drawn uniformly from its c // 2 compressed rows in random order.
Window positions map to distinct random SWA pages (64 positions a page), each selected compressed row to
its own random compressed page; every record a row reads is a valid fp8_ds_mla record of random keys
(N(0, 1), quantized by K1's rule), the other cache bytes are random.

Planted margins (the one discrete decision, the router's top 6, must not flip between the reference and a
correct kernel): the router weight is a random N(0, 0.02) matrix plus the least-squares correction that
gives every row 6 random winners whose selection keys (score + bias) are spaced over [1.55, 1.85] and the
other 378 experts logits in [-4, -0.5] (keys <= 0.74) for the row's FFN input (computed exactly as the
reference does), rounded to bf16: the 6th and 7th keys stay >= 0.5 apart and the winners' order >= 0.02
(checked).
"""
import torch

from dsv41_math import (
    EXPERTS, GROUP_K, HC, HEAD_DIM, HIDDEN, INTER, INTER_PAD, INTER_RANK, K_MIX, MIX, MX, N_HEADS,
    O_GROUPS, O_RANK, TOP_K, TOPK_KEYS, TP, WINDOW, encode_records, k2_ffn_input, key_table, page_bytes,
    route, write_records,
)

SWA_BLOCK, COMP_BLOCK = 64, 128
RATIO = 2  # compressed rows of a context of c tokens: c // 2
MAX_POSITIONS = 1 << 20
ROPE_THETA = 10000.0
Q_STD, SINK_MEAN = 1.5, 0.5
WIN_KEYS, LOSE_LOGITS = (1.55, 1.85), (-4.0, -0.5)
GATE_STD, BIAS_STD, BIAS_CLIP = 0.02, 0.02, 0.05
ROUTE_MARGIN, ORDER_GAP = 0.5, 0.02
E8M0_EXPERT = (117, 121)  # routed and shared expert scales 2^-10 .. 2^-7
E8M0_WOA, E8M0_WOB = (118, 123), (114, 119)
FP8_STD = 8.0  # E4M3 weight codes ~ N(0, 8) (clamped to +-448)
UNIT_E8M0 = 0x7F
HEADS_RANK = N_HEADS // TP
O_COLS = O_GROUPS // TP * O_RANK  # 2048: a rank's wo_a rows / wo_b columns
CHUNK = 1 << 18


def _uniform(shape, low, high, g, device):
    return low + (high - low) * torch.rand(shape, generator=g, device=device, dtype=torch.float64)


def _fp8(shape, g, device):
    return (FP8_STD * torch.randn(shape, generator=g, device=device)).clamp(-448, 448).to(torch.float8_e4m3fn)


def _codes(shape, bounds, g, device):
    return torch.randint(*bounds, shape, generator=g, device=device, dtype=torch.uint8)


def rope_table(device):
    """cos | sin of position p times 10000^(-j / 32), j = 0 .. 31, float32 [2^20, 64]."""
    inv = ROPE_THETA ** (-torch.arange(32, device=device, dtype=torch.float64) / 32)
    out = torch.empty(MAX_POSITIONS, 64, dtype=torch.float32, device=device)
    for p0 in range(0, MAX_POSITIONS, CHUNK):
        p1 = min(p0 + CHUNK, MAX_POSITIONS)
        ang = torch.arange(p0, p1, device=device, dtype=torch.float64)[:, None] * inv[None, :]
        out[p0:p1] = torch.cat([ang.cos(), ang.sin()], dim=1).float()
    return out


def make_step(rows, ctx, g, device):
    """Contexts, positions, the window and top-k slots of every row (see the module docstring)."""
    low = max(ctx // 2, 1)
    ctxs = [ctx] + torch.randint(low, ctx + 1, (rows - 1,), generator=g, device=device).tolist()
    swa_pages = 3 * rows + 1
    comp_pages = TOPK_KEYS * rows + 1
    swa_perm = (torch.randperm(swa_pages - 1, generator=g, device=device) + 1).tolist()
    comp_perm = (torch.randperm(comp_pages - 1, generator=g, device=device) + 1).tolist()
    swa_idx = torch.full((rows, WINDOW), -1, dtype=torch.int32, device=device)
    topk = torch.full((rows, TOPK_KEYS), -1, dtype=torch.int32, device=device)
    swa_lens = torch.zeros(rows, dtype=torch.int32, device=device)
    topk_lens = torch.zeros(rows, dtype=torch.int32, device=device)
    slot = torch.empty(rows, dtype=torch.int32, device=device)
    for t, c in enumerate(ctxs):
        nswa = min(c, WINDOW)
        first = c - nswa
        pages = {}
        win = []
        for p in range(first, c):
            if p // SWA_BLOCK not in pages:
                pages[p // SWA_BLOCK] = swa_perm.pop()
            win.append(pages[p // SWA_BLOCK] * SWA_BLOCK + p % SWA_BLOCK)
        swa_idx[t, :nswa] = torch.tensor(win, dtype=torch.int32, device=device)
        swa_lens[t] = nswa
        slot[t] = win[-1]  # the row's own new record
        n_comp = c // RATIO
        ntopk = min(TOPK_KEYS, n_comp)
        if ntopk:
            sel = torch.randperm(n_comp, generator=g, device=device)[:ntopk].tolist()
            topk[t, :ntopk] = torch.tensor([comp_perm.pop() * COMP_BLOCK + r % COMP_BLOCK for r in sel],
                                           dtype=torch.int32, device=device)
        topk_lens[t] = ntopk
    pos = torch.tensor([c - 1 for c in ctxs], dtype=torch.int64, device=device)
    return {"ctxs": ctxs, "pos": pos, "slot": slot, "swa_idx": swa_idx, "swa_lens": swa_lens, "topk": topk,
            "topk_lens": topk_lens, "swa_pages": swa_pages, "comp_pages": comp_pages}


def fill_cache(pages, block, slots, g, device):
    """A paged cache of random bytes whose records at ``slots`` hold random N(0, 1) keys."""
    cache = torch.randint(0, 256, (pages, page_bytes(block)), generator=g, device=device, dtype=torch.uint8)
    slots = slots[slots >= 0].long()
    if slots.numel():
        data, scales = encode_records(torch.randn(slots.numel(), HEAD_DIM, generator=g, device=device,
                                                  dtype=torch.float64))
        write_records(cache, slots, data, scales, block)
    return cache


def _sinkhorn(x, iters=20):
    for _ in range(iters):
        x = x / x.sum(-1, keepdim=True)
        x = x / x.sum(-2, keepdim=True)
    return x


def seam_state(rows, g, device):
    """K1's seam outputs as K2 reads them: the residual streams and the attention sublayer's mixes."""
    res = torch.randn(rows, HC, HIDDEN, generator=g, device=device, dtype=torch.float64)
    res = (res * torch.rsqrt(res.square().mean(-1, keepdim=True))).to(torch.bfloat16)
    comb = _sinkhorn(torch.rand(rows, HC, HC, generator=g, device=device, dtype=torch.float64) + 0.1).float()
    post = (2 * torch.sigmoid(torch.randn(rows, HC, generator=g, device=device, dtype=torch.float64))).float()
    pre = (torch.sigmoid(torch.randn(rows, HC, generator=g, device=device, dtype=torch.float64)) + 1e-6).float()
    return {"res_in": res, "post_in": post, "comb_in": comb.contiguous(), "pre_in": pre}


def mhc_weights(g, device):
    return {
        "hc_fn": (0.01 * torch.randn(MIX, K_MIX, generator=g, device=device)).float(),
        "hc_scale": (0.5 * torch.randn(3, generator=g, device=device)).abs().float() + 0.5,
        "hc_base": (0.1 * torch.randn(MIX, generator=g, device=device)).float(),
    }


def _plant_router(h, bias, g, device):
    """A bf16 router weight whose logits for the rows' FFN inputs ``h`` have 6 clear winners a row:
    winners' selection keys (score + bias) spaced over WIN_KEYS, losers' logits in LOSE_LOGITS."""
    rows = h.shape[0]
    logits = _uniform((rows, EXPERTS), *LOSE_LOGITS, g, device)
    b = bias.to(torch.float64)
    for t in range(rows):
        win = torch.randperm(EXPERTS, generator=g, device=device)[:TOP_K]
        keys = torch.linspace(*WIN_KEYS, TOP_K, device=device, dtype=torch.float64)
        keys = keys + _uniform((TOP_K,), -0.01, 0.01, g, device)
        score = keys - b[win]  # sqrt(softplus(logit)) = score  <=>  logit = log(expm1(score^2))
        logits[t, win] = torch.log(torch.expm1(score.square()))
    base = GATE_STD * torch.randn(EXPERTS, HIDDEN, generator=g, device=device, dtype=torch.float64)
    hh = h.to(torch.float64)
    # base + C h with (base + C h) h^T = logits^T: the minimum-norm correction in the rows' span
    corr = torch.linalg.solve(hh @ hh.T, (logits.T - base @ hh.T).T).T
    return (base + corr @ hh).to(torch.bfloat16)


def _check_routing(h, gate, bias):
    _, _, _, key = route(h, gate, bias)
    top = key.sort(dim=-1, descending=True).values
    margin = float((top[:, TOP_K - 1] - top[:, TOP_K]).min())
    gap = float((top[:, :TOP_K - 1] - top[:, 1:TOP_K]).min())
    if margin < ROUTE_MARGIN or gap < ORDER_GAP:
        raise RuntimeError(f"routing margins too small: top-6 margin {margin:.4f}, order gap {gap:.4f}")


def make_inputs(case, inputs, generator, device):
    axes, g = case["axes"], generator
    rows, ctx = axes["rows"], axes["ctx"]
    step = make_step(rows, ctx, g, device)
    kt, klen = key_table(step["swa_idx"], step["swa_lens"], step["topk"], step["topk_lens"], step["slot"])
    inp = {
        "q": (Q_STD * torch.randn(rows, N_HEADS, HEAD_DIM, generator=g, device=device)).to(torch.bfloat16),
        "swa": fill_cache(step["swa_pages"], SWA_BLOCK, step["swa_idx"].flatten(), g, device),
        "comp": fill_cache(step["comp_pages"], COMP_BLOCK, step["topk"].flatten(), g, device),
        "kt": kt, "klen": klen, "pos": step["pos"],
        "sink": (SINK_MEAN + torch.randn(N_HEADS, generator=g, device=device)).float(),
        "cos_sin": rope_table(device),
        "woa": _fp8((O_GROUPS * O_RANK, GROUP_K), g, device),
        "woa_s": _codes((O_GROUPS * O_RANK // MX, GROUP_K // MX), E8M0_WOA, g, device),
        "wob": _fp8((HIDDEN, O_GROUPS * O_RANK), g, device),
        "wob_s": _codes((HIDDEN // MX, O_GROUPS * O_RANK // MX), E8M0_WOB, g, device),
        **seam_state(rows, g, device), **mhc_weights(g, device),
        "ffn_norm": (1.0 + 0.1 * torch.randn(HIDDEN, generator=g, device=device)).to(torch.bfloat16),
        "bias": (BIAS_STD * torch.randn(EXPERTS, generator=g, device=device)).clamp(-BIAS_CLIP, BIAS_CLIP),
        "w13": torch.randint(0, 256, (EXPERTS, 2 * INTER, HIDDEN // 2), generator=g, device=device,
                             dtype=torch.uint8),
        "w13_s": _codes((EXPERTS, 2 * INTER, HIDDEN // MX), E8M0_EXPERT, g, device),
        "w2": torch.randint(0, 256, (EXPERTS, HIDDEN, INTER // 2), generator=g, device=device, dtype=torch.uint8),
        "w2_s": _codes((EXPERTS, HIDDEN, INTER // MX), E8M0_EXPERT, g, device),
        "sgu": (torch.randn(2 * INTER, HIDDEN, generator=g, device=device)).clamp(-6, 6).to(torch.float8_e4m3fn),
        "sgu_s": _codes((2 * INTER // MX, HIDDEN // MX), E8M0_EXPERT, g, device),
        "sw2": (torch.randn(HIDDEN, INTER, generator=g, device=device)).clamp(-6, 6).to(torch.float8_e4m3fn),
        "sw2_s": _codes((HIDDEN // MX, INTER // MX), E8M0_EXPERT, g, device),
    }
    ffn = k2_ffn_input({**inp, "swa_block": SWA_BLOCK, "comp_block": COMP_BLOCK})
    inp["gate_w"] = _plant_router(ffn["y"], inp["bias"], g, device)
    _check_routing(ffn["y"], inp["gate_w"], inp["bias"])
    return inp


# ------------------------------------------------------------------------------------------ layouts
def pack_mxfp4(q, gate_up):
    """An MXFP4 bank [E, N, K / 2] in aiter's gfx950 shuffle_weight layout (16-row tiles, a 64-byte K step
    as [4 lane groups][16 rows][16 B]); ``gate_up``: rows are [gate; up] halves, interleaved a 16-row tile
    each (is_guinterleave). The tile order matches mx.fp4_tile_load and moe.ug_wave_pair."""
    e, n, kp = q.shape
    if gate_up:
        w = q.view(e, 2, n // 32, 16, kp // 64, 4, 16).permute(0, 2, 1, 4, 5, 3, 6)
    else:
        w = q.view(e, n // 16, 16, kp // 64, 4, 16).permute(0, 1, 3, 4, 2, 5)
    return w.contiguous().view(e, n, kp)


def pack_mxfp4_scales(s, gate_up):
    """E8M0 scales [E, N, K / 32] in aiter's shuffle_scale layout (mx.scale_index: [row / 32][col / 8]
    [col % 4][row % 16][(col / 4) % 2][(row / 16) % 2]); ``gate_up``: the rows interleaved as pack_mxfp4's."""
    e, n, kb = s.shape
    if gate_up:
        t = s.reshape(e, 2, n // 32, 16, kb // 8, 2, 4).permute(0, 2, 4, 6, 3, 5, 1)
    else:
        t = s.reshape(e, n // 32, 2, 16, kb // 8, 2, 4).permute(0, 1, 4, 6, 3, 5, 2)
    return t.contiguous().view(e, n, kb)


def _rank_rows(t, start, size, pad):
    """Rows [start, start + size) of t's second axis, padded with zeros to ``pad`` rows."""
    out = torch.zeros(t.shape[0], pad, *t.shape[2:], dtype=t.dtype, device=t.device)
    out[:, :size] = t[:, start:start + size]
    return out


def shard_inputs(logical, *, rank, world_size):
    """Rank ``rank``'s kernel inputs before to_kernel_layout (see the module docstring)."""
    assert world_size == TP, world_size
    r = rank
    out = dict(logical)
    out["q"] = logical["q"][:, r * HEADS_RANK:(r + 1) * HEADS_RANK].contiguous()
    out["sink"] = logical["sink"][r * HEADS_RANK:(r + 1) * HEADS_RANK].contiguous()
    out["woa"] = logical["woa"][r * O_COLS:(r + 1) * O_COLS].contiguous()
    out["woa_s"] = logical["woa_s"][r * O_COLS // MX:(r + 1) * O_COLS // MX].contiguous()
    out["wob"] = logical["wob"][:, r * O_COLS:(r + 1) * O_COLS].contiguous()
    out["wob_s"] = logical["wob_s"][:, r * O_COLS // MX:(r + 1) * O_COLS // MX].contiguous()
    w13, s13 = logical["w13"], logical["w13_s"]
    out["w13"] = torch.cat([_rank_rows(w13, r * INTER_RANK, INTER_RANK, INTER_PAD),
                            _rank_rows(w13, INTER + r * INTER_RANK, INTER_RANK, INTER_PAD)], dim=1)
    out["w13_s"] = torch.cat([_rank_rows(s13, r * INTER_RANK, INTER_RANK, INTER_PAD),
                              _rank_rows(s13, INTER + r * INTER_RANK, INTER_RANK, INTER_PAD)],
                             dim=1).reshape(EXPERTS * 2 * INTER_PAD, HIDDEN // MX)
    w2 = torch.zeros(EXPERTS, HIDDEN, INTER_PAD // 2, dtype=torch.uint8, device=logical["w2"].device)
    w2[..., :INTER_RANK // 2] = logical["w2"][..., r * INTER_RANK // 2:(r + 1) * INTER_RANK // 2]
    out["w2"] = w2
    s2 = torch.full((EXPERTS, HIDDEN, 24), UNIT_E8M0, dtype=torch.uint8, device=w2.device)
    s2[..., :INTER_PAD // MX] = 0
    s2[..., :INTER_RANK // MX] = logical["w2_s"][..., r * INTER_RANK // MX:(r + 1) * INTER_RANK // MX]
    out["w2_s"] = s2.reshape(EXPERTS * HIDDEN, 24)
    sgu, sgu_s = logical["sgu"], logical["sgu_s"]
    out["sgu"] = torch.cat([sgu[r * INTER_RANK:(r + 1) * INTER_RANK],
                            sgu[INTER + r * INTER_RANK:INTER + (r + 1) * INTER_RANK]]).contiguous()
    out["sgu_s"] = torch.cat([sgu_s[r * INTER_RANK // MX:(r + 1) * INTER_RANK // MX],
                              sgu_s[(INTER + r * INTER_RANK) // MX:(INTER + (r + 1) * INTER_RANK) // MX]]).contiguous()
    out["sw2"] = logical["sw2"][:, r * INTER_RANK:(r + 1) * INTER_RANK].contiguous()
    out["sw2_s"] = logical["sw2_s"][:, r * INTER_RANK // MX:(r + 1) * INTER_RANK // MX].contiguous()
    return out


def to_kernel_layout(case, inputs):
    """A rank's MXFP4 experts as SGLang's gfx950 Fp8MoEMethod leaves them (outside timing)."""
    inputs = dict(inputs)
    inputs["w13"] = pack_mxfp4(inputs["w13"], gate_up=True)
    inputs["w13_s"] = pack_mxfp4_scales(inputs["w13_s"].view(EXPERTS, 2 * INTER_PAD, -1), gate_up=True) \
        .reshape(EXPERTS * 2 * INTER_PAD, -1)
    inputs["w2"] = pack_mxfp4(inputs["w2"], gate_up=False)
    inputs["w2_s"] = pack_mxfp4_scales(inputs["w2_s"].view(EXPERTS, HIDDEN, -1), gate_up=False) \
        .reshape(EXPERTS * HIDDEN, -1)
    return inputs


def rank_expected(logical_outputs, *, rank, world_size):
    """Every output is replicated: every rank writes the whole tensors."""
    assert world_size == TP, world_size
    return dict(logical_outputs)

