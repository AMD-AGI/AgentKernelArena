# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Logical inputs of one MiniMax-M3 sparse MoE layer step on a TP4 group, each rank's shard, the kernel-facing
weight layouts and each rank's expected outputs.

Logical inputs (what make_inputs returns and reference.py reads):
  replicated, as every rank sees them: ar, res [S, 6144] bf16 (the previous layer's all-reduced MLP output and
    its residual); positions, slot_mapping [S] int64; block_table [R, 1024] int32 (128-token page ids, a row a
    request); seq_lens [S] int32 (a row each); the norm gains g_in, g_post [6144] and g_q, g_k,
    g_iq, g_ik [128] bf16 (Gemma: the kernel applies gain + 1); cos_sin [131072, 64] bf16 (cos | sin of the 64
    rotary dims); k_scale, v_scale [1] fp32 (the layer's fixed cache scales); w_gate [128, 6144] bf16 and
    bias [128] fp32 (router); index_cache [slots, 128] E4M3 (the indexer's key cache, unit scale).
  rank-specific, stacked on a leading TP axis (rank r's slice of the model, as SGLang shards it):
    w_qkv [4, 2560, 6144] E4M3 + s_qkv [4, 2560] fp32: per-output-channel FP8 (amax / 448) of the rank's fused
      projection q (its 16 heads) | k | v (its KV head) | its index q head | the index k head (replicated);
    w_o [4, 6144, 2048] E4M3 + s_o [4, 6144] fp32: per-channel FP8 of the rank's o_proj columns;
    w13 [4, 129, 1536, 3072] uint8 + s13 [4, 129, 1536, 192] uint8: MXFP4 gate (rows 0-767) and up (768-1535)
      rows of the rank's intermediate slice of the 128 routed experts and the fused shared expert (id 128),
      e2m1 codes two per byte along the hidden axis (low nibble first), one e8m0 byte per 32;
    w2 [4, 129, 6144, 384] uint8 + s2 [4, 129, 6144, 24] uint8: the down projection's slice, MXFP4 along INTER;
    k_cache, v_cache [4, slots, 128] E4M3: the rank's KV head, token-major (slot = page * 128 + offset).

shard_inputs picks rank r's slices and writes its caches in the kernel's page-16 SHUFFLE layout (separate K and
V planes: K page p holds slots 16 p .. 16 p + 15 as [d / 16][token][d % 16] bytes, V page p as [d][token]);
to_kernel_layout preshuffles the weights as AITER / SGLang load them (shuffle_weight (16, 16) on the FP8 and
MXFP4 codes, e8m0_shuffle on the MXFP4 scales). rank_expected maps the reference's logical outputs to rank r's
tensors the same way.

Steps: S = R * q_len rows, R requests of q_len = S / R consecutive rows (1: decode, 4: EAGLE3 verify of 3 draft
tokens);
row j of a request with context c sees c - q_len + 1 + j keys and writes the key at position c - q_len + j.
Request 0 has the case's context, the others a context drawn in [ctx / 2, ctx] (ragged batch). Pages are a
random permutation of 1 .. num_pages - 1 (page 0 is the allocator's reserved page).

Planted margins (the two discrete decisions must not flip between the reference and a correct kernel):
  block selection: for every rank and request with a scored context (> 16 blocks), 15 random blocks below the
    request's first new key are made hot: each holds, for every (rank, row) whose hot set contains it, an index
    key from the dual basis of those rows' index q (computed exactly as the reference does), scaled so its dot
    with its own row's index q is alpha |q|^2 (alpha = 1, 0.96, .., 0.44 over the rank's 15 hot blocks, so their
    order is clear too, also after the FP8 rounding of the planted keys) and ~0 with the others'. Every other index key is 0.25 N(0, 1): the 15 hot blocks beat
    every other scored block by a wide margin. The generator checks the margins. The same slot of the rank's K
    cache holds a key its row's q heads attend to (mean logit 6, against ~N(0, 1) for the random keys), so the
    selected blocks carry most of the attention mass, as the indexer intends: a wrong block moves the output.
  routing: the router weight is a random N(0, 0.02) matrix plus the least-squares correction that gives every
    row 4 random winners with logits in [1.5, 3] and the rest in [-4, -0.25] for the row's normalized router
    input (computed exactly as the reference does), so sigmoid + bias separates the top 4 by >= 0.27; checked.
"""
import math

import torch

from m3_numerics import (
    EXPERTS, FP8_MAX, HEAD_DIM, HIDDEN, INTER, N_ROUTED, QKV_ROWS, ROTARY_DIM, SM_SCALE, SPARSE_BLOCK, TOP_K,
    TOPK_BLOCKS, TP, attention_block, k1, n_blocks, requests_of, route,
)

O_K = 2048
MAX_BLOCKS = 1024  # block-table width: the served context 131072 / 128
MAX_POSITIONS = 131072
PAGE16 = 16
ROPE_THETA = 5.0e6
AR_STD, RES_STD, GAIN_STD = 0.25, 1.0, 0.1
GATE_STD, BIAS_STD, BIAS_CLIP = 0.02, 0.02, 0.05
# expert scale bytes: gate / up 2^-9 .. 2^-6, down 2^-11 .. 2^-8 (MoE output rms ~1.5, like the residual)
E8M0_RANGE = {"w13": (118, 122), "w2": (116, 120)}
COLD_INDEX_STD = 0.25
HOT_ALPHA_STEP = 0.04  # 0.02 left a 0.049 order gap after the FP8 key rounding on MI355X-generated data
HOT_ATTENTION_SCORE = 6.0  # mean attention logit of a hot block's key for its row's q heads
WIN_LOGITS, LOSE_LOGITS = (1.5, 3.0), (-4.0, -0.25)
SELECT_MARGIN, ORDER_GAP, ROUTE_MARGIN = 1.0, 0.05, 0.1
CHUNK = 1 << 20
RANK_SPECIFIC = ("w_qkv", "s_qkv", "w_o", "s_o", "w13", "s13", "w2", "s2", "k_cache", "v_cache")


def _uniform(shape, low, high, g, device):
    return low + (high - low) * torch.rand(shape, generator=g, device=device, dtype=torch.float64)


def _gain(n, g, device):
    return (GAIN_STD * torch.randn(n, generator=g, device=device)).to(torch.bfloat16)


def _ptpc_fp8(w):
    """SGLang's load-time FP8 copy (weights.linear_to_ptpc_fp8): per output channel, scale = amax / 448."""
    amax = w.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    scale = amax / FP8_MAX
    return (w / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), scale.float()


def _fp8_cache(slots, scale, g, device):
    out = torch.empty(slots, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
    for s0 in range(0, slots, CHUNK):
        n = min(CHUNK, slots - s0)
        x = torch.randn(n, HEAD_DIM, generator=g, device=device) / scale
        out[s0:s0 + n] = x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return out


def _rope_table(device):
    inv = 1.0 / ROPE_THETA ** (torch.arange(0, ROTARY_DIM // 2, device=device, dtype=torch.float64) / (ROTARY_DIM // 2))
    ang = torch.arange(MAX_POSITIONS, device=device, dtype=torch.float64)[:, None] * inv[None, :]
    return torch.cat([ang.cos(), ang.sin()], dim=1).to(torch.bfloat16).contiguous()


def _step(rows, n_req, ctx, num_pages, g, device):
    """Contexts, block table, sequence lengths, positions and slots of a step's rows."""
    assert rows % n_req == 0, (rows, n_req)
    q_len = rows // n_req
    assert ctx >= q_len, (q_len, ctx)
    low = max(ctx // 2, q_len)
    ctxs = [ctx] + torch.randint(low, ctx + 1, (n_req - 1,), generator=g, device=device).tolist()
    need = sum(n_blocks(c) for c in ctxs)
    assert need <= num_pages - 1, (need, num_pages)
    pages = torch.randperm(num_pages - 1, generator=g, device=device).to(torch.int32) + 1
    table = torch.zeros(n_req, MAX_BLOCKS, dtype=torch.int32, device=device)
    seq = torch.empty(rows, dtype=torch.int32, device=device)
    used = 0
    for i, c in enumerate(ctxs):
        nb = n_blocks(c)
        rs = slice(i * q_len, (i + 1) * q_len)
        table[i, :nb] = pages[used:used + nb]
        used += nb
        seq[rs] = torch.arange(c - q_len + 1, c + 1, dtype=torch.int32, device=device)
    positions = (seq - 1).to(torch.int64)
    row_table = table.repeat_interleave(q_len, dim=0)
    page = row_table.gather(1, (positions // SPARSE_BLOCK).unsqueeze(1)).squeeze(1).to(torch.int64)
    slots = page * SPARSE_BLOCK + positions % SPARSE_BLOCK
    return table, seq, positions, slots


def _plant_hot_blocks(inp, q_len, g, device):
    """Hot index blocks per (rank, request): -> {(r, first row): [hot block ids]} (see the module docstring)."""
    pre = k1(inp)  # bf16 q / index q as the reference computes them
    iq = pre["iq"]  # [TP, S, 128]
    seq = inp["seq_lens"].long().tolist()
    hot = {}
    cache = inp["index_cache"]
    k_scale = float(inp["k_scale"])
    for first, last in requests_of(inp["seq_lens"], q_len):
        scored = [k for k in range(first, last + 1) if n_blocks(seq[k]) > TOPK_BLOCKS]
        if not scored:
            continue
        # blocks wholly below the request's first new key: old keys only, scored for every scored row
        n_cand = min(min(n_blocks(seq[k]) - 1 for k in scored), (seq[first] - 1) // SPARSE_BLOCK)
        assert n_cand >= TOPK_BLOCKS - 1, (n_cand, seq[first])
        alphas = {}
        for r in range(TP):
            blocks = torch.randperm(n_cand, generator=g, device=device)[:TOPK_BLOCKS - 1].tolist()
            hot[r, first] = blocks
            for m, b in enumerate(blocks):
                alphas[r, b] = 1.0 - HOT_ALPHA_STEP * m
        for b in sorted({b for (_, b) in alphas}):
            pairs = [(r, j) for r in range(TP) if (r, b) in alphas for j in range(first, last + 1)]
            q = torch.stack([iq[r][j] for r, j in pairs])  # [m, 128]
            scale = torch.tensor([alphas[r, b] for r, _ in pairs], dtype=torch.float64, device=device)
            keys = torch.linalg.solve(q @ q.T, q) * (scale * q.square().sum(-1)).unsqueeze(-1)
            offsets = torch.randperm(SPARSE_BLOCK, generator=g, device=device)[:len(pairs)].long()
            page = int(inp["block_table"][first // q_len, b])
            cache[page * SPARSE_BLOCK + offsets] = keys.float().to(torch.float8_e4m3fn)
            # and at the same slot of the rank's K cache, a key its row's 16 q heads attend to strongly
            for (r, j), slot in zip(pairs, (page * SPARSE_BLOCK + offsets).tolist()):
                heads = pre["q"][r][j]  # [16, 128]
                u = (heads / heads.norm(dim=-1, keepdim=True)).sum(0)
                u = u / u.norm()
                beta = HOT_ATTENTION_SCORE / (SM_SCALE * float((heads @ u).mean()))
                inp["k_cache"][r, slot] = (beta * u / k_scale).clamp(-FP8_MAX, FP8_MAX).float().to(
                    torch.float8_e4m3fn)
    return hot


def _check_selection(att, hot, q_len, seq_lens):
    seq = seq_lens.long().tolist()
    for first, last in requests_of(seq_lens, q_len):
        for k in range(first, last + 1):
            n_blk = n_blocks(seq[k])
            if n_blk <= TOPK_BLOCKS:
                continue
            for r in range(TP):
                s = att["scores"][r][k][:n_blk - 1]
                want = hot[r, first]
                mask = torch.zeros_like(s, dtype=torch.bool)
                mask[want] = True
                hot_s = s[mask].sort(descending=True).values
                margin = float(hot_s[-1] - s[~mask].max())
                gap = float((hot_s[:-1] - hot_s[1:]).min())
                if margin < SELECT_MARGIN or gap < ORDER_GAP or sorted(att["selection"][r][k][:-1]) != sorted(want):
                    raise RuntimeError(f"block selection margin too small: rank {r} row {k} margin {margin:.3f} "
                                       f"order gap {gap:.3f}")


def _plant_router(xn, g, device):
    """A bf16 router weight whose logits for the rows' normalized inputs ``xn`` have 4 clear winners a row."""
    rows = xn.shape[0]
    logits = _uniform((rows, N_ROUTED), *LOSE_LOGITS, g, device)
    for k in range(rows):
        win = torch.randperm(N_ROUTED, generator=g, device=device)[:TOP_K]
        logits[k, win] = _uniform((TOP_K,), *WIN_LOGITS, g, device)
    base = GATE_STD * torch.randn(N_ROUTED, HIDDEN, generator=g, device=device, dtype=torch.float64)
    # base + C xn with (base + C xn) xn^T = logits^T: the min-norm correction in the rows' span
    corr = torch.linalg.solve(xn @ xn.T, (logits.T - base @ xn.T).T).T
    return (base + corr @ xn).to(torch.bfloat16)


def _check_routing(xn, gate, bias):
    ids, _, logits = route(xn, gate, bias)
    key = (torch.sigmoid(logits) + bias.double()).sort(dim=-1, descending=True).values
    margin = float((key[:, TOP_K - 1] - key[:, TOP_K]).min())
    if margin < ROUTE_MARGIN:
        raise RuntimeError(f"routing margin too small: {margin:.4f}")
    return ids


def make_inputs(case, inputs, generator, device):
    axes, g = case["axes"], generator
    rows, n_req, ctx = axes["rows"], axes["requests"], axes["ctx"]
    q_len = rows // n_req
    num_pages = axes["num_pages"]
    slots = num_pages * SPARSE_BLOCK
    table, seq, positions, slot_mapping = _step(rows, n_req, ctx, num_pages, g, device)
    k_scale = _uniform((1,), 0.015, 0.03, g, device).float()
    v_scale = _uniform((1,), 0.015, 0.03, g, device).float()
    inp = {
        "ar": (AR_STD * torch.randn(rows, HIDDEN, generator=g, device=device)).to(torch.bfloat16),
        "res": (RES_STD * torch.randn(rows, HIDDEN, generator=g, device=device)).to(torch.bfloat16),
        "positions": positions, "slot_mapping": slot_mapping, "block_table": table, "seq_lens": seq,
        "g_in": _gain(HIDDEN, g, device), "g_q": _gain(HEAD_DIM, g, device), "g_k": _gain(HEAD_DIM, g, device),
        "g_iq": _gain(HEAD_DIM, g, device), "g_ik": _gain(HEAD_DIM, g, device), "g_post": _gain(HIDDEN, g, device),
        "cos_sin": _rope_table(device), "k_scale": k_scale, "v_scale": v_scale,
        "bias": (BIAS_STD * torch.randn(N_ROUTED, generator=g, device=device)).clamp(-BIAS_CLIP, BIAS_CLIP),
    }
    # the rank's fused projection; the index k head's rows are the same on every rank
    w = torch.randn(TP, QKV_ROWS, HIDDEN, generator=g, device=device) * HIDDEN ** -0.5
    w[1:, QKV_ROWS - HEAD_DIM:] = w[0, QKV_ROWS - HEAD_DIM:]
    inp["w_qkv"], inp["s_qkv"] = _ptpc_fp8(w)
    del w
    w = torch.randn(TP, HIDDEN, O_K, generator=g, device=device) * (TP * O_K) ** -0.5
    inp["w_o"], inp["s_o"] = _ptpc_fp8(w)
    del w
    for name, shape in (("w13", (2 * INTER, HIDDEN)), ("w2", (HIDDEN, INTER))):
        inp[name] = torch.randint(0, 256, (TP, EXPERTS, shape[0], shape[1] // 2), generator=g, device=device,
                                  dtype=torch.uint8)
        inp["s" + name[1:]] = torch.randint(*E8M0_RANGE[name], (TP, EXPERTS, shape[0], shape[1] // 32),
                                            generator=g, device=device, dtype=torch.uint8)
    inp["k_cache"] = torch.stack([_fp8_cache(slots, float(k_scale), g, device) for _ in range(TP)])
    inp["v_cache"] = torch.stack([_fp8_cache(slots, float(v_scale), g, device) for _ in range(TP)])
    inp["index_cache"] = _fp8_cache(slots, 1.0 / COLD_INDEX_STD, g, device)
    hot = _plant_hot_blocks(inp, q_len, g, device)
    att = attention_block(inp)
    _check_selection(att, hot, q_len, seq)
    inp["w_gate"] = _plant_router(att["xn"], g, device)
    _check_routing(att["xn"], inp["w_gate"], inp["bias"])
    return inp


# ------------------------------------------------------------------------------------------ layouts
def _bytes(t):
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def _like(u8, t):
    return u8.view(torch.float8_e4m3fn) if t.dtype == torch.float8_e4m3fn else u8


def shuffle_weight(t):
    """aiter.ops.shuffle.shuffle_weight(t, layout=(16, 16)) of a byte matrix [..., N, K bytes]: 16-row x 32-byte
    tiles in 16-byte halves, so lane l of an MFMA reads row l % 16, bytes 16 (l / 16) .. + 16 of a 64-byte k step."""
    x = _bytes(t)
    n, kb = x.shape[-2:]
    y = x.reshape(-1, n // 16, 16, kb // 32, 2, 16).permute(0, 1, 3, 4, 2, 5).contiguous().reshape(x.shape)
    return _like(y, t)


def e8m0_shuffle(t):
    """aiter e8m0_shuffle of e8m0 scales [..., rows, groups] taken as one [rows', groups] matrix."""
    rows, groups = t.shape[-2:]
    x = t.reshape(-1, groups)
    r = x.shape[0]
    y = x.view(r // 32, 2, 16, groups // 8, 2, 4).permute(0, 3, 5, 2, 4, 1).contiguous()
    return y.reshape(t.shape)


def k_page16(cache):
    """Token-major [slots, 128] -> page-16 SHUFFLE K plane [slots / 16, 2048]: [d / 16][token][d % 16]."""
    x = _bytes(cache)
    n = x.shape[0] // PAGE16
    return _like(x.reshape(n, PAGE16, HEAD_DIM // 16, 16).permute(0, 2, 1, 3).contiguous().reshape(n, -1), cache)


def v_page16(cache):
    """Token-major [slots, 128] -> page-16 SHUFFLE V plane [slots / 16, 2048]: [d][token]."""
    x = _bytes(cache)
    n = x.shape[0] // PAGE16
    return _like(x.reshape(n, PAGE16, HEAD_DIM).permute(0, 2, 1).contiguous().reshape(n, -1), cache)


def shard_inputs(logical, *, rank, world_size):
    """Rank ``rank``'s kernel inputs (before to_kernel_layout): its slices, the replicated tensors, its caches
    in the page-16 SHUFFLE layout and its own copy of the index cache."""
    assert world_size == TP, world_size
    out = {}
    for name, value in logical.items():
        out[name] = value[rank].contiguous() if name in RANK_SPECIFIC else value
    out["k_cache"] = k_page16(logical["k_cache"][rank])
    out["v_cache"] = v_page16(logical["v_cache"][rank])
    out["index_cache"] = logical["index_cache"].clone()
    return out


def to_kernel_layout(case, inputs):
    """A rank's weights as SGLang / AITER prepare them at load time (outside timing)."""
    inputs = dict(inputs)
    for name in ("w_qkv", "w_o", "w13", "w2"):
        inputs[name] = shuffle_weight(inputs[name])
    for name in ("s13", "s2"):
        inputs[name] = e8m0_shuffle(inputs[name])
    return inputs


def rank_expected(logical_outputs, *, rank, world_size):
    """Rank ``rank``'s expected tensors: the replicated rows and its caches in the kernel's layout."""
    assert world_size == TP, world_size
    out = logical_outputs
    return {"h_out": out["h_out"], "h_mid": out["h_mid"], "ar_out": out["ar_out"],
            "k_cache": k_page16(out["k_cache"][rank]), "v_cache": v_page16(out["v_cache"][rank]),
            "index_cache": out["index_cache"]}
