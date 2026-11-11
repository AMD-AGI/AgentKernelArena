# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Logical inputs of one GLM-5.3 indexed decoder layer, the rank shards and the per-rank expected outputs.

Logical inputs (what reference.py reads) are the whole layer in SGLang's checkpoint layouts, row-major:

  w_qkv_a [2624, 6144], w_q_b [16384, 2048] (64 heads x (192 nope | 64 rope)), w_o [6144, 16384],
  w_ik [128, 6144], w_iq [4096, 2048]: E4M3 codes with f32 scales per 128 x 128 block (s_*);
  w_uk = SGLang's w_kc [64, 192, 512] and w_uv = w_vc [64, 512, 256]: E4M3 with one f32 scale each;
  w_iw (indexer weights_proj) [32, 6144] and w_r (router) [256, 6144] bf16, bias [256] f32,
  g_ik / b_ik (indexer k LayerNorm) f32 [128];
  w_ug = w13 [257, 4096, 6144] (rows 0..2047 gate, 2048..4095 up; expert 256 is the shared expert) and
  w_dn = w2 [257, 6144, 2048]: E4M3 with f32 128 x 128 block scales.
The step: ``rows`` rows (a decode token, or an MTP-verify draft token of a 4-token chain), each with its
position, MLA cache slot, top-k offset (indptr), page-table row (req_ids, page_table) and the caches: the
bf16 MLA cache [slots, 576] and the uint8 paged index cache [pages, 64, 132] (a page's 64 x 128 E4M3 keys in
16-token x 16-byte tiles, then 64 f32 scales). ``step_guard`` is the kernel's guard block (epoch 1 and a
600 s wait deadline in 100 MHz ticks); it is read-only.

Requests: rows // 4 MTP-verify requests of 4 rows, then rows % 4 decode requests of one row. The first
request holds ``ctx`` cached tokens, the others between ctx / 2 and ctx (at least 2048), so every row
scores at least 2049 keys and selects exactly 2048. Pages are a random permutation of the pool (page 0, the
pad page, unused); the rows of one request share its page-table row, as SGLang's DSA backend gives them.

Decision margins (so the reference and a kernel make the same discrete choices):
* index selection: the indexer's weights_proj is planted (a random matrix plus a component along the
  rows' normalized inputs) so every row's 32 head weights are positive. Each request's cached keys are then
  tiers of per-token power-of-two scales: 2048 - L "A" keys whose scores exceed every other key's at least
  twice over, L - 1 "B" keys ordered 2x apart below the request's own new keys, and the rest ("C") at least
  2x below; the new keys a row sees are its own and its request's earlier rows'. Each row of a request
  therefore selects A, its visible new keys and the top B keys, with every boundary at a 2x score ratio.
  The generator checks the ratio of the 2048th to the 2049th score of every row (>= 2) and raises otherwise.
* routing: the router is planted on the rows' post-attention normalized states (computed in float64 by
  glm53_math) so each row's 8 experts (random, distinct) have logits 2..3 and every other logit is at most
  0.6; with bias in [-0.05, 0.05] the gap between the 8th and 9th sigmoid + bias is checked (>= 0.05).

The rank shard (tensor parallelism 8, as SGLang shards the layer) slices the attention heads (8 a rank: q_b
rows, W_UK / W_UV heads, o_proj columns) and the expert intermediates (256 a rank: w13 gate and up rows, w2
columns); everything else is replicated. Kernel layouts, applied in the shard: AITER's (16, 16) shuffle of
every in-place block-FP8 weight (q_a | kv_a, q_b, o_proj, indexer wq_b and wk, experts), the kernel's packed
16-row x 64-k tiles for W_UK, W_UV (``pack_fp8``) and the bf16 router and weights_proj (``pack_bf16``),
W_UK / W_UV's per-tensor scale broadcast to their block-scale tables. Every output is replicated.
"""
import math

import torch

from glm53_math import (FP8_MAX, HEADS, HIDDEN, INDEX_DIM, INDEX_HEADS, INTER, KV_LORA, N_ROUTED, NOPE_DIM,
                        PAGE, PE_DIM, Q_LORA, QK_DIM, QKV_A_ROWS, TOP_K, TOPK, V_DIM, attention, front,
                        index_raw, fp8_values, pack_index_pages, rms_norm, route, select, updated_index_cache,
                        check_margins)

WORLD_SIZE = 8
LOCAL_HEADS = HEADS // WORLD_SIZE
LOCAL_INTER = INTER // WORLD_SIZE
N_EXPERTS = N_ROUTED + 1
VERIFY_ROWS = 4                 # MTP: 3 draft tokens + 1, one chain a request
WEIGHT_CODE_STD = 64.0          # E4M3 weight codes ~ N(0, 64^2); block scales carry the magnitude
CACHE_CODE_STD = 64.0
ROPE_THETA = 1.0e6
DEADLINE_TICKS = 600 * 100_000_000   # the guard's wait deadline: 600 s of the 100 MHz s_memrealtime clock
CHUNK = 1 << 24


def requests_of(rows):
    """Rows per request: verify chains of 4 first, then one-row decode requests."""
    return [VERIFY_ROWS] * (rows // VERIFY_ROWS) + [1] * (rows % VERIFY_ROWS)


def request_pages(ctx):
    return -(-(ctx + VERIFY_ROWS) // PAGE) + 1


# ----------------------------------------------------------------------------------- random helpers
def _codes(shape, g, device, std=WEIGHT_CODE_STD):
    """E4M3 codes of N(0, std^2) values, drawn in chunks (float8 storage only)."""
    out = torch.empty(shape, dtype=torch.uint8, device=device)
    flat = out.view(-1)
    for start in range(0, flat.numel(), CHUNK):
        n = min(CHUNK, flat.numel() - start)
        v = torch.randn(n, generator=g, device=device)
        flat[start:start + n] = v.mul_(std).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
    return out.view(torch.float8_e4m3fn)


def _block_scales(shape, k, g, device):
    """f32 block scales: a GEMV of unit-RMS inputs over ``k`` gives unit-RMS outputs."""
    return ((0.5 + torch.rand(shape, generator=g, device=device)) / (WEIGHT_CODE_STD * math.sqrt(k))).float()


def _gamma(n, g, device, dtype=torch.bfloat16):
    return (1.0 + 0.1 * torch.randn(n, generator=g, device=device)).to(dtype)


def _plant(x, targets, noise):
    """noise [m, n] plus the least change that makes x [r, n] @ W^T equal targets [r, m] (float64)."""
    dual = torch.linalg.solve(x @ x.T, x).T              # [n, r]: x @ dual = I
    return noise + (targets - x @ noise.T).T @ dual.T


def _pow2_floor(x):
    return 2.0 ** max(-60, min(30, math.floor(math.log2(x))))


def _pow2_ceil(x):
    return 2.0 ** max(-60, min(30, math.ceil(math.log2(x))))


def _rope_tables(n, device):
    inv = ROPE_THETA ** (-torch.arange(0, PE_DIM, 2, dtype=torch.float64, device=device) / PE_DIM)
    ang = torch.arange(n, dtype=torch.float64, device=device)[:, None] * inv[None]
    return ang.cos().bfloat16().contiguous(), ang.sin().bfloat16().contiguous()


def _draw_keys(n, accept, iq, iw, typ, g, device):
    """``n`` random key code rows whose score ratios to the rows' typical scores pass ``accept``: (E4M3 codes
    [n, 128], raw scores [n, rows])."""
    codes, raws, have = [], [], 0
    for _ in range(4096):
        cand = _codes((max(4 * n, 256), INDEX_DIM), g, device, CACHE_CODE_STD)
        raw = index_raw(fp8_values(cand), iq, iw)
        ok = accept(raw / typ)
        codes.append(cand.view(torch.uint8)[ok])
        raws.append(raw[ok])
        have += int(ok.sum())
        if have >= n:
            return torch.cat(codes)[:n].view(torch.float8_e4m3fn), torch.cat(raws)[:n]
    raise ValueError("could not draw planted index keys")


# ----------------------------------------------------------------------------------- the generator
def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    rows, ctx = axes["rows"], axes["ctx"]
    g = generator
    plan = requests_of(rows)
    n_req = len(plan)
    assert n_req == axes["requests"] and rows <= axes["rows"]
    contexts = [ctx] + [int(torch.randint(max(ctx // 2, TOPK), ctx + 1, (1,), generator=g, device=device))
                        for _ in plan[1:]]
    per_req = request_pages(ctx)
    pages = 1 + n_req * per_req
    assert per_req == axes["req_pages"] and pages == axes["pages"]
    perm = (torch.randperm(pages - 1, generator=g, device=device) + 1).to(torch.int32).view(n_req, per_req)
    req_of, positions = [], []
    for r, n_rows in enumerate(plan):
        for j in range(n_rows):
            req_of.append(r)
            positions.append(contexts[r] + j)
    req_t = torch.tensor(req_of, device=device)
    pos_t = torch.tensor(positions, dtype=torch.int64, device=device)
    page_table = perm[req_t].contiguous()
    slots = page_table.long().gather(1, (pos_t // PAGE)[:, None])[:, 0] * PAGE + pos_t % PAGE

    x = {}
    x["hidden"] = torch.randn(rows, HIDDEN, generator=g, device=device).bfloat16()
    x["positions"] = pos_t
    x["slots"] = slots
    x["indptr"] = (torch.arange(rows + 1, dtype=torch.int32, device=device) * TOPK)
    x["req_ids"] = torch.arange(rows, dtype=torch.int32, device=device)
    x["page_table"] = page_table
    x["kv_cache"] = torch.randn(pages * PAGE, QK_DIM, generator=g, device=device).bfloat16()
    x["rope_cos"], x["rope_sin"] = _rope_tables(axes["rope_rows"], device)
    x["g_in"], x["g_q"] = _gamma(HIDDEN, g, device), _gamma(Q_LORA, g, device)
    x["g_kv"], x["g_post"] = _gamma(KV_LORA, g, device), _gamma(HIDDEN, g, device)
    x["w_qkv_a"] = _codes((QKV_A_ROWS, HIDDEN), g, device)
    x["s_qkv_a"] = _block_scales((-(-QKV_A_ROWS // 128), HIDDEN // 128), HIDDEN, g, device)
    x["w_q_b"] = _codes((HEADS * (NOPE_DIM + PE_DIM), Q_LORA), g, device)
    x["s_q_b"] = _block_scales((HEADS * (NOPE_DIM + PE_DIM) // 128, Q_LORA // 128), Q_LORA, g, device)
    x["w_uk"] = _codes((HEADS, NOPE_DIM, KV_LORA), g, device)
    x["s_uk"] = _block_scales((1,), NOPE_DIM, g, device)
    x["w_uv"] = _codes((HEADS, KV_LORA, V_DIM), g, device)
    x["s_uv"] = _block_scales((1,), KV_LORA, g, device)
    x["w_o"] = _codes((HIDDEN, HEADS * V_DIM), g, device)
    x["s_o"] = _block_scales((HIDDEN // 128, HEADS * V_DIM // 128), HEADS * V_DIM, g, device)
    x["w_ik"] = _codes((INDEX_DIM, HIDDEN), g, device)
    x["s_ik"] = _block_scales((1, HIDDEN // 128), HIDDEN, g, device)
    x["w_iq"] = _codes((INDEX_HEADS * INDEX_DIM, Q_LORA), g, device)
    x["s_iq"] = _block_scales((INDEX_HEADS * INDEX_DIM // 128, Q_LORA // 128), Q_LORA, g, device)
    x["g_ik"] = _gamma(INDEX_DIM, g, device, torch.float32)
    x["b_ik"] = (0.1 * torch.randn(INDEX_DIM, generator=g, device=device)).float()
    x["w_ug"] = _codes((N_EXPERTS, 2 * INTER, HIDDEN), g, device)
    x["s_ug"] = _block_scales((N_EXPERTS, 2 * INTER // 128, HIDDEN // 128), HIDDEN, g, device)
    x["w_dn"] = _codes((N_EXPERTS, HIDDEN, INTER), g, device)
    x["s_dn"] = _block_scales((N_EXPERTS, HIDDEN // 128, INTER // 128), INTER, g, device)

    # indexer head weights: positive for every row (planted on the rows' normalized inputs)
    xn = rms_norm(x["hidden"].double(), x["g_in"])
    iw_target = 0.5 + torch.rand(rows, INDEX_HEADS, generator=g, device=device, dtype=torch.float64)
    iw_noise = torch.randn(INDEX_HEADS, HIDDEN, generator=g, device=device, dtype=torch.float64) / math.sqrt(HIDDEN)
    x["w_iw"] = _plant(xn, iw_target, iw_noise).bfloat16()
    f = front(x)

    # the index cache: natural random keys everywhere, then each request's planted tiers
    codes = _codes((pages * PAGE, INDEX_DIM), g, device, CACHE_CODE_STD).view(torch.uint8)
    scales = torch.full((pages * PAGE,), 2.0**-7, dtype=torch.float32, device=device)
    row0 = 0
    for r, n_rows in enumerate(plan):
        rr = list(range(row0, row0 + n_rows))
        row0 += n_rows
        _plant_request(codes, scales, page_table[rr[0]], contexts[r], f, rr, g, device)
    x["index_cache"] = pack_index_pages(codes.view(pages, PAGE, INDEX_DIM), scales.view(pages, PAGE))

    picked, edge = select(updated_index_cache(x, f), x, f)
    a = attention(x, f, picked)

    # the router: each row's 8 experts planted on its post-attention normalized state
    z = rms_norm(a, x["g_post"])
    logits = (0.5 * torch.randn(rows, N_ROUTED, generator=g, device=device, dtype=torch.float64)).clamp(-1.5, 0.6)
    for s in range(rows):
        winners = torch.randperm(N_ROUTED, generator=g, device=device)[:TOP_K]
        logits[s, winners] = 2.0 + torch.rand(TOP_K, generator=g, device=device, dtype=torch.float64)
    r_noise = torch.randn(N_ROUTED, HIDDEN, generator=g, device=device, dtype=torch.float64) / math.sqrt(HIDDEN)
    x["w_r"] = _plant(z, logits, r_noise).bfloat16()
    x["bias"] = (0.1 * torch.rand(N_ROUTED, generator=g, device=device) - 0.05).float()
    _, _, keyed = route(x, a)
    check_margins(edge, keyed)

    guard = torch.zeros(8, dtype=torch.int64)
    guard[0] = 1
    guard[1], guard[2] = DEADLINE_TICKS & 0xFFFFFFFF, DEADLINE_TICKS >> 32
    guard = torch.where(guard >= 1 << 31, guard - (1 << 32), guard)
    x["step_guard"] = guard.to(torch.int32).to(device)
    return x


def _plant_request(codes, scales, pages_row, context, f, rr, g, device):
    """Overwrite one request's cached keys 0 .. context - 1 with the tiered keys (see the module docstring)."""
    n_rows = len(rr)
    iq, iw = f["iq"][rr], f["iw"][rr]
    slots = pages_row.long()[torch.arange(context, device=device) // PAGE] * PAGE + \
        torch.arange(context, device=device) % PAGE
    raw_c = index_raw(fp8_values(codes[slots].view(torch.float8_e4m3fn)), iq, iw)                  # [context, rows]
    typ = raw_c.median(0).values
    # this launch's keys: row j's key at position context + j; row s sees keys j <= s
    new_raw = index_raw(fp8_values(f["ik_codes"][rr]), iq, iw)          # [keys j, rows s]
    new_score = new_raw * f["ik_scale"][rr][:, None]
    visible = torch.ones(n_rows, n_rows, dtype=torch.bool, device=device).tril().T   # [j, s]: j <= s
    inf = torch.full_like(new_score, math.inf)
    new_min = torch.where(visible, new_score, inf).amin(0)              # [rows]
    new_max = torch.where(visible, new_score, -inf).amax(0)
    if bool((new_min <= 0).any()):
        raise ValueError("a new index key scores 0 for a row that sees it")
    chosen = torch.randperm(context, generator=g, device=device)[:TOPK - 1]
    a_pos, b_pos = chosen[:TOPK - n_rows], chosen[TOPK - n_rows:]
    a_codes, a_raw = _draw_keys(TOPK - n_rows, lambda q: q.amin(1) >= 0.6, iq, iw, typ, g, device)
    b_codes, b_raw = _draw_keys(n_rows - 1, lambda q: (q.amin(1) >= 0.8) & (q.amax(1) <= 1.25),
                                iq, iw, typ, g, device)
    # B_j below every row's new keys, each 2x below the previous; C 2x below the last B (or the new keys)
    upper = new_min / 2
    b_scales = []
    for j in range(n_rows - 1):
        sj = _pow2_floor(float((upper / b_raw[j]).min()))
        b_scales.append(sj)
        upper = sj * b_raw[j] / 2
    rest = torch.ones(context, dtype=torch.bool, device=device)
    rest[chosen] = False
    c_scale = _pow2_floor(float((upper / raw_c[rest].amax(0)).min()))
    a_floor = 2 * torch.maximum(new_max, (b_scales[0] * b_raw[0]) if b_scales else new_max)
    a_scale = _pow2_ceil(float((a_floor / a_raw.amin(0)).max()))
    tier_c = c_scale * torch.exp2(-torch.randint(0, 3, (context,), generator=g, device=device).double())
    scales[slots] = tier_c.float()
    codes[slots[a_pos]] = a_codes.view(torch.uint8)
    scales[slots[a_pos]] = a_scale
    if n_rows > 1:
        codes[slots[b_pos]] = b_codes.view(torch.uint8)
        scales[slots[b_pos]] = torch.tensor(b_scales, dtype=torch.float32, device=device)


# ----------------------------------------------------------------------------------- kernel layouts
def shuffle_16x16(w):
    """AITER shuffle_weight(w, layout=(16, 16)) of a 1-byte [..., N, K] weight: [N/16, K/32, 2, 16 rows, 16 B]."""
    shape = w.shape
    n, k = shape[-2], shape[-1]
    t = w.contiguous().view(torch.uint8).view(-1, n // 16, 16, k // 32, 2, 16)
    return t.permute(0, 1, 3, 4, 2, 5).contiguous().view(shape).view(w.dtype)


def pack_fp8(q):
    """The kernel's packed FP8 tiles of [rows, K] (16 rows x 64 k): flat, sgl-amd-mono packing.pack_fp8."""
    rows, k = q.shape
    w8 = q.contiguous().view(torch.uint8).reshape(rows // 16, 16, k // 64, 2, 4, 8)
    return w8.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1).view(q.dtype)


def pack_bf16(w):
    """The kernel's packed bf16 tiles of [rows, K] (16 rows x 64 k): flat, sgl-amd-mono packing.pack_bf16."""
    rows, k = w.shape
    w16 = w.contiguous().view(torch.int16).reshape(rows // 16, 16, k // 64, 2, 4, 8)
    return w16.permute(0, 2, 3, 4, 1, 5).contiguous().view(-1).view(torch.bfloat16)


def shard_inputs(inputs, *, rank, world_size):
    """One rank's kernel inputs: its TP shard of the heads and expert intermediates, in kernel layouts."""
    if world_size != WORLD_SIZE:
        raise ValueError(f"this scenario is tensor parallelism {WORLD_SIZE}, got {world_size}")
    x = inputs
    h0, h1 = rank * LOCAL_HEADS, (rank + 1) * LOCAL_HEADS
    i0, i1 = rank * LOCAL_INTER, (rank + 1) * LOCAL_INTER
    q_rows = NOPE_DIM + PE_DIM
    out = {name: x[name] for name in ("hidden", "positions", "slots", "indptr", "req_ids", "page_table",
                                      "rope_cos", "rope_sin", "g_in", "g_q", "g_kv", "g_post", "s_qkv_a",
                                      "s_ik", "s_iq", "g_ik", "b_ik", "bias", "step_guard")}
    out["kv_cache"] = x["kv_cache"].clone()
    out["index_cache"] = x["index_cache"].clone()
    out["w_qkv_a"] = shuffle_16x16(x["w_qkv_a"])
    out["w_q_b"] = shuffle_16x16(x["w_q_b"][h0 * q_rows:h1 * q_rows])
    out["s_q_b"] = x["s_q_b"][h0 * q_rows // 128:h1 * q_rows // 128].contiguous()
    uk = x["w_uk"][h0:h1].transpose(1, 2).reshape(LOCAL_HEADS * KV_LORA, NOPE_DIM)
    out["w_uk"] = pack_fp8(uk)
    out["s_uk"] = torch.full((LOCAL_HEADS * KV_LORA // 128, NOPE_DIM // 64), float(x["s_uk"][0]),
                             dtype=torch.float32, device=uk.device)
    uv = x["w_uv"][h0:h1].transpose(1, 2).reshape(LOCAL_HEADS * V_DIM, KV_LORA)
    out["w_uv"] = pack_fp8(uv)
    out["s_uv"] = torch.full((LOCAL_HEADS * V_DIM // 128, KV_LORA // 128), float(x["s_uv"][0]),
                             dtype=torch.float32, device=uv.device)
    out["w_o"] = shuffle_16x16(x["w_o"][:, h0 * V_DIM:h1 * V_DIM])
    out["s_o"] = x["s_o"][:, h0 * V_DIM // 128:h1 * V_DIM // 128].contiguous()
    out["w_ik"] = shuffle_16x16(x["w_ik"])
    out["w_iq"] = shuffle_16x16(x["w_iq"])
    out["w_iw"] = pack_bf16(x["w_iw"])
    out["w_r"] = pack_bf16(x["w_r"])
    w13 = torch.cat([x["w_ug"][:, i0:i1], x["w_ug"][:, INTER + i0:INTER + i1]], 1)
    out["w_ug"] = shuffle_16x16(w13)
    out["s_ug"] = torch.cat([x["s_ug"][:, i0 // 128:i1 // 128],
                             x["s_ug"][:, (INTER + i0) // 128:(INTER + i1) // 128]], 1).contiguous()
    out["w_dn"] = shuffle_16x16(x["w_dn"][:, :, i0:i1])
    out["s_dn"] = x["s_dn"][:, :, i0 // 128:i1 // 128].contiguous()
    return out


def rank_expected(outputs, *, rank, world_size):
    """Every output and cache is replicated: each rank writes the whole logical tensor."""
    return {name: outputs[name] for name in ("out", "topk", "kv_cache", "index_cache")}


def index_cache_values(cache):
    """The decode hook of ``compare.index_cache``: the paged index cache [pages, 64, 132] as float64 key
    values [pages, 64, 128] (tokens in page order) = E4M3 code x the token's f32 scale."""
    pages = cache.shape[0]
    flat = cache.reshape(pages, PAGE * (INDEX_DIM + 4))
    codes = flat[:, :PAGE * INDEX_DIM].reshape(pages, 4, 8, 16, 16).permute(0, 1, 3, 2, 4)
    codes = fp8_values(codes.reshape(pages, PAGE, INDEX_DIM).contiguous().view(torch.float8_e4m3fn))
    scales = flat[:, PAGE * INDEX_DIM:].contiguous().view(torch.float32).double().view(pages, PAGE, 1)
    return codes * scales
