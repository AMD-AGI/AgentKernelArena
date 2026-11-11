# SPDX-License-Identifier: Apache-2.0
"""Logical inputs, per-rank shards and per-rank expected outputs of one Muse Spark MXFP4 decoder layer
(attention -> K2 -> K3 -> next layer's K1) at TP8 / EP8, one DFlash2 verify step of R requests x 8 rows.

Logical inputs (``make_inputs``; identical on every rank, drawn from the case seed):

  q [rows, 8192] bf16, g [rows, 8192] bf16   this layer's K1 outputs for all 128 query heads (head h at
                                             columns 64 h ..): q per-head RMS-normed and scaled by
                                             qk_scale_factor 5.473, g the o_gate pre-activation
  k_cache, v_cache [kv_slots, 16, 64] fp8    this (full-attention) layer's page-16 NHD fp8 cache, all 16 KV
                                             heads, unit scale; it already holds the step's rows
  page_table [R, max_blocks] int32           page ids of each request (row padded with page 0)
  seq_lens [R] int32                         keys of each request before this step's 8 rows (the verify's
                                             committed prefix); request r attends seq_lens[r] + 8 keys
  res [rows, 8192] fp32                      the residual stream entering the layer (updated twice in place)
  w_o [8192, 8192] bf16                      o_proj (out x in)
  alpha, beta, alpha2, beta2 [8192] fp32     the post-attention / post-FFN residual gates
  pre_w, pfw, npw [8192], xn_w, pen_w [4096] bf16   effective norm weights (weight + gain centre)
  router_hi, router_lo [256, 8192] bf16     the fp32 sigmoid router [8192, 256] as its transposed bf16 hi + lo
                                             split (``weights._router_hi_lo``; the weight is hi + lo exactly)
  expert_bias [256] fp32                     the router's selection bias
  w_pool [4096, 8192] bf16, w_unpool [8192, 4096] bf16   pooler (8192 -> 4096) and unpooler
  w13 [256, 8192, 2048] u8, s13 [256, 8192, 128] u8      Quark MXFP4 experts: gate rows 0..4095, up rows
  w2 [256, 4096, 2048] u8, s2 [256, 4096, 128] u8        4096..8191; E2M1 codes two a byte (low nibble =
                                                          even k), E8M0 scales a 32-block (2^(s - 127))
  w_qkvg1 [18432, 8192] bf16                 the next layer's fused projection, rows q (8192) | k (1024) |
                                             v (1024) | o_gate (8192)
  cos_sin [ctx, 64] fp32                     the RoPE table (cos 32 | sin 32, theta 500000)
  positions, slots [rows] int64              the step's token positions and full-pool cache slots
  swa_map [kv_slots] int64                   the full-to-sliding slot map of the next (sliding) layer
  k_cache1, v_cache1 [SWA_SLOTS, 16, 64] fp8 the next layer's sliding-pool fp8 cache (random contents)
  guard [8] int32                            the runner's epoch / guard-word buffer: epoch 1, a 30 s wait
                                             deadline in 100 MHz ticks, clean error words (replicated)

Routing margins. The router's top-8 is a discrete decision that must not flip between the float64 reference
and the kernel (the kernel's routing key keeps 24 bits of the biased sigmoid score and its x differs from the
reference's by ~1e-4 relative). The generator plants each token's 8 experts: expert e owns hidden columns
32 e .. 32 e + 31; the residual stream of token t carries level L_j (85, 75, .., 15) along a random sign
pattern of the block of its j-th expert, over unit-normal noise; the router's column e reads that block
divided by (alpha * pre_w) (so the post-attention gate and the pre-FFN norm weight do not blur the levels),
plus N(0, 5e-4) everywhere. The 8 picks then stand apart from each other and from the other 248 experts by
several hundredths in the biased score (README.md gives the measured margins), far beyond the kernel's error,
and every token picks a different random set of experts (all 256 are exercised across a step).

Sharding (``shard_inputs``, rank r of 8; the runner's per-rank tensors):
  q, g, the o output: columns 1024 r .. (16 query heads); k_cache, v_cache, k_cache1, v_cache1: KV heads
  2 r, 2 r + 1; w_o: input columns 1024 r ..; w_qkvg1: q rows 1024 r .., k rows 8192 + 128 r .., v rows
  9216 + 128 r .., gate rows 10240 + 1024 r .. (2304 rows); the experts as TP slices of every expert
  (``weights.tp_expert_copies`` / ``test_muse_spark_k3_tp8.py``): w13 / s13 the gate rows 512 r .. and up rows
  4096 + 512 r .. (1024 rows), w2 / s2 the down rows 512 r .. (512 rows); everything else replicated
  (the pooler and unpooler whole: the kernel reads its rank's rows of them).

The kernels' mailbox scratch (whose contents depend on the implementation) is not an input: the baseline
keeps it in the framework's symmetric buffer, which is zeroed before every call (baseline/kernel.py).
"""
import math

import torch

HIDDEN = 8192
HEAD_DIM = 64
Q_HEADS = 128
KV_HEADS = 16
Q_LEN = 8
EXPERTS = 256
TOP_K = 8
MOE_HIDDEN = 4096
INTER = 4096
PAGE = 16
SWA_SLOTS = 32768
QK_SCALE = 5.4730064863875
RMS_EPS = 1e-5
ROPE_THETA = 500000.0
WORLD = 8
LOCAL_Q = HIDDEN // WORLD  # 1024: 16 query heads a rank
LOCAL_KV = KV_HEADS // WORLD  # 2 KV heads a rank
K_ROW0 = Q_HEADS * HEAD_DIM  # 8192: the fused projection's k rows
V_ROW0 = K_ROW0 + KV_HEADS * HEAD_DIM  # 9216
G_ROW0 = V_ROW0 + KV_HEADS * HEAD_DIM  # 10240
QKVG_ROWS = G_ROW0 + Q_HEADS * HEAD_DIM  # 18432
GUARD_WORDS = 8
DEADLINE_S = 30.0
CLOCK_HZ = 100_000_000  # the guard's s_memrealtime clock
# routing plant (see the module docstring)
ROUTE_BLOCK = HIDDEN // EXPERTS  # 32 hidden columns an expert's signature
ROUTE_LEVELS = (85.0, 75.0, 65.0, 55.0, 45.0, 35.0, 25.0, 15.0)
ROUTE_GAIN = 0.1
ROUTE_NOISE = 5e-4
BIAS_RANGE = 1e-3
WEIGHT_STD = 0.02
KV_CHUNK = 1 << 18  # cache slots drawn at a time (bounds the float32 temporary)


def _normal(shape, std, generator, device, dtype):
    out = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    return out.mul_(std).to(dtype)


def _uniform(shape, low, high, generator, device):
    return torch.rand(shape, generator=generator, device=device, dtype=torch.float32).mul_(high - low).add_(low)


def _norm_weight(n, generator, device):
    """An effective RMSNorm weight (weight + gain centre): 1 + N(0, 0.1), bfloat16."""
    return _normal((n,), 0.1, generator, device, torch.float32).add_(1.0).to(torch.bfloat16)


def _fp8_cache(slots, generator, device):
    """A page-16 NHD fp8 cache [slots, 16, 64] of unit-normal values, drawn in chunks."""
    out = torch.empty((slots, KV_HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device)
    for start in range(0, slots, KV_CHUNK):
        n = min(KV_CHUNK, slots - start)
        out[start:start + n] = torch.randn((n, KV_HEADS, HEAD_DIM), generator=generator, device=device,
                                           dtype=torch.float32).to(torch.float8_e4m3fn)
    return out


def _cos_sin(rows, device):
    """RoPE cos | sin [rows, 64] fp32 (interleaved pairs, theta 500000), computed in float64."""
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64, device=device) / HEAD_DIM))
    freqs = torch.arange(rows, dtype=torch.float64, device=device)[:, None] * inv[None, :]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.float32).contiguous()


def make_inputs(case, inputs, generator, device):
    """The logical inputs of one case (every rank draws the same from the case seed)."""
    axes = case["axes"]
    R, ctx = int(axes["requests"]), int(axes["ctx"])
    rows = R * Q_LEN
    max_blocks = int(axes["max_blocks"])
    kv_pages = int(axes["kv_pages"])
    kv_slots = kv_pages * PAGE
    bf16 = torch.bfloat16

    # ---- the step: ragged contexts (the longest is ctx; the others uniform in [ctx / 2, ctx]), pages, slots
    lens = [ctx]
    for _ in range(R - 1):
        lo = max(Q_LEN, ctx // 2)
        lens.append(int(torch.randint(lo, ctx + 1, (1,), generator=generator, device=device)))
    order = torch.randperm(R, generator=generator, device=device).tolist()
    lens = [lens[i] for i in order]
    perm = torch.randperm(kv_pages - 1, generator=generator, device=device) + 1  # page 0 stays unused
    page_table = torch.zeros((R, max_blocks), dtype=torch.int32, device=device)
    at = 0
    positions, slots = [], []
    for r, n_keys in enumerate(lens):
        n_pages = -(-n_keys // PAGE)
        page_table[r, :n_pages] = perm[at:at + n_pages].to(torch.int32)
        at += n_pages
        for j in range(Q_LEN):
            p = n_keys - Q_LEN + j
            positions.append(p)
            slots.append(int(page_table[r, p // PAGE]) * PAGE + p % PAGE)
    seq_lens = torch.tensor([n - Q_LEN for n in lens], dtype=torch.int32, device=device)
    positions = torch.tensor(positions, dtype=torch.int64, device=device)
    slots = torch.tensor(slots, dtype=torch.int64, device=device)
    swa_map = torch.randint(0, SWA_SLOTS, (kv_slots,), generator=generator, device=device, dtype=torch.int64)
    swa_map[slots] = torch.randperm(SWA_SLOTS, generator=generator, device=device)[:rows]  # distinct rows

    # ---- this layer's caches and its K1 outputs (q per-head normed and scaled; the o gate)
    k_cache = _fp8_cache(kv_slots, generator, device)
    v_cache = _fp8_cache(kv_slots, generator, device)
    q = torch.randn((rows, Q_HEADS, HEAD_DIM), generator=generator, device=device, dtype=torch.float32)
    q = (q * torch.rsqrt(q.square().mean(-1, keepdim=True) + RMS_EPS) * QK_SCALE).reshape(rows, HIDDEN).to(bf16)
    g = _normal((rows, HIDDEN), 1.0, generator, device, bf16)

    # ---- the layer's dense weights
    w_o = _normal((HIDDEN, HIDDEN), WEIGHT_STD, generator, device, bf16)
    alpha = _uniform((HIDDEN,), 0.5, 1.0, generator, device)
    beta = _uniform((HIDDEN,), 0.0, 0.5, generator, device)
    pre_w = _norm_weight(HIDDEN, generator, device)

    # ---- the residual stream and the router with planted picks
    picks = torch.stack([torch.randperm(EXPERTS, generator=generator, device=device)[:TOP_K]
                         for _ in range(rows)])  # [rows, 8]: token t's j-th expert
    signs = torch.randint(0, 2, (EXPERTS, ROUTE_BLOCK), generator=generator, device=device,
                          dtype=torch.int64).to(torch.float32).mul_(2).sub_(1).div_(math.sqrt(ROUTE_BLOCK))
    res = torch.randn((rows, HIDDEN), generator=generator, device=device, dtype=torch.float32)
    levels = torch.tensor(ROUTE_LEVELS, dtype=torch.float32, device=device)
    res3 = res.view(rows, EXPERTS, ROUTE_BLOCK)
    tok = torch.arange(rows, device=device)[:, None].expand(rows, TOP_K)
    res3[tok, picks] += levels[None, :, None] * signs[picks]
    router = torch.randn((HIDDEN, EXPERTS), generator=generator, device=device, dtype=torch.float32).mul_(ROUTE_NOISE)
    gain = (alpha * pre_w.float()).view(EXPERTS, ROUTE_BLOCK)
    e = torch.arange(EXPERTS, device=device)
    router.view(EXPERTS, ROUTE_BLOCK, EXPERTS)[e, :, e] += ROUTE_GAIN * signs / gain
    router_t = router.t().contiguous()  # weights._router_hi_lo: the fp32 router as bf16 hi + lo [EXPERTS, HIDDEN]
    router_hi = router_t.to(bf16)
    router_lo = (router_t - router_hi.float()).to(bf16)
    del router, router_t
    expert_bias = _uniform((EXPERTS,), -BIAS_RANGE, BIAS_RANGE, generator, device)

    # ---- pooler, experts, unpooler, FFN boundary
    w_pool = _normal((MOE_HIDDEN, HIDDEN), WEIGHT_STD, generator, device, bf16)
    xn_w = _norm_weight(MOE_HIDDEN, generator, device)
    w13 = torch.randint(0, 256, (EXPERTS, 2 * INTER, MOE_HIDDEN // 2), generator=generator, device=device,
                        dtype=torch.uint8)
    s13 = torch.randint(117, 122, (EXPERTS, 2 * INTER, MOE_HIDDEN // 32), generator=generator, device=device,
                        dtype=torch.uint8)
    w2 = torch.randint(0, 256, (EXPERTS, MOE_HIDDEN, INTER // 2), generator=generator, device=device,
                       dtype=torch.uint8)
    s2 = torch.randint(117, 122, (EXPERTS, MOE_HIDDEN, INTER // 32), generator=generator, device=device,
                       dtype=torch.uint8)
    pen_w = _norm_weight(MOE_HIDDEN, generator, device)
    w_unpool = _normal((HIDDEN, MOE_HIDDEN), WEIGHT_STD, generator, device, bf16)
    pfw = _norm_weight(HIDDEN, generator, device)
    alpha2 = _uniform((HIDDEN,), 0.5, 1.0, generator, device)
    beta2 = _uniform((HIDDEN,), 0.0, 0.5, generator, device)
    npw = _norm_weight(HIDDEN, generator, device)

    # ---- the next (sliding, RoPE) layer's K1
    w_qkvg1 = _normal((QKVG_ROWS, HIDDEN), WEIGHT_STD, generator, device, bf16)
    cos_sin = _cos_sin(ctx, device)
    k_cache1 = _fp8_cache(SWA_SLOTS, generator, device)
    v_cache1 = _fp8_cache(SWA_SLOTS, generator, device)

    return {
        "q": q, "g": g, "k_cache": k_cache, "v_cache": v_cache, "page_table": page_table, "seq_lens": seq_lens,
        "res": res, "w_o": w_o, "alpha": alpha, "beta": beta, "pre_w": pre_w, "router_hi": router_hi,
        "router_lo": router_lo,
        "expert_bias": expert_bias, "w_pool": w_pool, "xn_w": xn_w, "w13": w13, "s13": s13, "w2": w2, "s2": s2,
        "pen_w": pen_w, "pfw": pfw, "alpha2": alpha2, "beta2": beta2, "npw": npw, "w_unpool": w_unpool,
        "w_qkvg1": w_qkvg1, "cos_sin": cos_sin, "positions": positions, "slots": slots, "swa_map": swa_map,
        "k_cache1": k_cache1, "v_cache1": v_cache1, "guard": _guard_words(device),
    }


def _guard_words(device):
    """The runner's epoch buffer: the epoch (word 0), the wait deadline in 100 MHz ticks (words 1-2, an
    int64 as two int32), clean error words (3-5) and padding."""
    ticks = int(DEADLINE_S * CLOCK_HZ)
    lo = ticks & 0xFFFFFFFF
    words = [1, lo - (1 << 32) if lo >= 1 << 31 else lo, ticks >> 32] + [0] * (GUARD_WORDS - 3)
    return torch.tensor(words, dtype=torch.int32, device=device)


def _tp_rows13(t, r):
    """A rank's TP slice of the gate | up rows of every expert (``weights.tp_expert_copies``)."""
    rs = INTER // WORLD
    return torch.cat([t[:, r * rs:(r + 1) * rs], t[:, INTER + r * rs:INTER + (r + 1) * rs]], dim=1).contiguous()


def shard_inputs(logical, *, rank, world_size):
    """Rank ``rank``'s kernel inputs from the logical inputs (outside timing)."""
    assert world_size == WORLD, world_size
    r = rank
    qc = slice(r * LOCAL_Q, (r + 1) * LOCAL_Q)
    kvh = slice(r * LOCAL_KV, (r + 1) * LOCAL_KV)
    kvc = slice(r * LOCAL_KV * HEAD_DIM, (r + 1) * LOCAL_KV * HEAD_DIM)
    w = logical["w_qkvg1"]
    w_qkvg1 = torch.cat([w[qc], w[K_ROW0:V_ROW0][kvc], w[V_ROW0:G_ROW0][kvc], w[G_ROW0:][qc]], dim=0).contiguous()
    cs = MOE_HIDDEN // WORLD
    out = {
        "q": logical["q"][:, qc].contiguous(),
        "g": logical["g"][:, qc].contiguous(),
        "k_cache": logical["k_cache"][:, kvh].contiguous(),
        "v_cache": logical["v_cache"][:, kvh].contiguous(),
        "page_table": logical["page_table"],
        "seq_lens": logical["seq_lens"],
        "res": logical["res"].clone(),
        "w_o": logical["w_o"][:, qc].contiguous(),
        "w13": _tp_rows13(logical["w13"], r),
        "s13": _tp_rows13(logical["s13"], r),
        "w2": logical["w2"][:, r * cs:(r + 1) * cs].contiguous(),
        "s2": logical["s2"][:, r * cs:(r + 1) * cs].contiguous(),
        "w_qkvg1": w_qkvg1,
        "k_cache1": logical["k_cache1"][:, kvh].contiguous(),
        "v_cache1": logical["v_cache1"][:, kvh].contiguous(),
    }
    for name in ("alpha", "beta", "pre_w", "router_hi", "router_lo", "expert_bias", "w_pool", "xn_w", "pen_w", "pfw",
                 "alpha2", "beta2", "npw", "w_unpool", "cos_sin", "positions", "slots", "swa_map", "guard"):
        out[name] = logical[name]
    return out


def rank_expected(logical_outputs, *, rank, world_size):
    """What rank ``rank`` must write: its heads' slices of the sharded tensors, the replicated ones whole."""
    assert world_size == WORLD, world_size
    r = rank
    qc = slice(r * LOCAL_Q, (r + 1) * LOCAL_Q)
    kvh = slice(r * LOCAL_KV, (r + 1) * LOCAL_KV)
    kvc = slice(r * LOCAL_KV * HEAD_DIM, (r + 1) * LOCAL_KV * HEAD_DIM)
    out = logical_outputs
    return {
        "o": out["o"][:, qc],
        "ids": out["ids"],
        "scores": out["scores"],
        "xin": out["xin"],
        "attn_out": out["attn_out"],
        "k_out": out["k_out"][:, kvc],
        "v_out": out["v_out"][:, kvc],
        "q": out["q"][:, qc],
        "g": out["g"][:, qc],
        "res": out["res"],
        "k_cache1": out["k_cache1"][:, kvh],
        "v_cache1": out["v_cache1"][:, kvh],
    }
