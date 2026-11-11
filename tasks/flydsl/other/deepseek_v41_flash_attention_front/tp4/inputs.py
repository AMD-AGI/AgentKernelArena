# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Logical inputs of one DeepSeek-V4.1-Flash K1 launch (decode step) on a TP4 group, each rank's shard and
each rank's expected outputs.

Logical inputs (what make_inputs returns and reference.py reads):
  res_in [S, 4, 5120] bf16, pend [S, 5120] bf16: the previous layer's residual streams and its MoE output (its
    FFN post still owed); post_in, pre_in [S, 4] and comb_in [S, 4, 4] f32: that layer's FFN post, pre and comb
    mixes; hc_fn [24, 20480], hc_scale [3], hc_base [24] f32 and attn_norm [5120] bf16: this layer's attention
    mHC and input_layernorm;
  wqkv [1792, 5120] E4M3 + wqkv_s [56, 160]: the fused wq_a | wkv_a (replicated), 32 x 32 E8M0 blocks;
  q_norm [1280], kv_norm [512] bf16; wqb [32768, 1280] E4M3 + wqb_s [1024, 40]: wq_b of all 64 heads
    (rank r holds rows 8192 r ..); cos_sin [1048576, 64] f32 (32 cos, then 32 sin);
  pos [S] int64, slot [S] int32 (each row's new SWA slot); swa [swa_pages, 37440] uint8: the layer's paged SWA
    cache (64 records a page; replicated, every rank writes its own copy); swa_idx [S, 128], swa_lens [S],
    topk [S, 512], topk_lens [S] int32: the step's window slots and top-k compressed rows (SGLang's metadata).
shard_inputs gives rank r its wq_b rows; rank_expected its 16 heads of q; everything else is replicated.

Step model (decode, a row a request): as K2's task (row 0 at the case's context ctx, the others in
[ctx / 2, ctx]; window = the last min(c, 128) positions, the row's own new slot last; top-k = min(512, c // 2)
compressed rows). The cache bytes outside the step's records are random.

Planted exactness of the KV record (a packed uint8 cache, compared bitwise). Every rounding on the record's
path is kept at least REL = 2e-5 (relative to the magnitude of the terms it sums) from a rounding boundary, far
past the kernel's float32 error, so a correct kernel writes the reference's bytes exactly:
  * the seam: the generator computes R', x and y as the reference does and keeps the hidden columns whose bf16
    roundings (R' of all 4 streams, x, y) are safe for every row and whose MXFP8 block code survives a
    one-ulp move of the block's bf16 amax (an unsafe element may be the block's largest);
  * wkv_a (wqkv rows 1280 .. 1791) has one nonzero E4M3 weight a row, at a random safe column: kv_a is one
    product of two E4M3 values and two powers of two, exact in float32 and in bf16 (the q rows stay dense);
  * kv_norm: a dimension whose record value (E4M3 code of the NoPE dims, bf16 of the RoPE dims) or 64-block
    scale code is not safe for every row gets new random bf16 gains until it is (checked).
"""
import torch

from dsv41_math import (
    HC, HEAD_DIM, HIDDEN, K_MIX, KV_DIM, MIX, MX, N_HEADS, NOPE, Q_RANK, QKV_ROWS, TOPK_KEYS, TP, WINDOW,
    bf16, encode_records, fp8_block_dequant, k1_front, mx_codes, mx_quant, page_bytes, rope_rotate, seam,
)

SWA_BLOCK = 64
RATIO = 2
MAX_POSITIONS = 1 << 20
ROPE_THETA = 10000.0
REL = 2e-5
# an unsafe element of an x8 block may move the block's bf16 amax by one ulp: its code must survive that
REL_CODE = 2.0 ** -7
FP8_STD = 8.0
E8M0_QKV, E8M0_QB = (117, 122), (117, 122)
PEND_STD = 0.5
HEADS_RANK = N_HEADS // TP
QB_ROWS = HEADS_RANK * HEAD_DIM  # 8192: a rank's wq_b rows
CHUNK = 1 << 18
TRIES = 64


def _codes(shape, bounds, g, device):
    return torch.randint(*bounds, shape, generator=g, device=device, dtype=torch.uint8)


def _fp8(shape, g, device):
    return (FP8_STD * torch.randn(shape, generator=g, device=device)).clamp(-448, 448).to(torch.float8_e4m3fn)


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
    """Contexts, positions, slots, the window and top-k slots of every row (see the module docstring)."""
    low = max(ctx // 2, 1)
    ctxs = [ctx] + torch.randint(low, ctx + 1, (rows - 1,), generator=g, device=device).tolist()
    swa_pages = 3 * rows + 1
    swa_perm = (torch.randperm(swa_pages - 1, generator=g, device=device) + 1).tolist()
    comp_perm = (torch.randperm(TOPK_KEYS * rows, generator=g, device=device) + 1).tolist()
    swa_idx = torch.full((rows, WINDOW), -1, dtype=torch.int32, device=device)
    topk = torch.full((rows, TOPK_KEYS), -1, dtype=torch.int32, device=device)
    swa_lens = torch.zeros(rows, dtype=torch.int32, device=device)
    topk_lens = torch.zeros(rows, dtype=torch.int32, device=device)
    slot = torch.empty(rows, dtype=torch.int32, device=device)
    for t, c in enumerate(ctxs):
        nswa = min(c, WINDOW)
        pages, win = {}, []
        for p in range(c - nswa, c):
            if p // SWA_BLOCK not in pages:
                pages[p // SWA_BLOCK] = swa_perm.pop()
            win.append(pages[p // SWA_BLOCK] * SWA_BLOCK + p % SWA_BLOCK)
        swa_idx[t, :nswa] = torch.tensor(win, dtype=torch.int32, device=device)
        swa_lens[t] = nswa
        slot[t] = win[-1]
        n_comp = c // RATIO
        ntopk = min(TOPK_KEYS, n_comp)
        if ntopk:
            sel = torch.randperm(n_comp, generator=g, device=device)[:ntopk].tolist()
            topk[t, :ntopk] = torch.tensor([comp_perm.pop() * 128 + r % 128 for r in sel], dtype=torch.int32,
                                           device=device)
        topk_lens[t] = ntopk
    pos = torch.tensor([c - 1 for c in ctxs], dtype=torch.int64, device=device)
    return {"pos": pos, "slot": slot, "swa_idx": swa_idx, "swa_lens": swa_lens, "topk": topk,
            "topk_lens": topk_lens, "swa_pages": swa_pages}


def _sinkhorn(x, iters=20):
    for _ in range(iters):
        x = x / x.sum(-1, keepdim=True)
        x = x / x.sum(-2, keepdim=True)
    return x


# ------------------------------------------------------------------------------- margins
def _same(round_fn, v, slack):
    """Elementwise: v, v - slack and v + slack round alike."""
    r = round_fn(v)
    return (round_fn(v - slack) == r) & (round_fn(v + slack) == r)


def _fp8_value(v):
    return v.to(torch.float32).clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.float64)


def _code_safe(amax, rule, rel=REL):
    """Block codes that do not move under a relative perturbation ``rel`` of the block's amax."""
    a = amax.to(torch.float64)
    c = mx_codes(a.to(torch.float32), rule)
    return (mx_codes((a * (1 - rel)).to(torch.float32), rule) == c) & \
           (mx_codes((a * (1 + rel)).to(torch.float32), rule) == c)


def safe_columns(sm, res, pend, post, comb, pre):
    """Hidden columns whose x8 every row of the step reproduces exactly (see the module docstring)."""
    f = torch.float64
    res, pend, post, comb, pre = (t.to(f) for t in (res, pend, post, comb, pre))
    r_scale = (post[:, :, None] * pend[:, None, :]).abs() + torch.einsum("sho,shc->soc", comb.abs(), res.abs())
    ok = _same(bf16, sm["r_raw"], REL * r_scale).all(dim=1)
    x_scale = (pre[:, :, None] * sm["res_out"]).abs().sum(1)
    ok &= _same(bf16, sm["x_raw"], REL * x_scale)
    ok &= _same(bf16, sm["y_raw"], REL * sm["y_raw"].abs())
    amax = sm["y"].abs().view(sm["y"].shape[0], -1, MX).amax(-1)
    ok &= _code_safe(amax, "vllm", REL_CODE).repeat_interleave(MX, dim=-1)
    return ok.all(dim=0)


def _record_safe(kv_a, kv_norm, cs):
    """Per row: (NoPE dims [S, 448] safe, RoPE pair dims [S, 64] safe, NoPE 64-blocks [S, 7] safe)."""
    f = torch.float64
    rk = 1.0 / torch.sqrt(kv_a.square().mean(-1, keepdim=True) + 1e-20)
    kn = kv_a * rk * kv_norm.to(f)
    e = rope_rotate(kn, cs)
    blocks = e[:, :NOPE].view(e.shape[0], -1, 64)
    code = mx_codes(blocks.abs().amax(-1).to(torch.float32), "record")
    code_ok = _code_safe(blocks.abs().amax(-1), "record")
    mul = torch.exp2(127.0 - code.to(f)).repeat_interleave(64, dim=-1)
    nope_ok = _same(_fp8_value, e[:, :NOPE] * mul, REL * (e[:, :NOPE] * mul).abs())
    c, s = cs[:, :32], cs[:, 32:]
    k0, k1 = kn[:, NOPE::2], kn[:, NOPE + 1::2]
    # e0 = k0 c - k1 s and e1 = k0 s + k1 c, each against the magnitude of its own two terms
    rope_ok = torch.empty(e.shape[0], 64, dtype=torch.bool, device=e.device)
    rope_ok[:, 0::2] = _same(bf16, e[:, NOPE::2], REL * ((k0 * c).abs() + (k1 * s).abs()))
    rope_ok[:, 1::2] = _same(bf16, e[:, NOPE + 1::2], REL * ((k0 * s).abs() + (k1 * c).abs()))
    return nope_ok, rope_ok, code_ok


def _plant_kv_norm(kv_a, cs, g, device):
    """bf16 kv_norm gains whose record values are safe for every row (resampling the unsafe dims)."""
    kv_norm = (0.5 + torch.rand(KV_DIM, generator=g, device=device, dtype=torch.float64)).to(torch.bfloat16)
    for _ in range(TRIES):
        nope_ok, rope_ok, code_ok = _record_safe(kv_a, kv_norm, cs)
        bad = torch.cat([~nope_ok.all(0), ~rope_ok.all(0)])
        # a block whose code is unsafe: resample the gains of its largest elements' dims
        for b in (~code_ok.all(0)).nonzero().flatten().tolist():
            bad[64 * b:64 * (b + 1)] = True
        # a RoPE pair is resampled together
        pair = bad[NOPE:].view(-1, 2).any(-1)
        bad[NOPE:] = pair.repeat_interleave(2)
        if not bool(bad.any()):
            return kv_norm
        fresh = (0.5 + torch.rand(KV_DIM, generator=g, device=device, dtype=torch.float64)).to(torch.bfloat16)
        kv_norm = torch.where(bad, fresh, kv_norm)
    raise RuntimeError("could not plant safe KV record margins")


def _plant_wkv(safe, g, device):
    """wkv_a rows (wqkv rows 1280 ..): one nonzero E4M3 weight a row at a random safe column."""
    cols = safe.nonzero().flatten()
    if cols.numel() == 0:
        raise RuntimeError("no hidden column is safe for every row")
    pick = cols[torch.randint(0, cols.numel(), (KV_DIM,), generator=g, device=device)]
    vals = _fp8((KV_DIM,), g, device).to(torch.float32)
    vals = torch.where(vals == 0, torch.ones_like(vals), vals).to(torch.float8_e4m3fn)
    w = torch.zeros(KV_DIM, HIDDEN, dtype=torch.float8_e4m3fn, device=device)
    w[torch.arange(KV_DIM, device=device), pick] = vals
    return w


def make_inputs(case, inputs, generator, device):
    axes, g = case["axes"], generator
    rows, ctx = axes["rows"], axes["ctx"]
    step = make_step(rows, ctx, g, device)
    res = torch.randn(rows, HC, HIDDEN, generator=g, device=device, dtype=torch.float64)
    inp = {
        "res_in": (res * torch.rsqrt(res.square().mean(-1, keepdim=True))).to(torch.bfloat16),
        "pend": (PEND_STD * torch.randn(rows, HIDDEN, generator=g, device=device)).to(torch.bfloat16),
        "post_in": (2 * torch.sigmoid(torch.randn(rows, HC, generator=g, device=device))).float(),
        "comb_in": _sinkhorn(torch.rand(rows, HC, HC, generator=g, device=device, dtype=torch.float64) + 0.1)
        .float().contiguous(),
        "pre_in": (torch.sigmoid(torch.randn(rows, HC, generator=g, device=device)) + 1e-6).float(),
        "hc_fn": (0.01 * torch.randn(MIX, K_MIX, generator=g, device=device)).float(),
        "hc_scale": (0.5 * torch.randn(3, generator=g, device=device)).abs().float() + 0.5,
        "hc_base": (0.1 * torch.randn(MIX, generator=g, device=device)).float(),
        "attn_norm": (1.0 + 0.1 * torch.randn(HIDDEN, generator=g, device=device)).to(torch.bfloat16),
        "q_norm": (1.0 + 0.1 * torch.randn(Q_RANK, generator=g, device=device)).to(torch.bfloat16),
        "wqb": _fp8((N_HEADS * HEAD_DIM, Q_RANK), g, device),
        "wqb_s": _codes((N_HEADS * HEAD_DIM // MX, Q_RANK // MX), E8M0_QB, g, device),
        "cos_sin": rope_table(device),
        "pos": step["pos"], "slot": step["slot"],
        "swa": torch.randint(0, 256, (step["swa_pages"], page_bytes(SWA_BLOCK)), generator=g, device=device,
                             dtype=torch.uint8),
        "swa_idx": step["swa_idx"], "swa_lens": step["swa_lens"], "topk": step["topk"],
        "topk_lens": step["topk_lens"],
    }
    sm = seam(inp["res_in"], inp["pend"], inp["post_in"], inp["comb_in"], inp["pre_in"], inp["hc_fn"],
              inp["hc_scale"], inp["hc_base"], inp["attn_norm"])
    safe = safe_columns(sm, inp["res_in"], inp["pend"], inp["post_in"], inp["comb_in"], inp["pre_in"])
    wqkv = torch.empty(QKV_ROWS, HIDDEN, dtype=torch.float8_e4m3fn, device=device)
    wqkv[:Q_RANK] = _fp8((Q_RANK, HIDDEN), g, device)
    wqkv[Q_RANK:] = _plant_wkv(safe, g, device)
    inp["wqkv"] = wqkv
    inp["wqkv_s"] = _codes((QKV_ROWS // MX, HIDDEN // MX), E8M0_QKV, g, device)
    # kv_a exactly as the kernel forms it: one E4M3 product a dim, exact in float32 and bf16
    _, _, x8 = mx_quant(sm["y"], "vllm")
    w_kv = fp8_block_dequant(wqkv, inp["wqkv_s"])[Q_RANK:]
    kv_a = x8 @ w_kv.T
    if not torch.equal(bf16(kv_a), kv_a):
        raise RuntimeError("kv_a is not exact in bf16")
    cs = inp["cos_sin"][step["pos"]].to(torch.float64)
    inp["kv_norm"] = _plant_kv_norm(kv_a, cs, g, device)
    nope_ok, rope_ok, code_ok = _record_safe(kv_a, inp["kv_norm"], cs)
    if not (bool(nope_ok.all()) and bool(rope_ok.all()) and bool(code_ok.all())):
        raise RuntimeError("KV record margins too small")
    front = k1_front(sm["y"], {**inp, "pos": step["pos"]})
    if not torch.equal(front["kv_a"], kv_a):
        raise RuntimeError("kv_a of the reference path differs from the planted one")
    encode_records(front["kn"])  # the record path runs on these inputs
    return inp


def shard_inputs(logical, *, rank, world_size):
    """Rank ``rank``'s kernel inputs: its wq_b rows; everything else replicated."""
    assert world_size == TP, world_size
    out = dict(logical)
    out["wqb"] = logical["wqb"][rank * QB_ROWS:(rank + 1) * QB_ROWS].contiguous()
    out["wqb_s"] = logical["wqb_s"][rank * QB_ROWS // MX:(rank + 1) * QB_ROWS // MX].contiguous()
    out["swa"] = logical["swa"].clone()
    return out


def rank_expected(logical_outputs, *, rank, world_size):
    """Rank ``rank``'s expected tensors: its 16 heads of q; the rest (and its cache copy) replicated."""
    assert world_size == TP, world_size
    out = dict(logical_outputs)
    out["q"] = logical_outputs["q"][:, rank * HEADS_RANK:(rank + 1) * HEADS_RANK].contiguous()
    return out
