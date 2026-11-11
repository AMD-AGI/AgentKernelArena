# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Logical inputs of one DeepSeek-V4-Pro CSA decoder layer at TP8, their per-rank shards and the
per-rank expected outputs.

Logical inputs (what the reference reads) are the WHOLE layer, as the checkpoint stores it, plus the
replicated activations and the per-request caches at the case's context:

  weights   FP8 e4m3 matrices with float32 128x128 block scales (arbitrary positive floats; the
            checkpoint's are ue8m0 powers of two): w_qkv_a [2048, 7168] (q_a rows, then the shared KV row), w_q_b
            [128 heads * 512, 1536], w_i_q_b [64 * 128, 1536], w_o_a [16 groups * 1024, 4096],
            w_o_b [7168, 16 * 1024], the shared expert w_sug [2 * 3072, 7168] (gate rows, then up rows)
            and w_sdn [7168, 3072]. BF16: w_qkv_c [2560, 7168] (compressor kv | gate, indexer kv | gate),
            i_w [64, 7168], w_r [384, 7168]. MXFP4 routed experts (e2m1 codes, low nibble = even
            element, one e8m0 byte per 32 contracted elements): w_ug [384, 2 * 3072, 3584] (gate rows,
            then up rows) with s_ug [384, 6144, 224], w_dn [384, 7168, 1536] with s_dn [384, 7168, 96].
            mHC mixers hc_*_fn [24, 4 * 7168] float32 with hc_*_sb = [3 scales | 24 bases].
  caches    the SGLang two-pool FP8 unified KV (kv_cache: per row 448 e4m3 NoPE codes, 16 bytes of
            per-64-group e8m0 scales (each byte twice, see ``_kv_rows``), 48 zero bytes; kv_rope: the
            64 bf16 RoPE values), rows [0, rows * 128) the per-request sliding-window ring, then 64
            compressed rows per 256-token page; the FP4 index-key pool (i_cache / i_cache_s, see
            ``_index_pages``) of the same pages; the compressor and indexer-compressor state rings
            (c_state / i_state, [kv | score] float32 rows, 8 a request).

``shard_inputs`` slices rank r's tensor-parallel shard (heads 16r..16r+15, o_a groups 2r, 2r+1, the
o_b and down-projection K columns and the expert intermediates 384r..384r+383) and packs every matrix
into the kernel's MFMA tile layouts (``pack_*``: the logic of sgl-amd-mono 52486703
sgl_amd_mono/deepseek_v4/gfx950/packing.py, Apache-2.0, adapted from ROCm/ATOM#2559, MIT). Caches and
replicated activations are the same on every rank. ``rank_expected`` returns the replicated outputs.

Workload construction (the reference is exact, so the generator keeps discrete decisions clear):
  * Positions: row 0 decodes at position ``ctx`` (``ctx - 1`` when ``ctx + 1`` is a multiple of the
    compress ratio 4), every other row at a uniform position in [ctx / 2, ctx], moved down by one in the
    same case. No row ends a compressor window, so the step writes no compressed entry and no FP4
    index key (see README.md: those packed uint8 writes would need an exact comparison that an
    independent float64 reference cannot meet).
  * Router margins: each row's hidden state carries a planted component along 6 random experts'
    router directions (``PLANT``), so its 6 routed experts lead the 7th by a wide score margin and the
    top-6 never flips between the reference and a kernel.
  * Caches are filled with normalized random KV rows / index keys quantized exactly as the kernel
    stores them, so every e8m0 scale byte stays below 0x7F (no NaN when kv_cache is read as e4m3).
  * Indexer selection (``_plant_index_keys``): for every row with more than 1024 live compressed
    entries (c < (p + 1) // 4; the indexer scores all of them, fewer are all attended), 1024 random live
    entries get a planted index key. The row's FP4 index query iq [64 heads, 128] and head weights w
    are computed exactly as reference.py does (``_index_query``, with its helpers); a planted key is
    the min-norm solution of iq k = t (the dual basis of the 64 query heads), t_h = w_h x U(0.8, 1.2)
    for the heads of positive weight and t_h < 0 (relu off) for the others, rescaled to norm sqrt(128),
    that of the random N(0, 1) keys (neither the norm nor the e8m0 bytes tell a planted key from a
    random one), and quantized as the cache fill does (``_fp4_key_bytes``). Its score sum_h relu(iq_h .
    k) w_h is typically 1.5 to 2.5 times the best random key's, up to a 262K-entry context. Random keys
    of the row scoring above 60% of its lowest planted score (``PLANT_CAP``: none in most rows, a
    far-tail few in the others) are redrawn. Every live entry's score comes from the stored bytes, as
    the reference computes it, and ``_check_margin`` raises unless the lowest planted score beats every
    other entry by ``SELECT_MARGIN`` = 35% of itself. A kernel's FP4 index query with hundreds of codes
    one step off, bf16 head weights one ulp off and fp32 sums moves a score by under 10% of that
    (checked on synthetic queries), so the selected set cannot change. Only the set matters: the
    kernel and the reference both place the selected entries in index order. Their KV rows stay
    random: K = V in this attention, so a planted high-logit direction would be shared by every
    planted row and hide which of them are attended, while random rows give every attended entry an
    independent value at a near-uniform weight, so each wrong entry moves every head's output.
"""
import math

import torch

# the reference's own float64 helpers (task-root module), so the planted selection is the one it computes
from reference import (EPS, INDEX_TOPK, N_COMP, SCORE_CHUNK, _hadamard, bf16, decode_index_keys, fp4q, fp8_matrix,
                       fp8q, hc_coefficients, rms, rope)

HIDDEN = 7168
HC = 4
Q_LORA = 1536
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HEADS = 128
O_GROUPS = 16
O_LORA = 1024
N_EXPERTS = 384
TOP_K = 6
INTER = 3072
INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
WINDOW = 128
CR = 4
C_COFF = 2
C_RING = 8
SWA_RING = 128
PAGE_TOKENS = 256
K_PB = PAGE_TOKENS // CR
KV_ROW_BYTES = 512
HC_MIX = (2 + HC) * HC
HC_ROWS = 32
ROPE_THETA = 1.6e5  # compress_rope_theta: a compressing layer rotates everything on it
FP8_MAX = 448.0
FP4_MAX = 6.0
# deadline of every in-kernel wait (guard words 1-2 of the epoch buffer), in 100 MHz clock ticks
DEADLINE_TICKS = 20 * 100_000_000
EPOCH_STEP = 1

# weight scales (standard deviations before quantization)
STD = {"w_qkv_a": 0.012, "w_qkv_c": 0.012, "w_q_b": 0.0255, "w_i_q_b": 0.0255, "w_o_a": 0.3,
       "w_o_b": 0.004, "w_sug": 0.012, "w_sdn": 0.011, "w_r": 0.006, "i_w": 0.012, "hc_fn": 0.006}
# e8m0 bytes of the MXFP4 experts: up/gate about 2^-7, down about 2^-8 (outputs of order one)
UG_SCALE_BYTES = (119, 120, 121)
DN_SCALE_BYTES = (118, 119, 120)
# norm of the planted router component of every hidden row (the noise is unit normal)
PLANT = 24.0
PLANT_SPREAD = (0.7, 1.3)
# planted index keys: per-head target factor, the off heads' target (times the mean on-head target),
# the redraw cap for random keys and the checked margin, both relative to the row's lowest planted score
INDEX_SPREAD = (0.8, 1.2)
INDEX_OFF = 0.5
PLANT_CAP = 0.6
SELECT_MARGIN = 0.35
REDRAW_ROUNDS = 16
ROW_CHUNK = 1 << 16
EXPERT_CHUNK = 16


# --------------------------------------------------------------------------- quantization helpers
def _pow2_ceil(x):
    """Smallest power of two >= x (x > 0), exactly: the kernel's pow2_ceil."""
    mant, exp = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), torch.where(mant == 0.5, exp - 1, exp))


def _fp8_blocks(x, block, floor=None):
    """E4M3 codes and power-of-two scales of x [..., K] in 1 x ``block`` blocks: scale =
    pow2_ceil(amax / 448) (pow2_ceil(max(amax, floor) / 448) with ``floor``), codes = RNE(x / scale)."""
    shape = x.shape
    xb = x.reshape(*shape[:-1], shape[-1] // block, block)
    amax = xb.abs().amax(-1, keepdim=True)
    if floor is not None:
        amax = amax.clamp_min(floor)
    scale = torch.where(amax > 0, _pow2_ceil(amax / FP8_MAX), torch.ones_like(amax))
    codes = (xb / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return codes.reshape(shape), scale.squeeze(-1)


def _quant_fp8_matrix(rows, cols, std, generator, device):
    """A normal matrix quantized as FP8 weights: e4m3 codes [rows, cols] and float32 128 x 128 block
    scales [rows / 128, cols / 128] (amax / 448 times a uniform factor in [1, 1.25))."""
    codes = torch.empty(rows, cols, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty(rows // 128, cols // 128, dtype=torch.float32, device=device)
    step = max(128, (ROW_CHUNK * 64 // cols) // 128 * 128)
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        w = torch.randn(r1 - r0, cols, generator=generator, device=device, dtype=torch.float32) * std
        blocks = w.view((r1 - r0) // 128, 128, cols // 128, 128)
        amax = blocks.abs().amax(dim=(1, 3), keepdim=True)
        # an arbitrary positive float32 block scale at or above amax / 448 (the checkpoint's are powers
        # of two; the kernel multiplies by any float32, and a power-of-two draw would be bf16-exact)
        spread = 1.0 + 0.25 * torch.rand(amax.shape, generator=generator, device=device, dtype=torch.float32)
        scale = amax / FP8_MAX * spread
        codes[r0:r1] = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(r1 - r0, cols)
        scales[r0 // 128:r1 // 128] = scale.view((r1 - r0) // 128, cols // 128)
    return codes, scales


def e2m1_codes(q):
    """float32 values in [-6, 6] -> e2m1 codes (sign in bit 3), round to nearest even."""
    m = q.abs()
    code = ((m > 0.25).to(torch.uint8) + (m >= 0.75).to(torch.uint8) + (m > 1.25).to(torch.uint8)
            + (m >= 1.75).to(torch.uint8) + (m > 2.5).to(torch.uint8) + (m >= 3.5).to(torch.uint8)
            + (m > 5.0).to(torch.uint8))
    return code | (torch.signbit(q).to(torch.uint8) << 3)


def _bf16_values(shape, std, generator, device):
    return (torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * std).to(torch.bfloat16)


# --------------------------------------------------------------------------- caches
def scale_dwords(e8):
    """The 16 scale bytes of a KV row from its 7 per-64-group e8m0 bytes [..., 7]: dword k holds
    group 2k in bytes 0-1 and group 2k + 1 in bytes 2-3 (group 7 is absent: 0)."""
    pad = torch.cat([e8, torch.zeros_like(e8[..., :1])], dim=-1)  # [..., 8]
    return pad.repeat_interleave(2, dim=-1)  # [..., 16]


def _kv_rows(n, generator, device):
    """``n`` random KV rows as the kernel writes them: a normalized 512-vector, NoPE part FP8 per
    64-group with scale pow2_ceil(max(amax, 448 * 2^-126) / 448), RoPE part bf16."""
    nope = torch.empty(n, KV_ROW_BYTES, dtype=torch.uint8, device=device)
    rope = torch.empty(n, ROPE_DIM, dtype=torch.bfloat16, device=device)
    for r0 in range(0, n, ROW_CHUNK):
        r1 = min(n, r0 + ROW_CHUNK)
        v = torch.randn(r1 - r0, HEAD_DIM, generator=generator, device=device, dtype=torch.float32)
        codes, scale = _fp8_blocks(v[:, :NOPE_DIM], 64, floor=FP8_MAX * 2.0 ** -126)
        e8 = (torch.log2(scale).round().to(torch.int32) + 127).to(torch.uint8)
        nope[r0:r1, :NOPE_DIM] = codes.view(torch.uint8)
        nope[r0:r1, NOPE_DIM:NOPE_DIM + 16] = scale_dwords(e8)
        nope[r0:r1, NOPE_DIM + 16:] = 0
        rope[r0:r1] = v[:, NOPE_DIM:].to(torch.bfloat16)
    return nope.view(torch.float8_e4m3fn), rope


def _fp4_key_bytes(k):
    """float32 index keys [..., 128] -> (e2m1 codes [..., 4 groups, 16 bytes], e8m0 bytes [..., 4]) as the
    kernel's indexer compressor writes them: per 32-block scale pow2_ceil(max(amax, 6 * 2^-126) / 6),
    byte i of group g holding channels 32g + 2i (low nibble) and 32g + 2i + 1."""
    kb = k.reshape(*k.shape[:-1], INDEX_HEAD_DIM // 32, 32)
    amax = kb.abs().amax(-1, keepdim=True).clamp_min(FP4_MAX * 2.0 ** -126)
    sc = _pow2_ceil(amax / FP4_MAX)
    c = e2m1_codes((kb / sc).clamp(-FP4_MAX, FP4_MAX))  # [..., g, 32]
    packed = c[..., 0::2] | (c[..., 1::2] << 4)  # [..., g, 16]
    e8 = (torch.log2(sc.squeeze(-1)).round().to(torch.int32) + 127).to(torch.uint8)  # [..., g]
    return packed, e8


def _index_pages(n_pages, generator, device):
    """FP4 index keys of ``n_pages`` pages (64 entries each) from N(0, 1) draws of the 128 normalized
    (Hadamard-rotated) channels, quantized by ``_fp4_key_bytes``. Codes: [page][group g = channel // 32]
    [entry][16 bytes]; scales: [page][group][byte (e % 16) * 4 + e // 16]."""
    codes = torch.empty(n_pages, 4, K_PB, 16, dtype=torch.uint8, device=device)
    scales = torch.empty(n_pages, 4, K_PB, dtype=torch.uint8, device=device)
    perm = (torch.arange(K_PB, device=device) % 16) * 4 + torch.arange(K_PB, device=device) // 16
    step = max(1, ROW_CHUNK // K_PB)
    for p0 in range(0, n_pages, step):
        p1 = min(n_pages, p0 + step)
        k = torch.randn(p1 - p0, K_PB, INDEX_HEAD_DIM, generator=generator, device=device, dtype=torch.float32)
        packed, e8 = _fp4_key_bytes(k)  # [p, e, g, 16], [p, e, g]
        codes[p0:p1] = packed.permute(0, 2, 1, 3)
        s = torch.empty(p1 - p0, 4, K_PB, dtype=torch.uint8, device=device)
        s[:, :, perm] = e8.permute(0, 2, 1)
        scales[p0:p1] = s
    return codes.view(n_pages, 4 * K_PB * 16), scales.view(n_pages, 4 * K_PB)


def _write_index_keys(i_cache, i_cache_s, table, entries, keys):
    """Store float keys [n, 128] at a request's compressed entries ``entries`` (its block-table row
    ``table``), quantized as ``_index_pages`` quantizes the cache fill."""
    packed, e8 = _fp4_key_bytes(keys.float())
    page = table.long()[entries // K_PB]
    slot = entries % K_PB
    i_cache.view(-1, 4, K_PB, 16)[page, :, slot] = packed
    i_cache_s.view(-1, 4, K_PB)[page, :, (slot % 16) * 4 + slot // 16] = e8


def _index_scores(i_cache, i_cache_s, iq, wp, table, entries):
    """Indexer scores of a request's entries from the stored FP4 keys, as reference._select_entries
    computes them: sum_h relu(iq_h . key) w_h (float64)."""
    scores = torch.empty(entries.numel(), dtype=torch.float64, device=iq.device)
    for c0 in range(0, entries.numel(), SCORE_CHUNK):
        e = entries[c0:c0 + SCORE_CHUNK]
        keys = decode_index_keys(i_cache, i_cache_s, table[e // K_PB].long(), e % K_PB)
        scores[c0:c0 + SCORE_CHUNK] = (torch.relu(keys @ iq.T) * wp).sum(-1)
    return scores


def _n_live(p):
    """Live compressed entries of a row at position p (the indexer scores all of them)."""
    return min((p + 1) // CR, N_COMP)


def _index_query(out, inputs):
    """The indexer's FP4 query iq [S, 64, 128] and head weights wp [S, 64] of every row, computed exactly
    as reference.py's steps 1, 2, 4 and 5 compute them (float64, its helpers)."""
    h = out["hidden"].double()
    pos = out["positions"].long()
    cos = out["rope_cos"].double()[pos]
    sin = out["rope_sin"].double()[pos]
    pre_a, _, _ = hc_coefficients(h, out["hc_attn_fn"], out["hc_attn_sb"])
    xin = bf16((pre_a[:, :, None] * h).sum(1))
    rstd = torch.rsqrt(xin.square().mean(-1, keepdim=True) + EPS)
    xa = bf16(xin * rstd * inputs["g_in"].double())
    qkv = bf16(fp8q(xa, 128)[0] @ fp8_matrix(out["w_qkv_a"], out["s_qkv_a"]).T)
    qa8 = fp8q(rms(qkv[:, :Q_LORA]) * inputs["g_q"].double(), 128)[0]
    half = INDEX_HEAD_DIM // 2
    iq = bf16(qa8 @ fp8_matrix(out["w_i_q_b"], out["s_i_q_b"]).T).view(h.shape[0], INDEX_HEADS, INDEX_HEAD_DIM)
    iq = torch.cat([iq[..., :half], bf16(rope(iq[..., half:], cos[:, None], sin[:, None]))], -1)
    iq = fp4q(bf16(iq @ _hadamard(INDEX_HEAD_DIM, h.device)))
    wp = bf16((xin * rstd * inputs["g_in"].double()) @ inputs["i_w"].double().T) \
        * INDEX_HEAD_DIM ** -0.5 * INDEX_HEADS ** -0.5
    return iq, wp


def _plant_index_keys(i_cache, i_cache_s, iq, wp, positions, block_tables, generator):
    """Plant the top-1024 of every row with more than 1024 live entries (see the module docstring), in
    place, and check its margin -> {row: planted entries}. One scoring pass over the live entries a row
    (plus the few redrawn ones): the kernel scores exactly these entries."""
    device = i_cache.device
    hot = {}
    for s, p in enumerate(positions.tolist()):
        n_live = _n_live(p)
        if n_live <= INDEX_TOPK:
            continue
        q, w, table = iq[s], wp[s], block_tables[s]
        entries = torch.randperm(n_live, generator=generator, device=device)[:INDEX_TOPK]
        lo, hi = INDEX_SPREAD
        f = torch.rand(INDEX_TOPK, INDEX_HEADS, generator=generator, device=device, dtype=torch.float64)
        f = f * (hi - lo) + lo
        t = torch.where(w > 0, w, -INDEX_OFF * w.clamp_min(0).mean()) * f  # [n, heads] wanted iq_h . k
        k = torch.linalg.solve(q @ q.T, t.T).T @ q  # the min-norm k with q k = t (the dual basis of q)
        k = k * (INDEX_HEAD_DIM ** 0.5 / k.norm(dim=-1, keepdim=True))  # the random keys' norm
        _write_index_keys(i_cache, i_cache_s, table, entries, k)
        live = torch.arange(n_live, device=device)
        scores = _index_scores(i_cache, i_cache_s, q, w, table, live)
        planted = torch.zeros(n_live, dtype=torch.bool, device=device)
        planted[entries] = True
        cap = PLANT_CAP * float(scores[entries].min())
        for _ in range(REDRAW_ROUNDS):
            # random keys of the far tail: redraw them (the planted set must lead by a clear margin)
            over = live[(scores > cap) & ~planted]
            if over.numel() == 0:
                break
            fresh = torch.randn(over.numel(), INDEX_HEAD_DIM, generator=generator, device=device, dtype=torch.float32)
            _write_index_keys(i_cache, i_cache_s, table, over, fresh)
            scores[over] = _index_scores(i_cache, i_cache_s, q, w, table, over)
        _check_margin(scores, planted, s, p)
        hot[s] = entries
    return hot


def _check_margin(scores, planted, s, p):
    """Raise unless the planted entries lead every other live entry by SELECT_MARGIN of their lowest
    score. ``scores`` are every live entry's, from the stored keys as the reference computes them, so
    the planted set is the reference's top-1024 (no tie can reach the boundary)."""
    low = float(scores[planted].min())
    high = float(scores[~planted].max())
    if not low > 0 or low - high < SELECT_MARGIN * low:
        raise RuntimeError(f"indexer selection margin too small: row {s} (position {p}, {scores.numel()} live "
                           f"entries) lowest planted score {low:.4g}, best other {high:.4g}")


def _rope_tables(n, device):
    """cos / sin [n, 32] of positions 0..n-1 at compress_rope_theta (interleaved pairs). ``n`` = ctx + 4096: like
    SGLang's precomputed cache the table extends past the step's positions (a 2-row table at ctx 1 would be mostly
    the exact cos 1 of position 0)."""
    inv = ROPE_THETA ** (-torch.arange(0, ROPE_DIM, 2, dtype=torch.float64, device=device) / ROPE_DIM)
    pos = torch.arange(n, dtype=torch.float64, device=device)
    angle = pos[:, None] * inv[None, :]
    return angle.cos().float(), angle.sin().float()


def _positions(rows, ctx, generator, device):
    """Each row's decode position; none ends a compressor window ((p + 1) % 4 != 0)."""
    def off_boundary(p):
        return torch.where((p + 1) % CR == 0, p - 1, p)

    first = torch.tensor([ctx], dtype=torch.int64, device=device)
    if rows > 1:
        low = (ctx + 1) // 2
        rest = torch.randint(low, ctx + 1, (rows - 1,), generator=generator, device=device, dtype=torch.int64)
        first = torch.cat([first, rest])
    return off_boundary(first).clamp_min(0)


# --------------------------------------------------------------------------- the generator
def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    rows, ctx = axes["rows"], axes["ctx"]
    assert axes["hidden"] == HIDDEN and axes["n_experts"] == N_EXPERTS and axes["local_heads"] == HEADS // 8
    g = generator
    out = {}

    # ---- attention weights (whole layer)
    out["w_qkv_a"], out["s_qkv_a"] = _quant_fp8_matrix(Q_LORA + HEAD_DIM, HIDDEN, STD["w_qkv_a"], g, device)
    out["w_qkv_c"] = _bf16_values((2 * C_COFF * (HEAD_DIM + INDEX_HEAD_DIM), HIDDEN), STD["w_qkv_c"], g, device)
    out["w_q_b"], out["s_q_b"] = _quant_fp8_matrix(HEADS * HEAD_DIM, Q_LORA, STD["w_q_b"], g, device)
    out["w_i_q_b"], out["s_i_q_b"] = _quant_fp8_matrix(INDEX_HEADS * INDEX_HEAD_DIM, Q_LORA, STD["w_i_q_b"],
                                                       g, device)
    out["w_o_a"], out["s_o_a"] = _quant_fp8_matrix(O_GROUPS * O_LORA, HEADS * HEAD_DIM // O_GROUPS,
                                                   STD["w_o_a"], g, device)
    out["w_o_b"], out["s_o_b"] = _quant_fp8_matrix(HIDDEN, O_GROUPS * O_LORA, STD["w_o_b"], g, device)
    out["attn_sink"] = torch.rand(HEADS, generator=g, device=device, dtype=torch.float32) * 3.0 + 5.0
    for side in ("attn", "ffn"):
        out[f"hc_{side}_fn"] = torch.randn(HC_MIX, HC * HIDDEN, generator=g, device=device,
                                           dtype=torch.float32) * STD["hc_fn"]
        scales = torch.rand(3, generator=g, device=device, dtype=torch.float32) + 0.5
        bases = torch.randn(HC_MIX, generator=g, device=device, dtype=torch.float32) * 0.5
        out[f"hc_{side}_sb"] = torch.cat([scales, bases]).contiguous()

    # ---- FFN weights
    out["w_r"] = _bf16_values((N_EXPERTS, HIDDEN), STD["w_r"], g, device)
    out["w_sug"], out["s_sug"] = _quant_fp8_matrix(2 * INTER, HIDDEN, STD["w_sug"], g, device)
    out["w_sdn"], out["s_sdn"] = _quant_fp8_matrix(HIDDEN, INTER, STD["w_sdn"], g, device)
    out["w_ug"] = torch.empty(N_EXPERTS, 2 * INTER, HIDDEN // 2, dtype=torch.uint8, device=device)
    out["w_dn"] = torch.empty(N_EXPERTS, HIDDEN, INTER // 2, dtype=torch.uint8, device=device)
    for e0 in range(0, N_EXPERTS, EXPERT_CHUNK):
        e1 = min(N_EXPERTS, e0 + EXPERT_CHUNK)
        out["w_ug"][e0:e1] = torch.randint(0, 256, (e1 - e0, 2 * INTER, HIDDEN // 2), generator=g,
                                           device=device, dtype=torch.uint8)
        out["w_dn"][e0:e1] = torch.randint(0, 256, (e1 - e0, HIDDEN, INTER // 2), generator=g,
                                           device=device, dtype=torch.uint8)
    ug_bytes = torch.tensor(UG_SCALE_BYTES, dtype=torch.uint8, device=device)
    dn_bytes = torch.tensor(DN_SCALE_BYTES, dtype=torch.uint8, device=device)
    out["s_ug"] = ug_bytes[torch.randint(0, len(UG_SCALE_BYTES), (N_EXPERTS, 2 * INTER, HIDDEN // 32),
                                         generator=g, device=device)]
    out["s_dn"] = dn_bytes[torch.randint(0, len(DN_SCALE_BYTES), (N_EXPERTS, HIDDEN, INTER // 32),
                                         generator=g, device=device)]

    # ---- replicated activations with planted router winners
    noise = torch.randn(rows, HC, HIDDEN, generator=g, device=device, dtype=torch.float32)
    unit = out["w_r"].float()
    unit = unit / unit.norm(dim=-1, keepdim=True)
    plant = torch.zeros(rows, HIDDEN, dtype=torch.float32, device=device)
    for s in range(rows):
        winners = torch.randperm(N_EXPERTS, generator=g, device=device)[:TOP_K]
        lo, hi = PLANT_SPREAD
        weight = torch.rand(TOP_K, generator=g, device=device, dtype=torch.float32) * (hi - lo) + lo
        plant[s] = (weight[:, None] * unit[winners]).sum(0) * PLANT
    out["hidden"] = (noise + plant[:, None, :]).to(torch.bfloat16)

    # ---- the step's rows and the per-request caches at their contexts
    pos = _positions(rows, ctx, g, device)
    pages = axes["pages"]
    out["positions"] = pos.to(torch.int32)
    out["state_slots"] = torch.arange(rows, dtype=torch.int32, device=device)
    out["block_tables"] = torch.randperm(rows * pages, generator=g, device=device).to(torch.int32).view(rows, pages)
    swa_rows = rows * SWA_RING
    swa_row = torch.arange(rows, device=device) * SWA_RING + pos % SWA_RING
    out["dest_rows"] = torch.stack([swa_row, torch.full_like(swa_row, swa_rows)]).to(torch.int32).contiguous()
    q = pos[:, None] - (WINDOW - 1) + torch.arange(WINDOW, device=device)[None]
    win = torch.where(q >= 0, torch.arange(rows, device=device)[:, None] * SWA_RING + q.clamp_min(0) % SWA_RING, -1)
    tail = torch.full((rows, axes["n_keys"] - WINDOW), -1, dtype=torch.int64, device=device)
    out["indices"] = torch.cat([win, tail], 1).to(torch.int32).contiguous()
    out["kv_cache"], out["kv_rope"] = _kv_rows(axes["kv_rows"], g, device)
    out["i_cache"], out["i_cache_s"] = _index_pages(rows * pages, g, device)
    out["c_state"] = torch.randn(rows, C_RING, 2 * C_COFF * HEAD_DIM, generator=g, device=device)
    out["i_state"] = torch.randn(rows, C_RING, 2 * C_COFF * INDEX_HEAD_DIM, generator=g, device=device)
    out["rope_cos"], out["rope_sin"] = _rope_tables(axes["rope_rows"], device)
    epoch = torch.zeros(8, dtype=torch.int32, device=device)
    epoch[0] = EPOCH_STEP
    epoch[1] = DEADLINE_TICKS & 0x7FFFFFFF
    epoch[2] = DEADLINE_TICKS >> 32
    out["epoch"] = epoch

    # ---- the indexer's top-1024 of every row past the threshold, planted (drawn last)
    if any(_n_live(p) > INDEX_TOPK for p in pos.tolist()):
        iq, wp = _index_query(out, inputs)
        _plant_index_keys(out["i_cache"], out["i_cache_s"], iq, wp, pos, out["block_tables"], g)
    return out


# --------------------------------------------------------------------------- kernel layouts
def pack_fp8(q):
    """FP8 [N, K] (bytes) in the kernel's 16-row, 64-K MFMA tiles, flattened (packing.pack_fp8)."""
    rows, k = q.shape
    assert rows % 16 == 0 and k % 64 == 0
    w8 = q.contiguous().view(torch.uint8).reshape(rows // 16, 16, k // 64, 2, 4, 8)
    return w8.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1)


def pack_bf16(w):
    """BF16 [N, K] in the kernel's 16-row, 64-K MFMA tiles, flattened (packing.pack_bf16)."""
    rows, k = w.shape
    assert rows % 16 == 0 and k % 64 == 0
    w16 = w.contiguous().view(torch.int16).reshape(rows // 16, 16, k // 64, 2, 4, 8)
    return w16.permute(0, 2, 3, 4, 1, 5).contiguous().view(-1).view(torch.bfloat16)


def pack_mxfp4(q, gate_up):
    """An MXFP4 bank [E, N, K / 2] in ATOM's gfx950 layout (aiter shuffle_weight,
    is_guinterleave=True; packing.pack_mxfp4). ``gate_up``: rows are [gate; up] halves."""
    e, n, kp = q.shape
    assert n % (32 if gate_up else 16) == 0 and kp % 64 == 0
    if gate_up:
        w = q.view(e, 2, n // 32, 16, kp // 64, 4, 16).permute(0, 2, 1, 4, 5, 3, 6)
    else:
        w = q.view(e, n // 16, 16, kp // 64, 4, 16).permute(0, 1, 3, 4, 2, 5)
    return w.contiguous().view(e, n, kp)


def pack_mxfp4_scales(s, gate_up):
    """E8M0 scales [E, N, K / 32] in ATOM's gfx950 layout (aiter shuffle_scale,
    is_guinterleave=True; packing.pack_mxfp4_scales); K / 32 padded to a multiple of 8 with 0x7F."""
    e, n, kb = s.shape
    kb8 = -(-kb // 8) * 8
    if kb8 != kb:
        s = torch.cat([s, torch.full((e, n, kb8 - kb), 0x7F, dtype=torch.uint8, device=s.device)], dim=-1)
    if gate_up:
        t = s.reshape(e, 2, n // 32, 16, kb8 // 8, 2, 4).permute(0, 2, 4, 6, 3, 5, 1)
    else:
        t = s.reshape(e, n // 32, 2, 16, kb8 // 8, 2, 4).permute(0, 1, 4, 6, 3, 5, 2)
    return t.contiguous().view(e, n, kb8)


def pack_hc_fn(fn):
    """An fp32 mixer [24, K], padded to 32 rows, as bf16 [hi; lo] tiles (packing.pack_hc_fn)."""
    full = torch.zeros(HC_ROWS, fn.shape[1], dtype=torch.float32, device=fn.device)
    full[:fn.shape[0]] = fn.float()
    hi = full.to(torch.bfloat16)
    lo = (full - hi.float()).to(torch.bfloat16)
    return pack_bf16(torch.cat([hi, lo]))


def _halves(t, start, size, half):
    """Rows [start, start + size) of both halves (first ``half`` rows, then the rest) of t."""
    return torch.cat([t[..., start:start + size, :], t[..., half + start:half + start + size, :]], dim=-2)


def shard_inputs(logical, *, rank, world_size):
    heads = HEADS // world_size
    groups = O_GROUPS // world_size
    inter = INTER // world_size
    out = dict(logical)
    out["attn_sink"] = logical["attn_sink"][rank * heads:(rank + 1) * heads].contiguous()
    qb = heads * HEAD_DIM
    out["w_q_b"] = pack_fp8(logical["w_q_b"][rank * qb:(rank + 1) * qb])
    out["s_q_b"] = logical["s_q_b"][rank * qb // 128:(rank + 1) * qb // 128].contiguous()
    oa = groups * O_LORA
    out["w_o_a"] = pack_fp8(logical["w_o_a"][rank * oa:(rank + 1) * oa])
    out["s_o_a"] = logical["s_o_a"][rank * oa // 128:(rank + 1) * oa // 128].contiguous()
    out["w_o_b"] = pack_fp8(logical["w_o_b"][:, rank * oa:(rank + 1) * oa].contiguous())
    out["s_o_b"] = logical["s_o_b"][:, rank * oa // 128:(rank + 1) * oa // 128].contiguous()
    out["w_qkv_a"] = pack_fp8(logical["w_qkv_a"])
    out["w_i_q_b"] = pack_fp8(logical["w_i_q_b"])
    out["w_qkv_c"] = pack_bf16(logical["w_qkv_c"])
    out["w_r"] = pack_bf16(logical["w_r"])
    for side in ("attn", "ffn"):
        out[f"hc_{side}_fn"] = pack_hc_fn(logical[f"hc_{side}_fn"])
    # the shared expert (FP8, 128 x 128 scales): gate | up rows of this rank, then its down columns
    out["w_sug"] = pack_fp8(_halves(logical["w_sug"], rank * inter, inter, INTER))
    out["s_sug"] = _halves(logical["s_sug"], rank * inter // 128, inter // 128, INTER // 128).contiguous()
    out["w_sdn"] = pack_fp8(logical["w_sdn"][:, rank * inter:(rank + 1) * inter].contiguous())
    out["s_sdn"] = logical["s_sdn"][:, rank * inter // 128:(rank + 1) * inter // 128].contiguous()
    # the routed MXFP4 bank: gate | up rows and down K columns of this rank, in ATOM's layout
    out["w_ug"] = pack_mxfp4(_halves(logical["w_ug"], rank * inter, inter, INTER), True)
    out["s_ug"] = pack_mxfp4_scales(_halves(logical["s_ug"], rank * inter, inter, INTER), True)
    out["w_dn"] = pack_mxfp4(logical["w_dn"][:, :, rank * inter // 2:(rank + 1) * inter // 2].contiguous(), False)
    out["s_dn"] = pack_mxfp4_scales(logical["s_dn"][:, :, rank * inter // 32:(rank + 1) * inter // 32].contiguous(),
                                    False)
    return out


def rank_expected(logical_outputs, *, rank, world_size):
    """Every rank writes the whole (replicated) layer output and the same cache rows."""
    names = ("out", "kv_cache", "kv_rope", "c_state", "i_state")
    return {name: logical_outputs[name] for name in names}
