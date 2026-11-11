# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Float64 numerics of one MiniMax-M3 sparse MoE layer on a tensor-parallel-4 group, as the sgl-amd-mono
monokernel (``minimax_m3/gfx950/post_attn.py`` with K1 of ``pre_attn.py`` fused in front) defines it.

Shared by reference.py (the logical outputs) and inputs.py (which plants clear margins for the two discrete
decisions, block selection and routing). Never imports the baseline. Every product and sum is float64; the
dataflow roundings the kernel defines are applied exactly (bfloat16 stage outputs, per-token FP8, the FP8
attention probabilities, MXFP8 activations, the bfloat16 partial rows of both all-reduces); float32
accumulation order and the hardware exp2 / rcp / rsqrt approximations are not modelled.

Logical inputs (see inputs.py): replicated tensors as one rank sees them, rank-specific tensors stacked on a
leading ``TP`` axis (``w_qkv[r]`` is rank r's fused projection, ``k_cache[r]`` its KV head's cache), the caches
in plain token-major layout (slot = page * 128 + offset) and one block-table row per request (q_len = rows /
requests consecutive rows each).
"""
import math

import torch

TP = 4
HIDDEN = 6144
HEAD_DIM = 128
ROTARY_DIM = 64
LOCAL_Q_HEADS = 16
O_K = LOCAL_Q_HEADS * HEAD_DIM  # 2048
# a rank's fused projection rows: 16 q heads | k | v | its index q head | the (replicated) index k head
QKV_ROWS = (LOCAL_Q_HEADS + 4) * HEAD_DIM
K_OFF, V_OFF, IQ_OFF, IK_OFF = O_K, O_K + HEAD_DIM, O_K + 2 * HEAD_DIM, O_K + 3 * HEAD_DIM
N_ROUTED = 128
SHARED_EXPERT = N_ROUTED  # the fused shared expert's id
EXPERTS = N_ROUTED + 1
TOP_K = 4
INTER = 768  # expert intermediate per rank (3072 / TP)
SPARSE_BLOCK = 128
TOPK_BLOCKS = 16
INIT_BLOCKS = 0
LOCAL_BLOCKS = 1
SPLIT_KEYS = 256
FP8_MAX = 448.0
MX_BLOCK = 32
# MiniMax-M3 build constants (sgl_amd_mono/minimax_m3/spec.py _build and its config.json)
EPS = 1e-6
SM_SCALE = HEAD_DIM ** -0.5
ROUTE_SCALE = 2.0
SHARED_WEIGHT = 1.0
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
LOG2E = 1.4426950408889634
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def f32(x):
    """A Python float rounded to the nearest float32 (normal range), ties to even."""
    if x == 0.0 or not math.isfinite(x):
        return x
    m, e = math.frexp(x)
    return math.ldexp(round(m * (1 << 24)), e - 24)


# the indexer's fp32(fp32(sm_scale) * fp32(log2 e)) (index_score.index_scale_log2e)
INDEX_SCALE = f32(f32(SM_SCALE) * f32(LOG2E))


def bf16(x):
    return x.to(torch.float32).to(torch.bfloat16).to(torch.float64)


def fp8(x):
    return x.to(torch.float32).to(torch.float8_e4m3fn).to(torch.float64)


def sat(x):
    """The cache writers' clamp to the E4M3 range (fp8_sat)."""
    return x.clamp(-FP8_MAX, FP8_MAX)


def gemma_rms(x, gamma):
    """Gemma RMSNorm over the last axis: x * rsqrt(mean(x^2) + eps) * (gamma + 1)."""
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * (gamma.double() + 1.0)


def rope_neox(x, cos, sin):
    """Partial NeoX RoPE on the first ROTARY_DIM dims of the last axis (cos / sin: [..., ROTARY_DIM / 2])."""
    half = ROTARY_DIM // 2
    x1, x2 = x[..., :half], x[..., half:ROTARY_DIM]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., ROTARY_DIM:]], dim=-1)


def mxfp8(v):
    """MXFP8 of the last axis per 32-block, dequantized: e8m0 = ceil-pow2 code of fp32(amax * fp32(1/448))
    clamped to [1, 254] (aiter's RoundUp rule, ``mx8_scale``), code = E4M3(clamp(v * 2^(127 - e8m0), +-448))."""
    shape = v.shape
    blocks = v.reshape(*shape[:-1], shape[-1] // MX_BLOCK, MX_BLOCK).to(torch.float32)
    amax = blocks.abs().amax(-1)
    bits = (amax * torch.tensor(f32(1.0 / FP8_MAX), dtype=torch.float32, device=v.device)).view(torch.int32)
    code = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    code = code.clamp(1, 254).to(torch.float64)
    q = fp8((blocks.double() * torch.exp2(127.0 - code).unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX))
    return (q * torch.exp2(code - 127.0).unsqueeze(-1)).reshape(shape)


def mxfp4_dequant(codes, scales):
    """[N, K/2] e2m1 codes (low nibble = even element) and [N, K/32] e8m0 bytes -> float64 [N, K]."""
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=codes.device)
    values = torch.stack((lut[(codes & 0xF).long()], lut[(codes >> 4).long()]), dim=-1).reshape(codes.shape[0], -1)
    return values * torch.exp2(scales.to(torch.float64) - 127.0).repeat_interleave(MX_BLOCK, dim=-1)


def swiglu(g, u):
    """aiter swiglu_mul_batch: min(g, limit) * sigmoid(1.702 min(g, limit)) * (clamp(u, +-limit) + 1)."""
    g = g.clamp(max=SWIGLU_LIMIT)
    u = u.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    return g * torch.sigmoid(SWIGLU_ALPHA * g) * (u + 1.0)


def requests_of(seq_lens, q_len):
    """[(first row, last row)] of the step's requests: q_len consecutive rows each."""
    rows = seq_lens.numel()
    assert rows % q_len == 0, (rows, q_len)
    return [(i, i + q_len - 1) for i in range(0, rows, q_len)]


def query_len(inp):
    """Rows a request: the step's rows over its requests (block-table rows)."""
    rows, requests = inp["seq_lens"].numel(), inp["block_table"].shape[0]
    assert rows % requests == 0, (rows, requests)
    return rows // requests


def n_blocks(seq):
    return (max(int(seq), 1) + SPARSE_BLOCK - 1) // SPARSE_BLOCK


# ------------------------------------------------------------------------------------------------- K1
def k1(inp):
    """Input add + norm, per-token FP8, the fused FP8 qkv / index GEMV, head norms, RoPE, for every rank.

    -> dict: acc [S, H] (ar + res, exact), h (bf16(acc)), q [TP, S, 16, 128] (bf16), k / v [TP, S, 128]
    (the cache values before E4M3 rounding: clamp(bf16(k) / k_scale), clamp(v / v_scale)), iq [TP, S, 128]
    (bf16 index q), ik [S, 128] (bf16 index k, the cache value before E4M3 rounding, unit scale)."""
    acc = inp["ar"].double() + inp["res"].double()
    xn = gemma_rms(acc, inp["g_in"])
    amax = xn.abs().amax(-1, keepdim=True)
    x_scale = torch.where(amax == 0, torch.ones_like(amax), amax / FP8_MAX)
    x8 = fp8(xn / x_scale)
    rows = acc.shape[0]
    pos = inp["positions"].long()
    table = inp["cos_sin"].double()[pos]  # [S, 64]: cos | sin
    cos, sin = table[:, None, :ROTARY_DIM // 2], table[:, None, ROTARY_DIM // 2:]

    def head(e, gamma):  # e: [S, n, 128]
        return rope_neox(gemma_rms(e, gamma), cos, sin)

    k_scale = inp["k_scale"].double().reshape(())
    v_scale = inp["v_scale"].double().reshape(())
    qs, ks, vs, iqs, iks = [], [], [], [], []
    for r in range(TP):
        w = inp["w_qkv"][r].double()
        qkv = bf16((x8 @ w.T) * x_scale * inp["s_qkv"][r].double())
        qs.append(bf16(head(qkv[:, :O_K].reshape(rows, LOCAL_Q_HEADS, HEAD_DIM), inp["g_q"])))
        ks.append(sat(bf16(head(qkv[:, None, K_OFF:V_OFF], inp["g_k"])[:, 0]) / k_scale))
        vs.append(sat(qkv[:, V_OFF:IQ_OFF] / v_scale))
        iqs.append(bf16(head(qkv[:, None, IQ_OFF:IK_OFF], inp["g_iq"])[:, 0]))
        iks.append(bf16(head(qkv[:, None, IK_OFF:], inp["g_ik"])[:, 0]))
    return {"acc": acc, "h": bf16(acc), "q": torch.stack(qs), "k": torch.stack(ks), "v": torch.stack(vs),
            "iq": torch.stack(iqs), "ik": iks[0]}


def write_caches(inp, pre):
    """The caches after K1's writes: E4M3 working copies (what the kernel's readers see) and float32 outputs
    holding the written slots' values before their E4M3 rounding (the reference's output values)."""
    slots = inp["slot_mapping"].long()
    live = slots >= 0
    s = slots[live]
    k, v, ik = inp["k_cache"].clone(), inp["v_cache"].clone(), inp["index_cache"].clone()
    k_out, v_out, ik_out = k.float(), v.float(), ik.float()
    for r in range(TP):
        k[r, s] = pre["k"][r][live].float().to(torch.float8_e4m3fn)
        v[r, s] = pre["v"][r][live].float().to(torch.float8_e4m3fn)
        k_out[r, s], v_out[r, s] = pre["k"][r][live].float(), pre["v"][r][live].float()
    ik[s] = pre["ik"][live].float().to(torch.float8_e4m3fn)
    ik_out[s] = pre["ik"][live].float()
    return {"k": k, "v": v, "ik": ik}, {"k_cache": k_out, "v_cache": v_out, "index_cache": ik_out}


# ------------------------------------------------------------------------------------- indexer + select
def request_keys(cache, table_row, n_keys):
    """Keys 0 .. n_keys - 1 of a request (its block-table row) from a token-major cache [slots, 128]."""
    pos = torch.arange(n_keys, device=cache.device)
    slot = table_row.long()[pos // SPARSE_BLOCK] * SPARSE_BLOCK + pos % SPARSE_BLOCK
    return cache[slot].double()


def index_scores(inp, iq, ik_cache):
    """Indexer block scores: scores[r][k] = float64 [n_scored] for every row k whose request's last row has
    more than TOPK_BLOCKS blocks (blocks INIT_BLOCKS .. n_blk - LOCAL_BLOCKS - 1 of the last row), else None.
    Score of block b for row k = max over its keys key < seq_k of bf16(E4M3 index k) . bf16 index q, times
    fp32(sm_scale * log2 e)."""
    seq = inp["seq_lens"].long().tolist()
    q_len = query_len(inp)
    out = [[None] * len(seq) for _ in range(TP)]
    for first, last in requests_of(inp["seq_lens"], q_len):
        n_blk = n_blocks(seq[last])
        if n_blk <= TOPK_BLOCKS:
            continue
        n_scored = n_blk - INIT_BLOCKS - LOCAL_BLOCKS
        n_keys = (INIT_BLOCKS + n_scored) * SPARSE_BLOCK
        keys = request_keys(ik_cache, inp["block_table"][first // q_len], n_keys)
        cols = list(range(first, last + 1))
        pos = torch.arange(keys.shape[0], device=keys.device)
        for r in range(TP):
            dots = keys @ iq[r][cols].T  # [keys, cols]
            for j, k in enumerate(cols):
                d = dots[:, j].masked_fill(pos >= max(seq[k], 1), -math.inf)
                out[r][k] = d.view(-1, SPARSE_BLOCK).amax(-1)[INIT_BLOCKS:] * INDEX_SCALE
    return out


def select_blocks(seq_k, scores_k):
    """A row's sparse context, as the kernel places it: the TOPK_BLOCKS - 1 best scored blocks by (score, block
    id) descending, then the tail block (the one holding the row's last key, always pinned); a context of at
    most TOPK_BLOCKS blocks keeps every block in id order. -> (blocks in placement order, sparse key count)."""
    n_blk = n_blocks(seq_k)
    tail = n_blk - 1
    if n_blk <= TOPK_BLOCKS:
        order = list(range(n_blk))
    else:
        assert INIT_BLOCKS == 0 and LOCAL_BLOCKS == 1
        sc = scores_k[:tail]
        ids = torch.arange(tail - 1, -1, -1, device=sc.device)  # descending ids: a stable sort keeps them first
        ranked = ids[torch.sort(sc[ids], descending=True, stable=True).indices]
        order = ranked[:TOPK_BLOCKS - 1].tolist() + [tail]
    n_ctx = (len(order) - 1) * SPARSE_BLOCK + max(int(seq_k), 1) - tail * SPARSE_BLOCK
    return order, n_ctx


# ------------------------------------------------------------------------------------------- attention
def sparse_attention(q, k_cache, v_cache, table_row, order, n_ctx, k_scale, v_scale):
    """One row's sparse paged attention on one rank: FP8 q (unit scale) x E4M3 K (k_scale), 256-key splits in
    placement order, each with its own max, FP8 probabilities (P8 = E4M3(p * v_scale * 448 / (v_scale + 1e-8)),
    dequantized with v_scale / 448) and a bf16 partial output, then the splits' log-sum-exp merge and bf16.
    q: [16, 128] bf16 values -> [16, 128]."""
    pos = torch.cat([torch.arange(b * SPARSE_BLOCK, (b + 1) * SPARSE_BLOCK, device=q.device) for b in order])
    pos = pos[:n_ctx]
    slot = table_row.long()[pos // SPARSE_BLOCK] * SPARSE_BLOCK + pos % SPARSE_BLOCK
    kk, vv = k_cache[slot].double(), v_cache[slot].double()
    s = SM_SCALE * k_scale * (fp8(q) @ kk.T)  # [16, n_ctx]
    p_scale = v_scale * (FP8_MAX / (v_scale + 1e-8))
    ms, ls, os_ = [], [], []
    for t0 in range(0, n_ctx, SPLIT_KEYS):
        st = s[:, t0:t0 + SPLIT_KEYS]
        m = st.amax(-1, keepdim=True)
        p = torch.exp(st - m)
        l = p.sum(-1, keepdim=True)
        p8 = fp8(p * p_scale)
        os_.append(bf16((v_scale / FP8_MAX) * (p8 @ vv[t0:t0 + SPLIT_KEYS]) / l))
        ms.append(m)
        ls.append(l)
    m = torch.cat(ms, -1)
    w = torch.cat(ls, -1) * torch.exp(m - m.amax(-1, keepdim=True))
    w = w / w.sum(-1, keepdim=True)
    return bf16(sum(w[:, i:i + 1] * os_[i] for i in range(len(os_))))


def per_token_fp8(x):
    """Standalone per-token FP8 (scale = amax / 448; an all-zero row stays zero) -> (dequantized codes, scale)."""
    amax = x.abs().amax(-1, keepdim=True)
    scale = amax / FP8_MAX
    safe = torch.where(amax == 0, torch.ones_like(scale), scale)
    return torch.where(amax == 0, torch.zeros_like(x), fp8(x / safe)), scale


def attention_block(inp):
    """K1, the cache writes, the indexer, block selection, sparse attention, o_proj and the attention
    all-reduce. -> dict with the K1 results, the caches, 'scores', 'selection' [r][k], 'a' (bf16 all-reduced
    attention), 'h_mid' (bf16(a) + h, before its rounding) and 'xn' (the router's bf16 normalized input)."""
    pre = k1(inp)
    caches, cache_out = write_caches(inp, pre)
    scores = index_scores(inp, pre["iq"], caches["ik"])
    seq = inp["seq_lens"].long().tolist()
    rows = len(seq)
    q_len = query_len(inp)
    k_scale = inp["k_scale"].double().reshape(())
    v_scale = inp["v_scale"].double().reshape(())
    partial_sum = torch.zeros(rows, HIDDEN, dtype=torch.float64, device=pre["acc"].device)
    selection = [[None] * rows for _ in range(TP)]
    for r in range(TP):
        attn = []
        for k in range(rows):
            order, n_ctx = select_blocks(seq[k], scores[r][k])
            selection[r][k] = order
            table = inp["block_table"][k // q_len]
            attn.append(sparse_attention(pre["q"][r][k], caches["k"][r], caches["v"][r], table, order, n_ctx,
                                         k_scale, v_scale).reshape(O_K))
        a8, a_scale = per_token_fp8(torch.stack(attn))
        w_o = inp["w_o"][r].double()
        # the rank's bf16 partial rows, summed in rank order by the all-reduce
        partial_sum = partial_sum + bf16((a8 @ w_o.T) * a_scale * inp["s_o"][r].double())
    a = bf16(partial_sum)
    v = a + pre["h"]
    xn = bf16(gemma_rms(v, inp["g_post"]))
    return {**pre, "caches": caches, "cache_out": cache_out, "scores": scores, "selection": selection,
            "a": a, "h_mid": v, "xn": xn}


# ----------------------------------------------------------------------------------------------- MoE
def route(xn, gate, bias):
    """Sigmoid router with bias-corrected selection: logits = bf16(xn . gate), top-4 of sigmoid(logits) + bias,
    weights = the picks' sigmoids renormalized and scaled by route_scale. -> (ids [S, 4], weights [S, 4],
    logits [S, 128])."""
    logits = bf16(xn @ gate.double().T)
    sc = torch.sigmoid(logits)
    ids = (sc + bias.double()).topk(TOP_K, dim=-1).indices
    picked = sc.gather(1, ids)
    weights = picked * (ROUTE_SCALE / picked.sum(-1, keepdim=True).clamp_min(1e-20))
    return ids, weights, logits


def moe(inp, xn, ids, weights):
    """The routed experts and the fused shared expert of every rank (its INTER slice), MXFP8 activations x
    MXFP4 weights, swiglu, MXFP8 mids, route-weighted down projection; each rank's bf16 partial rows summed in
    rank order. -> float64 [S, H], before the all-reduce output's bf16 rounding."""
    rows = xn.shape[0]
    xq = mxfp8(xn)
    total = torch.zeros(rows, HIDDEN, dtype=torch.float64, device=xn.device)
    slots = torch.cat([ids, torch.full((rows, 1), SHARED_EXPERT, dtype=ids.dtype, device=ids.device)], 1)
    wts = torch.cat([weights, torch.full((rows, 1), SHARED_WEIGHT, dtype=weights.dtype, device=ids.device)], 1)
    for r in range(TP):
        part = torch.zeros(rows, HIDDEN, dtype=torch.float64, device=xn.device)
        for e in sorted(set(slots.flatten().tolist())):
            tok, slot = torch.where(slots == e)
            w13 = mxfp4_dequant(inp["w13"][r][e], inp["s13"][r][e])  # [2 * INTER, H]: gate rows, then up
            gu = xq[tok] @ w13.T
            mid = mxfp8(bf16(swiglu(gu[:, :INTER], gu[:, INTER:])))
            del w13
            w2 = mxfp4_dequant(inp["w2"][r][e], inp["s2"][r][e])  # [H, INTER]
            # a token routes to an expert at most once: distinct rows, a deterministic update
            part[tok] = part[tok] + (mid @ w2.T) * wts[tok, slot].unsqueeze(-1)
            del w2
        total = total + bf16(part)
    return total


def layer(inp):
    """The whole layer: -> (attention block dict, routing (ids, weights, logits), MoE output before rounding)."""
    att = attention_block(inp)
    ids, weights, logits = route(att["xn"], inp["w_gate"], inp["bias"])
    return att, (ids, weights, logits), moe(inp, att["xn"], ids, weights)
