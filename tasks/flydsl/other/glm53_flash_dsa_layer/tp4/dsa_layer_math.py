# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Float64 definition of one GLM-5.3-Flash sparse-attention (DSA) decoder layer, as the
sgl-amd-mono monokernel ``flash_ffn_kernel`` (``FlashBuild(dsa=True)``) defines it.

Shared by ``reference.py`` (the correctness oracle) and ``inputs.py`` (whose generator plants
the margins of the layer's two discrete decisions on the same math). It imports only torch.

Semantics, per step row s (``rows`` rows; a row with position -1 is a pad row):

attention-side mHC (``hc_seam`` + ``hc_gate``)
    R'[j] = sum_i comb[i][j] R[i] + post[j] x                       (four streams, j = 0..3)
    mixes m = fn . R' (fn [24, 4 * 4096], column j * 4096 + c), rinv = rsqrt(mean(R'^2) + 1e-5)
    pre = sigmoid(m[0:4] rinv sc0 + b[0:4]) + 1e-6, post' = 2 sigmoid(m[4:8] rinv sc1 + b[4:8])
    comb' = Sinkhorn(softmax over the last axis of m[8:24] rinv sc2 + b[8:24]): softmax + 1e-6,
        a column normalization, then 19 (row, column) normalizations (each sum + 1e-6)
    a = RMSNorm(sum_i pre[i] R'[i]) * input_norm
sparse MLA (NoPE, 64 heads, latent 512)
    q_a | kv_a = W_qkv_a a (FP8 128x128-block weights, exactly dequantized), ik | iw = W_idx a,
    ig = W_gate a; q_an = RMSNorm(q_a) g_qa; kv_n = RMSNorm(kv_a) g_kv (written to the MLA cache);
    kn = LayerNorm(ik; 1e-6) ln_w + ln_b; q_nope = W_qb q_an; q_lat[h] = W_UK[h]^T q_nope[h].
    Index query: iq = W_iq q_an, per head the Hadamard-128 (Sylvester order, / sqrt(128)), bf16, then
    UE8M0 FP8 (scale 2^ceil(log2(max(amax, 1e-4) / 448))); head weight w'_h = iw_h q_scale_h /
    sqrt(128 * 32). A row whose position completes a kpool pool (pos % 4 == 3) compresses it:
    per dim softmax over the pool's 4 tokens of (gate + ape[j]) weighting their k (three from the
    request's tail ring, its own (kn, ig)), bf16, Hadamard, bf16, UE8M0 FP8 into the index cache.
    Its (kn, ig) then go to ring slot pos % 4. A row with more than 512 complete pools scores them,
    score(p) = scale(p) sum_h w'_h relu(q8_h . k8_p), and attends to the 512 best pools' 2048 tokens
    plus its tail tokens (4 * bound .. pos); any other row attends to every token 0 .. pos.
    o_lat[h] = softmax(q_lat[h] . K / 16) K over the attended latent rows K; o[h] = W_UV[h] o_lat[h];
    x_attn = W_o o (FP8 block weights; the TP sum of the ranks' head shards). A pad row writes no
    cache and attends to nothing (x_attn = 0).
FFN-side mHC: R'' = seam(x_attn, R', post', comb') -> res_out; gate on R'' -> post_out, comb_out
    and the FFN input h (post_attention_layernorm).
MoE: logits = W_r h (bf16 weights), scores = sigmoid(logits); the 8 largest scores + bias (bias as
    bf16) are routed, weights = score / sum of the 8 scores * 2.5; the shared expert (slot 288) has
    weight 1. h is quantized MXFP8 per 32 (scale 2^ceil(log2(amax / 448)), E4M3 RNE), every expert
    is MXFP4 (E2M1 codes, low nibble first, E8M0 scale per 32): gate | up = W13[e] hq,
    mid = silu(min(gate, 10)) * clamp(up, -10, 10), quantized MXFP8 per 32 the same way,
    x_out = sum over slots of weight * W2[e] midq.

Everything is float64 on the dequantized operands, with the quantizations above emulated exactly
(from the float64 value rounded once to float32). The pool compression additionally rounds to
bf16 where the kernel stores bf16, so that, on inputs whose earlier ring keys are equal (the task's
generator plants them), the compressed FP8 key and its scale are bit-exact.
"""
import torch

HIDDEN = 4096
HC = 4
N_MIX = HC * (2 + HC)
EPS = 1e-5
HC_EPS = 1e-6
HC_POST_MULT = 2.0
SINKHORN_ITERS = 20
Q_LORA = 1536
KV_LORA = 512
NOPE = 256
V_DIM = 256
MLA_HEADS = 64
MLA_SCALE = NOPE ** -0.5
INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_EPS = 1e-6
KPOOL = 4
TOPK_POOLS = 512
TOPK_TOKENS = TOPK_POOLS * KPOOL
FWHT_SCALE = 0.08838834764831845
IQW_SCALE = INDEX_DIM ** -0.5 * INDEX_HEADS ** -0.5
N_EXPERTS = 288
SHARED = N_EXPERTS
TOP_K = 8
ROUTE_SCALE = 2.5
SWIGLU_LIMIT = 10.0
GLOBAL_INTER = 2048
FP8_MAX = 448.0
FP8_BLOCK = 128
MX_BLOCK = 32
# caches: 64-token MLA blocks (the page table's entries), 256-token index pages of 64 pools
BLOCK_TOKENS = 64
PAGE_TOKENS = 256
PAGE_POOLS = PAGE_TOKENS // KPOOL
POOL_BYTES = INDEX_DIM + 4
PAGE_BYTES = PAGE_POOLS * POOL_BYTES
# the step's row table: (position, MLA slot, index pool slot, tail ring slot, page-table row)
ROW_WORDS = 8
ROW_POS, ROW_SLOT, ROW_POOL_SLOT, ROW_TAIL_SLOT, ROW_BT = range(5)
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


# ------------------------------------------------------------------ quantization primitives
def _pow2_f32(exponent):
    """2^(exponent - 127) as float32, exponent an int32 tensor in 1..254."""
    return (exponent << 23).view(torch.float32)


def fp8_block_dequant(w, s):
    """FP8 E4M3 [N, K] with float32 [N / 128, K / 128] block scales -> float64 [N, K]."""
    n, k = w.shape
    blocks = w.double().view(n // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK)
    return (blocks * s.double()[:, None, :, None]).view(n, k)


def mxfp4_dequant(codes, scales):
    """MXFP4 [N, K / 2] packed E2M1 (low nibble = even element), [N, K / 32] E8M0 -> float64 [N, K]."""
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=codes.device)
    values = torch.stack((lut[(codes & 0xF).long()], lut[(codes >> 4).long()]), dim=-1)
    values = values.reshape(codes.shape[0], -1)
    return values * torch.exp2(scales.double() - 127.0).repeat_interleave(MX_BLOCK, dim=-1)


def mxfp8_quant_dequant(x64):
    """The kernel's MXFP8 rule over the last axis (blocks of 32), dequantized exactly in float64:
    e8m0 = biased exponent of float32(amax * float32(1/448)) rounded up when its mantissa is nonzero,
    clamped to 1..254; code = E4M3(clamp(x * 2^(127 - e8m0), +-448)) rounded to nearest even."""
    x = x64.float()
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // MX_BLOCK, MX_BLOCK)
    amax = blocks.abs().amax(dim=-1)
    y = amax * torch.tensor(1.0 / FP8_MAX, dtype=torch.float32, device=x.device)
    bits = y.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    exponent = exponent.clamp(1, 254)
    inverse = _pow2_f32(254 - exponent)
    codes = (blocks * inverse.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (codes.double() * _pow2_f32(exponent).double().unsqueeze(-1)).reshape(x.shape)


def fwht128(x):
    """Unnormalized Sylvester Hadamard transform over the last axis (128)."""
    shape = x.shape
    h = x.reshape(-1, INDEX_DIM)
    step = 1
    while step < INDEX_DIM:
        h = h.view(-1, INDEX_DIM // (2 * step), 2, step)
        a, b = h[:, :, 0, :], h[:, :, 1, :]
        h = torch.stack((a + b, a - b), dim=2).reshape(-1, INDEX_DIM)
        step *= 2
    return h.reshape(shape)


def hadamard_ue8m0(x64):
    """vLLM's index-key quantization of [..., 128] vectors: bf16(float32(H x) * float32(1/sqrt(128))),
    then FP8 with one power-of-two scale 2^ceil(log2(max(amax, 1e-4) / 448)) per vector.
    Returns (codes float8_e4m3fn, scale float32 [...])."""
    t = fwht128(x64).float() * torch.tensor(FWHT_SCALE, dtype=torch.float32, device=x64.device)
    t = t.to(torch.bfloat16).float()
    amax = t.abs().amax(dim=-1).clamp_min(torch.tensor(1.0e-4, dtype=torch.float32).item())
    y = amax * torch.tensor(1.0 / FP8_MAX, dtype=torch.float32, device=x64.device)
    bits = y.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    inverse = _pow2_f32(254 - exponent)
    codes = (t * inverse.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return codes, _pow2_f32(exponent)


# ------------------------------------------------------------------ index cache layout
def pool_byte_index(device):
    """[64, 128] byte of (pool offset, dim) in an index page: 16-pool x 16-byte tiles."""
    off = torch.arange(PAGE_POOLS, device=device)[:, None]
    d = torch.arange(INDEX_DIM, device=device)[None, :]
    return (off // 16) * (16 * INDEX_DIM) + (d // 16) * 256 + (off % 16) * 16 + d % 16


def index_keys(index_cache):
    """The index cache's FP8 keys [pages, 64, 128] (float8_e4m3fn) and scales [pages, 64] (float32)."""
    perm = pool_byte_index(index_cache.device).reshape(-1)
    keys = index_cache[:, perm].contiguous().view(torch.float8_e4m3fn).view(-1, PAGE_POOLS, INDEX_DIM)
    scales = index_cache[:, PAGE_POOLS * INDEX_DIM:].contiguous().view(torch.float32)
    return keys, scales


def write_pool(index_cache, pool_slot, codes, scale):
    """Store one pool's 128 FP8 codes and float32 scale at pool slot ``pool_slot`` (in place)."""
    page, off = pool_slot // PAGE_POOLS, pool_slot % PAGE_POOLS
    index_cache[page, pool_byte_index(index_cache.device)[off]] = codes.view(torch.uint8)
    start = PAGE_POOLS * INDEX_DIM + 4 * off
    index_cache[page, start:start + 4] = scale.reshape(1).contiguous().view(torch.uint8)


def pool_pages(page_table_row, n_pools):
    """(index page, pool offset) of pools 0 .. n_pools - 1 of a request: an index page is the entry of
    the first 64-token block of its 256 tokens over 4 (SGLang's 64-token page table)."""
    group = PAGE_TOKENS // BLOCK_TOKENS
    p = torch.arange(n_pools, device=page_table_row.device)
    page = page_table_row[(p // PAGE_POOLS) * group].long() // group
    return page, p % PAGE_POOLS


def kv_slots(page_table_row, tokens):
    """MLA cache rows of ``tokens`` (int64 tensor) of a request through its 64-token page table."""
    return page_table_row[tokens // BLOCK_TOKENS].long() * BLOCK_TOKENS + tokens % BLOCK_TOKENS


def row_keys(pos):
    """(complete pools, scored?) of a row at position ``pos``."""
    bound = (pos + 1) // KPOOL
    return bound, bound > TOPK_POOLS


# ------------------------------------------------------------------ layer pieces
def rms(x, weight, eps=EPS):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight


def hc_seam(x, streams, post, comb):
    """R'[s, j] = sum_i comb[s, i, j] streams[s, i] + post[s, j] x[s]."""
    return torch.einsum("sij,sic->sjc", comb, streams) + post * x[:, None, :]


def hc_gate(streams, fn, scale, base, norm_weight):
    """mHC gate of the streams [S, 4, H]: (normalized layer input [S, H], post [S, 4, 1], comb [S, 4, 4])."""
    rows = streams.shape[0]
    flat = streams.reshape(rows, HC * HIDDEN)
    mixes = flat @ fn.T
    rinv = torch.rsqrt(flat.square().mean(-1, keepdim=True) + EPS)
    pre = torch.sigmoid(mixes[:, :HC] * rinv * scale[0] + base[:HC]) + HC_EPS
    post = torch.sigmoid(mixes[:, HC:2 * HC] * rinv * scale[1] + base[HC:2 * HC]) * HC_POST_MULT
    logits = (mixes[:, 2 * HC:] * rinv * scale[2] + base[2 * HC:]).view(rows, HC, HC)
    comb = torch.softmax(logits, dim=-1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    mixed = (pre.unsqueeze(-1) * streams).sum(1)
    return rms(mixed, norm_weight), post.unsqueeze(-1), comb


_FRONT = ("x", "residual", "post", "comb", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "input_norm",
          "w_idx", "w_kpool_gate", "q_a_norm", "kv_a_norm", "k_norm_w", "k_norm_b", "w_kv_b", "w_index_q")


def attention_front(t):
    """Attention-side mHC and the projections every rank computes. ``t``: the logical inputs."""
    d = {k: t[k].double() for k in _FRONT}
    out = {}
    streams = hc_seam(d["x"], d["residual"], d["post"], d["comb"])
    a, post_a, comb_a = hc_gate(streams, d["hc_attn_fn"], d["hc_attn_scale"], d["hc_attn_base"], d["input_norm"])
    out.update(streams_a=streams, a=a, post_a=post_a, comb_a=comb_a)
    qkv = a @ fp8_block_dequant(t["w_qkv_a"], t["s_qkv_a"]).T
    q_a, kv_a = qkv[:, :Q_LORA], qkv[:, Q_LORA:]
    idx = a @ d["w_idx"].T
    out["ik"], out["iw"] = idx[:, :INDEX_DIM], idx[:, INDEX_DIM:]
    out["ig"] = a @ d["w_kpool_gate"].T
    q_an = rms(q_a, d["q_a_norm"])
    out["kv_n"] = rms(kv_a, d["kv_a_norm"])
    ik = out["ik"]
    mean = ik.mean(-1, keepdim=True)
    var = (ik - mean).square().mean(-1, keepdim=True)
    out["kn"] = (ik - mean) * torch.rsqrt(var + INDEX_EPS) * d["k_norm_w"] + d["k_norm_b"]
    rows = a.shape[0]
    q_nope = (q_an @ fp8_block_dequant(t["w_q_b"], t["s_q_b"]).T).view(rows, -1, NOPE)
    heads = q_nope.shape[1]
    kv_b = d["w_kv_b"].view(heads, NOPE + V_DIM, KV_LORA)
    out["q_lat"] = torch.einsum("hkl,shk->shl", kv_b[:, :NOPE], q_nope)
    iq = (q_an @ d["w_index_q"].T).view(rows, INDEX_HEADS, INDEX_DIM)
    q8, q_scale = hadamard_ue8m0(iq)
    out["q8"] = q8.double()
    out["iqw"] = out["iw"] * q_scale.double() * IQW_SCALE
    return out


def compress_pool(ring_k, ring_g, kn, ig, ape):
    """A completed pool's (FP8 codes [128], float32 scale): ring_k / ring_g [3, 128] the pool's first
    three tokens (float64 of bf16), kn / ig [128] the completing token's, ape [4, 128]."""
    k = torch.cat([ring_k, kn[None]], 0)
    s = torch.cat([ring_g, ig[None]], 0) + ape
    pooled = (torch.softmax(s, dim=0) * k).sum(0)
    pooled = pooled.float().to(torch.bfloat16).double()
    codes, scale = hadamard_ue8m0(pooled)
    return codes, scale


def index_scores(q8, iqw, keys, scales, pages, offs):
    """score(p) = scale(p) * sum_h w'_h relu(q8_h . k8_p) of one row over the pools (pages, offs)."""
    k = keys[pages, offs].double()
    return (k @ q8.T).clamp_min(0.0) @ iqw * scales[pages, offs].double()


def cache_updates(t, front):
    """The step's cache writes on copies of the caches: (kv_cache float32, index_cache uint8, tail_k and
    tail_g float64). The MLA cache is held in float32 (exact for its bf16 rows; the written rows are
    the float64 values rounded once to float32), as float64 copies of a long-context cache are large."""
    rows_tab = t["row_table"].long()
    kv = t["kv_cache"].float()
    idx = t["index_cache"].clone()
    tail_k = t["kpool_tail_k"].double().clone()
    tail_g = t["kpool_tail_gate"].double().clone()
    ring = tail_k.shape[1]
    ape = t["kpool_ape"].double()
    for s in range(rows_tab.shape[0]):
        pos, slot, pslot, tslot = (int(v) for v in rows_tab[s, :4])
        if pos < 0 or slot <= 0:
            continue
        kv[slot] = front["kv_n"][s]
        r = max(tslot, 0) // KPOOL
        if pslot >= 0 and pos % KPOOL == KPOOL - 1:
            prior = [(pos - (KPOOL - 1) + j) % ring for j in range(KPOOL - 1)]
            codes, scale = compress_pool(tail_k[r, prior], tail_g[r, prior], front["kn"][s], front["ig"][s], ape)
            write_pool(idx, pslot, codes, scale)
        if tslot >= 0:
            tail_k[r, pos % ring] = front["kn"][s]
            tail_g[r, pos % ring] = front["ig"][s]
    return kv, idx, tail_k, tail_g


def attended_tokens(s, pos, front, keys, scales, page_table_row):
    """Token positions row ``s`` attends to (int64)."""
    bound, scored = row_keys(pos)
    if not scored:
        return torch.arange(pos + 1, device=page_table_row.device)
    pages, offs = pool_pages(page_table_row, bound)
    score = index_scores(front["q8"][s], front["iqw"][s], keys, scales, pages, offs)
    pools = score.topk(TOPK_POOLS).indices
    picked = (pools[:, None] * KPOOL + torch.arange(KPOOL, device=pools.device)).reshape(-1)
    tail = torch.arange(bound * KPOOL, pos + 1, device=pools.device)
    return torch.cat([picked, tail])


def attention(t, front, kv, idx):
    """x_attn [S, H] (float64): sparse MLA over the updated caches, W_UV and o_proj, all heads."""
    rows_tab = t["row_table"].long()
    rows = rows_tab.shape[0]
    keys, scales = index_keys(idx)
    q_lat = front["q_lat"]
    heads = q_lat.shape[1]
    kv_b = t["w_kv_b"].double().view(heads, NOPE + V_DIM, KV_LORA)
    w_o = fp8_block_dequant(t["w_o"], t["s_o"])
    out = torch.zeros(rows, HIDDEN, dtype=torch.float64, device=q_lat.device)
    for s in range(rows):
        pos, slot = int(rows_tab[s, ROW_POS]), int(rows_tab[s, ROW_SLOT])
        if pos < 0 or slot <= 0:
            continue
        pt_row = t["page_table"][int(rows_tab[s, ROW_BT])]
        tokens = attended_tokens(s, pos, front, keys, scales, pt_row)
        lat = kv[kv_slots(pt_row, tokens)].double()
        p = torch.softmax((q_lat[s] @ lat.T) * MLA_SCALE, dim=-1)
        o_lat = p @ lat
        o = torch.einsum("hvl,hl->hv", kv_b[:, NOPE:], o_lat)
        out[s] = w_o @ o.reshape(-1)
    return out


def route(h, router_w, router_bias):
    """(expert ids [S, 8], route weights [S, 8]) of the FFN inputs h."""
    scores = torch.sigmoid(h @ router_w.double().T)
    keys = scores + router_bias.to(torch.bfloat16).double()
    ids = keys.topk(TOP_K, dim=-1).indices
    chosen = scores.gather(1, ids)
    return ids, chosen / chosen.sum(-1, keepdim=True) * ROUTE_SCALE


def moe(h, ids, weights, w13, w13_scale, w2, w2_scale):
    """x_out [S, H] of the shared + 8 routed MXFP4 experts on MXFP8 activations."""
    rows = h.shape[0]
    hq = mxfp8_quant_dequant(h)
    out = torch.zeros(rows, HIDDEN, dtype=torch.float64, device=h.device)
    slots = torch.cat([ids, torch.full((rows, 1), SHARED, dtype=ids.dtype, device=ids.device)], 1)
    coef = torch.cat([weights, torch.ones(rows, 1, dtype=weights.dtype, device=weights.device)], 1)
    for expert in torch.unique(slots).tolist():
        rs, ks = torch.where(slots == expert)
        up_gate = mxfp4_dequant(w13[expert], w13_scale[expert])
        inter = up_gate.shape[0] // 2
        gu = hq[rs] @ up_gate.T
        del up_gate
        gate = gu[:, :inter].clamp_max(SWIGLU_LIMIT)
        up = gu[:, inter:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        mid = mxfp8_quant_dequant(gate * torch.sigmoid(gate) * up)
        down = mxfp4_dequant(w2[expert], w2_scale[expert])
        out.index_add_(0, rs, (mid @ down.T) * coef[rs, ks].unsqueeze(-1))
        del down
    return out


def ffn_input(t, front, x_attn):
    """(R'' [S, 4, H], h [S, H], post_out [S, 4, 1], comb_out [S, 4, 4]) after the layer's attention."""
    d = {k: t[k].double() for k in ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "post_attn_norm")}
    streams = hc_seam(x_attn, front["streams_a"], front["post_a"], front["comb_a"])
    h, post, comb = hc_gate(streams, d["hc_ffn_fn"], d["hc_ffn_scale"], d["hc_ffn_base"], d["post_attn_norm"])
    return streams, h, post, comb


def layer(t):
    """Every output and final cache of the layer: a dict by the task's names."""
    front = attention_front(t)
    kv, idx, tail_k, tail_g = cache_updates(t, front)
    x_attn = attention(t, front, kv, idx)
    streams, h, post, comb = ffn_input(t, front, x_attn)
    ids, weights = route(h, t["router_w"], t["router_bias"])
    x_out = moe(h, ids, weights, t["w13"], t["w13_scale"], t["w2"], t["w2_scale"])
    return {"x_out": x_out, "res_out": streams, "post_out": post, "comb_out": comb,
            "kv_cache": kv, "index_cache": idx, "kpool_tail_k": tail_k.float(),
            "kpool_tail_gate": tail_g.float()}
