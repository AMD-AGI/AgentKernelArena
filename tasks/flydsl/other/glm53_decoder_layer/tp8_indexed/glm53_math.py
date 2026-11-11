# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Float64 PyTorch math of one GLM-5.3 indexed decoder layer, shared by reference.py and inputs.py.

The layer is the one sgl-amd-mono's ``glm5_monokernel`` runs for a decode or MTP-verify step (indexer layer,
block-FP8 attention weights, ATOM bf16 MLA cache, FP8 experts), written over the whole layer: all 64 attention
heads and all 2048 expert intermediates, so the tensor-parallel split does not appear. Everything is float64;
the quantizations the operator defines are applied exactly:

* block-FP8 weights (E4M3 codes times an f32 scale per 128 x 128 block) are dequantized exactly;
* the paged DSA index cache holds E4M3 keys with one power-of-two f32 scale per token; a new key is quantized
  with scale = 2^ceil(log2(max(amax, 1e-4) / 448)) and round-to-nearest-even E4M3 codes;
* the MLA cache is bf16: the attention reads the new rows as stored (bf16), the cache output is compared
  against the unrounded values;
* the expert input and the expert mid are quantized to E4M3 per 128-element block with scale = amax / 448.

Nothing here reads the baseline or a kernel language.
"""
import math

import torch

HIDDEN = 6144
Q_LORA = 2048
KV_LORA = 512
PE_DIM = 64
NOPE_DIM = 192
V_DIM = 256
QK_DIM = KV_LORA + PE_DIM
HEADS = 64
N_ROUTED = 256
SHARED_EXPERT = N_ROUTED
TOP_K = 8
INTER = 2048
ROUTE_SCALE = 2.5
EPS = 1e-5
INDEX_EPS = 1e-6
INDEX_HEADS = 32
INDEX_DIM = 128
TOPK = 2048
PAGE = 64
PAGE_BYTES = PAGE * (INDEX_DIM + 4)
FP8_MAX = 448.0
SOFTMAX_SCALE = (NOPE_DIM + PE_DIM) ** -0.5
QKV_A_ROWS = Q_LORA + KV_LORA + PE_DIM
BLOCK = 128


# ----------------------------------------------------------------------------------- number formats
def fp8_values(t):
    """float8_e4m3fn codes -> their float64 values."""
    return t.float().double()


def fp8_round(x):
    """Round float64 values (|x| <= 448) to E4M3 (round to nearest even); returns the codes."""
    return x.clamp(-FP8_MAX, FP8_MAX).float().to(torch.float8_e4m3fn)


def bf16_round(x):
    return x.float().bfloat16().double()


def dequant_blocks(w, scale, rows_per_block=BLOCK, cols_per_block=BLOCK):
    """E4M3 [..., rows, cols] with f32 block scales [..., ceil(rows / rb), ceil(cols / cb)] -> float64."""
    rows, cols = w.shape[-2], w.shape[-1]
    s = scale.double().repeat_interleave(rows_per_block, -2)[..., :rows, :]
    s = s.repeat_interleave(cols_per_block, -1)[..., :cols]
    return fp8_values(w) * s


def block_quant(x, block=BLOCK):
    """Per-``block`` E4M3 quantization along the last axis with scale amax / 448 (1 for an all-zero block);
    returns the dequantized float64 values."""
    shape = x.shape
    b = x.reshape(*shape[:-1], shape[-1] // block, block)
    amax = b.abs().amax(-1, keepdim=True)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    return (fp8_values(fp8_round(b / scale)) * scale).reshape(shape)


def index_key_quant(v):
    """The index cache's quantization of keys [..., 128]: (E4M3 codes, float64 power-of-two scales)."""
    amax = v.abs().amax(-1)
    ratio = amax.clamp_min(1e-4) / FP8_MAX
    scale = torch.exp2(torch.ceil(torch.log2(ratio)))
    return fp8_round(v / scale[..., None]), scale


# ----------------------------------------------------------------------------------- elementwise pieces
def rms_norm(x, g, eps=EPS):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * g.double()


def layer_norm(x, g, b, eps=INDEX_EPS):
    mean = x.mean(-1, keepdim=True)
    c = x - mean
    return c * torch.rsqrt(c.square().mean(-1, keepdim=True) + eps) * g.double() + b.double()


def rope_interleaved(x, cos, sin):
    """Rotate the pairs (2i, 2i + 1) of x [..., 2n] by cos / sin [..., n] (broadcast over the middle axes)."""
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


def _sylvester(n, device):
    h = torch.ones(1, 1, dtype=torch.float64, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def hadamard(x):
    """The orthonormal Walsh-Hadamard transform (Sylvester order) of the last axis."""
    n = x.shape[-1]
    return x @ _sylvester(n, x.device) * n**-0.5


# ----------------------------------------------------------------------------------- paged index cache
def index_value_bytes(off, d):
    """Byte of key ``d`` of token ``off`` in a 64-token page: 16-token x 16-byte tiles (SGLang's gfx950 layout)."""
    return (off // 16) * (16 * INDEX_DIM) + (off % 16) * 16 + (d // 16) * 256 + d % 16


def index_scale_bytes(off):
    return PAGE * INDEX_DIM + off * 4


def _token_bytes(slots):
    """[n, 132] byte columns (within the page) of each slot's 128 key bytes then its 4 scale bytes."""
    off = (slots % PAGE)[:, None]
    d = torch.arange(INDEX_DIM, device=slots.device)[None]
    keys = index_value_bytes(off, d)
    scale = index_scale_bytes(off) + torch.arange(4, device=slots.device)[None]
    return torch.cat([keys, scale], 1)


def read_index_tokens(cache, slots):
    """(codes as float64 [n, 128], scales float64 [n]) of cache slots ``slots`` (int64)."""
    flat = cache.reshape(cache.shape[0], PAGE_BYTES)
    raw = flat[(slots // PAGE)[:, None], _token_bytes(slots)]
    codes = fp8_values(raw[:, :INDEX_DIM].contiguous().view(torch.float8_e4m3fn))
    scales = raw[:, INDEX_DIM:].contiguous().view(torch.float32).double().flatten()
    return codes, scales


def write_index_tokens(cache, slots, codes, scales):
    """Store E4M3 ``codes`` [n, 128] and float ``scales`` [n] at cache slots ``slots`` (in place)."""
    flat = cache.view(cache.shape[0], PAGE_BYTES)
    data = torch.cat([codes.contiguous().view(torch.uint8),
                      scales.float().contiguous().view(-1, 1).view(torch.uint8)], 1)
    flat[(slots // PAGE)[:, None], _token_bytes(slots)] = data


def pack_index_pages(codes, scales):
    """E4M3 codes [pages, 64, 128] and f32 scales [pages, 64] -> the uint8 cache [pages, 64, 132]."""
    pages = codes.shape[0]
    values = codes.contiguous().view(torch.uint8).view(pages, 4, 16, 8, 16).permute(0, 1, 3, 2, 4)
    out = torch.empty(pages, PAGE_BYTES, dtype=torch.uint8, device=codes.device)
    out[:, :PAGE * INDEX_DIM] = values.reshape(pages, PAGE * INDEX_DIM)
    out[:, PAGE * INDEX_DIM:] = scales.float().contiguous().view(torch.uint8).view(pages, PAGE * 4)
    return out.view(pages, PAGE, INDEX_DIM + 4)


def unpack_index_pages(cache, page_ids):
    """(E4M3 codes [n * 64, 128], f32 scales [n * 64]) of whole pages ``page_ids``, tokens in page order."""
    flat = cache.reshape(cache.shape[0], PAGE_BYTES)[page_ids.long()]
    n = flat.shape[0]
    codes = flat[:, :PAGE * INDEX_DIM].reshape(n, 4, 8, 16, 16).permute(0, 1, 3, 2, 4)
    codes = codes.reshape(n * PAGE, INDEX_DIM).contiguous().view(torch.float8_e4m3fn)
    scales = flat[:, PAGE * INDEX_DIM:].contiguous().view(torch.float32).reshape(n * PAGE)
    return codes, scales


def row_key_slots(page_table_row, bound):
    """Cache slots of keys 0 .. bound - 1 of a row whose request's pages are ``page_table_row``."""
    keys = torch.arange(bound, device=page_table_row.device)
    return page_table_row.long()[keys // PAGE] * PAGE + keys % PAGE


# ----------------------------------------------------------------------------------- the layer
def front(inp):
    """Everything before the selection, for every row: a dict of float64 tensors.

    xn: the input RMSNorm; q_nope [S, 64, 192], q_pe [S, 64, 64] (rotated); kv_new [S, 576] (the MLA cache
    row, unrounded); iq [S, 32, 128] (index query, rotated and transformed); iw [S, 32] (index head weights);
    ik_codes (E4M3) [S, 128] and ik_scale [S] (the new index-cache entries)."""
    h = inp["hidden"].double()
    pos = inp["positions"].long()
    cos, sin = inp["rope_cos"].double()[pos], inp["rope_sin"].double()[pos]
    xn = rms_norm(h, inp["g_in"])
    qkv = xn @ dequant_blocks(inp["w_qkv_a"], inp["s_qkv_a"]).T
    q_a, kv_a, k_pe = qkv[:, :Q_LORA], qkv[:, Q_LORA:Q_LORA + KV_LORA], qkv[:, Q_LORA + KV_LORA:]
    q_an = rms_norm(q_a, inp["g_q"])
    q = (q_an @ dequant_blocks(inp["w_q_b"], inp["s_q_b"]).T).view(h.shape[0], HEADS, NOPE_DIM + PE_DIM)
    q_nope = q[..., :NOPE_DIM]
    q_pe = rope_interleaved(q[..., NOPE_DIM:], cos[:, None], sin[:, None])
    kv_new = torch.cat([rms_norm(kv_a, inp["g_kv"]), rope_interleaved(k_pe, cos, sin)], -1)
    iq = (q_an @ dequant_blocks(inp["w_iq"], inp["s_iq"]).T).view(h.shape[0], INDEX_HEADS, INDEX_DIM)
    iq = torch.cat([rope_interleaved(iq[..., :PE_DIM], cos[:, None], sin[:, None]), iq[..., PE_DIM:]], -1)
    iq = hadamard(iq)
    iw = xn @ inp["w_iw"].double().T
    ik = layer_norm(xn @ dequant_blocks(inp["w_ik"], inp["s_ik"]).T, inp["g_ik"], inp["b_ik"])
    ik = hadamard(torch.cat([rope_interleaved(ik[:, :PE_DIM], cos, sin), ik[:, PE_DIM:]], -1))
    ik_codes, ik_scale = index_key_quant(ik)
    return {"xn": xn, "q_nope": q_nope, "q_pe": q_pe, "kv_new": kv_new, "iq": iq, "iw": iw,
            "ik_codes": ik_codes, "ik_scale": ik_scale}


def index_raw(codes, iq_rows, iw_rows):
    """sum_h w_h relu(q_h . k) of keys ``codes`` [n, 128] for rows (iq [r, 32, 128], iw [r, 32]): [n, r]."""
    r = iq_rows.shape[0]
    dots = codes @ iq_rows.reshape(r * INDEX_HEADS, INDEX_DIM).T
    return (dots.clamp_min(0).view(-1, r, INDEX_HEADS) * iw_rows[None]).sum(-1)


def updated_index_cache(inp, f):
    cache = inp["index_cache"].clone()
    write_index_tokens(cache, inp["slots"].long(), f["ik_codes"], f["ik_scale"])
    return cache


def row_scores(cache, inp, f, s):
    """(cache slots, float64 scores) of row s's keys 0 .. position (the new entries already in ``cache``)."""
    bound = int(inp["positions"][s]) + 1
    row = inp["page_table"][s]
    slots = row_key_slots(row, bound)
    codes, scales = unpack_index_pages(cache, row[:(bound + PAGE - 1) // PAGE])
    raw = index_raw(fp8_values(codes[:bound]), f["iq"][s:s + 1], f["iw"][s:s + 1])[:, 0]
    return slots, raw * scales[:bound].double()


def select(cache, inp, f):
    """Each row's top-2048 cache slots [S, 2048] (score descending, then position), and the scores of the
    2048th and 2049th keys [S, 2] (the selection's margin)."""
    rows = inp["hidden"].shape[0]
    picked, edge = [], []
    for s in range(rows):
        slots, scores = row_scores(cache, inp, f, s)
        order = torch.sort(scores, descending=True, stable=True).indices
        picked.append(slots[order[:TOPK]])
        nxt = scores[order[TOPK]] if scores.numel() > TOPK else torch.zeros((), dtype=scores.dtype,
                                                                                device=scores.device)
        edge.append(torch.stack([scores[order[TOPK - 1]], nxt]))
    return torch.stack(picked), torch.stack(edge)


def attention(inp, f, picked):
    """The attention residual a = hidden + o_proj(sparse MLA), float64 [S, 6144]."""
    h = inp["hidden"].double()
    rows = h.shape[0]
    new_slots = inp["slots"].long()
    w_uk = fp8_values(inp["w_uk"]) * inp["s_uk"].double()      # [64, 192, 512]
    w_uv = fp8_values(inp["w_uv"]) * inp["s_uv"].double()      # [64, 512, 256]
    kv_cache = inp["kv_cache"]
    o = torch.empty(rows, HEADS, V_DIM, dtype=torch.float64, device=h.device)
    for s in range(rows):
        sel = picked[s]
        kv = kv_cache[sel].double()
        # a key this launch writes is read as the bf16 row it stores
        for r in range(rows):
            hit = sel == new_slots[r]
            if bool(hit.any()):
                kv[hit] = bf16_round(f["kv_new"][r])
        q_lat = torch.einsum("hd,hdc->hc", f["q_nope"][s], w_uk)
        logits = (q_lat @ kv[:, :KV_LORA].T + f["q_pe"][s] @ kv[:, KV_LORA:].T) * SOFTMAX_SCALE
        p = torch.softmax(logits, -1)
        o[s] = torch.einsum("hc,hcv->hv", p @ kv[:, :KV_LORA], w_uv)
    return h + o.reshape(rows, HEADS * V_DIM) @ dequant_blocks(inp["w_o"], inp["s_o"]).T


def route(inp, a):
    """(expert ids [S, 8], route weights [S, 8], sigmoid + bias [S, 256]) of the router on rows ``a``."""
    z = rms_norm(a, inp["g_post"])
    sig = torch.sigmoid(z @ inp["w_r"].double().T)
    keyed = sig + inp["bias"].double()
    ids = torch.sort(keyed, dim=-1, descending=True, stable=True).indices[:, :TOP_K]
    picked = sig.gather(1, ids)
    return ids, picked / picked.sum(-1, keepdim=True) * ROUTE_SCALE, keyed


def moe(inp, a):
    """The routed + shared FP8 experts on the post-attention RMSNorm of ``a``: float64 [S, 6144]."""
    rows = a.shape[0]
    z = rms_norm(a, inp["g_post"])
    xq = block_quant(z)
    ids, weights, _ = route(inp, a)
    slots = torch.cat([ids, torch.full((rows, 1), SHARED_EXPERT, dtype=ids.dtype, device=ids.device)], 1)
    wts = torch.cat([weights, torch.ones(rows, 1, dtype=weights.dtype, device=weights.device)], 1)
    out = torch.zeros(rows, HIDDEN, dtype=torch.float64, device=a.device)
    for e in torch.unique(slots).tolist():
        r, k = (slots == e).nonzero(as_tuple=True)
        w13 = dequant_blocks(inp["w_ug"][e], inp["s_ug"][e])
        gu = xq[r] @ w13.T
        del w13
        mid = torch.nn.functional.silu(gu[:, :INTER]) * gu[:, INTER:]
        y = block_quant(mid) @ dequant_blocks(inp["w_dn"][e], inp["s_dn"][e]).T
        out.index_add_(0, r, y * wts[r, k][:, None])
    return out


def layer(inp):
    """The layer's outputs from the logical inputs: (out f64, topk slots int64, kv_cache f64, index_cache u8)."""
    f = front(inp)
    cache = updated_index_cache(inp, f)
    picked, _ = select(cache, inp, f)
    a = attention(inp, f, picked)
    out = a + moe(inp, a)
    kv = inp["kv_cache"].double()
    kv[inp["slots"].long()] = f["kv_new"]
    return out, picked, kv, cache


def check_margins(edge, keyed, min_ratio=2.0, min_gap=0.05):
    """Fail when a discrete decision has no clear margin: the 2048th / 2049th index score ratio and the gap
    between each row's 8th and 9th router key (sigmoid + bias)."""
    ratio = edge[:, 0] / edge[:, 1].clamp_min(1e-300)
    if bool((ratio < min_ratio).any()):
        raise ValueError(f"index selection margin {float(ratio.min()):.3g} below {min_ratio}")
    top = torch.sort(keyed, -1, descending=True).values
    gap = top[:, TOP_K - 1] - top[:, TOP_K]
    if bool((gap < min_gap).any()):
        raise ValueError(f"router margin {float(gap.min()):.3g} below {min_gap}")


def ceil_log2(x):
    return math.ceil(math.log2(x))
