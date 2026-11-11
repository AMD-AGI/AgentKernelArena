# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Shared by inputs.py and reference.py: the model constants, the routed experts' MXFP4 bytes (a
deterministic function of a key the generator draws) and the float64 layer the reference computes.

Routed-expert bytes. The 896 experts' MXFP4 weights are 15.7 GB for the whole TP group (each rank holds
its 1/8 TP slice of every expert, 1.97 GB). The generator draws one 31-bit key a tensor; every byte is a
fixed hash of (key, expert, row, column), so each rank's shard computes only its own slice and the
reference only the experts the step picked, on any device, bit for bit the same. The hash is the
`lowbias32` integer mixer (multiplications split in 16-bit halves so no int64 product overflows):

    rowkey(e, row) = mix32((mix32(e * 8192 + row) + key) mod 2^32)
    word(e, row, g) = mix32((rowkey + g * 0x9E3779B9) mod 2^32)      g = column // 4 (codes) or column (scales)

E2M1 codes take the word's four bytes (little endian: column 4g + i is byte i); E8M0 scale bytes are
118 + word mod 5 (2^-9 .. 2^-5, the repository's own synthetic range). Logical coordinates are those of the
full TP group: gate rows 0..3071 and up rows 3072..6143 of w13 (3584 latent columns, 1792 bytes), w2
[3584 latent rows, 3072 intermediate columns, 1536 bytes]; rank r holds gate / up rows 384 r .. 384 r + 383 and
w2 columns 384 r .. 384 r + 383.

Numerics (float64; the production tensors' bfloat16 hand-offs and the MXFP4 quantizations are applied
exactly; see README.md).
"""
import torch

TP = 8
HIDDEN = 7168
CORE = 1536                 # a rank's attention core width (12 heads x 128)
LAT = 3584                  # routed-expert (latent) width
EXPERTS = 896
TOP_K = 16
RI = 384                    # routed intermediate a rank
INTER = TP * RI             # 3072
SI = 768                    # shared intermediate a rank
UP_N = HIDDEN // TP         # 896 hidden columns a rank up-projects
MAX_BLOCKS = 8              # attention-residual bank rows allocated
NBLOCKS = 4                 # bank rows live at this layer's MLP seam (a mid-model layer)
EPS = 1e-5                  # mlp_res_norm, post_attention_layernorm, the latent RMSNorm (K2Build defaults)
BETA, LINEAR_BETA = 4.0, 25.0
SCALE_LO, SCALE_SPAN = 118, 5

_M32 = (1 << 32) - 1
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ---------------------------------------------------------------- the expert byte hash
def _mul32(x, c):
    """(x * c) mod 2^32 for int64 x in [0, 2^32) and a 32-bit constant c, without int64 overflow."""
    lo = x * (c & 0xFFFF)
    hi = ((x * (c >> 16)) & 0xFFFF) << 16
    return (lo + hi) & _M32


def mix32(x):
    """lowbias32 (C. Wellons): a 32-bit integer mixer on int64 tensors holding values in [0, 2^32)."""
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846CA68B)
    return x ^ (x >> 16)


def _words(key, experts, rows, groups, device):
    """int64 [len(experts), len(rows), len(groups)] hash words of (key, expert, row, group)."""
    e = torch.as_tensor(experts, dtype=torch.int64, device=device).view(-1, 1, 1)
    r = torch.as_tensor(rows, dtype=torch.int64, device=device).view(1, -1, 1)
    g = torch.as_tensor(groups, dtype=torch.int64, device=device).view(1, 1, -1)
    rowkey = mix32((mix32(e * 8192 + r) + int(key)) & _M32)
    return mix32((rowkey + g * 0x9E3779B9) & _M32)


def code_bytes(key, experts, rows, col0, ncols, device):
    """uint8 [E, R, ncols] E2M1 code bytes at columns col0 .. col0 + ncols (both multiples of 4)."""
    assert col0 % 4 == 0 and ncols % 4 == 0
    w = _words(key, experts, rows, range(col0 // 4, (col0 + ncols) // 4), device)
    w = torch.where(w >= 1 << 31, w - (1 << 32), w).to(torch.int32)
    return w.contiguous().view(torch.uint8).view(w.shape[0], w.shape[1], ncols)


def scale_bytes(key, experts, rows, col0, ncols, device):
    """uint8 [E, R, ncols] E8M0 scale bytes (118 .. 122) at columns col0 .. col0 + ncols."""
    w = _words(key, experts, rows, range(col0, col0 + ncols), device)
    return (w % SCALE_SPAN + SCALE_LO).to(torch.uint8)


def chunked(fn, key, experts, rows, col0, ncols, device, chunk=32):
    """``fn`` over expert chunks (bounds the int64 temporaries), concatenated."""
    experts = list(experts)
    parts = [fn(key, experts[i:i + chunk], rows, col0, ncols, device) for i in range(0, len(experts), chunk)]
    return torch.cat(parts, 0) if len(parts) > 1 else parts[0]


def gate_up_rows(rank):
    """Logical w13 rows of rank ``rank``'s slice: its gate rows, then its up rows."""
    return list(range(rank * RI, (rank + 1) * RI)) + list(range(INTER + rank * RI, INTER + (rank + 1) * RI))


def rank_experts(keys, rank, device):
    """Rank ``rank``'s routed-expert tensors in the logical (unshuffled) per-rank layout:
    w13 [E, 768, 1792], w13s [E, 768, 112], w2 [E, 3584, 192], w2s [E, 3584, 12]."""
    every = range(EXPERTS)
    gu = gate_up_rows(rank)
    lat = range(LAT)
    return {
        "w13": chunked(code_bytes, keys["w13"], every, gu, 0, LAT // 2, device),
        "w13s": chunked(scale_bytes, keys["w13s"], every, gu, 0, LAT // 32, device),
        "w2": chunked(code_bytes, keys["w2"], every, lat, rank * RI // 2, RI // 2, device),
        "w2s": chunked(scale_bytes, keys["w2s"], every, lat, rank * RI // 32, RI // 32, device),
    }


# ---------------------------------------------------------------- MXFP4
def e2m1_values(codes):
    """float64 values of packed E2M1 bytes [..., n] -> [..., 2 n] (low nibble = even element)."""
    table = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), dtype=torch.float64, device=codes.device)
    c = codes.to(torch.int64)
    return torch.stack([table[c & 15], table[c >> 4]], -1).flatten(-2)


def e8m0_values(scales):
    return torch.exp2(scales.to(torch.float64) - 127.0)


def dequant(codes, scales):
    """MXFP4 bytes [..., K/2] and scale bytes [..., K/32] -> float64 [..., K]."""
    v = e2m1_values(codes)
    return (v.view(*v.shape[:-1], -1, 32) * e8m0_values(scales).unsqueeze(-1)).flatten(-2)


def e8m0_roundup_code(amax):
    """aiter's ``_e8m0_roundup(amax, 6)`` as the kernel computes it: the float32 amax times float32(1/6),
    the exponent field rounded up when any mantissa bit is set, at most 254 (int64 codes)."""
    a = amax.to(torch.float32) * torch.tensor(1.0 / 6.0, dtype=torch.float32, device=amax.device)
    bits = a.view(torch.int32).to(torch.int64) & _M32
    return torch.clamp(((bits + 0x7FFFFF) >> 23) & 0xFF, max=254)


def e2m1_round(x):
    """Round-to-nearest-even onto the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} (saturating), sign kept."""
    a = x.abs()
    q = torch.full_like(a, 6.0)
    for limit, value, inclusive in ((5.0, 4.0, True), (3.5, 3.0, False), (2.5, 2.0, True), (1.75, 1.5, False),
                                    (1.25, 1.0, True), (0.75, 0.5, False), (0.25, 0.0, True)):
        q = torch.where(a <= limit if inclusive else a < limit, torch.full_like(a, value), q)
    return torch.copysign(q, x)


def mxfp4_quant(x):
    """MXFP4 quantize-dequantize of x [..., K] in 32-element blocks: the kernel's block scale
    (``e8m0_roundup_code`` of the block's absolute maximum) and E2M1 rounding of x / scale."""
    blocks = x.reshape(*x.shape[:-1], -1, 32)
    code = e8m0_roundup_code(blocks.abs().amax(-1))
    scale = torch.exp2(code.to(torch.float64) - 127.0).unsqueeze(-1)
    return (e2m1_round(blocks / scale) * scale).flatten(-2)


# ---------------------------------------------------------------- the layer
def bf16(x):
    """Round to bfloat16 (nearest even) and back to float64."""
    return x.to(torch.bfloat16).to(torch.float64)


def rms_norm(x, weight, eps=EPS):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight


def situ(g, u):
    """SiTU / SiTUv2 (beta 4, linear beta 25): (2 tanh(g / 4)) (1 + tanh(g / 2)) 25 tanh(u / 25)."""
    return (0.5 * BETA * torch.tanh(g / BETA)) * (1.0 + torch.tanh(0.5 * g)) * (LINEAR_BETA * torch.tanh(u / LINEAR_BETA))


def route(x, w_gate, bias):
    """Sigmoid router with the score-correction bias: the top-16 of sigmoid + bias (ties to the lower
    expert), weights the picked sigmoids renormalized. Returns (ids [S, 16] int64, weights, keys)."""
    sig = torch.sigmoid(x @ w_gate.double().T)
    keys = sig + bias.double()
    order = torch.argsort(-keys, dim=-1, stable=True)
    ids = order[:, :TOP_K]
    wts = sig.gather(1, ids)
    return ids, wts / wts.sum(-1, keepdim=True), keys


def seam(blocks, prefix, ares_nw, ares_qk, in_nw):
    """The MLP AttnRes seam: the live bank rows and the updated prefix scored through the score RMSNorm
    and the one-row projection, softmax-mixed, then post_attention_layernorm. Returns x before bf16."""
    rows = torch.cat([blocks[:, :NBLOCKS].double(), prefix.unsqueeze(1)], 1)
    scores = (rms_norm(rows, ares_nw.double()) * ares_qk.double()).sum(-1)
    probs = torch.softmax(scores, -1)
    mixed = (probs.unsqueeze(-1) * rows).sum(1)
    return rms_norm(mixed, in_nw.double())


def expert_tensors(keys, expert, device):
    """Expert ``expert``'s full TP-group weights, float64: gate [3072, 3584], up [3072, 3584],
    down [3584, 3072]."""
    gu_rows = range(2 * INTER)
    w13 = code_bytes(keys["w13"], [expert], gu_rows, 0, LAT // 2, device)[0]
    w13s = scale_bytes(keys["w13s"], [expert], gu_rows, 0, LAT // 32, device)[0]
    w2 = code_bytes(keys["w2"], [expert], range(LAT), 0, INTER // 2, device)[0]
    w2s = scale_bytes(keys["w2s"], [expert], range(LAT), 0, INTER // 32, device)[0]
    gate_up = dequant(w13, w13s)
    return gate_up[:INTER], gate_up[INTER:], dequant(w2, w2s)


def layer(inp):
    """The logical outputs of one layer launch on the TP8 group: (out, prefix) before their final
    bfloat16 rounding, and the routing (ids, keys) for the margin checks."""
    s = inp["core"].shape[0]
    device = inp["core"].device
    keys = {name: int(inp[name].reshape(-1)[0]) for name in ("w13", "w13s", "w2", "w2s")}
    # o_proj: each rank's bf16 partial over its 1536 core columns, all-reduced (fp32 sum, bf16)
    core, w_o = inp["core"].double(), inp["w_o"].double()
    attn = sum(bf16(core[:, r * CORE:(r + 1) * CORE] @ w_o[:, r * CORE:(r + 1) * CORE].T) for r in range(TP))
    attn = bf16(attn)
    prefix_exact = inp["prefix_in"].double() + attn
    prefix = bf16(prefix_exact)
    x = bf16(seam(inp["blocks"], prefix, inp["ares_nw"], inp["ares_qk"], inp["in_nw"]))
    ids, wts, scores = route(x, inp["w_gate"], inp["bias"])
    # the latent (bf16), MXFP4-quantized as the experts' input
    lat = bf16(x @ inp["w_ld"].double().T)
    latq = mxfp4_quant(lat)
    # the shared experts, each rank's TP slice: gate | up (bf16), SiTU (bf16), down (bf16 partial)
    w_sgu, w_sd = inp["w_sgu"].double(), inp["w_sd"].double()
    shared = []
    for r in range(TP):
        g = bf16(x @ w_sgu[r * SI:(r + 1) * SI].T)
        u = bf16(x @ w_sgu[TP * SI + r * SI:TP * SI + (r + 1) * SI].T)
        h = bf16(situ(g, u))
        shared.append(bf16(h @ w_sd[:, r * SI:(r + 1) * SI].T))
    # the routed experts: each rank's weighted sum over the token's picks, a bf16 partial
    routed = torch.zeros(TP, s, LAT, dtype=torch.float64, device=device)
    for e in torch.unique(ids).tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        gate, up, down = expert_tensors(keys, e, device)
        a = latq[tok]
        act = situ(a @ gate.T, a @ up.T)                          # [n, 3072] (SiTUv2 on exact sums)
        for r in range(TP):
            q = mxfp4_quant(act[:, r * RI:(r + 1) * RI])          # 12 blocks of 32 a rank
            y = q @ down[:, r * RI:(r + 1) * RI].T                # [n, 3584]
            routed[r].index_add_(0, tok, y * wts[tok, slot].unsqueeze(-1))
    routed = bf16(routed)
    lat_sum = bf16(routed.sum(0))
    ln = bf16(rms_norm(lat_sum, inp["ln_w"].double()))
    up_all = ln @ inp["w_up"].double().T                          # [S, 7168]: rank r owns 896 r ..
    final = []
    for r in range(TP):
        f = shared[r].clone()
        cols = slice(r * UP_N, (r + 1) * UP_N)
        f[:, cols] = bf16(f[:, cols] + up_all[:, cols])
        final.append(f)
    moe = bf16(sum(final))
    out_exact = moe + prefix
    return {"out": out_exact, "prefix": prefix_exact, "ids": ids, "keys": scores}


def margins(ids, scores):
    """The routing margins: min over tokens of (16th - 17th biased score) and of the smallest gap between
    consecutive picks."""
    top = torch.sort(scores, -1, descending=True).values
    return float((top[:, TOP_K - 1] - top[:, TOP_K]).min()), float((top[:, :TOP_K - 1] - top[:, 1:TOP_K]).min())

