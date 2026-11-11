# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Inputs of one GLM-5.3-Flash sparse-attention (DSA) decoder layer at tensor parallelism 4.

``make_inputs`` draws the logical (whole-model) layer once per case and seed; ``shard_inputs`` cuts one
rank's call from it, ``rank_expected`` gives the tensors that rank writes, and ``to_kernel_layout``
turns a rank's row-major weights into the layouts the production kernel reads in place.

Logical inputs (the reference reads these; every rank draws the same values from the same seed):
  x [rows, 4096] bf16, residual [rows, 4, 4096] bf16, post [rows, 4, 1] f32, comb [rows, 4, 4] f32: the
      previous layer's output with its FFN post still owed (unit-normal x and streams, post in [0, 2),
      comb row-softmax of unit normals; as the upstream layer test draws them).
  hc_attn_* / hc_ffn_*: the layer's two mHC gates (fn [24, 16384] N(0, 0.01), scale U(0.5, 1.5),
      base N(0, 0.1)); input_norm / post_attn_norm 1 + N(0, 0.1) (bf16).
  w_qkv_a [2048, 4096], w_q_b [64 * 256, 1536], w_o [4096, 64 * 256]: FP8 E4M3 with float32 128 x 128
      block scales (amax / 448 per block, as the upstream test quantizes N(0, 1 / fan_in) weights).
  w_idx [160, 4096] bf16 (the indexer's wk rows, then its 32 weights_proj rows), w_kpool_gate [128, 4096],
      w_index_q [4096, 1536], w_kv_b [64 * 512, 512] (kv_b_proj: per head 256 W_UK rows, then 256 W_UV
      rows), w_uk [64, 512, 256] = W_UK^T packed from w_kv_b (the copy production makes at load).
  kv_cache [256 * (1 + requests * pages), 512] bf16: SGLang's MLA latent cache (256-token pages; page 0 is
      the padding page); index_cache [1 + requests * pages, 64 * 132] uint8: the kpool index cache, per
      page 64 pools' FP8 keys in 16-pool x 16-byte tiles, then their float32 scales; kpool_tail_k /
      kpool_tail_gate [requests + 1, 4, 128] bf16: the kpool ring of every request (row 0 unused).
  row_table [rows, 8] int32: (position, MLA slot, index pool slot, ring slot, page-table row) of every
      row, -1 (and page-table row 0) for the width's pad rows; page_table [rows, 4 * pages] int32:
      SGLang's 64-token page table (a 256-token page p is entries 4p .. 4p + 3).
  router_w [288, 4096] bf16, router_bias [288] f32, w13 [289, 2 * 2048, 2048] uint8 (gate rows, then up
      rows; MXFP4 codes, two per byte), w13_scale [289, 4096, 128] uint8 (E8M0), w2 [289, 4096, 1024],
      w2_scale [289, 4096, 64]: the 288 routed experts and the shared expert (slot 288), whole width.
  guard [8] int32: the launch's guard block (epoch 0, a 10 s wait deadline in 100 MHz ticks).

Request lengths: the first request holds ``ctx`` tokens after this step's token; the others between
ctx / 2 and ctx (ragged batch). The second request's length is a multiple of 4, so its token completes a
kpool pool (the first's does when ctx % 4 == 0).

Planted margins (the two discrete decisions must not flip between the reference and the kernel):
  * Pool selection. The 32 indexer head weights of every row are positive (weights_proj rows solved so
    that weights_proj . a = U(0.5, 2) for the rows' attention inputs, plus N(0, 0.3 / 64) noise), so every
    pool's unscaled score u = sum_h w'_h relu(q8_h . k8) is nonnegative. Each scored row's pools get
    power-of-two scales: its 512 best pools score in [2T, 4T), every other pool below T / 4 (a factor 8),
    from the reference's own index query. A completed pool's scale is the kernel's (it is written this
    step), so T is chosen to put it at 3T.
  * Routing. The router rows are solved so that every row's logits are U(3, 6) for 8 random experts and
    N(0, 0.25) clipped to +-0.75 for the others, on the reference's own FFN input (bias U(-0.1, 0.1)): the
    8th and 9th keys differ by more than 0.07.
  * Pool compression. A completing request's three earlier ring entries share one key on the 1/64 grid
    (|k| in [0.25, 4)) and gates 16 above its own token's, so the softmax-weighted key is that key exactly
    in bf16, its Hadamard is exact, and the compressed FP8 key and scale are bit-exact on any correct
    implementation (``index_cache`` is compared exactly).
Experts: uniform random MXFP4 codes with E8M0 scales 116..120 (gate and up of about unit size); in the
``_clamp`` case 119..123, so that about a tenth of the gate and up values pass the SwiGLU limit of 10.
"""
import torch

import dsa_layer_math as m

DEADLINE_TICKS = 10 * 100_000_000  # the guard's wait deadline: 10 s of the 100 MHz s_memrealtime clock
GUARD_WORDS = 8
RING = 4
INDEX_WEIGHT_RANGE = (0.5, 2.0)
ROUTED_LOGITS = (3.0, 6.0)
OTHER_LOGIT_STD, OTHER_LOGIT_MAX = 0.25, 0.75
ROUTER_BIAS = 0.1
RING_GATE = 24.0
RING_GATE_MARGIN = 16.0


def _normal(shape, g, device, std=1.0):
    return torch.randn(shape, generator=g, device=device, dtype=torch.float32) * std


def _uniform(shape, g, device, low, high):
    return torch.rand(shape, generator=g, device=device, dtype=torch.float32) * (high - low) + low


def _fp8_block(w):
    """FP8 E4M3 with float32 128 x 128 block scales (amax / 448 per block): (codes, scales)."""
    n, k = w.shape
    blocks = w.view(n // 128, 128, k // 128, 128)
    scales = blocks.abs().amax((1, 3)) / 448.0 + 1e-12
    codes = (blocks / scales[:, None, :, None]).to(torch.float8_e4m3fn)
    return codes.view(n, k), scales.contiguous()


def _solve_rows(noise, basis, targets):
    """noise [N, K] + C basis (basis [S, K]) with (noise + C basis) basis^T = targets^T (targets [S, N])."""
    gram = basis @ basis.T
    rhs = targets - basis @ noise.T
    return noise + torch.linalg.solve(gram, rhs).T @ basis


def _row_lengths(requests, ctx, g, device):
    lengths = [ctx]
    for r in range(1, requests):
        low = max(1, ctx // 2)
        n = int(torch.randint(low, ctx + 1, (1,), generator=g, device=device))
        if r == 1 and ctx >= 4:
            n = max(4, n // 4 * 4)  # completes a pool
        lengths.append(n)
    return lengths


def _tables(requests, rows, pages, lengths, g, device):
    """(page_table, row_table, per-request ring rows) of the step."""
    order = torch.randperm(requests * pages, generator=g, device=device) + 1
    req_pages = order.view(requests, pages)
    page_table = torch.zeros(rows, 4 * pages, dtype=torch.int32, device=device)
    page_table[:requests] = (req_pages[:, :, None] * 4 + torch.arange(4, device=device)).view(requests, -1).int()
    ring_rows = torch.randperm(requests, generator=g, device=device) + 1
    row_table = torch.full((rows, m.ROW_WORDS), -1, dtype=torch.int32, device=device)
    row_table[:, m.ROW_BT] = 0
    for r, n in enumerate(lengths):
        pos = n - 1
        slot = int(page_table[r, pos // m.BLOCK_TOKENS]) * m.BLOCK_TOKENS + pos % m.BLOCK_TOKENS
        row_table[r, :5] = torch.tensor([pos, slot, slot // m.KPOOL, int(ring_rows[r]) * m.KPOOL, r],
                                        dtype=torch.int32, device=device)
    return page_table, row_table


def _index_cache(n_pages, g, device):
    """Random FP8 keys (N(0, 64^2) clipped to +-448) and power-of-two scales 2^-6 .. 2^-9."""
    cache = torch.empty(n_pages, m.PAGE_BYTES, dtype=torch.uint8, device=device)
    values = _normal((n_pages, m.PAGE_POOLS * m.INDEX_DIM), g, device, 64.0).clamp(-448, 448)
    perm = m.pool_byte_index(device).reshape(-1)
    cache[:, perm] = values.to(torch.float8_e4m3fn).view(torch.uint8)
    exps = torch.randint(6, 10, (n_pages, m.PAGE_POOLS), generator=g, device=device)
    cache[:, m.PAGE_POOLS * m.INDEX_DIM:] = torch.exp2(-exps.float()).view(torch.uint8)
    return cache


def _plant_completion(t, front, s, g, device):
    """Ring entries of row s's pool (its three earlier tokens): one shared key on the 1/64 grid and
    gates RING_GATE_MARGIN above its own token's (plus the ape differences)."""
    pos, tslot = int(t["row_table"][s, m.ROW_POS]), int(t["row_table"][s, m.ROW_TAIL_SLOT])
    r = tslot // m.KPOOL
    grid = torch.randint(16, 256, (m.INDEX_DIM,), generator=g, device=device).float()
    sign = torch.randint(0, 2, (m.INDEX_DIM,), generator=g, device=device).float() * 2 - 1
    key = (grid * sign / 64.0).to(torch.bfloat16)
    ape = t["kpool_ape"].double()
    need = (front["ig"][s] + ape[3] - ape[:3]).max().item() + RING_GATE_MARGIN
    gate = max(RING_GATE, float(int(need) + 1))
    assert gate <= 256.0, "ring gate is not an exact bf16 integer"
    for j in range(m.KPOOL - 1):
        slot = (pos - (m.KPOOL - 1) + j) % RING
        t["kpool_tail_k"][r, slot] = key
        t["kpool_tail_gate"][r, slot] = gate


def _plant_pool_scales(t, front, s, keys, g):
    """Power-of-two pool scales of scored row s: its 512 best pools in [2T, 4T), the others below T / 4."""
    rows_tab = t["row_table"]
    pos, pslot = int(rows_tab[s, m.ROW_POS]), int(rows_tab[s, m.ROW_POOL_SLOT])
    bound, _ = m.row_keys(pos)
    pt_row = t["page_table"][int(rows_tab[s, m.ROW_BT])]
    pages, offs = m.pool_pages(pt_row, bound)
    q8, iqw = front["q8"][s], front["iqw"][s]
    u = (keys[pages, offs].double() @ q8.T).clamp_min(0.0) @ iqw
    completing = pos % m.KPOOL == m.KPOOL - 1
    candidates = torch.arange(bound - 1 if completing else bound, device=u.device)
    score_c = 0.0
    if completing:
        # the pool this step completes: its key and scale are the kernel's compression
        r = int(rows_tab[s, m.ROW_TAIL_SLOT]) // m.KPOOL
        prior = [(pos - (m.KPOOL - 1) + j) % RING for j in range(m.KPOOL - 1)]
        codes, scale = m.compress_pool(t["kpool_tail_k"][r, prior].double(), t["kpool_tail_gate"][r, prior].double(),
                                       front["kn"][s], front["ig"][s], t["kpool_ape"].double())
        score_c = float(((codes.double() @ q8.T).clamp_min(0.0) @ iqw) * scale.double())
    n_win = m.TOPK_POOLS - (1 if score_c > 0 else 0)
    order = candidates[u[candidates].argsort(descending=True)]
    winners, losers = order[:n_win], order[n_win:]
    assert bool((u[winners] > 0).all()), "a planted winner pool has no positive head score"
    scale_t = score_c / 3.0 if score_c > 0 else float(u[winners[-1]]) * 2.0 ** -6
    exps = torch.zeros(bound, dtype=torch.float64, device=u.device)
    exps[winners] = torch.ceil(torch.log2(2.0 * scale_t / u[winners]))
    if losers.numel():
        lu = u[losers]
        positive = lu > 0
        cap = torch.floor(torch.log2(scale_t / (4.0 * lu[positive].median()))) if positive.any() else \
            torch.tensor(0.0, dtype=torch.float64, device=u.device)
        own = torch.where(positive, torch.floor(torch.log2(scale_t / (4.0 * lu.clamp_min(1e-300)))), cap)
        exps[losers] = torch.minimum(own, cap)
    assert bool((exps[candidates].abs() < 120).all()), "a planted pool scale leaves the float32 range"
    scales = torch.exp2(exps[candidates]).float()
    idx = t["index_cache"]
    start = m.PAGE_POOLS * m.INDEX_DIM
    byte = start + 4 * offs[candidates][:, None] + torch.arange(4, device=u.device)
    idx[pages[candidates][:, None], byte] = scales.view(torch.uint8).view(-1, 4)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    g = generator
    requests, rows, ctx, pages = axes["requests"], axes["rows"], axes["ctx"], axes["pages"]
    hidden, heads = axes["hidden"], axes["mla_heads"]
    assert hidden == m.HIDDEN and heads == m.MLA_HEADS and axes["n_experts"] == m.N_EXPERTS
    clamp = case["id"].endswith("_clamp")
    t = {}
    # the previous layer's output, its FFN post owed
    t["x"] = _normal((rows, hidden), g, device).to(torch.bfloat16)
    t["residual"] = _normal((rows, m.HC, hidden), g, device).to(torch.bfloat16)
    t["post"] = _uniform((rows, m.HC, 1), g, device, 0.0, 2.0)
    t["comb"] = torch.softmax(_normal((rows, m.HC, m.HC), g, device), dim=-1)
    for side, norm in (("attn", "input_norm"), ("ffn", "post_attn_norm")):
        t[f"hc_{side}_fn"] = _normal((m.N_MIX, m.HC * hidden), g, device, 0.01)
        t[f"hc_{side}_scale"] = _uniform((3,), g, device, 0.5, 1.5)
        t[f"hc_{side}_base"] = _normal((m.N_MIX,), g, device, 0.1)
        t[norm] = (1.0 + _normal((hidden,), g, device, 0.1)).to(torch.bfloat16)
    # sparse attention weights (the whole model: 64 heads)
    t["w_qkv_a"], t["s_qkv_a"] = _fp8_block(_normal((m.Q_LORA + m.KV_LORA, hidden), g, device, hidden ** -0.5))
    w_idx = _normal((m.INDEX_DIM + m.INDEX_HEADS, hidden), g, device, hidden ** -0.5)
    t["w_kpool_gate"] = _normal((m.INDEX_DIM, hidden), g, device, hidden ** -0.5).to(torch.bfloat16)
    t["q_a_norm"] = (1.0 + _normal((m.Q_LORA,), g, device, 0.1)).to(torch.bfloat16)
    t["kv_a_norm"] = (1.0 + _normal((m.KV_LORA,), g, device, 0.1)).to(torch.bfloat16)
    t["k_norm_w"] = 1.0 + _normal((m.INDEX_DIM,), g, device, 0.1)
    t["k_norm_b"] = _normal((m.INDEX_DIM,), g, device, 0.1)
    t["kpool_ape"] = _normal((m.KPOOL, m.INDEX_DIM), g, device, 0.5)
    t["w_q_b"], t["s_q_b"] = _fp8_block(_normal((heads * m.NOPE, m.Q_LORA), g, device, m.Q_LORA ** -0.5))
    t["w_index_q"] = _normal((m.INDEX_HEADS * m.INDEX_DIM, m.Q_LORA), g, device, m.Q_LORA ** -0.5).to(torch.bfloat16)
    kv_b = _normal((heads * (m.NOPE + m.V_DIM), m.KV_LORA), g, device, m.KV_LORA ** -0.5).to(torch.bfloat16)
    t["w_kv_b"] = kv_b
    t["w_uk"] = kv_b.view(heads, m.NOPE + m.V_DIM, m.KV_LORA)[:, :m.NOPE].transpose(1, 2).contiguous()
    t["w_o"], t["s_o"] = _fp8_block(_normal((hidden, heads * m.V_DIM), g, device, (heads * m.V_DIM) ** -0.5))
    # the step's requests, tables and caches
    lengths = _row_lengths(requests, ctx, g, device)
    t["page_table"], t["row_table"] = _tables(requests, rows, pages, lengths, g, device)
    n_pages = 1 + requests * pages
    t["kv_cache"] = _normal((n_pages * m.PAGE_TOKENS, m.KV_LORA), g, device).to(torch.bfloat16)
    t["index_cache"] = _index_cache(n_pages, g, device)
    t["kpool_tail_k"] = _normal((requests + 1, RING, m.INDEX_DIM), g, device).to(torch.bfloat16)
    t["kpool_tail_gate"] = _normal((requests + 1, RING, m.INDEX_DIM), g, device).to(torch.bfloat16)
    # positive indexer head weights on the rows' attention inputs
    d = {k: t[k].double() for k in ("x", "residual", "post", "comb", "hc_attn_fn", "hc_attn_scale",
                                     "hc_attn_base", "input_norm")}
    a, _, _ = m.hc_gate(m.hc_seam(d["x"], d["residual"], d["post"], d["comb"]), d["hc_attn_fn"],
                        d["hc_attn_scale"], d["hc_attn_base"], d["input_norm"])
    targets = _uniform((rows, m.INDEX_HEADS), g, device, *INDEX_WEIGHT_RANGE).double()
    noise = w_idx[m.INDEX_DIM:].double() * 0.3
    w_idx[m.INDEX_DIM:] = _solve_rows(noise, a, targets).float()
    t["w_idx"] = w_idx.to(torch.bfloat16)
    front = m.attention_front(t)
    assert bool((front["iqw"] > 0).all()), "an indexer head weight is not positive"
    live = [s for s in range(requests)]
    for s in live:
        if lengths[s] % m.KPOOL == 0:
            _plant_completion(t, front, s, g, device)
    keys, _ = m.index_keys(t["index_cache"])
    for s in live:
        if m.row_keys(lengths[s] - 1)[1]:
            _plant_pool_scales(t, front, s, keys, g)
    # the reference's own FFN input, for the router margins
    kv, idx, _, _ = m.cache_updates(t, front)
    x_attn = m.attention(t, front, kv, idx)
    del kv, idx
    _, h, _, _ = m.ffn_input(t, front, x_attn)
    logits = (_normal((rows, m.N_EXPERTS), g, device, OTHER_LOGIT_STD)
              .clamp(-OTHER_LOGIT_MAX, OTHER_LOGIT_MAX).double())
    for s in range(rows):
        picks = torch.randperm(m.N_EXPERTS, generator=g, device=device)[:m.TOP_K]
        logits[s, picks] = _uniform((m.TOP_K,), g, device, *ROUTED_LOGITS).double()
    router_noise = _normal((m.N_EXPERTS, hidden), g, device, 0.02).double()
    t["router_w"] = _solve_rows(router_noise, h, logits).to(torch.bfloat16)
    t["router_bias"] = _uniform((m.N_EXPERTS,), g, device, -ROUTER_BIAS, ROUTER_BIAS)
    # experts: the 288 routed and the shared one (slot 288), the whole intermediate width
    n_slots, inter = m.N_EXPERTS + 1, axes["global_inter"]
    lo, hi = (119, 124) if clamp else (116, 121)
    t["w13"] = torch.randint(0, 256, (n_slots, 2 * inter, hidden // 2), generator=g, device=device,
                             dtype=torch.uint8)
    t["w13_scale"] = torch.randint(lo, hi, (n_slots, 2 * inter, hidden // 32), generator=g, device=device,
                                   dtype=torch.uint8)
    t["w2"] = torch.randint(0, 256, (n_slots, hidden, inter // 2), generator=g, device=device, dtype=torch.uint8)
    t["w2_scale"] = torch.randint(lo, hi, (n_slots, hidden, inter // 32), generator=g, device=device,
                                  dtype=torch.uint8)
    guard = torch.zeros(GUARD_WORDS, dtype=torch.int32, device=device)
    guard[1], guard[2] = DEADLINE_TICKS & 0xFFFFFFFF, DEADLINE_TICKS >> 32
    t["guard"] = guard
    return t


# ------------------------------------------------------------------ one rank's call
def shard_inputs(logical, *, rank, world_size):
    """Rank ``rank``'s inputs: its 64 / world_size MLA heads (q_b rows, kv_b rows, W_UK^T, o_proj columns
    and their block scales) and its 2048 / world_size expert intermediate columns (w13 gate and up rows,
    w2 and scale columns); everything else is replicated."""
    out = dict(logical)
    heads = m.MLA_HEADS // world_size
    h0, h1 = rank * heads, (rank + 1) * heads
    out["w_q_b"] = logical["w_q_b"][h0 * m.NOPE:h1 * m.NOPE].contiguous()
    out["s_q_b"] = logical["s_q_b"][h0 * m.NOPE // 128:h1 * m.NOPE // 128].contiguous()
    out["w_kv_b"] = logical["w_kv_b"][h0 * (m.NOPE + m.V_DIM):h1 * (m.NOPE + m.V_DIM)].contiguous()
    out["w_uk"] = logical["w_uk"][h0:h1].contiguous()
    out["w_o"] = logical["w_o"][:, h0 * m.V_DIM:h1 * m.V_DIM].contiguous()
    out["s_o"] = logical["s_o"][:, h0 * m.V_DIM // 128:h1 * m.V_DIM // 128].contiguous()
    full = logical["w13"].shape[1] // 2
    inter = full // world_size
    lo, hi = rank * inter, (rank + 1) * inter
    for name in ("w13", "w13_scale"):
        w = logical[name]
        out[name] = torch.cat([w[:, lo:hi], w[:, full + lo:full + hi]], dim=1).contiguous()
    out["w2"] = logical["w2"][:, :, lo // 2:hi // 2].contiguous()
    out["w2_scale"] = logical["w2_scale"][:, :, lo // 32:hi // 32].contiguous()
    return out


def rank_expected(logical_outputs, *, rank, world_size):
    """Every rank writes the whole outputs (the TP sums run in place, in rank order) and its own replica
    of the caches: the logical outputs, unchanged."""
    return dict(logical_outputs)


# ------------------------------------------------------------------ kernel-facing layouts
def shuffle_16x16(w):
    """AITER ``shuffle_weight(w, layout=(16, 16))`` of a 1-byte [..., N, K] tensor: 16-row x 32-byte tiles,
    each split in two 16-byte halves ([N / 16, K / 32, 2, 16, 16])."""
    *lead, n, k = w.shape
    x = w.view(torch.uint8).reshape(-1, n // 16, 16, k // 32, 2, 16)
    return x.permute(0, 1, 3, 4, 2, 5).contiguous().view(*lead, n, k).view(w.dtype)


def e8m0_shuffle(s):
    """AITER ``e8m0_shuffle`` of a [rows, groups] E8M0 matrix (rows % 32 == 0, groups % 8 == 0)."""
    rows, groups = s.shape
    x = s.view(rows // 32, 2, 16, groups // 8, 2, 4)
    return x.permute(0, 3, 5, 2, 4, 1).contiguous().view(rows, groups)


def pack_bf16(w):
    """The kernel's router packing (16-row x 64-K MFMA tiles) of a bf16 [N, K] matrix, same shape."""
    n, k = w.shape
    w16 = w.view(torch.int16).reshape(n // 16, 16, k // 64, 2, 4, 8)
    return w16.permute(0, 2, 3, 4, 1, 5).contiguous().view(torch.bfloat16).view(n, k)


def to_kernel_layout(case, inputs):
    """Load-time weight preparation (production does it once): the FP8 projections in AITER's (16, 16)
    preshuffled layout (SGLang's gfx950 block-FP8 linear method), the MXFP4 experts and scales in AITER's
    A4W4 layout (shuffle_weight (16, 16), e8m0_shuffle), the router in the kernel's MFMA tiles."""
    out = dict(inputs)
    for name in ("w_qkv_a", "w_q_b", "w_o"):
        out[name] = shuffle_16x16(inputs[name])
    for name in ("w13", "w2"):
        out[name] = shuffle_16x16(inputs[name])
    for name in ("w13_scale", "w2_scale"):
        e, n, gcount = inputs[name].shape
        out[name] = e8m0_shuffle(inputs[name].reshape(e * n, gcount)).view(e, n, gcount)
    out["router_w"] = pack_bf16(inputs["router_w"])
    return out
