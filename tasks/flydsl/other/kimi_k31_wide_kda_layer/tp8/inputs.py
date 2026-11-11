"""Logical inputs, rank shards and rank outputs of one Kimi-K3.1 wide-row KDA layer launch at TP8.

A DSpark verify step of ``requests`` R requests x 8 tokens is padded to its row bucket of ``rows`` m rows
(16, 32, 64, 128, 256 or 384: the smallest of 8 x {2, 4, 8, 16, 32, 48} that holds 8 R), as
``wide.Wide.backbone`` pads it; rows 8 R .. m are padding (no KDA state, zero activations).

Logical tensors (what reference.py reads; W = 8 ranks, a rank's slice in brackets):

  lat_p   [W m, 3584] bf16   every rank's routed-experts latent partial of the previous layer [rows r m ..]
  sh_p    [W m, 7168] bf16   every rank's shared-expert partial of the previous layer [rows r m ..]
  w_up    [7168, 3584] bf16  the previous layer's routed_expert_up_proj [rows 896 r ..]
  prefix  [m, 7168] bf16     the residual stream before the previous layer's MoE output (replicated; a rank
                             reads its 896 columns)
  bank    [m, 8, 7168] bf16  the AttnRes bank (rows 0 .. 3 live) [columns 896 r ..]
  cw_a, cw_m [7168] f32      self_attention_res_norm.weight x self_attention_res_proj.weight (mlp_res_*)
  w_in, w_ln [7168] bf16     input_layernorm / post_attention_layernorm weights
  w_ip    [W 6288, 7168] bf16 every rank's fused in_proj rows: q | k | v | g (1536 each, the rank's 12 heads),
                             f_a (128, the same rows on every rank), beta (12), 4 zero pad rows [block r]
  w_fb    [W 1536, 128] bf16 f_b_proj rows of every head [block r]
  conv_w  [W 4608, 4] f32    qkv_conv1d weights, a rank's channels q | k | v [block r]
  a_log   [W 12] f32, dt_bias [W 1536] f32 [block r];  on_w [128] bf16 (o_norm weight)
  conv_state [64, 3, W 4608] bf16  SGLang's conv cache [slot, CONV_W - 1, channel] [channels 4608 r ..]
  rstate  [64, W 12, 128, 128] f32 the committed recurrent state [heads 12 r ..] (ReplaySSM: read only)
  st_idx  [m / 8, 8] int32   a request's state slot (each of its 8 columns; -1: a padding request)
  num_acc [m / 8] int32      1 (the ring's accepted-token index + 1 is 1 for every request)
  win_row [m / 8] int32      a request's row of the verify conv-window buffer (-1: none)
  wo      [7168, W 1536] bf16 o_proj [columns 1536 r ..: the rank's K slice]
  wf      [W 2096, 7168] bf16 every rank's MoE front rows: router (experts 112 r ..), latent down (rows
                             448 r ..), shared gate (768) | up (768) of the two shared experts [block r]
  bias    [896 + 3584] f32   e_score_correction_bias, then the previous layer's latent (routed_expert_norm)
                             weight (the launch's ``up_norm`` layout)
  w_sd    [7168, W 768] bf16 shared experts' down_proj [columns 768 r ..]
  epoch   [8] int32          the launch's epoch / guard block: epoch 1, a 120 s wait deadline, clean words
Mutable (updated in place, compared):
  prefix_out [m, 7168] bf16  zeros; a rank writes its 896 columns of the stream after the attention add
  ring_k, ring_v [64, W 12, 16, 128] bf16; ring_g [64, W 12, 16, 128] f32; ring_b [64, W 12, 16] f32
                             the ReplaySSM ring: a live request's slot gets tokens 0 .. 7 [heads 12 r ..]
  conv_win [64, 8, 3, W 4608] bf16  the verify conv windows SGLang's accept reads, at win_row [channels]
  ip_buf [m, 6288], acc_u [m, 896] bf16  the launch's merge buffers: zero before and after

Routing margins: the launch ranks 896 router scores and keeps 16 in order (topk_ids is compared exactly),
so the scores are planted. 128 experts U of each case own 8 hidden columns each (expert e: 8 e .. 8 e + 7,
random signs); on those 1024 columns every live row's prefix and bank snapshots carry only the plant -- its
16 experts (drawn from U) at levels giving router logits up to 3.0, 2.85, .., 0.75 -- and the previous
layer's shared partials, latent up rows and o_proj rows are scaled by 1e-3 there, so the MLP seam's output
keeps the plant. Router rows of U read their expert's columns (sign / post_attention_layernorm weight), plus
N(0, 2e-4) everywhere. The score-correction bias is a permutation of a 2^-22 grid (|bias| < 1.1e-4, offset by
0.37 steps so that 0.5 + bias stays distinct in float32): padding rows (all-zero activations, sigmoid 0.5)
rank by it alone. Measured margins are in README.md.
"""

import math

import torch

WORLD = 8
L = 8
H, SL, LAT, LSL = 7168, 896, 3584, 448
E, E_R, TOPK = 896, 112, 16
NH, HD, PROJ = 12, 128, 1536
QKV = 3 * PROJ  # 4608 conv channels a rank
G2_0, FA_0, BETA_0, NPROJ = QKV, QKV + PROJ, QKV + PROJ + HD, 6288
NBETA = 12
NF, SI = 2096, 768
NBK, NVB = 8, 4
SLOTS, WIN_ROWS, RING = 64, 64, 16
GUARD_WORDS = 8
CLOCK_HZ = 100_000_000  # the guard's s_memrealtime clock
DEADLINE_S = 120.0

U_EXPERTS = 128  # experts whose columns carry the router plant
LOGIT_TOP, LOGIT_STEP = 3.0, 0.15  # planted router logits: 3.0, 2.85, .., 0.75 (at the largest row scale)
ROUTER_GAIN = 0.1  # router weight on a planted expert's columns (x the inverse norm weight)
# logit per unit plant level (rstd x ROUTER_GAIN x 8 columns): measured 2.2 .. 4.45 over rows on CPU (the
# seam's mixture energy varies by row), so levels are set for the largest and the sigmoid never saturates
PLANT_SCALE = 4.4
ROUTER_NOISE = 2e-4
QUIET = 1e-3  # the plant columns' scale of the previous layer's shared / up and of o_proj
SRC_STD = 0.2  # prefix / bank noise off the plant columns
BIAS_STEP = 2.0 ** -22
BF = torch.bfloat16


def _normal(shape, std, gen, device, dtype=BF):
    return (torch.randn(shape, generator=gen, device=device) * std).to(dtype)


def _norm_w(n, gen, device, dtype=BF):
    return (1.0 + 0.1 * torch.randn(n, generator=gen, device=device)).to(dtype)


def _bf16_rows(rows, cols, std, gen, device, chunk=2048):
    """A [rows, cols] bf16 normal draw, made in row chunks (bounded float32 transients)."""
    out = torch.empty(rows, cols, dtype=BF, device=device)
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        out[r0:r1] = (torch.randn(r1 - r0, cols, generator=gen, device=device) * std).to(BF)
    return out


def _guard_words(device):
    """The epoch buffer: the epoch (word 0), the wait deadline in 100 MHz ticks (words 1-2, an int64 as two
    int32), clean error words (3-5) and padding."""
    ticks = int(DEADLINE_S * CLOCK_HZ)
    lo = ticks & 0xFFFFFFFF
    words = [1, lo - (1 << 32) if lo >= 1 << 31 else lo, ticks >> 32] + [0] * (GUARD_WORDS - 3)
    return torch.tensor(words, dtype=torch.int32, device=device)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    reqs, m, nreq = axes["requests"], axes["rows"], axes["breq"]
    assert axes["world_size"] == WORLD and m == nreq * L and reqs * L <= m, axes
    live = reqs * L
    g = generator

    # ---- the router plant (see the module docstring)
    u = torch.randperm(E, generator=g, device=device)[:U_EXPERTS]
    signs = torch.randint(0, 2, (E, 8), generator=g, device=device).to(torch.float32) * 2 - 1
    plant_cols = (u.view(-1, 1) * 8 + torch.arange(8, device=device)).reshape(-1)
    on_plant = torch.zeros(H, dtype=torch.bool, device=device)
    on_plant[plant_cols] = True
    plant = torch.zeros(m, H, device=device)
    levels = torch.tensor([(LOGIT_TOP - LOGIT_STEP * k) / PLANT_SCALE for k in range(TOPK)], device=device)
    for t in range(live):
        picks = u[torch.randperm(U_EXPERTS, generator=g, device=device)[:TOPK]]
        cols = (picks.view(-1, 1) * 8 + torch.arange(8, device=device)).reshape(-1)
        plant[t, cols] = (levels.view(-1, 1) * signs[picks]).reshape(-1)
    plant = plant.to(BF).float()

    def stream_rows(std):
        """[m, H] bf16: the plant on U's columns, noise elsewhere; padding rows zero."""
        x = torch.randn(m, H, generator=g, device=device) * std
        x = torch.where(on_plant, plant, x)
        x[live:] = 0
        return x.to(BF)

    prefix = stream_rows(SRC_STD)
    bank = torch.empty(m, NBK, H, dtype=BF, device=device)
    for b in range(NBK):
        bank[:, b] = stream_rows(SRC_STD) if b < NVB else _normal((m, H), SRC_STD, g, device)
    bank[live:] = 0
    quiet = torch.where(on_plant, torch.tensor(QUIET, device=device), torch.tensor(1.0, device=device))

    # ---- T1: the previous layer's MoE tail
    lat_p = _normal((WORLD * m, LAT), 0.3, g, device).view(WORLD, m, LAT)
    lat_p[:, live:] = 0
    sh_p = (torch.randn(WORLD, m, H, generator=g, device=device) * 0.07 * quiet).to(BF)
    sh_p[:, live:] = 0
    w_up = (torch.randn(H, LAT, generator=g, device=device) * (0.0033 * quiet.view(-1, 1))).to(BF)
    score_bias = ((torch.randperm(E, generator=g, device=device).to(torch.float64) - E // 2 + 0.37)
                  * BIAS_STEP).to(torch.float32)
    lat_norm_w = 1.0 + 0.1 * torch.randn(LAT, generator=g, device=device)  # float32 (not bf16-rounded)
    bias = torch.cat([score_bias, lat_norm_w]).contiguous()

    # ---- the two AttnRes seams
    def seam_w():
        w_s = _norm_w(H, g, device)
        w_p = _normal(H, 4.0 / math.sqrt(H), g, device)
        return (w_s.float() * w_p.float()).contiguous()

    cw_a, cw_m = seam_w(), seam_w()
    w_in, w_ln = _norm_w(H, g, device), _norm_w(H, g, device)

    # ---- in_proj and KDA
    w_ip = torch.zeros(WORLD, NPROJ, H, dtype=BF, device=device)
    f_a = _bf16_rows(HD, H, 1.0 / math.sqrt(H), g, device)
    for r in range(WORLD):
        w_ip[r, :FA_0] = _bf16_rows(FA_0, H, 1.0 / math.sqrt(H), g, device)
        w_ip[r, FA_0:BETA_0] = f_a
        w_ip[r, BETA_0:BETA_0 + NBETA] = _bf16_rows(NBETA, H, 1.0 / math.sqrt(H), g, device)
    w_fb = _normal((WORLD * PROJ, HD), 0.02, g, device)
    conv_w = (torch.randn(WORLD * QKV, 4, generator=g, device=device) * 0.3).contiguous()
    a_log = (torch.randn(WORLD * NH, generator=g, device=device) * 0.1).contiguous()
    dt_bias = (torch.randn(WORLD * PROJ, generator=g, device=device) * 0.1).contiguous()
    on_w = _norm_w(HD, g, device)
    conv_state = _normal((SLOTS, 3, WORLD * QKV), 0.5, g, device)
    rstate = torch.randn(SLOTS, WORLD * NH, HD, HD, generator=g, device=device) * 0.05
    slots = torch.randperm(SLOTS, generator=g, device=device)[:reqs].to(torch.int32)
    rows_w = torch.randperm(WIN_ROWS, generator=g, device=device)[:reqs].to(torch.int32)
    st_idx = torch.full((nreq, L), -1, dtype=torch.int32, device=device)
    st_idx[:reqs] = slots.view(-1, 1)
    num_acc = torch.ones(nreq, dtype=torch.int32, device=device)
    win_row = torch.full((nreq,), -1, dtype=torch.int32, device=device)
    win_row[:reqs] = rows_w
    ring_k = _normal((SLOTS, WORLD * NH, RING, HD), 1.0, g, device)
    ring_v = _normal((SLOTS, WORLD * NH, RING, HD), 1.0, g, device)
    ring_g = -5.0 * torch.rand(SLOTS, WORLD * NH, RING, HD, generator=g, device=device)
    ring_b = torch.rand(SLOTS, WORLD * NH, RING, generator=g, device=device)
    conv_win = _normal((WIN_ROWS, L, 3, WORLD * QKV), 0.5, g, device)

    # ---- o_proj, the MoE front and the shared down
    wo = (torch.randn(H, WORLD * PROJ, generator=g, device=device) * (0.0036 * quiet.view(-1, 1))).to(BF)
    wf = torch.empty(WORLD, NF, H, dtype=BF, device=device)
    for r in range(WORLD):
        router = torch.randn(E_R, H, generator=g, device=device) * ROUTER_NOISE
        for i in range(E_R):
            e = r * E_R + i
            if bool((u == e).any()):
                router[i, 8 * e:8 * e + 8] += ROUTER_GAIN * signs[e] / w_ln[8 * e:8 * e + 8].float()
        wf[r, :E_R] = router.to(BF)
        wf[r, E_R:] = _bf16_rows(NF - E_R, H, 1.0 / math.sqrt(H), g, device)
    w_sd = _bf16_rows(H, WORLD * SI, 1.0 / math.sqrt(SI), g, device)

    return {
        "lat_p": lat_p.reshape(WORLD * m, LAT), "sh_p": sh_p.reshape(WORLD * m, H), "w_up": w_up,
        "prefix": prefix, "bank": bank, "cw_a": cw_a, "w_in": w_in, "w_ip": w_ip.view(WORLD * NPROJ, H),
        "w_fb": w_fb, "conv_w": conv_w, "a_log": a_log, "dt_bias": dt_bias, "on_w": on_w,
        "conv_state": conv_state, "rstate": rstate, "st_idx": st_idx, "num_acc": num_acc, "win_row": win_row,
        "wo": wo, "cw_m": cw_m, "w_ln": w_ln, "wf": wf.view(WORLD * NF, H), "bias": bias, "w_sd": w_sd,
        "epoch": _guard_words(device),
        "prefix_out": torch.zeros(m, H, dtype=BF, device=device),
        "ip_buf": torch.zeros(m, NPROJ, dtype=BF, device=device),
        "acc_u": torch.zeros(m, SL, dtype=BF, device=device),
        "ring_k": ring_k, "ring_v": ring_v, "ring_g": ring_g, "ring_b": ring_b, "conv_win": conv_win,
    }


REPLICATED = ("prefix", "cw_a", "w_in", "on_w", "st_idx", "num_acc", "win_row", "cw_m", "w_ln", "bias", "epoch",
              "prefix_out", "ip_buf", "acc_u")


def shard_inputs(logical, *, rank, world_size):
    """Rank ``rank``'s kernel inputs from the logical inputs (outside timing)."""
    assert world_size == WORLD, world_size
    r = rank
    m = logical["prefix"].shape[0]
    cols = slice(r * SL, (r + 1) * SL)
    heads = slice(r * NH, (r + 1) * NH)
    chans = slice(r * QKV, (r + 1) * QKV)
    out = {
        "lat_p": logical["lat_p"][r * m:(r + 1) * m].contiguous(),
        "sh_p": logical["sh_p"][r * m:(r + 1) * m].contiguous(),
        "w_up": logical["w_up"][cols].contiguous(),
        "bank": logical["bank"][:, :, cols].contiguous(),
        "w_ip": logical["w_ip"][r * NPROJ:(r + 1) * NPROJ].contiguous(),
        "w_fb": logical["w_fb"][r * PROJ:(r + 1) * PROJ].contiguous(),
        "conv_w": logical["conv_w"][chans].contiguous(),
        "a_log": logical["a_log"][heads].contiguous(),
        "dt_bias": logical["dt_bias"][r * PROJ:(r + 1) * PROJ].contiguous(),
        "conv_state": logical["conv_state"][:, :, chans].contiguous(),
        "rstate": logical["rstate"][:, heads].contiguous(),
        "wo": logical["wo"][:, r * PROJ:(r + 1) * PROJ].contiguous(),
        "wf": logical["wf"][r * NF:(r + 1) * NF].contiguous(),
        "w_sd": logical["w_sd"][:, r * SI:(r + 1) * SI].contiguous(),
        "ring_k": logical["ring_k"][:, heads].contiguous(),
        "ring_v": logical["ring_v"][:, heads].contiguous(),
        "ring_g": logical["ring_g"][:, heads].contiguous(),
        "ring_b": logical["ring_b"][:, heads].contiguous(),
        "conv_win": logical["conv_win"][..., chans].contiguous(),
    }
    for name in REPLICATED:
        out[name] = logical[name]
    return out


def rank_expected(outputs, *, rank, world_size):
    """What rank ``rank`` must write: its slices of the sharded tensors, the replicated ones whole."""
    assert world_size == WORLD, world_size
    r = rank
    m = outputs["mid"].shape[0]
    cols = slice(r * SL, (r + 1) * SL)
    heads = slice(r * NH, (r + 1) * NH)
    stream = torch.zeros_like(outputs["prefix_out"])
    stream[:, cols] = outputs["prefix_out"][:, cols]
    return {
        "mid": outputs["mid"][:, cols],
        "core": outputs["core"][:, r * PROJ:(r + 1) * PROJ],
        "lat_out": outputs["lat_out"],
        "topk_ids": outputs["topk_ids"],
        "topk_w": outputs["topk_w"],
        "logits_out": outputs["logits_out"],
        "sh_out": outputs["sh_out"][r * m:(r + 1) * m],
        "prefix_out": stream,
        "ip_buf": outputs["ip_buf"],
        "acc_u": outputs["acc_u"],
        "ring_k": outputs["ring_k"][:, heads],
        "ring_v": outputs["ring_v"][:, heads],
        "ring_g": outputs["ring_g"][:, heads],
        "ring_b": outputs["ring_b"][:, heads],
        "conv_win": outputs["conv_win"][..., r * QKV:(r + 1) * QKV],
    }
