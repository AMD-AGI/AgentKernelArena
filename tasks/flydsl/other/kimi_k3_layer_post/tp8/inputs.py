# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Inputs of the Kimi-K3 layer-post launch on a TP8 group: the generator (logical inputs), each rank's
shard, each rank's expected outputs, and the kernel-facing (aiter A16W4) layout of the routed experts.

Logical inputs (what the reference reads), S = ``rows``:

  core      [S, 12288] bf16   the attention's core output of all 96 heads (rank r: columns 1536 r ..)
  w_o       [7168, 12288]     o_proj (rank r: its 1536 input columns)
  prefix_in [S, 7168]         the residual stream the seam reads (SGLang's written residual), replicated
  blocks    [S, 8, 7168]      the attention-residual bank; rows 0 .. 3 are live at this layer, replicated
  w_gate    [896, 7168]       the router; bias [896] float32 its e_score_correction_bias, replicated
  w_sgu     [12288, 7168]     the shared experts' gate (rows 0 .. 6143) | up (6144 ..); rank r: 768 of each
  w_sd      [7168, 6144]      the shared experts' down (rank r: its 768 input columns)
  w_up      [7168, 3584]      the latent up projection (rank r: its 896 output rows)
  w13, w13s, w2, w2s          int64 [1] keys: the routed experts' MXFP4 bytes are a hash of the key
                              (k3_numerics.py); rank r's slice is computed by shard_inputs
  guard     [8] int32         the launch's epoch / guard words: epoch 1, a 30 s wait deadline in 100 MHz
                              ticks (int64 in words 1-2), clean error words
  built-in: ares_nw, ares_qk, in_nw (bf16 [7168]), w_ld (bf16 [3584, 7168]), ln_w (bf16 [3584])

Routing margins. The top-16 of 896 must not flip between reference and kernel, and the score-correction
bias must decide some picks, so every token's routing is planted. Expert e owns hidden columns 8 e .. 8 e +
7; every live seam source (the 4 bank rows and prefix_in) carries level L * sign_e[j] on the owned columns of
the token's 20 planted experts and N(0, 0.35) noise elsewhere; the router row of e reads its columns as
sign_e[j] * 0.061 / in_nw[c] plus N(0, 0.001) on every column, so the router logit is about 1.0 .. 1.4 L.
The bias classes are 224 "boost" experts (+0.12), 134 "decoy" experts (-0.30) and the rest (U(-0.01,
0.01)). A token plants 12 plain winners from the neutral class (L 2.0 .. 3.1), 4 boosted winners from the
boost class (L 1.0 .. 1.15, sigmoid ~0.75: picked only through their bias) and 4 decoys from the decoy class
(L 1.6 .. 1.9, sigmoid ~0.86: above the boosted winners without the bias, below every winner with it). An
implementation that ignores the bias picks the decoys. The o_proj output adds N(0, ~0.11) to the prefix.
Measured margins are in README.md.

Kernel-facing layout (to_kernel_layout), as SGLang's Mxfp4MoEMethod leaves the experts for Kimi-K3 under
AITER's gate / up interleaved SiTUv2 path and as the kernel reads them: ROCm/aiter
652f2d4b768239aefd515c929e84817681ff598d aiter/ops/shuffle.py (MIT, Copyright (C) 2024-2026 Advanced
Micro Devices, Inc.): ``shuffle_weight_a16w4(w13, 16, gate_up=True)``, ``shuffle_scale_a16w4(w13s, E,
gate_up=True)``, ``shuffle_weight_a16w4(w2, 16, gate_up=False)``, ``shuffle_scale_a16w4(w2s, E,
gate_up=False)``; the permutations below are aiter's, transcribed. w2s's 12 scale bytes a row are padded
to 16 with 0x7F (E8M0 1.0) by shard_inputs, as aiter's shuffle pads them; the kernel never reads the pad.
"""
import torch

from k3_numerics import (
    CORE,
    EXPERTS,
    HIDDEN,
    LAT,
    MAX_BLOCKS,
    NBLOCKS,
    SI,
    TP,
    UP_N,
    rank_experts,
)

DEADLINE_S = 30.0
CLOCK_HZ = 100_000_000  # the guard's s_memrealtime clock
GUARD_WORDS = 8
NOISE = 0.35
ROUTER_GAIN = 0.061
ROUTER_NOISE = 0.001
# planted routing: (start, step) of each kind's level ladder (the level is about the router logit)
N_PLAIN, N_BOOSTED, N_DECOYS = 12, 4, 4
PLAIN_LEVELS, BOOSTED_LEVELS, DECOY_LEVELS = (2.0, 0.1), (1.0, 0.05), (1.6, 0.1)
N_BOOST_CLASS, N_DECOY_CLASS = 224, 134       # experts with a +0.12 / -0.30 score-correction bias
BOOST_BIAS, DECOY_BIAS, BIAS_JITTER = 0.12, -0.30, 0.01
CORE_STD, W_O_STD = 1.0, 0.001
W_SGU_STD, W_SD_STD, W_UP_STD = 0.02, 0.008, 0.02
COLS = HIDDEN // EXPERTS   # 8 hidden columns an expert owns
KEY_NAMES = ("w13", "w13s", "w2", "w2s")


def _normal(shape, std, generator, device):
    t = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    return t.mul_(std).to(torch.bfloat16)


def _guard_words(device):
    ticks = int(DEADLINE_S * CLOCK_HZ)
    lo = ticks & 0xFFFFFFFF
    words = [1, lo - (1 << 32) if lo >= 1 << 31 else lo, ticks >> 32] + [0] * (GUARD_WORDS - 3)
    return torch.tensor(words, dtype=torch.int32, device=device)


def make_inputs(case, inputs, generator, device):
    s = case["axes"]["rows"]
    assert case["axes"]["world_size"] == TP
    # ---- the planted routing (see the module docstring): the bias classes, then per token its 12 plain and
    # 4 boosted winners and 4 decoys, their levels, and every expert's column signs
    cls = torch.rand(EXPERTS, generator=generator, device=device).argsort()
    boost, decoy, neutral = cls[:N_BOOST_CLASS], cls[N_BOOST_CLASS:N_BOOST_CLASS + N_DECOY_CLASS], \
        cls[N_BOOST_CLASS + N_DECOY_CLASS:]
    bias = (torch.rand(EXPERTS, generator=generator, device=device) * 2 - 1) * BIAS_JITTER
    bias[boost] += BOOST_BIAS
    bias[decoy] += DECOY_BIAS

    def draw(pool, n):
        return pool[torch.rand(s, pool.numel(), generator=generator, device=device).argsort(-1)[:, :n]]

    def ladder(start, step, n):
        return start + step * torch.rand(s, n, generator=generator, device=device).argsort(-1)

    planted_ids = torch.cat([draw(neutral, N_PLAIN), draw(boost, N_BOOSTED), draw(decoy, N_DECOYS)], 1)
    levels = torch.cat([ladder(*PLAIN_LEVELS, N_PLAIN), ladder(*BOOSTED_LEVELS, N_BOOSTED),
                        ladder(*DECOY_LEVELS, N_DECOYS)], 1)
    signs = torch.where(torch.rand(EXPERTS, COLS, generator=generator, device=device) < 0.5, -1.0, 1.0)
    planted = torch.zeros(s, EXPERTS, COLS, device=device)
    planted.scatter_(1, planted_ids.unsqueeze(-1).expand(-1, -1, COLS), levels.unsqueeze(-1) * signs[planted_ids])
    owned = torch.zeros(s, EXPERTS, dtype=torch.bool, device=device)
    owned.scatter_(1, planted_ids, True)
    owned = owned.repeat_interleave(COLS, 1)                     # [S, 7168]
    planted = planted.view(s, HIDDEN)

    def source():
        noise = torch.randn(s, HIDDEN, generator=generator, device=device) * NOISE
        return torch.where(owned, planted, noise).to(torch.bfloat16)

    prefix_in = source()
    blocks = torch.empty(s, MAX_BLOCKS, HIDDEN, dtype=torch.bfloat16, device=device)
    for b in range(MAX_BLOCKS):
        # rows past the live ones are allocated (the bank holds every block of the model) and never read
        blocks[:, b] = source() if b < NBLOCKS else _normal((s, HIDDEN), NOISE, generator, device)
    w_gate = torch.randn(EXPERTS, HIDDEN, generator=generator, device=device) * ROUTER_NOISE
    gain = (signs.view(EXPERTS, 1, COLS) * ROUTER_GAIN / inputs["in_nw"].float().view(EXPERTS, 1, COLS))
    w_gate.view(EXPERTS, EXPERTS, COLS).diagonal(0, 0, 1).add_(gain.squeeze(1).T)
    out = {
        "core": _normal((s, TP * CORE), CORE_STD, generator, device),
        "w_o": _normal((HIDDEN, TP * CORE), W_O_STD, generator, device),
        "prefix_in": prefix_in,
        "blocks": blocks,
        "w_gate": w_gate.to(torch.bfloat16),
        "bias": bias,
        "w_sgu": _normal((2 * TP * SI, HIDDEN), W_SGU_STD, generator, device),
        "w_sd": _normal((HIDDEN, TP * SI), W_SD_STD, generator, device),
        "w_up": _normal((HIDDEN, LAT), W_UP_STD, generator, device),
        "guard": _guard_words(device),
    }
    for name in KEY_NAMES:
        out[name] = torch.randint(0, 1 << 31, (1,), generator=generator, device=device, dtype=torch.int64)
    return out


def shard_inputs(inputs, *, rank, world_size):
    """Logical inputs -> rank ``rank``'s kernel inputs (before the layout transform)."""
    assert world_size == TP, world_size
    r = rank
    device = inputs["core"].device
    gate, up = inputs["w_sgu"][:TP * SI], inputs["w_sgu"][TP * SI:]
    out = dict(inputs)
    out.update(
        core=inputs["core"][:, r * CORE:(r + 1) * CORE],
        w_o=inputs["w_o"][:, r * CORE:(r + 1) * CORE],
        w_sgu=torch.cat([gate[r * SI:(r + 1) * SI], up[r * SI:(r + 1) * SI]], 0),
        w_sd=inputs["w_sd"][:, r * SI:(r + 1) * SI],
        w_up=inputs["w_up"][r * UP_N:(r + 1) * UP_N],
    )
    keys = {name: int(inputs[name].reshape(-1)[0]) for name in KEY_NAMES}
    experts = rank_experts(keys, r, device)
    pad = torch.full((EXPERTS, LAT, 4), 0x7F, dtype=torch.uint8, device=device)
    experts["w2s"] = torch.cat([experts["w2s"], pad], -1)        # 12 -> 16 scale bytes a row
    out.update(experts)
    return out


def rank_expected(outputs, *, rank, world_size):
    """Both outputs are replicated: every rank writes the full tensors."""
    assert world_size == TP, world_size
    return {"out": outputs["out"], "prefix": outputs["prefix"]}


# ---------------------------------------------------------------- aiter's A16W4 layouts (transcribed)
def shuffle_weight_a16w4(x, gate_up, n_lane=16):
    """aiter ``shuffle_weight(x, layout=(n_lane, 16), is_guinterleave=True, gate_up=gate_up)`` of uint8
    [E, N, K_pk]."""
    experts, n, k_pk = x.shape
    if gate_up:
        n //= 2
    k_pack = 16
    k_lane = 64 // n_lane
    n0, k0 = n // n_lane, k_pk // (k_lane * k_pack)
    if gate_up:
        y = x.view(experts, 2, n0, n_lane, k0, k_lane, k_pack).permute(0, 2, 1, 4, 5, 3, 6)
    else:
        y = x.view(experts, n0, n_lane, k0, k_lane, k_pack).permute(0, 1, 3, 4, 2, 5)
    return y.contiguous().view(x.shape)


def shuffle_scale_a16w4(src, experts, gate_up):
    """aiter ``shuffle_scale(src, experts_cnt=experts, is_guinterleave=True, gate_up=gate_up)`` of a uint8
    [E * N, K / 32] matrix whose K / 32 is a multiple of 8 (no padding needed)."""
    rows, k_ = src.shape
    assert k_ % 8 == 0, k_
    n_ = rows // experts
    k_pack, n_pack, n_lane, k_lane = 2, 2, 16, 4
    k1, n1 = k_ // k_pack // k_lane, n_ // n_lane // n_pack
    if gate_up:
        y = src.view(experts, n_pack, n1, n_lane, k1, k_pack, k_lane).permute(0, 2, 4, 6, 3, 5, 1)
    else:
        y = src.view(experts, n1, n_pack, n_lane, k1, k_pack, k_lane).permute(0, 1, 4, 6, 3, 5, 2)
    return y.contiguous().view(src.shape)


def to_kernel_layout(case, inputs):
    inputs = dict(inputs)
    e = EXPERTS
    inputs["w13"] = shuffle_weight_a16w4(inputs["w13"], gate_up=True)
    inputs["w2"] = shuffle_weight_a16w4(inputs["w2"], gate_up=False)
    for name, gate_up in (("w13s", True), ("w2s", False)):
        t = inputs[name]
        inputs[name] = shuffle_scale_a16w4(t.reshape(-1, t.shape[-1]), e, gate_up).view(t.shape)
    return inputs
