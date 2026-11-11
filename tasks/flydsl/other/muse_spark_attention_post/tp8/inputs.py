# SPDX-License-Identifier: Apache-2.0
"""Logical inputs, per-rank shards and per-rank expected outputs of the Muse Spark K2 launch (the
attention output through the experts' input) at TP8 / EP8, one DFlash2 verify step of ``rows`` rows
(``rows`` / 8 requests of 8 draft tokens).

Logical inputs (``make_inputs``; identical on every rank, drawn from the case seed):

  o [rows, 8192] bf16                 the attention output of all 128 query heads (head h at columns 64 h ..)
  g [rows, 8192] bf16                 K1's o_gate pre-activation of all 128 heads
  res [rows, 8192] fp32               the residual stream entering the attention boundary (updated in place)
  w_o [8192, 8192] bf16               o_proj (out x in)
  alpha, beta [8192] fp32             the attention residual gate's stream and branch coefficients
  pre_w [8192] bf16                   pre_feedforward_layernorm's effective weight (weight + gain centre)
  router_hi, router_lo [256, 8192]    the fp32 sigmoid router [8192, 256] as its transposed bf16 hi + lo
                       bf16           split (``weights._router_hi_lo``: the weight is hi + lo exactly)
  expert_bias [256] fp32              the router's selection bias
  w_pool [4096, 8192] bf16            pre_expert_proj (the pooler, 8192 -> 4096)
  xn_w [4096] bf16                    pre_expert_norm's effective weight
  guard [8] int32                     the runner's epoch / guard-word buffer: epoch 1, a 30 s wait deadline in
                                      100 MHz ticks, clean error words (replicated)

Routing margins. The router's top-8 is a discrete decision that must not flip between the float64
reference and the kernel (the kernel's routing key keeps 24 bits of the biased sigmoid score, and its
normed row x differs from the reference's by a few 1e-4 relative). The generator plants each token's 8
experts: expert e owns hidden columns 32 e .. 32 e + 31; the residual stream of token t carries level L_j
(85, 75, .., 15) along a random sign pattern of the block of its j-th expert, over unit-normal noise; the
router's column e reads that block divided by (alpha * pre_w) (so the attention gate and the pre-FFN norm
weight do not blur the levels), plus N(0, 5e-4) everywhere. The 8 picks then stand apart from each other
and from the other 248 experts by 0.0127 or more in the biased score (measured; README.md), far beyond
the kernel's error, and every token picks a different random set of experts.

Sharding (``shard_inputs``, rank r of 8; the runner's per-rank tensors): o and g columns 1024 r .. (the
rank's 16 query heads), w_o input columns 1024 r .. (the RowParallel K shard); everything else
replicated, the pooler whole (the kernel reads its rank's rows of it at S <= 16).

The kernel's mailbox scratch (whose contents depend on the implementation) is not an input: the
baseline keeps it in the framework's symmetric buffer, which is zeroed before every call.
"""
import math

import torch

HIDDEN = 8192
HEAD_DIM = 64
Q_HEADS = 128
EXPERTS = 256
TOP_K = 8
MOE_HIDDEN = 4096
WORLD = 8
LOCAL_Q = HIDDEN // WORLD  # 1024: 16 query heads a rank
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


def _normal(shape, std, generator, device, dtype):
    out = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    return out.mul_(std).to(dtype)


def _uniform(shape, low, high, generator, device):
    return torch.rand(shape, generator=generator, device=device, dtype=torch.float32).mul_(high - low).add_(low)


def _norm_weight(n, generator, device):
    """An effective RMSNorm weight (weight + gain centre): 1 + N(0, 0.1), bfloat16."""
    return _normal((n,), 0.1, generator, device, torch.float32).add_(1.0).to(torch.bfloat16)


def guard_words(device):
    """The runner's epoch buffer: the epoch (word 0), the wait deadline in 100 MHz ticks (words 1-2, an
    int64 as two int32), clean error words (3-5) and padding."""
    ticks = int(DEADLINE_S * CLOCK_HZ)
    lo = ticks & 0xFFFFFFFF
    words = [1, lo - (1 << 32) if lo >= 1 << 31 else lo, ticks >> 32] + [0] * (GUARD_WORDS - 3)
    return torch.tensor(words, dtype=torch.int32, device=device)


def make_inputs(case, inputs, generator, device):
    """The logical inputs of one case (every rank draws the same from the case seed)."""
    rows = int(case["axes"]["rows"])
    bf16 = torch.bfloat16

    o = _normal((rows, HIDDEN), 1.0, generator, device, bf16)
    g = _normal((rows, HIDDEN), 1.0, generator, device, bf16)
    w_o = _normal((HIDDEN, HIDDEN), WEIGHT_STD, generator, device, bf16)
    alpha = _uniform((HIDDEN,), 0.5, 1.0, generator, device)
    beta = _uniform((HIDDEN,), 0.0, 0.5, generator, device)
    pre_w = _norm_weight(HIDDEN, generator, device)

    # the residual stream and the router with planted picks
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

    w_pool = _normal((MOE_HIDDEN, HIDDEN), WEIGHT_STD, generator, device, bf16)
    xn_w = _norm_weight(MOE_HIDDEN, generator, device)

    return {
        "o": o, "g": g, "res": res, "w_o": w_o, "alpha": alpha, "beta": beta, "pre_w": pre_w,
        "router_hi": router_hi, "router_lo": router_lo, "expert_bias": expert_bias, "w_pool": w_pool,
        "xn_w": xn_w, "guard": guard_words(device),
    }


def shard_inputs(logical, *, rank, world_size):
    """Rank ``rank``'s kernel inputs from the logical inputs (outside timing)."""
    assert world_size == WORLD, world_size
    qc = slice(rank * LOCAL_Q, (rank + 1) * LOCAL_Q)
    out = {
        "o": logical["o"][:, qc].contiguous(),
        "g": logical["g"][:, qc].contiguous(),
        "w_o": logical["w_o"][:, qc].contiguous(),
    }
    for name in ("res", "alpha", "beta", "pre_w", "router_hi", "router_lo", "expert_bias", "w_pool", "xn_w",
                 "guard"):
        out[name] = logical[name]
    return out


def rank_expected(logical_outputs, *, rank, world_size):
    """What rank ``rank`` must write: every output is replicated (each rank computes the whole rows)."""
    assert world_size == WORLD, world_size
    return {name: logical_outputs[name] for name in ("ids", "scores", "xin", "res")}
