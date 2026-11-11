# SPDX-License-Identifier: Apache-2.0
"""Logical inputs, per-rank shards and per-rank expected outputs of the Muse Spark K3 launch (the MXFP4
experts through the next layer's attention input) at TP8 / EP8 with TP experts, one DFlash2 verify step of
``rows`` rows (``rows`` / 8 requests of 8 draft tokens).

Logical inputs (``make_inputs``; identical on every rank, drawn from the case seed):

  xin [rows, 4096] bf16          K2's expert input (pooled, pre-expert normed: unit RMS)
  ids [rows, 8] int32            K2's routing: 8 distinct experts a token of 256, uniformly at random
  scores [rows, 8] fp32          K2's routing weights: sigmoid-like picks in [0.5, 0.75], normalized to sum 1
  w13 [256, 8192, 2048] u8       Quark MXFP4 gate | up rows of every expert (gate rows 0..4095, up rows
  s13 [256, 8192, 128] u8        4096..8191): E2M1 codes two a byte (low nibble = even k), E8M0 scales a
  w2 [256, 4096, 2048] u8        32-block (2^(s - 127)); w2 / s2 the down rows (out x in). Codes are uniform
  s2 [256, 4096, 128] u8         bytes; scales uniform in 118..121 (2^-9 .. 2^-6)
  pen_w [4096] bf16              the experts' post norm weight (the finalize's per-slot norm)
  res [rows, 8192] fp32          the residual stream entering the FFN boundary (updated in place)
  pfw [8192] bf16                post_feedforward_layernorm's effective weight (the unpooled branch)
  alpha, beta [8192] fp32        the FFN residual gate's stream and branch coefficients
  npw [8192] bf16                the next layer's input_layernorm effective weight
  w_unpool [8192, 4096] bf16     post_expert_proj (the unpooler, 4096 -> 8192)
  guard [8] int32                the runner's epoch / guard-word buffer: epoch 1, a 30 s wait deadline in
                                 100 MHz ticks, clean error words (replicated)

The routing is an input here (K2 makes it), so no discrete decision is taken inside the launch; the
picks spread over all 256 experts (about 57 / 102 / 163 / 222 distinct experts at 8 / 16 / 32 / 64 rows).
The E8M0 scale range keeps the gate / up dot products exactly representable in float32 (the products of
E2M1 codes are multiples of 2^-2 of their scales; measured in README.md), so the MXFP4 re-quantization of
the mids does not depend on the accumulation order.

Sharding (``shard_inputs``, rank r of 8; ``weights.tp_expert_copies`` / ``test_muse_spark_k3_tp8.py``):
w13 / s13 the gate rows 512 r .. and up rows 4096 + 512 r .. of every expert (1024 rows), w2 / s2 the
down rows 512 r .. (512 rows); everything else replicated (the unpooler whole: the kernel reads its
rank's rows of it).

The kernel's mailbox scratch (whose contents depend on the implementation) is not an input: the
baseline keeps it in the framework's symmetric buffer, which is zeroed before every call.
"""
import torch

HIDDEN = 8192
EXPERTS = 256
TOP_K = 8
MOE_HIDDEN = 4096
INTER = 4096
WORLD = 8
GUARD_WORDS = 8
DEADLINE_S = 30.0
CLOCK_HZ = 100_000_000  # the guard's s_memrealtime clock
SCALE_LOW, SCALE_HIGH = 118, 121  # E8M0 exponents of the experts' 32-blocks (inclusive)
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


def _mxfp4(experts, rows, k, generator, device):
    """Quark MXFP4 weights: packed E2M1 codes [experts, rows, k / 2] and E8M0 scales [experts, rows, k / 32]."""
    codes = torch.randint(0, 256, (experts, rows, k // 2), generator=generator, device=device, dtype=torch.uint8)
    scales = torch.randint(SCALE_LOW, SCALE_HIGH + 1, (experts, rows, k // 32), generator=generator, device=device,
                           dtype=torch.uint8)
    return codes, scales


def make_inputs(case, inputs, generator, device):
    """The logical inputs of one case (every rank draws the same from the case seed)."""
    rows = int(case["axes"]["rows"])
    bf16 = torch.bfloat16

    xin = _normal((rows, MOE_HIDDEN), 1.0, generator, device, bf16)
    ids = torch.stack([torch.randperm(EXPERTS, generator=generator, device=device)[:TOP_K]
                       for _ in range(rows)]).to(torch.int32)
    picked = _uniform((rows, TOP_K), 0.5, 0.75, generator, device)
    scores = picked / picked.sum(-1, keepdim=True)
    w13, s13 = _mxfp4(EXPERTS, 2 * INTER, MOE_HIDDEN, generator, device)
    w2, s2 = _mxfp4(EXPERTS, MOE_HIDDEN, INTER, generator, device)
    pen_w = _norm_weight(MOE_HIDDEN, generator, device)
    res = torch.randn((rows, HIDDEN), generator=generator, device=device, dtype=torch.float32)
    pfw = _norm_weight(HIDDEN, generator, device)
    alpha = _uniform((HIDDEN,), 0.5, 1.0, generator, device)
    beta = _uniform((HIDDEN,), 0.0, 0.5, generator, device)
    npw = _norm_weight(HIDDEN, generator, device)
    w_unpool = _normal((HIDDEN, MOE_HIDDEN), WEIGHT_STD, generator, device, bf16)
    return {
        "xin": xin, "ids": ids, "scores": scores, "w13": w13, "s13": s13, "w2": w2, "s2": s2, "pen_w": pen_w,
        "res": res, "pfw": pfw, "alpha": alpha, "beta": beta, "npw": npw, "w_unpool": w_unpool,
        "guard": guard_words(device),
    }


def _tp_rows13(t, r):
    """A rank's TP slice of the gate | up rows of every expert (``weights.tp_expert_copies``)."""
    rs = INTER // WORLD
    return torch.cat([t[:, r * rs:(r + 1) * rs], t[:, INTER + r * rs:INTER + (r + 1) * rs]], dim=1).contiguous()


def shard_inputs(logical, *, rank, world_size):
    """Rank ``rank``'s kernel inputs from the logical inputs (outside timing)."""
    assert world_size == WORLD, world_size
    cs = MOE_HIDDEN // WORLD
    out = {
        "w13": _tp_rows13(logical["w13"], rank),
        "s13": _tp_rows13(logical["s13"], rank),
        "w2": logical["w2"][:, rank * cs:(rank + 1) * cs].contiguous(),
        "s2": logical["s2"][:, rank * cs:(rank + 1) * cs].contiguous(),
    }
    for name in ("xin", "ids", "scores", "pen_w", "res", "pfw", "alpha", "beta", "npw", "w_unpool", "guard"):
        out[name] = logical[name]
    return out


def rank_expected(logical_outputs, *, rank, world_size):
    """What rank ``rank`` must write: both tensors are replicated (each rank finishes the whole rows)."""
    assert world_size == WORLD, world_size
    return {name: logical_outputs[name] for name in ("attn_out", "res")}
