# SPDX-License-Identifier: Apache-2.0
"""Logical inputs, per-rank shards and per-rank expected outputs of the Muse Spark DFlash2 draft's D-MLP
launch (one draft layer's MLP half) at TP8, one draft step of ``rows`` rows (``rows`` / 8 requests, a block
of 8 rows each).

Logical inputs (``make_inputs``; identical on every rank, drawn from the case seed), the distributions of
sgl-amd-mono's test/kernels/test_muse_spark_dmlp.py with the whole (TP1) MLP:

  res [rows, 8192] bf16             the residual stream entering the post-attention norm: N(0, 1)
  attn_out [rows, 8192] bf16        the attention half's output: 0.5 N(0, 1)
  w_post [8192] bf16                post_attention_layernorm.weight: 1 + 0.1 N(0, 1)
  w_kp [2048, 8192] bf16            mlp_conv.kernel_projection.weight, rows (side, tap, group of 16): 0.01 N(0, 1)
  base [2, 2, 8192] bf16            mlp_conv.base_kernel (side, tap, channel): tap 0 1 + 0.05 N(0, 1), tap 1
                                    0.05 N(0, 1)
  w_gu [57344, 8192] bf16           mlp.gate_up_proj.weight: gate rows 0..28671, up rows 28672..: 0.02 N(0, 1)
  w_down [8192, 28672] bf16         mlp.down_proj.weight (out x in): 0.01 N(0, 1) (the test's 0.02 over its
                                    one-rank 3584 columns; halved for the 8x wider sum)
  guard [8] int32                   the runner's epoch / guard-word buffer: epoch 1, a 30 s wait deadline in
                                    100 MHz ticks, clean error words (replicated)

Sharding (``shard_inputs``, rank r of 8; SGLang's MergedColumnParallelLinear / RowParallelLinear shards):
w_gu the gate rows 3584 r .. and up rows 28672 + 3584 r .. (7168 rows: gate first), w_down the input
columns 3584 r ..; everything else replicated.

The kernel's mailbox scratch (whose contents depend on the implementation) is not an input: the
baseline keeps it in the framework's symmetric buffer, which is zeroed before every call.
"""
import torch

H = 8192
INTER = 28672  # the draft's intermediate width
WORLD = 8
I_RANK = INTER // WORLD  # 3584
COEF_ROWS = 2048  # 2 sides x 2 taps x 512 groups
TAPS = 2
SIDES = 2
GUARD_WORDS = 8
DEADLINE_S = 30.0
CLOCK_HZ = 100_000_000  # the guard's s_memrealtime clock


def _normal(shape, std, generator, device, dtype):
    out = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    return out.mul_(std).to(dtype)


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
    res = _normal((rows, H), 1.0, generator, device, bf16)
    attn_out = _normal((rows, H), 0.5, generator, device, bf16)
    w_post = _normal((H,), 0.1, generator, device, torch.float32).add_(1.0).to(bf16)
    w_kp = _normal((COEF_ROWS, H), 0.01, generator, device, bf16)
    base = torch.randn((SIDES, TAPS, H), generator=generator, device=device, dtype=torch.float32).mul_(0.05)
    base[:, 0] += 1.0  # tap 0 passes the row through, tap 1 mixes in the block's previous row
    base = base.to(bf16)
    w_gu = _normal((2 * INTER, H), 0.02, generator, device, bf16)
    w_down = _normal((H, INTER), 0.01, generator, device, bf16)
    return {"res": res, "attn_out": attn_out, "w_post": w_post, "w_kp": w_kp, "base": base, "w_gu": w_gu,
            "w_down": w_down, "guard": guard_words(device)}


def shard_inputs(logical, *, rank, world_size):
    """Rank ``rank``'s kernel inputs from the logical inputs (outside timing)."""
    assert world_size == WORLD, world_size
    rows = slice(rank * I_RANK, (rank + 1) * I_RANK)
    w_gu = logical["w_gu"]
    out = {
        "w_gu": torch.cat([w_gu[:INTER][rows], w_gu[INTER:][rows]], dim=0).contiguous(),
        "w_down": logical["w_down"][:, rows].contiguous(),
    }
    for name in ("res", "attn_out", "w_post", "w_kp", "base", "guard"):
        out[name] = logical[name]
    return out


def rank_expected(logical_outputs, *, rank, world_size):
    """What rank ``rank`` must write: both outputs are replicated (each rank finishes the whole rows)."""
    assert world_size == WORLD, world_size
    return {name: logical_outputs[name] for name in ("res_out", "out")}
