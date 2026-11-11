# SPDX-License-Identifier: Apache-2.0
"""Logical inputs, per-rank shards and per-rank expected outputs of the fused all-gather + in_proj GEMM of
Kimi-K3.1 at TP8 (m rows, hidden 7168, 6288 in_proj rows a rank).

Logical inputs (``make_inputs``; identical on every rank, drawn from the case seed), the distributions of
sgl-amd-mono's test/kernels/kimi_k31/ag_gemm_bench.py:

  x [m, 7168] bf16          the step's activations (the gathered hidden state): N(0, 1)
  w [8 * 6288, 7168] bf16   the in_proj weight of all 8 ranks, rank r's rows at 6288 r ..: N(0, 1) / sqrt(7168)
  epoch [8] int32           the launch's epoch / guard-word buffer: epoch 1, a 60 s wait deadline in 100 MHz
                            ticks, clean error words (replicated)

Sharding (``shard_inputs``, rank r of 8): x the columns 896 r .. 896 (r + 1) (the column-sharded residual
stream's slice this rank owns), w the rows 6288 r .. 6288 (r + 1) (the rank's in_proj rows, its own
weight); epoch replicated. Rank r writes out = x @ w_r^T, the logical output's columns 6288 r ..
(``rank_expected``).

The kernel's peer regions, flags and arrival counter (whose contents depend on the implementation) are not
inputs: the baseline keeps them in the framework's symmetric buffer, which is zeroed before every call.
"""

import torch

TP = 8
N = 6288  # a rank's in_proj rows
K = 7168  # hidden size
COLS = K // TP  # a rank's columns of x
GUARD_WORDS = 8
DEADLINE_S = 60.0
CLOCK_HZ = 100_000_000  # the guard's s_memrealtime clock


def guard_words(device):
    """The epoch buffer: the epoch (word 0), the wait deadline in 100 MHz ticks (words 1-2, an int64 as two
    int32), clean error words (3-5) and padding."""
    ticks = int(DEADLINE_S * CLOCK_HZ)
    lo = ticks & 0xFFFFFFFF
    words = [1, lo - (1 << 32) if lo >= 1 << 31 else lo, ticks >> 32] + [0] * (GUARD_WORDS - 3)
    return torch.tensor(words, dtype=torch.int32, device=device)


def make_inputs(case, inputs, generator, device):
    """The logical inputs of one case (every rank draws the same from the case seed)."""
    m = int(case["axes"]["m"])
    world = int(case["axes"].get("world_size", TP))
    assert world == TP, world
    x = torch.randn((m, K), generator=generator, device=device, dtype=torch.float32).to(torch.bfloat16)
    w = torch.empty((TP * N, K), dtype=torch.bfloat16, device=device)
    for r in range(TP):  # a rank's rows at a time: the float32 draw stays at one rank's 180 MB
        part = torch.randn((N, K), generator=generator, device=device, dtype=torch.float32)
        w[r * N:(r + 1) * N] = part.mul_(K ** -0.5).to(torch.bfloat16)
        del part
    return {"x": x, "w": w, "epoch": guard_words(device)}


def shard_inputs(logical, *, rank, world_size):
    """Rank ``rank``'s kernel inputs from the logical inputs (outside timing)."""
    assert world_size == TP, world_size
    return {
        "x": logical["x"][:, rank * COLS:(rank + 1) * COLS].contiguous(),
        "w": logical["w"][rank * N:(rank + 1) * N],
        "epoch": logical["epoch"],
    }


def rank_expected(logical_outputs, *, rank, world_size):
    """What rank ``rank`` must write: its own in_proj rows' columns of the logical output."""
    assert world_size == TP, world_size
    return {"out": logical_outputs["out"][:, rank * N:(rank + 1) * N]}
