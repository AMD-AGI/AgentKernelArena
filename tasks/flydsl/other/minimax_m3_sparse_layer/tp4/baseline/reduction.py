# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 reduction order shared with its unfused all-reduce epilogue."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from .ops import butterfly


@flyc.jit
def ar_block_sums(parts, lane, wave, red):
    """Per row ``(p0, p1)`` of ``parts``, the sum of a 6144-wide bf16 row's 768 per-pack
    partials in the 1-stage fused all-reduce + RMSNorm's order, so rstd is bit-exact
    with the unfused path.

    24 warps of 32, a 16..1 butterfly per warp, then the 24 warp sums butterflied again
    in one warp; every row goes behind one set of barriers. Logical warp
    ``3 * wave + lane // 32`` holds ``p0``; logical warp ``3 * wave + 2`` holds ``p1``
    in lanes < 32 (``WAVES`` = 8). ``red`` needs 24 words per row.
    """
    bs = [
        (butterfly(p0, (16, 8, 4, 2, 1)), butterfly(p1, (16, 8, 4, 2, 1)))
        for p0, p1 in parts
    ]
    gpu.barrier()
    for k in range_constexpr(len(bs)):
        b0, b1 = bs[k]
        if lane % 32 == 0:
            fx.ptr_store(b0, red + (24 * k + wave * 3 + lane // 32))
        if lane == 0:
            fx.ptr_store(b1, red + (24 * k + wave * 3 + 2))
    gpu.barrier()
    li = lane % 32
    tots = [
        butterfly(
            (li < 24).select(
                fx.ptr_load(red + (24 * k + fx.min(li, 23))), fx.Float32(0.0)
            ),
            (16, 8, 4, 2, 1),
        )
        for k in range(len(parts))
    ]
    gpu.barrier()
    return tots
