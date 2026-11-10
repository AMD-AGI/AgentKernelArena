# SPDX-License-Identifier: Apache-2.0
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# Written for AgentKernelArena (not part of sgl-amd-mono): the task defines the logits past each row's
# bound as -inf, which the plugin leaves unwritten (SGLang's top-k reads only the visible columns).
"""-inf into each row's columns at or past its bound: the part of the [rows, ctx] logits K2i does not write.

A CTA takes 4096 columns of one row and returns at once unless some of them lie at or past the row's bound,
so every element of the logits is written once: below the bound by K2i, from it by this kernel.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int32, Int64

from . import buffer_ops as bo
from .k2i import SOURCES, kernel_symbol
from .ops import ld_i32, rsrc

_STREAM = fx.Stream(None)
THREADS = 256
COLS = 16 * THREADS  # columns a CTA
NEG_INF_BITS = -8388608  # 0xff800000: float32 -inf as an int32 word


def blocks(width: int) -> int:
    """The column blocks of a row of ``width`` columns."""
    return -(-width // COLS)


@functools.lru_cache(maxsize=None)
def build_tail_fill():
    """The launch over (rows, blocks(lstride)) CTAs."""
    name = kernel_symbol("dsv41_k2i_tail_fill", c=COLS)
    keyed = (("cols", COLS), ("sources", SOURCES))

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def tail_fill(bounds: Int64, logits: Int64, lstride: Int32):
        _ = keyed
        t = fx.block_idx.x
        start = fx.block_idx.y * COLS
        bound = ld_i32(bounds, t)
        if start + COLS > bound:
            # the row's buffer ends at its last column, so stores past it are dropped
            out = rsrc(logits + fx.Int64(t) * fx.Int64(lstride) * 4, lstride * 4)
            for i in range_constexpr(COLS // THREADS):
                c = start + i * THREADS + fx.thread_idx.x
                bo.buffer_store(fx.Int32(NEG_INF_BITS), out, c, mask=c >= bound)

    @flyc.jit
    def launch(
        bounds: Int64, logits: Int64, lstride: Int32, rows: Int32, nblocks: Int32, stream: fx.Stream = _STREAM
    ):
        _ = keyed
        tail_fill(bounds, logits, lstride).launch(grid=(rows, nblocks, 1), block=(THREADS, 1, 1), stream=stream)

    return launch
