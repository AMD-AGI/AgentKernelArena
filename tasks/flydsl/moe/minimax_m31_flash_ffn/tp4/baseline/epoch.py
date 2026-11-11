# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/common/epoch.py). Modified for
# AgentKernelArena (FlyDSL 0.3.2 runtime): imports are package-relative; ``gstore`` goes through
# ops.global_store (an llvm-dialect store: FlyDSL 0.3.2 has no fx.generic_store); ``kernel_symbol``
# and ``epoch_begin`` (unused by the FFN kernel) are dropped.
"""The launch epoch: every CTA marks it, and CTA 0 moves it on once all have.

The epoch buffer starts with the guard block (``guard.py``): the epoch, the
wait deadline and the error words, then a mark per CTA."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import T

from . import buffer_ops as bo
from .guard import GUARD_WORDS
from .ops import CM_DEV, global_store, rsrc
from .plan import BLOCKS
from .sync import _spin


def gstore(addr, value, words=4):
    """Store ``words`` i32 (an Int32 or a Vector) at a 64-bit global address."""
    global_store(addr, value, words=words)


def spin_until(load, pending):
    """Poll ``load()``, sleeping between, until ``pending`` clears; return the value."""
    return _spin(load, pending)


EPOCH_MARKS = GUARD_WORDS  # word offset of the per-CTA marks, past the guard block


@flyc.jit
def epoch_end(c, epoch, ep, reset=None):
    """CTA 0: once every CTA marked the epoch, run ``reset()`` and move the epoch on."""
    if c["bid"] == 0:
        tid = c["tid"]
        if tid < BLOCKS:
            r = rsrc(epoch)

            def load_mark():
                return fx.Int32(
                    bo.buffer_load(
                        r,
                        EPOCH_MARKS + tid,
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )

            _ = spin_until(load_mark, lambda v: v != ep + 1)
        gpu.barrier()
        if tid == 0:
            if const_expr(reset is not None):
                reset()
            gstore(epoch, ep + 1, words=1)
