# SPDX-License-Identifier: Apache-2.0
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/common/stamps.py). Modified for
# AgentKernelArena: imports are package-relative; a mark stores through ops.global_store
# (FlyDSL 0.3.2 has no fx.generic_store or fx.Pointer.ir_type). Timeline builds are never made here.
"""A timeline build's stage marks (debug only).

Thread 0 of each CTA stores the 100 MHz clock at mark k into its CTA's row of a
[CTA][TL_POINTS] int64 scratch region; a build without the region records nothing.
Adapted from ROCm/ATOM (atom/mono/device/stamps.py), which keeps the row in LDS
until the launch ends: here each mark is a global store, a small skew on the waits
after it."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from .ops import clock, global_store

TL_POINTS = 48  # marks a CTA row: K2's attention 0 .., its FFN 16 .., K1's 32 ..


def timeline_row(scratch, offset: int, bid):
    """The address of CTA ``bid``'s row of the timeline at byte ``offset`` of ``scratch``."""
    return scratch + fx.Int64(offset) + fx.Int64(bid) * fx.Int64(TL_POINTS * 8)


@flyc.jit
def stamp(c, point: int):
    """Mark ``point`` on this CTA's row (``c["tl"]``), if the build has one."""
    if const_expr(c.get("tl") is not None):  # noqa: SIM102
        if c["tid"] == 0:
            at = c["tl"] + fx.Int64(point * 8)
            global_store(at, clock(), words=2)
