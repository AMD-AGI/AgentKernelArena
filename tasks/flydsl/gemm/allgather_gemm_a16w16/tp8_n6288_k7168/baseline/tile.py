# SPDX-License-Identifier: Apache-2.0
"""bf16 row-tile GEMM stage of the wide-row layer launch (16 .. 384 rows): its epilogues.

AKA: only ``atomic_lin`` is vendored, unchanged: the lane-contiguous packed-bf16 atomic adds that the
``mma_tile`` epilogue of ``gemm_coll.ag_gemm`` ("linbf") merges its K slices with. The row-tile GEMM
itself (``tile_acc`` / ``tile_reduce``), the other epilogues and the standalone builder are not used by
this launch.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, rocdl

from .ops import rsrc


def atomic_lin(out, ldo, n0, m, n, bf16):
    """Lane-contiguous adds for an LDS-shuffled epilogue: (elements a lane,
    ``add(row, col, vals)``); consecutive lanes take consecutive columns, so a
    wave's add covers whole 64 B lines."""
    eb = 2 if bf16 else 4
    size = m * ldo * eb
    r_o = rsrc(out, size)
    z = fx.Int32(0)

    def add(row, col, vals):
        ok = (row < m) & (n0 + col < n)
        off = ok.select((row * ldo + n0 + col) * eb, fx.Int32(size))
        if const_expr(bf16):
            pair = fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16)
            rocdl.raw_ptr_buffer_atomic_fadd(pair, r_o, off, z, z)
        else:
            rocdl.raw_ptr_buffer_atomic_fadd(vals[0], r_o, off, z, z)

    return (2 if bf16 else 1), add
