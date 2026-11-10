# SPDX-License-Identifier: Apache-2.0
"""The output epilogues of the tile GEMM stage (bf16 tiles, f32 K-slice partials).

Vendored from sgl-project/sgl-amd-mono at 52486703, ``sgl_amd_mono/kimi_k31/gfx950/stages/tile.py``:
``bf16_emit`` and ``f32_emit`` only, unchanged apart from the imports. The rest of that file
(the register-operand row-tile GEMM ``tile_acc`` / ``tile_reduce``, an alternative implementation
of this operator, and the atomic epilogues) is not used by this baseline and is not vendored.
"""

import flydsl.expr as fx

from . import buffer_ops as bo
from .ops import bf16_pair, rsrc


def bf16_emit(out, ldo, n0, m, n):
    """``emit`` storing bf16 into the row-major [m, ldo] at ``out``, columns n0 + col."""
    r_o = rsrc(out, m * ldo * 2)  # sized: a masked store is an out-of-range offset

    def emit(row, col, v, live):
        words = fx.Vector.from_elements(
            [
                bf16_pair(v[0], v[1]).bitcast(fx.Int32),
                bf16_pair(v[2], v[3]).bitcast(fx.Int32),
            ],
            fx.Int32,
        )
        ok = (row < m) & (n0 + col < n)
        ok = ok if live is None else ok & live
        bo.buffer_store(words, r_o, (row * ldo + n0 + col) // 2, mask=ok)

    return emit


def f32_emit(out, ldo, n0, m, n):
    """``emit`` storing f32 partials into the row-major [m, ldo] at ``out``."""
    r_o = rsrc(out, m * ldo * 4)

    def emit(row, col, v, live):
        ok = (row < m) & (n0 + col < n)
        ok = ok if live is None else ok & live
        bo.buffer_store(v, r_o, row * ldo + n0 + col, mask=ok)

    return emit
