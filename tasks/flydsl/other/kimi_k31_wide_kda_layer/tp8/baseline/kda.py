# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's KDA constants and load helpers, as the wide-row launch reads them.

AKA: the part of sgl-amd-mono ``sgl_amd_mono/kimi_k3/gfx950/attention/kda.py`` (the K1 ``kda_pre``
launch) that ``kimi_k31``'s wide layer imports (``kda as k1`` in ``layer.py`` / ``pre_attn.py``,
``ld_bf`` / ``ld_f32`` / ``sigmoid`` in ``kda_wide.py`` / ``moe_front.py``), copied verbatim. The K1
launch itself (its stages, build and host launcher) is not part of this task and is left out.
"""

import flydsl.expr as fx
from flydsl.expr.typing import T

from . import buffer_ops as bo
from .dims import HD, LOG2E, NH, PROJ  # noqa: F401 (k1.NH, k1.HD, k1.PROJ: read through this module)
from .ops import hw_exp2, rsrc

QKV = 3 * PROJ
G2_0 = QKV
FA_0 = G2_0 + PROJ  # 6144
BETA_0 = FA_0 + HD  # 6272
NPROJ = 6288  # in_proj rows a rank holds (6284 + 4 pad)
VCH = 16  # state rows a KDA task
NVCH = HD // VCH  # 8
CONV_W = 4


def ld_bf(ptr, i, cm=0):
    return fx.Float32(
        fx.BFloat16(
            bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16, cache_modifier=cm)
        )
    )


def ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))
