# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang#43984 at 134919476c (kernels/ops/mono/gfx950/
# deepseek_v41/indexer/k2i.py), whose scorer comes from ROCm/ATOM at 68506823
# (MIT License, Copyright (C) 2026, Advanced Micro Devices, Inc.)
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/deepseek_v41/gfx950/indexer/k2i.py).
# Modified for AgentKernelArena: imports are package-relative; the build-key source tag is a
# constant (upstream hashes the traced modules' source files with plan.kernel_sources, which the
# candidate rules forbid reading; FlyDSL hashes each jit and kernel function's own source itself);
# the kernel symbol is formatted here (common/epoch.py's kernel_symbol, aiter's kernel_signature
# written out); the launch scores FULL index-source layers only, so the ``reindex`` argument and
# the step's row count (used only by the removed REINDEX split) are gone and one build serves
# every row count; the one-member LDS union is allocated as its struct.
"""K2i: ATOM K2a's FP4 scorer (``score_fp4``) as its own launch between K1 and K2.

Each CTA scores the task the step's walk plan gives it, none waiting on another, into ``ilog``
[S, lstride] f32 for SGLang's top-k. ``itab`` is a FULL layer's page table (an entry 2 **
``ishift`` virtual 8-row pages, read at row ``ibatch[t]``, the request's first).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int32, Int64

from .plan import BLOCKS, THREADS
from .score_fp4 import ROWS_LDS_WORDS, run_scores4

# AgentKernelArena: the build key's source tag (bump it when a vendored helper changes).
SOURCES = "aka-dsv41-k2i-52486703-1"

_STREAM = fx.Stream(None)
MAX_ROWS = 48  # a step's rows at most (8 DSpark requests of 6)


def kernel_symbol(stem, **params):
    """Kernel symbol that profilers demangle to ``aiter::<stem>_<params>`` (common/epoch.py, with
    aiter's ``kernel_signature`` written out: the parameters joined as ``<name><value>``)."""
    name = stem + "_" + "_".join(f"{k}{v}" for k, v in params.items())
    return f"_ZN5aiter{len(name)}{name}E"


@functools.lru_cache(maxsize=None)
def build_k2i():
    """K2i for a FULL layer's step of up to MAX_ROWS rows (the rows come from the walk plan)."""
    name = kernel_symbol("dsv41_mono_k2i", f=1)
    keyed = (("full", 1), ("sources", SOURCES))

    @fx.struct
    class RowsLds:
        rows: fx.Array[fx.Int32, ROWS_LDS_WORDS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2i(
        iq: Int64, iqs: Int64, iw: Int64, plane: Int64, pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32, ibound: Int64, ilog: Int64,
        lstride: Int32,
    ):  # fmt: skip
        _ = keyed
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(RowsLds).peek()
        args = {
            "iq": iq, "iqs": iqs, "iw": iw, "plane": plane, "pscale": pscale, "iplan": iplan,
            "itab": itab, "itab_stride": itab_stride, "itab_len": itab_len, "ibatch": ibatch,
            "ishift": ishift, "ibound": ibound, "ilog": ilog, "lstride": lstride,
        }  # fmt: skip
        c = {
            "tid": tid, "lane": tid % 64, "wave": tid // 64, "args": args,
            "rows_lds": lds.rows.ptr,
        }  # fmt: skip
        run_scores4(c, fx.block_idx.x)

    @flyc.jit
    def launch(
        iq: Int64, iqs: Int64, iw: Int64, plane: Int64, pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32, ibound: Int64, ilog: Int64,
        lstride: Int32, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        k2i(
            iq, iqs, iw, plane, pscale, iplan, itab, itab_stride, itab_len, ibatch, ishift, ibound, ilog,
            lstride,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
