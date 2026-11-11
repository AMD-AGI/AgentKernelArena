# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/common/epoch.py). Modified for
# AgentKernelArena: imports are package-relative; gstore stores through ops.global_store (an
# llvm-dialect store: FlyDSL 0.3.2 has no fx.generic_store); kernel_signature is a local
# stand-in for aiter's kernels_common.kernel_signature.
"""The launch epoch: every CTA marks it, and CTA 0 moves it on once all have.

The epoch buffer starts with the guard block (``guard.py``): the epoch, the
wait deadline and the error words, then a mark per CTA."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import T

from .guard import GUARD_WORDS
from .ops import (
    CM_DEV,
    global_store,
    rsrc,
)
from .plan import BLOCKS
from .sync import _spin


def kernel_signature(**params: object) -> str:
    """Render build parameters into a kernel-name suffix (booleans as 0 / 1, no hyphens): a
    stand-in for aiter's ``kernels_common.kernel_signature``."""
    parts = [f"{name}{int(value) if isinstance(value, bool) else value}" for name, value in params.items()]
    return "_".join(parts).replace("-", "_")


def kernel_symbol(stem, **params):
    """Kernel symbol that profilers demangle to ``aiter::<stem>_<params>``."""
    name = f"{stem}_{kernel_signature(**params)}"
    return f"_ZN5aiter{len(name)}{name}E"


def gstore(addr, value, words=4):
    """Store ``words`` i32 (an Int32 or a Vector) at a 64-bit global address."""
    global_store(addr, value, words=words)


def spin_until(load, pending):
    """Poll ``load()``, sleeping between, until ``pending`` clears; return the value."""
    return _spin(load, pending)


EPOCH_MARKS = GUARD_WORDS  # word offset of the per-CTA marks, past the guard block


@flyc.jit
def epoch_begin(c, epoch, ep):
    """Mark that this CTA read epoch ``ep``."""
    if c["tid"] == 0:
        bo.buffer_store(
            ep + 1, rsrc(epoch), EPOCH_MARKS + c["bid"], cache_modifier=CM_DEV
        )


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
