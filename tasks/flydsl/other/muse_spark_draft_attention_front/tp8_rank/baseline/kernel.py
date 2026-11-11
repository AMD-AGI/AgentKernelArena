# SPDX-License-Identifier: Apache-2.0
"""The Muse Spark DFlash2 draft layer's D-pre launch (sgl-amd-mono
``muse_spark/gfx950/draft_attn_half.py::build_dpre``) in the arena ``run()`` interface, on one rank of
the draft's TP8 group (D-pre exchanges nothing with the other ranks).

One persistent launch of 256 CTAs x 512 threads: the input norm's fused add and RMSNorm, the kernel
projection (both sides' conv coefficients, ``coef``), conv0, the rank's q | k | v projection, the
per-head q / k RMSNorm and neox RoPE from the table, q out and k / v into the fp8 cache at the rows'
slots. Stages hand values over through tagged-pair mailboxes in a scratch buffer.

The build and the arguments are the draft runner's (``muse_spark/draft.py`` ``attn_kernels`` and
``launch_attn``): ``DPreBuild(rows)``; int64 positions and cache slots, as SGLang passes them.
Placement differs from the runner, host-side only: the runner keeps one scratch and one epoch for the
draft's launches and moves the epoch on once a step (``epoch[0] += 1``), so a mailbox pair of an
earlier step never carries the current tag. Here the scratch and the epoch / guard words (word 0 the
epoch, words 1-2 the wait deadline) share one device buffer, created zeroed at the first (eager) call
and reused, and every call moves the epoch on after its launch with a one-thread FlyDSL kernel (the
runner's per-step ``add_``, which the candidate rules leave to a kernel), so every call's tag is new.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr.typing import Int64, T

from . import buffer_ops as bo
from .draft_attn_half import DPreBuild, build_dpre, dp_scratch_layout
from .guard import GUARD_DEADLINE, deadline_ticks
from .layout import scratch_bytes
from .ops import CM_DEV, rsrc

LAYER = 0  # the mailbox tag is epoch * LAYER_SLOTS + layer + 1
GUARD_BYTES = 256  # the epoch / guard words, then the scratch (256-byte aligned)
DEADLINE_S = 30.0  # a wait that never completes stops the launch
_STREAM = fx.Stream(None)
_BUFFERS = {}  # device index -> the epoch / guard words and the scratch (sized for every row count)


@functools.lru_cache(maxsize=None)
def _launcher(rows: int):
    return build_dpre(DPreBuild(tokens=rows))


@functools.lru_cache(maxsize=None)
def _epoch_step():
    """A one-thread launch that moves the epoch (word 0) on by one, as the runner's end of step does."""

    @flyc.kernel(name="muse_spark_dpre_epoch_step", known_block_size=[64, 1, 1])
    def epoch_step(epoch: Int64):
        if fx.thread_idx.x == 0:
            r = rsrc(epoch)
            ep = fx.Int32(bo.buffer_load(r, 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
            bo.buffer_store(ep + 1, r, 0, cache_modifier=CM_DEV)

    @flyc.jit
    def launch(epoch: Int64, stream: fx.Stream = _STREAM):
        epoch_step(epoch).launch(grid=(1,), block=(64,), stream=stream)

    return launch


def _buffer(device):
    buf = _BUFFERS.get(device.index)
    if buf is None:
        buf = torch.zeros(GUARD_BYTES + scratch_bytes(dp_scratch_layout()), dtype=torch.uint8, device=device)
        words = buf[:GUARD_BYTES].view(torch.int32)
        ticks = deadline_ticks(DEADLINE_S)
        low = ticks & 0xFFFFFFFF
        words[GUARD_DEADLINE:GUARD_DEADLINE + 1].fill_(low - (1 << 32) if low >= 1 << 31 else low)
        words[GUARD_DEADLINE + 1:GUARD_DEADLINE + 2].fill_(ticks >> 32)
        _BUFFERS[device.index] = buf
    return buf


def run(*, x, res, first, w_in, w_kp, base, w_qkv, q_norm, k_norm, cos_sin, positions, slots,
        k_cache, v_cache, res_out, q, coef) -> None:
    rows = x.shape[0]
    launch = _launcher(rows)
    buf = _buffer(x.device)
    base_ptr = buf.data_ptr()
    stream = torch.cuda.current_stream()
    launch(
        x.data_ptr(), res.data_ptr(), int(first), w_in.data_ptr(), w_kp.data_ptr(), base.data_ptr(),
        w_qkv.data_ptr(), q_norm.data_ptr(), k_norm.data_ptr(), cos_sin.data_ptr(), positions.data_ptr(),
        int(positions.dtype == torch.int64), slots.data_ptr(), int(slots.dtype == torch.int64),
        k_cache.data_ptr(), v_cache.data_ptr(), res_out.data_ptr(), q.data_ptr(), coef.data_ptr(),
        base_ptr + GUARD_BYTES, base_ptr, LAYER, stream=stream,
    )  # fmt: skip
    _epoch_step()(base_ptr, stream=stream)
