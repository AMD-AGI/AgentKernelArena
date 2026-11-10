# SPDX-License-Identifier: Apache-2.0
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""The MiniMax-M3.1-Flash FFN monokernel on one rank of a TP4 group: the entry point of this baseline.

run() is sgl-amd-mono's ``FfnRunner.__call__`` (sgl_amd_mono/minimax_m31/runner.py at 52486703) for
one sparse layer, without SGLang: one launch of ``kffn`` (ffn.py), built per row count at TP 4, on
the current stream. The kernel reads the weights in the layouts SGLang leaves after loading
(inputs.py prepares them outside timing) and exchanges its two TP all-reduces through the
framework's symmetric peer buffers (``peers`` / ``symmetric``).

The persistent-kernel state the runner keeps across steps is made fresh inside every call instead,
in one workspace: the epoch / guard block (epoch 0, a 10 s wait deadline, no error, no CTA mark),
the step's slot mapping (every row a real token: slot 0 is valid, a negative slot would mark a
padded row) and the mailbox scratch, whose tagged-pair regions are zeroed (no pair of an earlier
call carries this call's tag; the data regions they announce are written before they are read).
The framework zeroes the symmetric buffer (the attention and MoE partials of both parities and
the guard's peer flags) before every call, so the fixed epoch-0 tags are safe. Every computation
runs in the FlyDSL kernel; the host only allocates, zeroes and launches.
"""

import functools

import torch

from . import ffn
from .guard import CLOCK_HZ, GUARD_DEADLINE

TP = 4
SYMMETRIC_BYTES = ffn.peer_bytes(TP)  # 5,243,136: both parity halves and the guard's peer flags
DEADLINE_TICKS = 10 * CLOCK_HZ  # every wait's deadline (runner.set_deadline): 10 s, < 2^31 ticks
EPOCH_OFF = 0  # the epoch / guard block: ffn.EPOCH_WORDS int32
SLOTS_OFF = 2048  # MAX_TOKENS int64 slots, zero: every row is a real token
SCRATCH_OFF = 4096  # the mailbox scratch (ffn.scratch_layout), 256-byte aligned
assert ffn.EPOCH_WORDS * 4 <= SLOTS_OFF and SLOTS_OFF + 8 * ffn.MAX_TOKENS <= SCRATCH_OFF


@functools.lru_cache(maxsize=None)
def _launcher(rows: int):
    return ffn.build_ffn(ffn.FfnBuild(tokens=rows, tp=TP))


@functools.lru_cache(maxsize=None)
def _workspace(rows: int) -> tuple[int, int]:
    """(bytes, zeroed prefix bytes) of a call's workspace: the guard block, the slots and the tagged
    pairs of the scratch are zeroed; its data regions (from ``sres`` on) need not be."""
    layout = ffn.scratch_layout(rows)
    return SCRATCH_OFF + ffn.scratch_bytes(rows), SCRATCH_OFF + layout["sres"][0]


def run(*, attn, res, wo, wos, norm_w, router_w, bias, w13, s13, w2, s2, w13s, s13s, w2s, s2s, out, res_out,
        rank, world_size, peers, symmetric):
    rows = attn.shape[0]
    assert world_size == TP and 1 <= rows <= ffn.MAX_TOKENS
    assert attn.shape[1] == ffn.ATTN_K and res.shape == (rows, ffn.HIDDEN)
    assert symmetric.numel() >= SYMMETRIC_BYTES
    nbytes, zeroed = _workspace(rows)
    ws = torch.empty(nbytes, dtype=torch.uint8, device=attn.device)
    ws[:zeroed].zero_()
    # the deadline's low word (DEADLINE_TICKS < 2^31: the high word stays 0)
    ws.view(torch.int32)[GUARD_DEADLINE:GUARD_DEADLINE + 1].fill_(DEADLINE_TICKS)
    base = ws.data_ptr()
    _launcher(rows)(
        attn.data_ptr(), res.data_ptr(), res_out.data_ptr(), out.data_ptr(), base + SLOTS_OFF,
        norm_w.data_ptr(), router_w.data_ptr(), bias.data_ptr(),
        w13.data_ptr(), s13.data_ptr(), w2.data_ptr(), s2.data_ptr(),
        w13s.data_ptr(), s13s.data_ptr(), w2s.data_ptr(), s2s.data_ptr(),
        wo.data_ptr(), wos.data_ptr(),
        base + SCRATCH_OFF, symmetric.data_ptr(), peers.data_ptr(), int(rank), base + EPOCH_OFF, 1,
        torch.cuda.current_stream(),
    )
