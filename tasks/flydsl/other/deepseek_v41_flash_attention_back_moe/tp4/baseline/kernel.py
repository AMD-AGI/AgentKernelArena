# SPDX-License-Identifier: Apache-2.0
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1-Flash K2 on one rank of a TP4 group: the entry point of this baseline.

The device code is sgl-amd-mono's K2 monokernel (``layer.build_mono_k2``, vendored from
sgl-project/sgl-amd-mono at 52486703fbf27e77f05b573e4d931d4eec87118d): one persistent launch of
256 CTAs x 512 threads that runs the attention back (sparse MQA decode with the attention sink
over the 128-key window and the top-512 compressed rows of K1's key table, combine, inverse
GPT-J RoPE), wo_a and wo_b, the attention's TP reduce, the FFN mHC seam, the router (top 6 of
384), the shared and routed MXFP4 experts and the MoE's TP all-reduce.

run() is ``DSV41MonoRunner.k2`` (gfx950/runner.py) for a non-index layer with a compressed
cache (``MonoBuild(kt=True, compressed=True, index=False)``) without SGLang:

* the build is the runner's for this width (``route_at``: one FP4 group an ug task below 7
  rows, else two; row-major wo_b, as SGLang loads it without the MXFP8 aiter preshuffle);
* the persistent-kernel state the runner keeps across launches is made fresh in every call:
  one zeroed allocation holds the epoch buffer (epoch 0; its zero deadline words stand for a
  10 s wait deadline, guard.DEFAULT_DEADLINE_TICKS) and the scratch mailboxes, so the call's
  fixed mailbox tags never meet an earlier call's; the framework zeroes the symmetric peer
  buffer (the TP exchange regions and the guard's error flags) before every call;
* the zero record an absent key reads (``zero_rec``) is allocated zeroed in the call.

Every computation runs in the FlyDSL kernel; the host only allocates, views and launches.
"""

import functools

import torch

from . import guard
from .layer import EPOCH_WORDS, MAX_TOKENS, MonoBuild, build_mono_k2, peer_half_bytes, scratch_bytes
from .moe_shape import EXPERTS, TOPK

TP = 4
# below this many rows an ug task takes one FP4 group (runner.FINE_UG_ROWS)
FINE_UG_ROWS = 7
# SGLang's paged V4 caches as the task lays them out: 64-record SWA pages, 128-record compressed pages
SWA_BLOCK, COMP_BLOCK = 64, 128
# float32 bits of 512 ** -0.5 * log2(e), the runner's qk_scale
QK_SCALE_BITS = 1031967726
# the wait deadline (runner.set_deadline): 10 s in 100 MHz ticks, the epoch buffer's words 1-2
# the call's epoch buffer, then its scratch, in one zeroed allocation (one memset in the graph); the
# guard reads the zero deadline words as guard.DEFAULT_DEADLINE_TICKS (10 s)
EPOCH_BYTES = -(-EPOCH_WORDS * 4 // 4096) * 4096
ZERO_RECORD_BYTES = 1024  # runner.zero_rec
SYMMETRIC_BYTES = 2 * peer_half_bytes(TP) + guard.FLAG_BYTES


def route_at(rows: int) -> dict:
    """DSV41MonoRunner.route_at for the target's route."""
    return {"experts": EXPERTS, "topk": TOPK, "ug_groups": 1 if rows < FINE_UG_ROWS else 2}


@functools.lru_cache(maxsize=None)
def _launcher(rows: int):
    return build_mono_k2(MonoBuild(rows, TP, kt=True, index=False, compressed=True, **route_at(rows)))


def run(*, q, swa, comp, kt, klen, pos, sink, cos_sin, woa, woa_s, wob, wob_s, res_in, post_in, comb_in,
        pre_in, hc_fn, hc_scale, hc_base, ffn_norm, gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s,
        out, res_out, post_out, comb_out, pre_out, rank, world_size, peers, symmetric):
    rows = q.shape[0]
    assert world_size == TP and 1 <= rows <= MAX_TOKENS
    assert symmetric.numel() >= SYMMETRIC_BYTES
    dev = q.device
    state = torch.zeros(EPOCH_BYTES + scratch_bytes(rows, TP, **route_at(rows)), dtype=torch.uint8, device=dev)
    epoch, scratch = state.data_ptr(), state.data_ptr() + EPOCH_BYTES
    zrec = torch.zeros(ZERO_RECORD_BYTES, dtype=torch.uint8, device=dev)
    p = lambda t: t.data_ptr()  # noqa: E731
    _launcher(rows)(
        p(q), p(swa), swa.stride(0), SWA_BLOCK, p(comp), comp.stride(0), COMP_BLOCK,
        p(kt), p(klen),
        # an index source's slots and keys: unread by this build (the runner passes klen)
        p(klen), 1, p(klen), p(klen), p(klen), p(klen),
        p(pos), p(sink), QK_SCALE_BITS, p(cos_sin), p(woa), p(woa_s), p(wob), p(wob_s), p(zrec),
        p(res_in), p(post_in), p(comb_in), p(pre_in), p(hc_fn), p(hc_scale), p(hc_base), p(ffn_norm),
        p(res_out), p(post_out), p(comb_out), p(pre_out),
        p(gate_w), p(bias), p(w13), p(w13_s), p(w2), p(w2_s), p(sgu), p(sgu_s), p(sw2), p(sw2_s),
        p(out), scratch, p(symmetric), p(peers), int(rank), epoch,
        stream=torch.cuda.current_stream(),
    )
