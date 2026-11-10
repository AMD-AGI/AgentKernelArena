# SPDX-License-Identifier: Apache-2.0
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1-Flash K1 on one rank of a TP4 group: the entry point of this baseline.

The device code is sgl-amd-mono's K1 monokernel (``layer.build_mono_k1``, vendored from
sgl-project/sgl-amd-mono at 52486703fbf27e77f05b573e4d931d4eec87118d): one persistent launch of
256 CTAs x 512 threads that runs the attention-side mHC seam (the previous layer's owed FFN post
folded into the residual streams, the 24 mix projections, the Sinkhorn gate, input RMSNorm and
its MXFP8 quantization), the fused wqkv_a GEMV, the q and kv RMSNorms, GPT-J RoPE, the V4.1
fp8_ds_mla SWA KV record write at each row's slot, K2's key table and the rank's wq_b GEMV.

run() is ``DSV41MonoRunner.k1`` (gfx950/runner.py) for a non-index layer whose previous layer owes
its post (``MonoBuild(kt=True, fold=True, index=False)``) without SGLang:

* the build is the runner's for this width (``route_at``; row-major wqkv and wq_b, as SGLang
  loads them without the MXFP8 aiter preshuffle);
* one zeroed allocation holds the epoch buffer (epoch 0; its zero deadline words stand for a 10 s
  wait deadline, guard.DEFAULT_DEADLINE_TICKS) and the scratch mailboxes in every call (the runner
  keeps them across launches and K2 moves the epoch on);
  the framework zeroes the symmetric peer buffer (K1 uses its guard flags only below 12 rows);
* the DSpark capture (aux), the index-source outputs (normed, qr) and the index query are off.

Every computation runs in the FlyDSL kernel; the host only allocates, views and launches.
"""

import functools

import torch

from . import guard
from .layer import EPOCH_WORDS, HIDDEN, MAX_TOKENS, MonoBuild, build_mono_k1, peer_half_bytes, scratch_bytes
from .moe_shape import EXPERTS, TOPK

TP = 4
FINE_UG_ROWS = 7  # runner.FINE_UG_ROWS: the route (and so the scratch layout) of a width
SWA_BLOCK = 64  # SGLang's SWA page as the task lays it out: 64 records
# the call's epoch buffer, then its scratch, in one zeroed allocation (one memset in the graph); the
# guard reads the zero deadline words as guard.DEFAULT_DEADLINE_TICKS (10 s)
EPOCH_BYTES = -(-EPOCH_WORDS * 4 // 4096) * 4096
SYMMETRIC_BYTES = 2 * peer_half_bytes(TP) + guard.FLAG_BYTES


def route_at(rows: int) -> dict:
    """DSV41MonoRunner.route_at for the target's route."""
    return {"experts": EXPERTS, "topk": TOPK, "ug_groups": 1 if rows < FINE_UG_ROWS else 2}


@functools.lru_cache(maxsize=None)
def _launcher(rows: int):
    return build_mono_k1(MonoBuild(rows, TP, kt=True, fold=True, index=False, **route_at(rows)))


def run(*, res_in, pend, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, attn_norm, wqkv, wqkv_s, q_norm,
        kv_norm, wqb, wqb_s, cos_sin, pos, slot, swa, swa_idx, swa_lens, topk, topk_lens,
        q, res_out, post_out, comb_out, pre_out, kt, klen, rank, world_size, peers, symmetric):
    rows = res_in.shape[0]
    assert world_size == TP and 1 <= rows <= MAX_TOKENS
    assert symmetric.numel() >= SYMMETRIC_BYTES
    dev = res_in.device
    state = torch.zeros(EPOCH_BYTES + scratch_bytes(rows, TP, **route_at(rows)), dtype=torch.uint8, device=dev)
    epoch, scratch = state.data_ptr(), state.data_ptr() + EPOCH_BYTES
    p = lambda t: t.data_ptr()  # noqa: E731
    _launcher(rows)(
        p(res_in), p(pend), p(post_in), p(comb_in), p(pre_in), p(hc_fn), p(hc_scale), p(hc_base), p(attn_norm),
        p(res_out),
        0, HIDDEN,  # aux_out (DSpark's capture: none), its row stride
        p(post_out), p(comb_out), p(pre_out),
        p(wqkv), p(wqkv_s), p(q_norm), p(kv_norm), p(wqb), p(wqb_s), p(cos_sin), p(pos),
        p(slot), 1,  # int32 slots: one word a slot
        p(swa), swa.stride(0), SWA_BLOCK, p(swa_idx), p(swa_lens), p(topk), p(topk_lens), p(kt), p(klen),
        p(q),
        0, 0,  # normed_out, qr_out (an index source's)
        0, 0, 0, 0,  # iq_w, iq_ws, iw_w, iw_scale (the index query: not built)
        cos_sin.shape[0],
        0, 0, 0,  # iq_out, iqs_out, iw_out
        scratch, p(symmetric), p(peers), int(rank), epoch,
        stream=torch.cuda.current_stream(),
    )
