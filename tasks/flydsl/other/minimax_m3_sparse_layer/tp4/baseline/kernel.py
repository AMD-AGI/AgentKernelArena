# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""The MiniMax-M3 sparse MoE layer monokernel on one rank of a TP4 group: the entry point of this baseline.

run() is sgl-amd-mono's ``M3MonoRunner.launch`` for one sparse layer (sgl_amd_mono/minimax_m3/gfx950/runner.py
at 52486703) without SGLang: one launch of ``post_attn_kernel`` (K1 of pre_attn.py fused in front of K4), built
per row count with MiniMax-M3's constants (spec._build: sm_scale 128^-0.5, eps 1e-6, routed scaling 2, shared
weight 1, swiglu limit 7, selection init 0 / local 1, one index q head a rank). The persistent-kernel state the
runner keeps across steps is made fresh inside every call instead: the K1 / K4 mailbox scratch is zeroed, the
epoch / guard words hold step 0 and a 10 s wait deadline, so the fixed mailbox tag of every call is safe; the
framework zeroes the symmetric peer buffer (the all-reduce mailboxes and the guard's peer flags) before every
call. The per-request block table is expanded to the kernel's per-row table (the row expansion of
metadata.prepare_token_rows). Every computation runs in the FlyDSL kernel.
"""

import functools

import torch

from . import guard
from .config import HEAD_DIM, LOCAL_Q_HEADS, MAX_TOKENS, ONE_INDEX_HEAD, TP
from .layout import SCRATCH_BYTES, sym_layout
from .post_attn import build_post_attn_kernel
from .pre_attn import SCRATCH_BYTES as K1_SCRATCH_BYTES

# MiniMax-M3's build constants (sgl_amd_mono/minimax_m3/spec.py _build, test_minimax_m3_compile.BUILD)
BUILD = dict(
    npes=TP,
    sm_scale=HEAD_DIM ** -0.5,
    eps=1e-6,
    route_scale=2.0,
    shared_weight=1.0,
    swiglu_limit=7.0,
    init_blocks=0,
    local_blocks=1,
    heads=ONE_INDEX_HEAD,
)
SYMMETRIC_BYTES = sym_layout(TP)["_bytes"] + guard.FLAG_BYTES
DEADLINE_TICKS = 10 * guard.CLOCK_HZ  # every wait's deadline (runner.set_deadline), < 2^31 ticks
LAYER = 0  # the layer slot of the mailbox tag


@functools.lru_cache(maxsize=None)
def _launcher(rows):
    return build_post_attn_kernel(tokens=rows, **BUILD)


def run(*, ar, res, positions, slot_mapping, block_table, seq_lens, g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik,
        cos_sin, k_scale, v_scale, w_o, s_o, g_post, w_gate, bias, w13, s13, w2, s2, k_cache, v_cache, index_cache,
        h_out, h_mid, ar_out, rank, world_size, peers, symmetric):
    rows, requests = seq_lens.shape[0], block_table.shape[0]
    assert world_size == TP and 1 <= rows <= MAX_TOKENS and rows % requests == 0
    assert symmetric.numel() >= SYMMETRIC_BYTES
    q_len = rows // requests
    dev = ar.device
    # the kernel's per-row block table: a request's row repeated for its q_len rows
    width = block_table.shape[1]
    rows_table = block_table[:, None, :].expand(requests, q_len, width).contiguous().view(rows, width)
    q = torch.empty(rows, LOCAL_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=dev)
    iq = torch.empty(rows, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    scratch1 = torch.zeros(K1_SCRATCH_BYTES, dtype=torch.uint8, device=dev)
    scratch4 = torch.zeros(SCRATCH_BYTES, dtype=torch.uint8, device=dev)
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    epoch[guard.GUARD_DEADLINE:guard.GUARD_DEADLINE + 1].fill_(DEADLINE_TICKS)
    _launcher(rows)(
        h_out.data_ptr(), q.data_ptr(), rows_table.data_ptr(), seq_lens.data_ptr(),
        k_cache.data_ptr(), v_cache.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(),
        w_o.data_ptr(), s_o.data_ptr(), g_post.data_ptr(), w_gate.data_ptr(), bias.data_ptr(),
        w13.data_ptr(), s13.data_ptr(), w2.data_ptr(), s2.data_ptr(),
        h_mid.data_ptr(), ar_out.data_ptr(), scratch4.data_ptr(), symmetric.data_ptr(), peers.data_ptr(),
        epoch.data_ptr(), int(rank), LAYER, width, q_len,
        # K1's pointers: ar_in, g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik, cos_sin, index_cache, iq_out, scratch1
        ar.data_ptr(), g_in.data_ptr(), w_qkv.data_ptr(), s_qkv.data_ptr(), g_q.data_ptr(), g_k.data_ptr(),
        g_iq.data_ptr(), g_ik.data_ptr(), cos_sin.data_ptr(), index_cache.data_ptr(), iq.data_ptr(),
        scratch1.data_ptr(),
        positions.data_ptr(), slot_mapping.data_ptr(), res.data_ptr(),
        torch.cuda.current_stream(),
    )
