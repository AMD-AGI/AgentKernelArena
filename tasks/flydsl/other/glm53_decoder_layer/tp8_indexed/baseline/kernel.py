# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# run() is the host side of sgl-amd-mono 52486703 sgl_amd_mono/glm53/gfx950/runner.py (Glm53Runtime and
# Glm53LayerOp.launch for an indexer layer), with the runtime's own buffers replaced by the task's inputs and
# the framework's symmetric memory.
"""One GLM-5.3 indexed decoder layer on one tensor-parallel rank: a single glm5_monokernel launch.

Build flags are the ones SGLang's GLM-5.3 spec passes for an indexer layer (gfx950/runner.py): block-FP8
attention weights read in place in AITER's (16, 16) layout, the ATOM bf16 MLA cache, SGLang's paged index cache
(64-token pages of 132-byte tokens, 16 x 16 tiles), FP8 experts, uv scale rows 128, one launch a step.

Persistent-kernel state: the guard block is the read-only ``step_guard`` input (epoch 1, so every mailbox
tag is 2, and a 600 s deadline); the peer mailboxes are the framework's symmetric buffer (zeroed before every
call) and the kernel's hand-off scratch is the part of it past the peer regions (the production runtime keeps
both in uncached memory too). ``indices`` points at the index cache and ``out_indices`` at ``topk``.
"""
import functools

import torch

from .glm5_mono import build_glm5_monokernel
from .layout import layout

TOPK = 2048
INDEX_MAX_SEQ = 1 << 20      # keys a row scores at most: the served context of GLM-5.3 (1M positions)
INDEX_PAGE_TOKENS = 64
INDEX_PAGE_BYTES = INDEX_PAGE_TOKENS * (128 + 4)
GLOBAL_HEADS = 64
GLOBAL_INTER = 2048
SCRATCH_ALIGN = 4096


@functools.lru_cache(maxsize=8)
def _build(rows, world_size):
    heads, inter = GLOBAL_HEADS // world_size, GLOBAL_INTER // world_size
    _, sy = layout(rows, heads, world_size, TOPK, True, INDEX_MAX_SEQ, inter=inter, output_heads=heads)
    launch = build_glm5_monokernel(
        rows,
        heads,
        world_size,
        TOPK,
        launches_per_step=1,
        with_indexer=True,
        index_max_seq=INDEX_MAX_SEQ,
        attention_weight="fp8_block128",
        kv_cache_layout="atom",
        kv_cache_dtype="bf16",
        inter=inter,
        output_heads=heads,
        uv_scale_rows=128,
        index_paged=True,
        index_block_size=INDEX_PAGE_TOKENS,
        index_block_bytes=INDEX_PAGE_BYTES,
        index_shuffled=True,
        fp8_shuffled=True,
        stop=True,
    )
    scratch_offset = (sy["_bytes"] + SCRATCH_ALIGN - 1) // SCRATCH_ALIGN * SCRATCH_ALIGN
    return launch, scratch_offset


def run(*, hidden, positions, slots, indptr, req_ids, page_table, kv_cache, index_cache, rope_cos, rope_sin,
        g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o, s_o, w_ik, s_ik,
        w_iw, w_iq, s_iq, g_ik, b_ik, w_r, bias, w_ug, s_ug, w_dn, s_dn, step_guard, out, topk, rank,
        world_size, peers, symmetric):
    launch, scratch_offset = _build(hidden.shape[0], world_size)
    p = lambda t: t.data_ptr()  # noqa: E731
    launch(
        p(hidden),
        p(out),
        p(positions),          # cur_pos: read only by the non-ATOM cache layouts
        p(positions),
        p(slots),
        p(indptr),
        p(kv_cache),
        p(kv_cache),           # the ATOM layout keeps k_pe in the latent row
        p(index_cache),
        p(rope_cos),
        p(rope_sin),
        p(g_in),
        p(g_q),
        p(g_kv),
        p(g_post),
        p(w_qkv_a),
        p(s_qkv_a),
        p(w_q_b),
        p(s_q_b),
        p(w_uk),
        p(s_uk),
        p(w_uv),
        p(s_uv),
        p(w_o),
        p(s_o),
        p(w_r),
        p(bias),
        p(w_ug),
        p(s_ug),
        p(w_dn),
        p(s_dn),
        p(symmetric) + scratch_offset,
        p(symmetric),
        p(peers),
        0,                     # the pointer table: only the NextN entry and the timeline use it
        p(step_guard),
        p(page_table),
        p(req_ids),
        p(topk),
        p(w_ik),
        p(s_ik),
        p(w_iw),
        p(w_iq),
        p(s_iq),
        p(g_ik),
        p(b_ik),
        rank,
        0,                     # layer: the launch's ordinal among its kind's launches in a step
        page_table.stride(0),
        torch.cuda.current_stream(),
    )
