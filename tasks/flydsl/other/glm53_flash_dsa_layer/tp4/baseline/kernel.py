# SPDX-License-Identifier: Apache-2.0
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""One GLM-5.3-Flash sparse-attention (DSA) decoder layer on one tensor-parallel rank: the
sgl-amd-mono monokernel (``flash_kernel.py``, vendored from sgl-project/sgl-amd-mono at
52486703fbf27e77f05b573e4d931d4eec87118d) built as ``FlashBuild(dsa=True)``, launched the way
``glm53_flash/gfx950/runner.py`` (``FlashLayerOp.launch``) launches a sparse-attention layer.

Host code only takes views and launches: the persistent kernel (256 CTAs x 512 threads) runs the
whole layer, its cross-CTA mailboxes in the framework's symmetric buffer past the TP exchange
regions (production keeps them in a separate uncached buffer of the same kind), and its TP
exchanges through ``peers``.
"""
import functools

import torch

from .config import GLOBAL_INTER, HC_MULT, HIDDEN, N_EXPERTS, FlashShard
from .flash_kernel import FlashBuild, build_flash_kernel
from .layout import ROW_WORDS

# kpool pools a row scores at most: the served context's (1M tokens), whole 256-pool score tasks
MAX_POOLS = 262144
# the launch's ordinal among its kind's launches of a step, and those launches: one layer a call
LAYER, LAUNCHES_PER_STEP = 0, 1
# the scratch mailboxes start at this alignment past the symmetric exchange regions
SCRATCH_ALIGN = 4096
# SGLang's DSA cache geometry (runner.py): 64-token MLA blocks (page-table entries), index
# pages of 64 pools (one per 256 tokens, four page-table entries)
BLOCK_TOKENS, INDEX_PAGE_POOLS = 64, 64
IDX_BT_GROUP = INDEX_PAGE_POOLS * 4 // BLOCK_TOKENS


@functools.lru_cache(maxsize=None)
def _launcher(rows: int, tp: int):
    """The compiled launch of a ``rows``-row sparse-attention layer at TP ``tp``, with the byte
    offset of its scratch in the symmetric buffer and the bytes it needs there."""
    key = FlashBuild(
        rows=rows,
        tp=tp,
        launches_per_step=LAUNCHES_PER_STEP,
        inter=GLOBAL_INTER // tp,
        dsa=True,
        max_pools=MAX_POOLS,
        # SGLang's gfx950 block-FP8 linear method leaves the projections AITER-shuffled
        qkv_shuffled=True,
        qb_shuffled=True,
        o_shuffled=True,
    )
    launch, sc, sy = build_flash_kernel(key)
    offset = -(-sy["_bytes"] // SCRATCH_ALIGN) * SCRATCH_ALIGN
    return launch, offset, offset + sc["_bytes"]


def run(*, x, residual, post, comb, hc_attn_fn, hc_attn_scale, hc_attn_base, input_norm, w_qkv_a, s_qkv_a,
        w_idx, w_kpool_gate, q_a_norm, kv_a_norm, k_norm_w, k_norm_b, kpool_ape, w_q_b, s_q_b, w_index_q,
        w_uk, w_kv_b, w_o, s_o, kv_cache, index_cache, kpool_tail_k, kpool_tail_gate, row_table, page_table,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base, post_attn_norm, router_w, router_bias, w13, w13_scale, w2,
        w2_scale, guard, x_out, res_out, post_out, comb_out, rank, world_size, peers, symmetric):
    rows = x.shape[0]
    FlashShard(world_size)  # TP 2, 4 or 8
    assert w13.shape[0] == N_EXPERTS + 1 and x.shape[1] == HIDDEN and residual.shape[1] == HC_MULT
    assert row_table.shape == (rows, ROW_WORDS)
    launch, scratch_offset, needed = _launcher(rows, world_size)
    assert symmetric.numel() >= needed, "the symmetric buffer cannot hold the exchange regions and scratch"
    p = lambda t: t.data_ptr()  # noqa: E731
    # the shared expert is the fused slot N_EXPERTS of the routed tensors
    shared = [p(t[N_EXPERTS]) for t in (w13, w13_scale, w2, w2_scale)]
    # the sparse-attention argument table (layout.DSA_TAB order): weights and caches, then geometry
    table = [
        p(w_qkv_a), p(s_qkv_a), p(w_idx), p(w_kpool_gate), p(q_a_norm), p(kv_a_norm), p(k_norm_w),
        p(k_norm_b), p(kpool_ape), p(w_q_b), p(s_q_b), p(w_index_q), p(w_uk), p(w_kv_b), p(w_o), p(s_o),
        p(kv_cache), p(index_cache), p(kpool_tail_k), p(kpool_tail_gate),
        IDX_BT_GROUP,
        BLOCK_TOKENS * kv_cache.stride(0),
        BLOCK_TOKENS,
        kv_cache.stride(0),
        index_cache.stride(0),
        INDEX_PAGE_POOLS,
        kpool_tail_k.stride(0),
    ]
    sym = p(symmetric)
    launch(
        # the previous layer's output, its FFN post owed (the attention seam reads x plainly)
        p(x), p(residual), p(post), p(comb),
        # the FFN side: its mHC, its norm, the outputs, the router and the experts
        p(hc_ffn_fn), p(hc_ffn_scale), p(hc_ffn_base), p(post_attn_norm),
        p(res_out), p(post_out), p(comb_out), p(x_out),
        p(router_w), p(router_bias), p(w13), p(w13_scale), p(w2), p(w2_scale), *shared,
        # the attention-side mHC; the KDA arguments are unused by a sparse-attention build
        p(hc_attn_fn), p(hc_attn_scale), p(hc_attn_base), p(input_norm),
        *[0] * 17,
        # dsa_tab (unused: the table entries follow), the table, the row table, the page table
        0, *table, p(row_table), p(page_table), page_table.stride(0),
        sym + scratch_offset, sym, p(peers), p(guard), rank, LAYER,
        torch.cuda.current_stream(),
    )
