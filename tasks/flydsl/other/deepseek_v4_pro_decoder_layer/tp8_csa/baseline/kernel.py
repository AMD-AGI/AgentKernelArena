# SPDX-License-Identifier: Apache-2.0
# Launch wrapper for AgentKernelArena around sgl-amd-mono's DeepSeek-V4 layer kernel (52486703,
# sgl_amd_mono/deepseek_v4/gfx950/runner.py Variant / Dsv4MonoRunner.launch, adapted from ROCm/ATOM#2559,
# MIT, Copyright (C) 2026 Advanced Micro Devices, Inc.).
"""One rank's DeepSeek-V4-Pro CSA decoder layer in one persistent FlyDSL launch (TP8 decode).

``run`` builds (once per row count) the kernel for this rank's shard and launches it on the current
stream with the task's tensors as raw addresses, in the argument order of ``Dsv4MonoRunner.launch``:

* the framework's ``symmetric`` buffer holds, in order, the two TP all-reduce regions the peers push
  into (``layout()``'s ``sym``: attention, then FFN), the guard's error flags (256 bytes) and the
  kernel's mailbox scratch. The evaluator zeroes it before every call, so the fixed launch tag of the
  read-only ``epoch`` buffer (word 0 = step, words 1-2 = the wait deadline) never meets a stale pair;
  production keeps the scratch in ordinary device memory and moves the step on instead (and scrubs);
* ``peers`` is the table of every rank's symmetric base the kernel's waves push to;
* the compressor states are SGLang's [kv | score] rows: the score pointers are the kv pointers plus
  one half-row; the layer is a scored (not hash-routed) CSA layer, so tok_ids / tid2eid are null.
"""
import functools

import torch

from .dsv4_layer import build_dsv4_kernel
from .dsv4_layout import layout
from .mono_guard import FLAG_BYTES

HEADS = 16  # local: 128 / 8
WINDOW = 128
HEAD_DIM = 512
INDEX_HEAD_DIM = 128
MAX_SEQ = 1 << 20  # SGLang's DeepSeek-V4 MAX_CONTEXT: sizes the indexer's compressed slots
COMPRESS_RATIO = 4
C_RING = 8
LAYER = 10  # a CSA layer past the hash-routed layers 0-2: only its launch tag depends on it
DIMS = {
    "hidden": 7168, "q_lora": 1536, "head_dim": HEAD_DIM, "o_groups": 2, "o_lora": 1024, "hc_mult": 4,
    "compress_ratio": COMPRESS_RATIO, "n_keys": 1152, "c_coff": 2, "index_head_dim": INDEX_HEAD_DIM,
    "index_heads": 64, "max_seq": MAX_SEQ, "index_topk": 1024, "indexer_hadamard": True,
    "n_experts": 384, "top_k": 6, "inter": 384,
}


@functools.lru_cache(maxsize=None)
def _variant(rows, world_size):
    """The rows-wide launch of this shard and the byte offset of its scratch in ``symmetric``."""
    _scratch, sym = layout(rows, HEADS, world_size, WINDOW, **DIMS)
    launch = build_dsv4_kernel(
        rows, HEADS, world_size, WINDOW,
        scale=HEAD_DIM ** -0.5, spec_steps=0, swiglu_limit=10.0, hc_sinkhorn_iters=20, hc_eps=1e-6,
        window_rows=WINDOW + MAX_SEQ // COMPRESS_RATIO, c_ring=C_RING, fault_site=0, fault_after=0,
        **DIMS,
    )
    return launch, sym["_bytes"] + FLAG_BYTES


def run(*, hidden, positions, kv_cache, kv_rope, dest_rows, indices, rope_cos, rope_sin, g_in, g_q, g_kv,
        g_post, attn_sink, ape, g_ckv, c_state, i_ape, g_ickv, i_state, i_cache, hc_attn_fn, hc_attn_sb,
        hc_ffn_fn, hc_ffn_sb, w_qkv_a, s_qkv_a, w_qkv_c, w_q_b, s_q_b, w_i_q_b, s_i_q_b, i_w, w_o_a, s_o_a,
        w_o_b, s_o_b, w_r, bias, w_ug, s_ug, w_dn, s_dn, w_sug, s_sug, w_sdn, s_sdn, epoch, state_slots,
        block_tables, i_cache_s, out, rank, world_size, peers, symmetric):
    launch, scratch_offset = _variant(hidden.shape[0], world_size)
    sym = symmetric.data_ptr()
    c_kv = c_state.data_ptr()
    i_kv = i_state.data_ptr()
    launch(
        hidden.data_ptr(), out.data_ptr(), positions.data_ptr(), kv_cache.data_ptr(), kv_rope.data_ptr(),
        dest_rows.data_ptr(), indices.data_ptr(), rope_cos.data_ptr(), rope_sin.data_ptr(),
        g_in.data_ptr(), g_q.data_ptr(), g_kv.data_ptr(), g_post.data_ptr(), attn_sink.data_ptr(),
        ape.data_ptr(), g_ckv.data_ptr(), c_kv, c_kv + c_state.shape[-1] // 2 * 4,
        i_ape.data_ptr(), g_ickv.data_ptr(), i_kv, i_kv + i_state.shape[-1] // 2 * 4, i_cache.data_ptr(),
        hc_attn_fn.data_ptr(), hc_attn_sb.data_ptr(), hc_ffn_fn.data_ptr(), hc_ffn_sb.data_ptr(),
        w_qkv_a.data_ptr(), s_qkv_a.data_ptr(), w_qkv_c.data_ptr(),
        w_q_b.data_ptr(), s_q_b.data_ptr(), w_i_q_b.data_ptr(), s_i_q_b.data_ptr(), i_w.data_ptr(),
        w_o_a.data_ptr(), s_o_a.data_ptr(), w_o_b.data_ptr(), s_o_b.data_ptr(),
        w_r.data_ptr(), bias.data_ptr(),
        w_ug.data_ptr(), s_ug.data_ptr(), w_dn.data_ptr(), s_dn.data_ptr(),
        w_sug.data_ptr(), s_sug.data_ptr(), w_sdn.data_ptr(), s_sdn.data_ptr(),
        sym + scratch_offset, sym, peers.data_ptr(), epoch.data_ptr(),
        state_slots.data_ptr(), state_slots.data_ptr(),
        0, 0,  # tok_ids, tid2eid: a scored layer
        block_tables.data_ptr(), i_cache_s.data_ptr(),
        rank, LAYER, c_state.stride(0), i_state.stride(0), 0,  # rank, layer, st_kv, st_i, use_hash
        block_tables.stride(0), 256 // COMPRESS_RATIO, 0,  # bt_stride, env_rows, run_pos (no verify rows)
        stream=torch.cuda.current_stream(),
    )
