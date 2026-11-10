# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""High-precision reference of one MiniMax-M3 sparse MoE layer on a tensor-parallel-4 group, used only for
correctness. It receives the logical inputs (inputs.py: rank-specific tensors stacked on a leading TP axis,
caches token-major, a block-table row per request) and returns the logical outputs; inputs.rank_expected maps
them to each rank's tensors.

The numerics are in m3_numerics.py (float64, the kernel's dataflow roundings applied exactly). Returned values
are taken before their final rounding: h_out = ar + res, h_mid = bf16(attention all-reduce) + bf16(h_out),
ar_out = the rank-ordered sum of the ranks' bf16 MoE partial rows, and the caches' newly written slots before
their E4M3 rounding (old slots keep their E4M3 values).
"""
from m3_numerics import layer


def reference(*, ar, res, positions, slot_mapping, block_table, seq_lens, g_in, w_qkv, s_qkv, g_q, g_k,
              g_iq, g_ik, cos_sin, k_scale, v_scale, w_o, s_o, g_post, w_gate, bias, w13, s13, w2, s2, k_cache,
              v_cache, index_cache, **_reserved):
    inp = {"ar": ar, "res": res, "positions": positions, "slot_mapping": slot_mapping, "block_table": block_table,
           "seq_lens": seq_lens, "g_in": g_in, "w_qkv": w_qkv, "s_qkv": s_qkv, "g_q": g_q,
           "g_k": g_k, "g_iq": g_iq, "g_ik": g_ik, "cos_sin": cos_sin, "k_scale": k_scale, "v_scale": v_scale,
           "w_o": w_o, "s_o": s_o, "g_post": g_post, "w_gate": w_gate, "bias": bias, "w13": w13, "s13": s13,
           "w2": w2, "s2": s2, "k_cache": k_cache, "v_cache": v_cache, "index_cache": index_cache}
    att, _, ar_out = layer(inp)
    return {"h_out": att["acc"], "h_mid": att["h_mid"], "ar_out": ar_out, **att["cache_out"]}
