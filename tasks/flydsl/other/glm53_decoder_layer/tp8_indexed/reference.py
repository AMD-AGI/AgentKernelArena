# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Float64 reference of one GLM-5.3 indexed decoder layer over the whole (untensor-parallel) layer.

Per row s (a decode token or an MTP-verify draft token at ``positions[s]``):

  xn      = RMSNorm(hidden[s]) * g_in                                   (eps 1e-5)
  q_a | kv_a | k_pe = xn @ W_qkv_a^T                                    (block-FP8 weights, dequantized)
  q       = RMSNorm(q_a) * g_q @ W_q_b^T -> 64 heads x (192 nope | 64 rope, interleaved RoPE at the position)
  kv row  = [RMSNorm(kv_a) * g_kv | RoPE(k_pe)]  -> written to the MLA cache at slots[s]
  index   q_i = Hadamard(RoPE_64(q_an @ W_iq^T)) [32 x 128], w_i = xn @ W_iw^T [32],
          k_i = Hadamard(RoPE_64(LayerNorm(xn @ W_ik^T))) quantized to E4M3 with a power-of-two scale and
          written to the paged index cache at slots[s]
  score(key) = scale_key * sum_h w_i[h] * relu(q_i[h] . codes_key) over the row's keys 0 .. positions[s]
          (page_table[s] maps a key to its cache slot); topk[s] = the 2048 best keys' slots
  attention: 64 absorbed MLA heads over the 2048 selected cache rows (bf16; this launch's rows as written),
          softmax scale 1/16, o = W_UV(softmax . latent), attn = o @ W_o^T, a = hidden + attn
  MoE     z = RMSNorm(a) * g_post; router sigmoid(z @ W_r^T), top-8 of sigmoid + bias, weights
          sigmoid / sum * 2.5, plus the shared expert (weight 1); experts on per-128 E4M3 z, SiLU(gate) * up,
          per-128 E4M3 mid, down projection; out = a + sum of weighted expert outputs.

Returns the outputs (out, topk) and the mutable inputs' final values (kv_cache, index_cache). The math is in
glm53_math.py (shared with the input generator, which uses it to plant clear decision margins).
"""
from glm53_math import layer


def reference(*, hidden, positions, slots, indptr, req_ids, page_table, kv_cache, index_cache, rope_cos,
              rope_sin, g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o,
              s_o, w_ik, s_ik, w_iw, w_iq, s_iq, g_ik, b_ik, w_r, bias, w_ug, s_ug, w_dn, s_dn, step_guard):
    inputs = dict(hidden=hidden, positions=positions, slots=slots, page_table=page_table, kv_cache=kv_cache,
                  index_cache=index_cache, rope_cos=rope_cos, rope_sin=rope_sin, g_in=g_in, g_q=g_q,
                  g_kv=g_kv, g_post=g_post, w_qkv_a=w_qkv_a, s_qkv_a=s_qkv_a, w_q_b=w_q_b, s_q_b=s_q_b,
                  w_uk=w_uk, s_uk=s_uk, w_uv=w_uv, s_uv=s_uv, w_o=w_o, s_o=s_o, w_ik=w_ik, s_ik=s_ik,
                  w_iw=w_iw, w_iq=w_iq, s_iq=s_iq, g_ik=g_ik, b_ik=b_ik, w_r=w_r, bias=bias, w_ug=w_ug,
                  s_ug=s_ug, w_dn=w_dn, s_dn=s_dn)
    out, topk, kv, cache = layer(inputs)
    return {"out": out, "topk": topk.int(), "kv_cache": kv, "index_cache": cache}
