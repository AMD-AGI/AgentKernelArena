# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""float64 reference of one Kimi-K3.1 MLA (DSA) layer's projection front on one TP8 rank (12 of the
96 query heads), used only for correctness.

* AttnRes seam: the sources of token t are the bank rows blocks[t, 0 .. nblocks - 1] and the prefix
  prefix[t]; score_s = sum_h v_s[h] attn_res_norm[h] attn_res_query[h] / rms(v_s) with
  rms(v) = sqrt(mean(v^2) + 1e-5); mixed = sum_s softmax(score)_s v_s;
  x = mixed / sqrt(mean(mixed^2) + 1e-5) * input_norm.
* lat = x @ w_qkv_a^T (2112: q latent 1536 | kv latent 512 | rope key 64); latent = lat[:, 1536:].
* gate = x @ w_gate^T; index_k = x @ w_index_k^T; index_weights = x @ w_index_weights^T.
* q_lora = lat_q / sqrt(mean(lat_q^2) + 1e-6) * q_a_norm; k_nope = lat_kv / sqrt(mean(lat_kv^2) +
  1e-6) * kv_a_norm.
* q = q_lora @ w_q_b^T (12 heads x (128 nope | 64 rope)); index_q = q_lora @ w_index_q^T.
* q_nope_absorbed[t, h, c] = sum_n q[t, 192 h + n] w_kc[h, c, n] (n < 128): the MLA absorb.
"""
import torch

HIDDEN = 7168
QL, KV, ROPE = 1536, 512, 64
NHQ, QHD, QN = 12, 192, 128
EPS = 1e-5  # AttnRes norm and input_layernorm
LAT_EPS = 1e-6  # q_a_layernorm and kv_a_layernorm
CHUNK = 1024  # weight rows converted to float64 at a time (bounds the reference's memory)


def _rms(v, eps):
    return torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)


def _linear(x, w):
    """x @ w^T in float64, converting w a block of rows at a time."""
    return torch.cat([x @ w[i:i + CHUNK].double().T for i in range(0, w.shape[0], CHUNK)], dim=-1)


def seam(prefix, blocks, attn_res_norm, attn_res_query, input_norm):
    """The attention-side AttnRes seam and input_layernorm -> x [rows, 7168] (float64)."""
    src = torch.cat([blocks.double(), prefix.double().unsqueeze(1)], dim=1)  # [rows, nblocks + 1, H]
    wq = attn_res_norm.double() * attn_res_query.double()
    scores = (src * wq).sum(-1) * _rms(src, EPS).squeeze(-1)
    mixed = (torch.softmax(scores, dim=-1).unsqueeze(-1) * src).sum(1)
    return mixed * _rms(mixed, EPS) * input_norm.double()


def reference(*, prefix, blocks, attn_res_norm, attn_res_query, input_norm, w_qkv_a, w_gate, w_index_k,
              w_index_weights, w_q_b, w_index_q, q_a_norm, kv_a_norm, w_kc):
    rows = prefix.shape[0]
    x = seam(prefix, blocks, attn_res_norm, attn_res_query, input_norm)
    lat = _linear(x, w_qkv_a)
    gate = _linear(x, w_gate)
    index_k = _linear(x, w_index_k)
    index_weights = _linear(x, w_index_weights)
    lat_q, lat_kv = lat[:, :QL], lat[:, QL:QL + KV]
    q_lora = lat_q * _rms(lat_q, LAT_EPS) * q_a_norm.double()
    k_nope = lat_kv * _rms(lat_kv, LAT_EPS) * kv_a_norm.double()
    q = _linear(q_lora, w_q_b)
    index_q = _linear(q_lora, w_index_q)
    q_nope = q.reshape(rows, NHQ, QHD)[:, :, :QN]
    absorbed = torch.einsum("thn,hcn->thc", q_nope, w_kc.double())
    return q, lat[:, QL:], k_nope, gate, index_q, index_k, index_weights, absorbed
