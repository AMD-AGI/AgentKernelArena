# SPDX-License-Identifier: Apache-2.0
"""Float64 reference of the Muse Spark K2 launch at TP8 (logical: all 128 query heads, the whole o_proj).

The math of the ROCm model's ops that K2 replaces, as sgl-amd-mono's test states it
(test/kernels/test_muse_spark_k2.py ``_reference``), computed in float64 except where the model
materializes a bfloat16 tensor:

  og      = bf16(o rstd(o) sigmoid(g)) per 64-wide head (eps 1e-5; the fused output norm-gate)
  attn    = bf16(og w_o^T)                       (o_proj; at TP8 the all-reduce of the ranks' K shards)
  nb      = bf16(attn rstd(attn))                (the weightless post norm, eps 1e-8)
  res'    = alpha res + beta nb                  (the gated residual stream, fp32, in place)
  x       = bf16(res' rstd(res') pre_w)          (the pre-FFN norm, eps 1e-5)
  s       = sigmoid(0.17677669529663687 x (router_hi + router_lo)^T)
  ids     = the 8 largest s + expert_bias, best first (a tie to the lower expert)
  scores  = s[ids] / (sum s[ids] + 1e-15)
  pooled  = bf16(x w_pool^T)                     (the pooler, 8192 -> 4096)
  xin     = pooled rstd(pooled) xn_w             (the pre-expert norm, eps 1e-5), a bfloat16 tensor

Returned bfloat16 outputs (xin) are the float64 values before their final rounding.
"""
import torch

HEAD_DIM = 64
TOP_K = 8
RMS_EPS = 1e-5
POST_EPS = 1e-8
ROUTE_EPS = 1e-15
ROUTE_SCALE = 1.0
ROUTER_MULT = 0.17677669529663687


def _bf(t):
    """Round to bfloat16 (nearest even) and back to float64."""
    return t.to(torch.bfloat16).to(torch.float64)


def _rstd(t, eps):
    return torch.rsqrt(t.square().mean(-1, keepdim=True) + eps)


def reference(*, o, g, res, w_o, alpha, beta, pre_w, router_hi, router_lo, expert_bias, w_pool, xn_w, guard):
    del guard  # the launch's epoch / deadline words: no effect on the result
    f64 = torch.float64
    rows = o.shape[0]
    ob = o.to(f64).reshape(rows, -1, HEAD_DIM)
    gate = torch.sigmoid(g.to(f64).reshape(rows, -1, HEAD_DIM))
    og = _bf(ob * _rstd(ob, RMS_EPS) * gate).reshape(rows, -1)
    attn = _bf(og @ w_o.to(f64).t())
    nb = _bf(attn * _rstd(attn, POST_EPS))
    new = alpha.to(f64) * res.to(f64) + beta.to(f64) * nb
    x = _bf(new * _rstd(new, RMS_EPS) * pre_w.to(f64))
    router = router_hi.to(f64) + router_lo.to(f64)  # [EXPERTS, HIDDEN]: the fp32 router's bf16 hi + lo split
    sig = torch.sigmoid((x @ router.t()) * ROUTER_MULT)
    biased = sig + expert_bias.to(f64)
    ids = torch.argsort(biased, dim=-1, descending=True, stable=True)[:, :TOP_K]
    picked = sig.gather(1, ids)
    scores = picked / (picked.sum(-1, keepdim=True) + ROUTE_EPS) * ROUTE_SCALE
    pooled = _bf(x @ w_pool.to(f64).t())
    xin = pooled * _rstd(pooled, RMS_EPS) * xn_w.to(f64)
    return {"ids": ids.to(torch.int32), "scores": scores, "xin": xin, "res": new}
