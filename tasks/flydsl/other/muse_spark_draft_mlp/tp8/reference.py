# SPDX-License-Identifier: Apache-2.0
"""Float64 reference of the Muse Spark DFlash2 draft's D-MLP launch at TP8 (logical: the whole MLP).

The SGLang draft layer's MLP half (``sglang.srt.models.dflash``: the fused add + RMSNorm, the grouped
two-tap convolution around the MLP, gate / up, silu(g) u, down, the finish convolution), as
sgl-amd-mono's test states it (test/kernels/test_muse_spark_dmlp.py ``_reference``), computed in float64
except where the layer materializes a bfloat16 tensor that the kernel also rounds:

  res'   = bf16(res + attn_out)                              (the fused add's residual, an output)
  h      = bf16(res' rstd(res') w_post)                      (eps 1e-5)
  delta  = h w_kp^T, as [rows, side, tap, group]             (the kernel projection: 2 x 2 x 512)
  conv_s(x)[t, c] = (base[s, 0, c] + delta[t, s, 0, c // 16]) x[t, c]
                  + (base[s, 1, c] + delta[t, s, 1, c // 16]) x[t - 1, c]   (t - 1 in t's 8-row block only)
  hc     = bf16(conv_0(h))
  g, u   = bf16(hc w_gu[:28672]^T), bf16(hc w_gu[28672:]^T)
  a      = bf16(bf16(silu(g)) u)
  d      = a w_down^T                                         (at TP8 the all-reduce of the ranks' K shards)
  out    = conv_1(d)                                          (bf16)

Returned bfloat16 outputs (res_out, out) are the float64 values before their final rounding.
"""
import torch

INTER = 28672
GROUP = 16
GROUPS = 512
TAPS = 2
SIDES = 2
BLOCK = 8  # rows a request: the convolution's tap 1 reads the previous row of the block
EPS = 1e-5
CHUNK = 4096  # weight rows converted to float64 at a time


def _bf(t):
    """Round to bfloat16 (nearest even) and back to float64."""
    return t.to(torch.bfloat16).to(torch.float64)


def _matmul_t(x, w):
    """x [rows, k] float64 times w [n, k]^T, converting w to float64 a chunk of rows at a time."""
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.float64, device=x.device)
    for start in range(0, w.shape[0], CHUNK):
        out[:, start:start + CHUNK] = x @ w[start:start + CHUNK].to(torch.float64).t()
    return out


def _conv(x, delta, base):
    """The grouped two-tap convolution within 8-row blocks: x [rows, H], delta [rows, TAPS, GROUPS],
    base [TAPS, H] (one side)."""
    rows, hidden = x.shape
    coef = base.to(torch.float64)[None] + delta.repeat_interleave(GROUP, dim=-1)  # [rows, TAPS, H]
    prev = torch.zeros_like(x)
    prev[1:] = x[:-1]
    prev[torch.arange(rows, device=x.device) % BLOCK == 0] = 0.0
    return coef[:, 0] * x + coef[:, 1] * prev


def reference(*, res, attn_out, w_post, w_kp, base, w_gu, w_down, guard):
    del guard  # the launch's epoch / deadline words: no effect on the result
    f64 = torch.float64
    rows = res.shape[0]
    res2 = res.to(f64) + attn_out.to(f64)
    r = _bf(res2)
    h = _bf(r * torch.rsqrt(r.square().mean(-1, keepdim=True) + EPS) * w_post.to(f64))
    delta = (h @ w_kp.to(f64).t()).reshape(rows, SIDES, TAPS, GROUPS)
    hc = _bf(_conv(h, delta[:, 0], base[0]))
    gu = _matmul_t(hc, w_gu)
    g, u = _bf(gu[:, :INTER]), _bf(gu[:, INTER:])
    a = _bf(_bf(g * torch.sigmoid(g)) * u)
    d = _matmul_t(a, w_down)
    out = _conv(d, delta[:, 1], base[1])
    return {"res_out": res2, "out": out}
