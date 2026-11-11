# SPDX-License-Identifier: Apache-2.0
"""Float64 reference of the Muse Spark K3 launch at TP8 (logical: every expert whole, every pick).

The compact-A4 chain of the ROCm model that K3 replaces, as sgl-amd-mono's tests state it
(test/kernels/test_muse_spark_k3.py ``_reference`` and its 8-rank form test_muse_spark_k3_tp8.py),
computed in float64 except where the operator defines a quantization or a materialized bfloat16 tensor:

  per (token t, slot j) with expert e = ids[t, j]:
    gu    = MXFP4(xin[t]) . dequant(w13[e], s13[e])^T    (aiter's _mxfp4_quant_op rule on the bf16 input;
                                                         the E2M1 x E8M0 weights decoded exactly)
    mid   = MXFP4(bf16(silu(gate) up))                   (gate = gu[:4096], up = gu[4096:])
    y     = bf16(mid . dequant(w2[e], s2[e])^T)
    v     = bf16(y pen_w)
    out[t] += scores[t, j] v / sqrt(mean(v^2) + 1e-8)   (the finalize's per-slot post norm)
  part   = bf16(out)
  branch = bf16(part w_unpool^T)                         (the unpooler, 4096 -> 8192)
  post   = bf16(branch pfw)
  nb     = bf16(post rstd(post))                         (eps 1e-8)
  res'   = alpha res + beta nb                           (the gated residual stream, fp32, in place)
  attn_out = res' rstd(res') npw                         (the next layer's pre-attention norm, eps 1e-5)

Returned bfloat16 outputs (attn_out) are the float64 values before their final rounding.
"""
import torch

INTER = 4096
POST_EPS = 1e-8
RMS_EPS = 1e-5
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _bf(t):
    """Round to bfloat16 (nearest even) and back to float64."""
    return t.to(torch.bfloat16).to(torch.float64)


def _rstd(t, eps):
    return torch.rsqrt(t.square().mean(-1, keepdim=True) + eps)


def _mx_dequant(x):
    """x (bfloat16 values as float64, [..., N]) through aiter's MXFP4 rule and back: 32-value blocks, the E8M0
    scale from the block max rounded at mantissa bit 21 (unbiased exponent floor(log2) - 2, at least -127),
    E2M1 codes of x * 2^-unbiased rounded half to even and saturating at 6 (the fp32 bit arithmetic of the
    rule, exact on bfloat16 inputs)."""
    shape = x.shape
    xb = x.to(torch.float32).reshape(*shape[:-1], -1, 32)
    amax = xb.abs().amax(-1, keepdim=True)
    bits = (amax.view(torch.int32) + 0x200000) & -0x800000
    unb = (((bits >> 23) & 0xFF) - 129).clamp(min=-127)
    qx = xb * torch.ldexp(torch.ones_like(amax), -unb)
    a = qx.abs()
    ab = a.view(torch.int32)
    den = ((a + 4194304.0).view(torch.int32) - (149 << 23)) & 0xFF
    nrm = ((ab + ((-126 << 23) + 0x1FFFFF) + ((ab >> 22) & 1)) >> 22) & 0xFF
    code = torch.where(a >= 6.0, torch.full_like(den, 7), torch.where(a < 1.0, den, nrm))
    mag = torch.tensor(E2M1, dtype=torch.float64, device=x.device)[code.long()]
    val = torch.where(qx < 0, -mag, mag)
    return (val * torch.ldexp(torch.ones_like(amax, dtype=torch.float64), unb.to(torch.float64))).reshape(shape)


def _dequant_e2m1(packed, scales):
    """Quark MXFP4 [rows, k / 2] codes (low nibble = even k) with E8M0 scales [rows, k / 32] -> float64."""
    rows = packed.shape[0]
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(rows, -1).long()
    lut = torch.tensor(E2M1, dtype=torch.float32, device=packed.device)
    mag = lut[codes & 7]
    val = torch.where((codes & 8) != 0, -mag, mag)
    scale = torch.ldexp(torch.ones(scales.shape, dtype=torch.float32, device=packed.device),
                        scales.to(torch.int32) - 127)
    return (val.reshape(rows, -1, 32) * scale[..., None]).reshape(rows, -1).to(torch.float64)


def reference(*, xin, ids, scores, w13, s13, w2, s2, pen_w, res, pfw, alpha, beta, npw, w_unpool, guard):
    del guard  # the launch's epoch / deadline words: no effect on the result
    f64 = torch.float64
    rows = xin.shape[0]
    xq = _mx_dequant(xin.to(f64))
    out = torch.zeros((rows, w2.shape[1]), dtype=f64, device=xin.device)
    pen = pen_w.to(f64)
    ids = ids.long()
    for e in torch.unique(ids).tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        gu = xq[tok] @ _dequant_e2m1(w13[e], s13[e]).t()
        gt, up = gu[:, :INTER], gu[:, INTER:]
        mid = _mx_dequant(_bf(gt * torch.sigmoid(gt) * up))
        y = _bf(mid @ _dequant_e2m1(w2[e], s2[e]).t())
        v = _bf(y * pen)
        weight = scores[tok, slot].to(f64) / torch.sqrt(v.square().mean(-1) + POST_EPS)
        out.index_add_(0, tok, weight[:, None] * v)
    part = _bf(out)
    branch = _bf(part @ w_unpool.to(f64).t())
    post = _bf(branch * pfw.to(f64))
    nb = _bf(post * _rstd(post, POST_EPS))
    new = alpha.to(f64) * res.to(f64) + beta.to(f64) * nb
    attn_out = new * _rstd(new, RMS_EPS) * npw.to(f64)
    return {"attn_out": attn_out, "res": new}
