"""Float64 reference of the Muse Spark DFlash2 draft layer's attention front (D-pre), one TP8 rank.

Rows come in blocks of 8 (one request's draft block). With H = 8192, 512 conv groups of 16 channels
and two taps:

    r      = x if first else bf16(res + x)                   (the input norm's fused add; res_out)
    h      = r * rsqrt(mean(r^2) + 1e-5) * w_in              (RMSNorm)
    coef   = h @ w_kp^T                                      ([S, 2048] = [side, tap, group])
    hc[t]  = (base[0, 0] + coef[t, 0, 0, g]) * h[t]
             + [t % 8 >= 1] (base[0, 1] + coef[t, 0, 1, g]) * h[t - 1]   (conv0, channel c in group g = c // 16)
    qkv    = hc @ w_qkv^T                                    ([S, 1280] = 8 q heads | k | v, 128 each)
    q, k   : per head x * rsqrt(mean(x^2) + 1e-5) * w_{q,k}_norm, then neox RoPE with
             cos = cos_sin[pos, :64], sin = cos_sin[pos, 64:]
    k_cache[slots[t], 0] = k[t], v_cache[slots[t], 0] = v[t] (the other slots unchanged)

Everything else is float64 from the bf16 / fp8 inputs. The fused add's bf16 rounding is part of the
operator (the residual is a bf16 tensor, which the norm reads, as SGLang's fused add-norm and the kernel
do); the other outputs are returned before their final rounding (bf16 q, float32 coef, float8_e4m3fn
cache rows).
"""

import torch

H = 8192
GROUP = 16
GROUPS = H // GROUP
TAPS = 2
BLOCK = 8
HEAD_DIM = 128
Q_HEADS = 8
EPS = 1e-5


def _rms(x, w):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * w


def reference(*, x, res, first, w_in, w_kp, base, w_qkv, q_norm, k_norm, cos_sin, positions, slots,
              k_cache, v_cache):
    S = x.shape[0]
    r = x.double() if first else (res.double() + x.double()).to(torch.bfloat16).double()
    h = _rms(r, w_in.double())
    coef = h @ w_kp.double().T
    delta = coef.reshape(S, 2, TAPS, GROUPS)[:, 0]  # side 0: [S, tap, group]
    c = base.double()[0].reshape(1, TAPS, GROUPS, GROUP) + delta[..., None]  # [S, tap, group, 16]
    hb = h.reshape(S, GROUPS, GROUP)
    prev = torch.cat([torch.zeros_like(hb[:1]), hb[:-1]])
    first_row = (torch.arange(S, device=x.device) % BLOCK == 0)[:, None, None]
    prev = torch.where(first_row, torch.zeros_like(prev), prev)
    hc = (c[:, 0] * hb + c[:, 1] * prev).reshape(S, H)
    qkv = hc @ w_qkv.double().T
    q, k, v = qkv[:, :Q_HEADS * HEAD_DIM], qkv[:, Q_HEADS * HEAD_DIM:-HEAD_DIM], qkv[:, -HEAD_DIM:]
    cs = cos_sin.double()[positions.long()]  # [S, 128]
    cos, sin = cs[:, None, :HEAD_DIM // 2], cs[:, None, HEAD_DIM // 2:]

    def norm_rope(t, w):
        n = _rms(t.reshape(S, -1, HEAD_DIM), w.double())
        x1, x2 = n[..., :HEAD_DIM // 2], n[..., HEAD_DIM // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).reshape(S, -1)

    q_out = norm_rope(q, q_norm)
    k_out = norm_rope(k, k_norm)
    k_final, v_final = k_cache.double().clone(), v_cache.double().clone()
    k_final[slots.long(), 0] = k_out
    v_final[slots.long(), 0] = v
    return {"res_out": r, "q": q_out, "coef": coef, "k_cache": k_final, "v_cache": v_final}
