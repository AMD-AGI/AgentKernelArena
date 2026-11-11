"""float64 reference of one Kimi-K3.1 wide-row KDA layer launch (all eight TP ranks, logical tensors).

The operator of sgl-amd-mono ``kimi_k31/gfx950/layer.py`` (a covered layer in the middle of a run: the previous
layer's MoE tail, no block write, no capture), rewritten from the repository's float32 references
(``test/kernels/kimi_k31/layer_bench.py``, ``pre_attn_bench.kda_ref`` / ``win_ref``, ``chain_bench``) in float64.
bfloat16 tensors that the launch hands between its stages keep their rounding (``bf``); every sum, norm and
nonlinearity is float64. The AttnRes seams follow the launch's one-round form (``stages/attnres.py``): scores
x_r . cw x rsqrt(mean(x_r^2) + eps) unrounded, the mixture rounded to bfloat16 before the output norm, whose
mean square is that of the unrounded mixture. Returned values are those before their own final rounding.
"""

import torch

from inputs import BETA_0, E, E_R, FA_0, G2_0, H, HD, LAT, L, LSL, NF, NH, NPROJ, NVB, PROJ, QKV, SI, TOPK, WORLD

EPS = 1e-5  # the seams' norms (input_layernorm / post_attention_layernorm / *_res_norm variance_epsilon)
EPS_N = 1e-5  # o_norm and the previous layer's latent RMSNorm
LOWER = -5.0  # KDA's lower-bound gate
F64 = torch.float64


def bf(x):
    return x.to(torch.bfloat16).to(F64)


def _seam(sources, prefix, cw, w_out):
    """AttnRes over [the NVB bank snapshots, prefix] (rows [m, H], float64 of bfloat16 values) -> bf16 x."""
    xs = torch.cat([sources, prefix.unsqueeze(1)], dim=1)  # [m, R, H]
    dots = xs @ cw.to(F64)  # [m, R]
    gram = xs @ xs.transpose(1, 2)  # [m, R, R]
    ss = torch.diagonal(gram, dim1=1, dim2=2)
    probs = torch.softmax(dots * torch.rsqrt(ss / H + EPS), dim=-1)
    mixed = bf((probs.unsqueeze(-1) * xs).sum(1))
    msq = torch.einsum("mr,mrs,ms->m", probs, gram, probs) / H
    return bf(mixed * torch.rsqrt(msq + EPS).unsqueeze(-1) * w_out.to(F64))


def _situ(g, u):
    """vLLM's situ_and_mul (beta 4, linear beta 25)."""
    return (2.0 * torch.tanh(g * 0.25)) * (1.0 + torch.tanh(g * 0.5)) * (25.0 * torch.tanh(u * 0.04))


def reference(*, lat_p, sh_p, w_up, prefix, bank, cw_a, w_in, w_ip, w_fb, conv_w, a_log, dt_bias, on_w,
              conv_state, rstate, st_idx, num_acc, win_row, wo, cw_m, w_ln, wf, bias, w_sd, epoch,
              prefix_out, ip_buf, acc_u, ring_k, ring_v, ring_g, ring_b, conv_win, **_reserved):
    m = prefix.shape[0]
    reqs = int((st_idx[:, 0] >= 0).sum())
    live = reqs * L
    bank_live = bank[:, :NVB].to(F64)

    # ---- T1: the previous layer's MoE tail (latent all-reduce, RMSNorm, up on owner columns, shared AR)
    lat = bf(lat_p.to(F64).view(WORLD, m, LAT).sum(0))
    rms = torch.rsqrt(lat.square().mean(-1, keepdim=True) + EPS_N)
    lag = bf(lat * bias[E:].to(F64))  # the gathered latent, scaled by the norm weight
    up = (lag @ w_up.to(F64).T) * rms
    shs = sh_p.to(F64).view(WORLD, m, H).sum(0)
    y = bf(up + shs)
    mid_pre = y + prefix.to(F64)
    mid = bf(mid_pre)

    # ---- T2 / T3: the attention seam -> in_proj, per rank
    x = _seam(bank_live, mid, cw_a, w_in)
    proj = torch.stack([bf(x @ w_ip[r * NPROJ:(r + 1) * NPROJ].to(F64).T) for r in range(WORLD)])  # [W, m, 6288]

    # ---- KDA verify on the ReplaySSM ring: one (request, head) task each, live requests only
    core_pre = torch.zeros(m, WORLD * PROJ, dtype=F64, device=prefix.device)
    core_bf = torch.zeros(WORLD, m, PROJ, dtype=F64, device=prefix.device)
    ring_k = ring_k.to(F64).clone()
    ring_v = ring_v.to(F64).clone()
    ring_g = ring_g.to(F64).clone()
    ring_b = ring_b.to(F64).clone()
    conv_win = conv_win.to(F64).clone()
    if reqs:
        slots = st_idx[:reqs, 0].long()
        p = proj[:, :live].view(WORLD, reqs, L, NPROJ)
        win = conv_state[slots].to(F64).view(reqs, 3, WORLD, QKV).permute(2, 0, 1, 3)  # [W, R, 3, 4608]
        seq = torch.cat([win, p[..., :QKV]], dim=2)  # [W, R, 3 + L, 4608]
        cw = conv_w.to(F64).view(WORLD, 1, QKV, 4)
        conv = torch.stack([sum(cw[..., q] * seq[:, :, j + q] for q in range(4)) for j in range(L)], dim=2)
        act = conv * torch.sigmoid(conv)  # [W, R, L, 4608]
        o = bf(act)
        wr = win_row[:reqs].long()
        for i in range(reqs):
            s = slots[i]
            hk = act[:, i, :, PROJ:2 * PROJ].view(WORLD, L, NH, HD).permute(0, 2, 1, 3)  # [W, 12, L, 128]
            hv = act[:, i, :, 2 * PROJ:].view(WORLD, L, NH, HD).permute(0, 2, 1, 3)
            ring_k[s, :, :L] = hk.reshape(WORLD * NH, L, HD)
            ring_v[s, :, :L] = hv.reshape(WORLD * NH, L, HD)
            if wr[i] >= 0:
                w = torch.stack([seq[:, i, j + 1:j + 4] for j in range(L)], dim=1)  # [W, L, 3, 4608]
                conv_win[wr[i]] = w.permute(1, 2, 0, 3).reshape(L, 3, WORLD * QKV)
        q = o[..., :PROJ].view(WORLD, reqs, L, NH, HD)
        k = o[..., PROJ:2 * PROJ].view(WORLD, reqs, L, NH, HD)
        v = o[..., 2 * PROJ:].view(WORLD, reqs, L, NH, HD)
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * HD ** -0.5
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        fa = p[..., FA_0:FA_0 + HD]  # bf16 values
        gfb = bf(torch.einsum("wrlk,wnk->wrln", fa, w_fb.to(F64).view(WORLD, PROJ, HD)))
        gt = gfb + dt_bias.to(F64).view(WORLD, 1, 1, PROJ)
        aa = torch.exp(a_log.to(F64)).view(WORLD, 1, 1, NH, 1)
        gate = LOWER * torch.sigmoid(aa * gt.view(WORLD, reqs, L, NH, HD))  # [W, R, L, 12, 128]
        decay = torch.exp(gate)
        beta = torch.sigmoid(p[..., BETA_0:BETA_0 + NH])  # [W, R, L, 12]
        for i in range(reqs):
            s = slots[i]
            ring_g[s, :, :L] = gate[:, i].permute(0, 2, 1, 3).reshape(WORLD * NH, L, HD)
            ring_b[s, :, :L] = beta[:, i].permute(0, 2, 1).reshape(WORLD * NH, L)
        st = rstate[slots].to(F64).view(reqs, WORLD, NH, HD, HD).transpose(0, 1)  # [W, R, 12, V, K]
        outs = []
        for j in range(L):
            st = st * decay[:, :, j].unsqueeze(-2)
            kj, qj = k[:, :, j], q[:, :, j]
            dl = (v[:, :, j] - torch.einsum("wrhvk,wrhk->wrhv", st, kj)) * beta[:, :, j].unsqueeze(-1)
            st = st + dl.unsqueeze(-1) * kj.unsqueeze(-2)
            outs.append(bf(torch.einsum("wrhvk,wrhk->wrhv", st, qj)))
        ob = torch.stack(outs, dim=2)  # [W, R, L, 12, 128]
        g2 = p[..., G2_0:G2_0 + PROJ].view(WORLD, reqs, L, NH, HD)
        yk = (ob * torch.rsqrt(ob.square().mean(-1, keepdim=True) + EPS_N) * on_w.to(F64)
              * torch.sigmoid(g2)).reshape(WORLD, live, PROJ)
        core_bf[:, :live] = bf(yk)
        core_pre[:live] = yk.permute(1, 0, 2).reshape(live, WORLD * PROJ)

    # ---- T4: o_proj, reduce-scatter in rank order + the residual
    parts = sum(bf(core_bf[r] @ wo[:, r * PROJ:(r + 1) * PROJ].to(F64).T) for r in range(WORLD))
    pout_pre = parts + mid
    pout = bf(pout_pre)

    # ---- T5 / T6: the MLP seam -> the MoE front, the route, the shared expert
    x2 = _seam(bank_live, pout, cw_m, w_ln)
    logits = torch.empty(m, E, dtype=F64, device=prefix.device)
    latent = torch.empty(m, LAT, dtype=F64, device=prefix.device)
    sh_out = torch.empty(WORLD * m, H, dtype=F64, device=prefix.device)
    for r in range(WORLD):
        f = x2 @ wf[r * NF:(r + 1) * NF].to(F64).T  # [m, 2096]
        logits[:, r * E_R:(r + 1) * E_R] = f[:, :E_R]
        latent[:, r * LSL:(r + 1) * LSL] = f[:, E_R:E_R + LSL]
        sgu = bf(f[:, E_R + LSL:])
        h = bf(_situ(sgu[:, :SI], sgu[:, SI:]))
        sh_out[r * m:(r + 1) * m] = h @ w_sd[:, r * SI:(r + 1) * SI].to(F64).T
    sig = torch.sigmoid(logits)
    ids = torch.topk(sig + bias[:E].to(F64), TOPK, dim=-1, sorted=True).indices
    picked = sig.gather(-1, ids)
    weights = picked / picked.sum(-1, keepdim=True)

    return {
        "mid": mid_pre,
        "core": core_pre,
        "lat_out": latent,
        "topk_ids": ids.to(torch.int64),
        "topk_w": weights,
        "logits_out": logits,
        "sh_out": sh_out,
        "prefix_out": pout_pre,
        "ip_buf": torch.zeros_like(ip_buf, dtype=F64),
        "acc_u": torch.zeros_like(acc_u, dtype=F64),
        "ring_k": ring_k,
        "ring_v": ring_v,
        "ring_g": ring_g,
        "ring_b": ring_b,
        "conv_win": conv_win,
    }

