# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""float64 reference of one Kimi-K3 KDA layer front for one request of ``rows`` verify tokens (the
TP8 rank slice: 12 heads of 128), used only for correctness.

* AttnRes seam: the sources of token t are the bank rows blocks[t, 0 .. nblocks - 1] and the prefix
  prefix[t]; score_s = sum_h v_s[h] attn_res_norm[h] attn_res_query[h] / rms(v_s) with
  rms(v) = sqrt(mean(v^2) + 1e-5); mixed = sum_s softmax(score)_s v_s;
  x = mixed / sqrt(mean(mixed^2) + 1e-5) * input_norm.
* proj = x @ w_in^T (6288 rows: q 0..1535 | k 1536..3071 | v 3072..4607 | g 4608..6143 |
  f_a 6144..6271 | beta 6272..6283 | 4 unused pad rows).
* Short conv: seq = [conv_state[0 .. 2]; proj[:, 0 .. 4607]] (the committed window, then the step's
  inputs); conv_t[c] = sum_i conv_weight[c, i] seq[t + i, c]; q | k | v = silu(conv).
* q, k of each head l2-normalized (eps 1e-6 inside the sqrt), q scaled by 128^-1/2.
* Gate of head h, key column k: g = (f_a @ w_fb[128 h .. 128 h + 127]^T)[k] + dt_bias[128 h + k];
  decay = exp(-5 sigmoid(exp(a_log[h]) g)); beta = sigmoid(proj[:, 6272 + h]).
* Delta rule from S = ssm_state[h] (S[v, k]), token by token: S = S * decay[t, None, :];
  S = S + ((v_t - S k_t) beta_t) k_t^T; o_t = S q_t; the state after token t is ssm_states[state_rows[t]].
* core_out[t, 128 h + v] = o_t[v] / sqrt(mean_v(o_t^2) + 1e-5) * o_norm[v] * sigmoid(g2[t, 128 h + v]),
  g2 = proj[:, 4608 .. 6143].
* conv_state (mutable): the launch shifts its buffer by one token and appends the step's conv inputs:
  final rows 0 .. 2 = initial rows 1 .. 3, final rows 3 .. = proj[:, 0 .. 4607].
"""
import torch

HIDDEN, NH, HD = 7168, 12, 128
PROJ = NH * HD  # 1536
CONV_DIM = 3 * PROJ  # 4608
G2_0, FA_0, BETA_0 = 3 * PROJ, 4 * PROJ, 4 * PROJ + HD
CONV_W = 4
EPS = 1e-5  # AttnRes norm, input_layernorm and the output RMSNorm
L2_EPS = 1e-6
LOWER_BOUND = -5.0
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


def reference(*, prefix, blocks, attn_res_norm, attn_res_query, input_norm, w_in, w_fb, conv_weight,
              a_log, dt_bias, o_norm, ssm_state, state_rows, conv_state):
    rows = prefix.shape[0]
    x = seam(prefix, blocks, attn_res_norm, attn_res_query, input_norm)
    proj = _linear(x, w_in)  # [rows, 6288]
    xin = proj[:, :CONV_DIM]
    window = conv_state.double()[:CONV_W - 1]
    seq = torch.cat([window, xin], dim=0)  # [rows + 3, 4608]
    cw = conv_weight.double()  # [4608, 4]
    conv = torch.stack([(cw * seq[t:t + CONV_W].T).sum(-1) for t in range(rows)])  # [rows, 4608]
    act = conv * torch.sigmoid(conv)
    q = act[:, :PROJ].reshape(rows, NH, HD)
    k = act[:, PROJ:2 * PROJ].reshape(rows, NH, HD)
    v = act[:, 2 * PROJ:].reshape(rows, NH, HD)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + L2_EPS) * HD ** -0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + L2_EPS)
    fa = proj[:, FA_0:FA_0 + HD]
    g2 = proj[:, G2_0:G2_0 + PROJ].reshape(rows, NH, HD)
    beta = torch.sigmoid(proj[:, BETA_0:BETA_0 + NH])  # [rows, 12]
    g = (fa @ w_fb.double().T).reshape(rows, NH, HD) + dt_bias.double().reshape(NH, HD)
    decay = torch.exp(LOWER_BOUND * torch.sigmoid(torch.exp(a_log.double())[:, None] * g))  # [rows, 12, 128]
    state = ssm_state.double().clone()  # [12, V, K]
    states = torch.empty(rows, NH, HD, HD, dtype=torch.float64, device=prefix.device)
    o = torch.empty(rows, NH, HD, dtype=torch.float64, device=prefix.device)
    for t in range(rows):
        state = state * decay[t].unsqueeze(1)
        p = torch.einsum("hvk,hk->hv", state, k[t])
        vn = (v[t] - p) * beta[t].unsqueeze(-1)
        state = state + vn.unsqueeze(-1) * k[t].unsqueeze(1)
        o[t] = torch.einsum("hvk,hk->hv", state, q[t])
        states[t] = state
    core = o * _rms(o, EPS) * o_norm.double() * torch.sigmoid(g2)
    ssm_states = torch.empty_like(states)
    ssm_states[state_rows.long()] = states
    conv_final = torch.cat([conv_state.double()[1:CONV_W], xin], dim=0)
    return core.reshape(rows, PROJ), ssm_states, conv_final
