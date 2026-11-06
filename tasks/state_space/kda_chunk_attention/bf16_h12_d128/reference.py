"""Float64 token-by-token Kimi Delta Attention recurrence, used only for correctness.

For every sequence n and head h, with the state S[v, k] starting at initial_state[n, h]:

    q_t = q_t / sqrt(|q_t|^2 + 1e-6) * head_dim^-1/2,   k_t = k_t / sqrt(|k_t|^2 + 1e-6)
    g_t = lower_bound * sigmoid(exp(A_log[h]) * (raw_g_t + dt_bias[h]))     (per channel k)
    S   = S * exp(g_t)[None, :]                       decay every key channel
    S   = S + sigmoid(raw_beta_t) * (v_t - S k_t) k_t^T   delta-rule write
    o_t = S q_t

final_state is S after the last token of the sequence.
"""
import torch

L2NORM_EPS = 1e-6
STEPS_PER_BLOCK = 256  # tokens whose float64 operands are prepared together


def reference(*, q, k, v, raw_g, raw_beta, A_log, dt_bias, initial_state, lower_bound):
    n_seqs, seq_len, heads, dim = q.shape
    scale = dim ** -0.5
    a = torch.exp(A_log.double()).view(1, 1, heads, 1)
    bias = dt_bias.double().view(1, 1, heads, dim)
    state = initial_state.double().clone()                       # [N, H, V, K]
    out = torch.empty(n_seqs, seq_len, heads, dim, dtype=torch.float64, device=q.device)
    for start in range(0, seq_len, STEPS_PER_BLOCK):
        stop = min(start + STEPS_PER_BLOCK, seq_len)
        qb = q[:, start:stop].double()
        kb = k[:, start:stop].double()
        qb = qb / torch.sqrt(qb.square().sum(-1, keepdim=True) + L2NORM_EPS) * scale
        kb = kb / torch.sqrt(kb.square().sum(-1, keepdim=True) + L2NORM_EPS)
        vb = v[:, start:stop].double()
        decay = torch.exp(lower_bound * torch.sigmoid(a * (raw_g[:, start:stop].double() + bias)))
        beta = torch.sigmoid(raw_beta[:, start:stop].double())
        for t in range(stop - start):
            state = state * decay[:, t, :, None, :]
            k_t = kb[:, t, :, :, None]                            # [N, H, K, 1]
            delta = (vb[:, t] - torch.matmul(state, k_t)[..., 0]) * beta[:, t, :, None]
            state = state + delta[..., :, None] * kb[:, t, :, None, :]
            out[:, start + t] = torch.matmul(state, qb[:, t, :, :, None])[..., 0]
    return {"o": out, "final_state": state}
