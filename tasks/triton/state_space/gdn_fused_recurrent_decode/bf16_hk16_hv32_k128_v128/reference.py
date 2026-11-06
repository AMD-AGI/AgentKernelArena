"""High-precision GDN packed decode reference used only for correctness."""
import torch

SEQUENCES_PER_STEP = 32  # bounds the float64 state temporaries
SOFTPLUS_THRESHOLD = 20.0
L2_EPS = 1e-6


def reference(*, mixed_qkv, a, b, A_log, dt_bias, scale, ssm_state, ssm_state_indices):
    B = mixed_qkv.shape[0]
    HV, V, K = ssm_state.shape[1:]
    H = (mixed_qkv.shape[1] - HV * V) // 2 // K
    rep = HV // H
    out = torch.zeros(B, 1, HV, V, dtype=torch.float64, device=mixed_qkv.device)
    final = ssm_state.double().clone()
    slots = ssm_state_indices.long()
    for start in range(0, B, SEQUENCES_PER_STEP):
        rows = torch.arange(start, min(start + SEQUENCES_PER_STEP, B), device=mixed_qkv.device)
        rows = rows[slots[rows] >= 0]
        if rows.numel() == 0:
            continue
        x = mixed_qkv[rows].double()
        q = x[:, :H * K].reshape(-1, H, K).repeat_interleave(rep, dim=1)             # [n, HV, K]
        k = x[:, H * K:2 * H * K].reshape(-1, H, K).repeat_interleave(rep, dim=1)
        v = x[:, 2 * H * K:].reshape(-1, HV, V)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + L2_EPS) * float(scale)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + L2_EPS)
        pre = a[rows].double() + dt_bias.double()
        softplus = torch.where(pre <= SOFTPLUS_THRESHOLD, torch.log1p(torch.exp(pre)), pre)
        decay = torch.exp(-torch.exp(A_log.double()) * softplus)                     # exp(g), [n, HV]
        beta = torch.sigmoid(b[rows].double())
        state = ssm_state[slots[rows]].double() * decay[:, :, None, None]           # [n, HV, V, K]
        delta = (v - (state * k[:, :, None, :]).sum(-1)) * beta[:, :, None]
        state = state + delta[..., None] * k[:, :, None, :]
        out[rows, 0] = (state * q[:, :, None, :]).sum(-1)
        final[slots[rows]] = state
    return out, final
