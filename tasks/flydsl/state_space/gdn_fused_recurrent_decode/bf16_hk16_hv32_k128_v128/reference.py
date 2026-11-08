"""High-precision GDN decode-step reference used only for correctness."""
import torch

SEQUENCES_PER_STEP = 32  # bounds the float64 state temporaries
SOFTPLUS_THRESHOLD = 20.0
L2_EPS = 1e-6


def reference(*, query, key, value, a, b, dt_bias, A_log, state_indices, state):
    B, _, H, K = query.shape
    HV, V = value.shape[2], value.shape[3]
    rep = HV // H
    scale = K ** -0.5
    out = torch.zeros(B, 1, HV, V, dtype=torch.float64, device=query.device)
    final = state.double().clone()
    slots = state_indices.long()
    for start in range(0, B, SEQUENCES_PER_STEP):
        rows = torch.arange(start, min(start + SEQUENCES_PER_STEP, B), device=query.device)
        q = query[rows, 0].double().repeat_interleave(rep, dim=1)                    # [n, HV, K]
        k = key[rows, 0].double().repeat_interleave(rep, dim=1)
        v = value[rows, 0].double()                                                  # [n, HV, V]
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + L2_EPS) * scale
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + L2_EPS)
        pre = a[rows, 0].double() + dt_bias.double()
        softplus = torch.where(pre <= SOFTPLUS_THRESHOLD, torch.log1p(torch.exp(pre)), pre)
        decay = torch.exp(-torch.exp(A_log.double()) * softplus)                     # exp(g), [n, HV]
        beta = torch.sigmoid(b[rows, 0].double())
        s = state[slots[rows]].double() * decay[:, :, None, None]                    # [n, HV, V, K]
        delta = (v - (s * k[:, :, None, :]).sum(-1)) * beta[:, :, None]
        s = s + delta[..., None] * k[:, :, None, :]
        out[rows, 0] = (s * q[:, :, None, :]).sum(-1)
        final[slots[rows]] = s
    return out, final
