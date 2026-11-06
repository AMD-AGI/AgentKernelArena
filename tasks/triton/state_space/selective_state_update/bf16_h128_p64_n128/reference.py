"""High-precision reference of the Mamba2 single-step selective state update, used only for correctness."""
import torch


def reference(*, state, x, dt, dt_bias, A, B, C, D, z, state_batch_indices):
    batch, nheads, headdim = x.shape
    ratio = nheads // B.shape[1]
    out = torch.empty(batch, nheads, headdim, dtype=torch.float64, device=x.device)
    new_state = state.to(torch.float32)  # slots outside state_batch_indices stay unchanged
    step = 16  # sequences per step, to bound float64 temporaries
    for start in range(0, batch, step):
        sl = slice(start, min(start + step, batch))
        idx = state_batch_indices[sl].long()
        s = state[idx].double()                                        # [c, H, P, N]
        dtv = torch.nn.functional.softplus(dt[sl].double() + dt_bias.double()[None, :])  # [c, H]
        decay = torch.exp(A.double()[None, :] * dtv)                   # [c, H]
        Bh = B[sl].double().repeat_interleave(ratio, dim=1)            # [c, H, N]
        Ch = C[sl].double().repeat_interleave(ratio, dim=1)
        xv = x[sl].double()
        s = s * decay[:, :, None, None] + (dtv[:, :, None] * Bh)[:, :, None, :] * xv[:, :, :, None]
        y = (s * Ch[:, :, None, :]).sum(-1) + xv * D.double()[None, :, None]
        zv = z[sl].double()
        out[sl] = y * (zv * torch.sigmoid(zv))
        new_state[idx] = s.to(torch.float32)
    return out, new_state
