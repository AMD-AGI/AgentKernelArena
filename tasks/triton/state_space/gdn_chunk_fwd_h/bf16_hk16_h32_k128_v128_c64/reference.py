"""High-precision GDN chunk state reference used only for correctness."""
import torch

CHUNK_SIZE = 64


def reference(*, k, w, u, g, initial_state, initial_state_indices):
    B, T, Hg, K = k.shape
    H, V = u.shape[2], u.shape[3]
    NT = -(-T // CHUNK_SIZE)
    rep = H // Hg
    idx = initial_state_indices.long()
    state = initial_state[idx].double()                       # [B, H, V, K]
    h = torch.empty(B, NT, H, V, K, dtype=torch.float64, device=k.device)
    v_new = torch.empty(B, T, H, V, dtype=torch.float64, device=k.device)
    for i in range(NT):
        t0, t1 = i * CHUNK_SIZE, min((i + 1) * CHUNK_SIZE, T)
        h[:, i] = state
        wc = w[:, t0:t1].double()                             # [B, L, H, K]
        uc = u[:, t0:t1].double()                             # [B, L, H, V]
        kc = k[:, t0:t1].double().repeat_interleave(rep, dim=2)   # [B, L, H, K]
        gc = g[:, t0:t1].double()                             # [B, L, H]
        g_last = gc[:, -1]                                    # [B, H]
        delta = uc - torch.einsum("blhk,bhvk->blhv", wc, state)
        v_new[:, t0:t1] = delta
        diff = g_last[:, None] - gc
        gate = torch.where(diff <= 0, torch.exp(diff), torch.zeros_like(diff))
        state = state * torch.exp(g_last)[:, :, None, None] \
            + torch.einsum("blhv,blhk->bhvk", delta * gate[..., None], kc)
    final = initial_state.double().clone()
    final[idx] = state
    return h, v_new, final
