"""High-precision GDN chunk state reference used only for correctness."""
import torch

CHUNK_SIZE = 64


def reference(*, k, w, u, g, initial_state):
    _, T, Hg, K = k.shape
    H, V = u.shape[1], u.shape[3]
    NT = -(-T // CHUNK_SIZE)
    rep = H // Hg
    state = initial_state.double()                            # [1, H, V, K]
    h = torch.empty(1, NT, H, V, K, dtype=torch.float64, device=k.device)
    v_new = torch.empty(1, H, T, V, dtype=torch.float64, device=k.device)
    for i in range(NT):
        t0, t1 = i * CHUNK_SIZE, min((i + 1) * CHUNK_SIZE, T)
        h[:, i] = state
        wc = w[:, :, t0:t1].double()                          # [1, H, L, K]
        uc = u[:, :, t0:t1].double()                          # [1, H, L, V]
        kc = k[:, t0:t1].double().repeat_interleave(rep, dim=2)   # [1, L, H, K]
        gc = g[:, :, t0:t1].double()                          # [1, H, L], base-2 cumulative log-gate
        g_last = gc[:, :, -1]                                 # [1, H]
        delta = uc - torch.einsum("bhlk,bhvk->bhlv", wc, state)
        v_new[:, :, t0:t1] = delta
        diff = g_last[:, :, None] - gc                        # <= 0 inside a chunk
        gate = torch.where(diff <= 0, torch.exp2(diff), torch.zeros_like(diff))
        state = state * torch.exp2(g_last)[:, :, None, None] \
            + torch.einsum("bhlv,blhk->bhvk", delta * gate[..., None], kc)
    return h, v_new, state
