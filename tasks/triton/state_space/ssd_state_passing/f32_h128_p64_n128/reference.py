"""High-precision Mamba2 state passing reference used only for correctness."""
import torch


def reference(*, states, dA_cumsum, last_chunk_indices, initial_states):
    out = torch.empty(states.shape, dtype=torch.float64, device=states.device)
    decay = torch.exp(dA_cumsum[:, :, -1].double())                  # [H, chunks]
    first = 0
    for b, last in enumerate(last_chunk_indices.tolist()):
        state = initial_states[b].double()                           # [H, dim]
        for c in range(first, last + 1):
            state = decay[:, c, None] * state + states[c].double()
            out[c] = state
        first = last + 1
    return out
