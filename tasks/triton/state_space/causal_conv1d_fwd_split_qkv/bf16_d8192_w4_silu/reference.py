import torch

KEY_DIM = 2048  # query and key channels each; the value channels are the remaining dim - 2 * KEY_DIM


def reference(*, x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state):
    width = weight.shape[1]
    state_len = width - 1
    w = weight.double()
    b = bias.double()
    out = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    states = conv_states.double().clone()
    bounds = query_start_loc.tolist()
    slots = cache_indices.tolist()
    has_state = has_initial_state.tolist()
    for i, slot in enumerate(slots):
        start, end = bounds[i], bounds[i + 1]
        length = end - start
        prior = states[slot].clone() if has_state[i] else torch.zeros_like(states[slot])
        z = torch.cat([prior, x[start:end].double()], dim=0)  # (state_len + length, dim)
        acc = b.expand(length, -1).clone()
        for j in range(width):
            acc += z[j:j + length] * w[:, j]
        out[start:end] = acc * torch.sigmoid(acc)
        states[slot] = z[length:length + state_len]
    return out[:, :KEY_DIM], out[:, KEY_DIM:2 * KEY_DIM], out[:, 2 * KEY_DIM:], states
