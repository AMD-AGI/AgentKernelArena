"""High-precision causal conv1d (split q/k/v) reference used only for correctness."""
import torch

# The channels are the concatenation [q | k | v]: dim = 2 * key_dim + value_dim = 2 * 2048 + 4096.


def reference(*, x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state, chunk_sequence_ids,
              chunk_ids):
    # The chunk schedule only tells the kernel how to split the work; the result does not depend on it.
    dim, n_tokens = x.shape
    width = weight.shape[1]
    state_len = width - 1
    key_dim, value_dim = dim // 4, dim // 2
    w = weight.double()
    b = bias.double()
    device = x.device
    query = torch.empty(n_tokens, key_dim, dtype=torch.float64, device=device)
    key = torch.empty(n_tokens, key_dim, dtype=torch.float64, device=device)
    value = torch.empty(n_tokens, value_dim, dtype=torch.float64, device=device)
    states = conv_states.double().clone()                      # [lines, state_len, dim]
    bounds = query_start_loc.tolist()
    slots = cache_indices.tolist()
    has_state = has_initial_state.tolist()
    for i, slot in enumerate(slots):
        start, end = bounds[i], bounds[i + 1]
        length = end - start
        prior = states[slot].t().clone() if has_state[i] else torch.zeros(dim, state_len, dtype=torch.float64,
                                                                           device=device)
        z = torch.cat([prior, x[:, start:end].double()], dim=1)     # [dim, state_len + length]
        acc = b[:, None].expand(dim, length).clone()
        for j in range(width):
            acc += w[:, j:j + 1] * z[:, j:j + length]
        act = acc * torch.sigmoid(acc)                              # [dim, length]
        query[start:end] = act[:key_dim].t()
        key[start:end] = act[key_dim:2 * key_dim].t()
        value[start:end] = act[2 * key_dim:].t()
        states[slot] = z[:, length:length + state_len].t()
    return query, key, value, states
