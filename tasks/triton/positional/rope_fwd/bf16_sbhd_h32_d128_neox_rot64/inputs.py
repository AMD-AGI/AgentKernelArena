import torch


def make_inputs(case, inputs, generator, device):
    """Angles freqs[s, k] = (start + s) * 10000^(-k / freq_dim): position times the standard RoPE inverse frequency."""
    axes = case["axes"]
    seq_len, half_dim = axes["seq_len"], axes["freq_dim"]
    inv_freq = 10000.0 ** (-torch.arange(half_dim, dtype=torch.float64, device=device) / half_dim)
    # Positions start at a random nonzero offset (as when decoding past a KV cache), so even a one-position
    # case draws full-precision angles instead of exact zeros.
    start = int(torch.randint(1, 4096, (1,), generator=generator, device=device))
    positions = torch.arange(seq_len, dtype=torch.float64, device=device) + start
    angles = positions[:, None] * inv_freq[None, :]
    return {"freqs": angles.to(torch.float32).reshape(seq_len, 1, 1, half_dim)}
