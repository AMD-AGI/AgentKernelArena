import math

import torch


def make_inputs(case, inputs, generator, device):
    """Ragged sequence offsets and the RoPE angle table.

    cu_seqlens splits n_tokens into num_seqs non-empty segments at random cut points, so lengths are
    ragged (from a few tokens to most of the batch). freqs[p, 0, 0, j] = p * 10000^(-j / half_dim) in
    float32, the standard uncached angle table (one angle per rotated pair).
    """
    a = case["axes"]
    n_tokens, num_seqs = int(a["n_tokens"]), int(a["num_seqs"])
    max_position, half_dim = int(a["max_position"]), int(a["half_dim"])
    cuts = torch.randperm(n_tokens - 1, generator=generator, device=device)[: num_seqs - 1] + 1
    cuts, _ = torch.sort(cuts)
    zero = torch.zeros(1, dtype=torch.int64, device=device)
    total = torch.full((1,), n_tokens, dtype=torch.int64, device=device)
    cu_seqlens = torch.cat((zero, cuts, total)).to(torch.int32)
    inv_freq = 10000.0 ** (-torch.arange(half_dim, dtype=torch.float64, device=device) / half_dim)
    pos = torch.arange(max_position, dtype=torch.float64, device=device)
    freqs = (pos[:, None] * inv_freq[None, :]).to(torch.float32).reshape(max_position, 1, 1, half_dim)
    return {"cu_seqlens": cu_seqlens, "freqs": freqs}
