"""Stage-1 GEMM output and the block-padded MoE sorting table of a balanced router.

x holds [gate | up] per (token, slot) row. The kernel computes silu(gate) * up with hardware approximations of exp2 and
reciprocal (about 5e-7 relative error), so a value whose magnitude lies within a few 1e-6 of a rounding decision of the
MXFP4 quantization (the group-scale exponent threshold at mantissa 1.5, the FP4 grid midpoints at mantissas 1, 1.25, 1.5
and 1.75 of an octave) could legitimately be rounded either way. The generator therefore moves such rare values (a
fraction of about 1e-4) to the next bfloat16 neighbour of `up`, so that the exact comparison of the packed codes and
scales is decided by the specified rounding rather than by approximation noise.

The (token, slot) rows are assigned to experts uniformly at random with exactly balanced counts, sorted by expert in
token order and padded per expert to a multiple of 32 rows with a sentinel entry; the padded length is the declared
sorted_len.
"""
import math

import torch

BLOCK = 32
MARGIN = 5.0e-6          # relative distance to a rounding decision below which a value is moved
CHUNK_ROWS = 1 << 16
_THRESHOLDS = [0.0] + [math.log2(m) for m in (1.25, 1.5, 1.75)]   # fractional log2 of the decisions


def _unsafe(gate, up):
    gate, up = gate.double(), up.double()
    act = (gate * torch.sigmoid(gate) * up).abs()
    frac = torch.log2(act.clamp(min=1e-300)) % 1.0
    close = torch.zeros_like(act, dtype=torch.bool)
    for t in _THRESHOLDS:
        distance = (frac - t).abs()
        close |= torch.minimum(distance, 1.0 - distance) < MARGIN * 1.4427
    return close & (act > 0)


def _bump(value):
    """Next bfloat16 value of larger magnitude."""
    return (value.view(torch.int16) + 1).view(torch.bfloat16)


def _safe_activations(x, inter_dim):
    for start in range(0, x.shape[0], CHUNK_ROWS):
        block = x[start:start + CHUNK_ROWS]
        gate, up = block[:, :inter_dim], block[:, inter_dim:]
        for _ in range(64):
            bad = _unsafe(gate, up)
            if not bool(bad.any()):
                break
            up.copy_(torch.where(bad, _bump(up), up))
        else:
            raise RuntimeError("could not move all values away from rounding decisions")


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_tokens, top_k, experts, inter_dim = axes["n_tokens"], axes["top_k"], axes["num_experts"], axes["inter_dim"]
    rows = n_tokens * top_k
    x = (torch.randn(rows, 2 * inter_dim, generator=generator, device=device, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    _safe_activations(x, inter_dim)

    expert_of_row = (torch.arange(rows, device=device) % experts)[torch.randperm(rows, generator=generator, device=device)]
    order = torch.sort(expert_of_row, stable=True).indices          # row index (token * top_k + slot), grouped by expert
    sorted_expert = expert_of_row[order]
    counts = torch.bincount(expert_of_row, minlength=experts)
    padded = (counts + BLOCK - 1) // BLOCK * BLOCK
    group_start = torch.cumsum(counts, 0) - counts
    padded_start = torch.cumsum(padded, 0) - padded
    position = padded_start[sorted_expert] + torch.arange(rows, device=device) - group_start[sorted_expert]
    sorted_len = int(padded.sum())
    assert sorted_len == axes["sorted_len"], (sorted_len, axes["sorted_len"])
    sorted_ids = torch.full((sorted_len,), (top_k << 24) | n_tokens, dtype=torch.int64, device=device)
    sorted_ids[position] = ((order % top_k) << 24) | (order // top_k)
    num_valid_ids = torch.tensor([sorted_len], dtype=torch.int32, device=device)
    return {"x": x, "sorted_ids": sorted_ids.to(torch.int32), "num_valid_ids": num_valid_ids}
