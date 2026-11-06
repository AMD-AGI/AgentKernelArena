"""Gates and triangular inverses of the Kimi Delta Attention chunked prefill.

``gk`` is the chunk-local cumulative per-channel log-decay scaled to the log2 domain, drawn like the
production gates: per-token log2-decays in [-0.25, 0] cumulated inside consecutive 64-token chunks
in float32, so ``gk`` is non-increasing in every chunk.

``A`` is the per-chunk inverse (I + L)^-1 of a unit lower-triangular matrix, as produced by the
triangular solve of production. The strictly lower part ``L`` is small (N(0, 0.05^2) entries, the
size of beta * k k^T products) and confined to the valid tokens of the chunk, so ``A`` is unit lower
triangular, zero above the diagonal and, in a partial last chunk, the identity in its padding.
"""
import torch

CHUNK_SIZE = 64


def _gates(tokens, heads, dim, generator, device):
    gk = torch.empty(tokens, heads, dim, dtype=torch.float32, device=device)
    rows = 8192 * CHUNK_SIZE  # whole chunks per step, to bound temporaries
    for start in range(0, tokens, rows):
        stop = min(start + rows, tokens)
        step = -0.25 * torch.rand(stop - start, heads, dim, generator=generator, device=device,
                                  dtype=torch.float32)
        pad = (-(stop - start)) % CHUNK_SIZE
        if pad:
            step = torch.cat([step, step.new_zeros(pad, heads, dim)])
        acc = step.view(-1, CHUNK_SIZE, heads, dim).cumsum(dim=1).view(-1, heads, dim)
        gk[start:stop] = acc[: stop - start]
    return gk


def _inverses(tokens, heads, generator, device):
    chunks = -(-tokens // CHUNK_SIZE)
    out = torch.empty(chunks * CHUNK_SIZE, heads, CHUNK_SIZE, dtype=torch.bfloat16, device=device)
    eye = torch.eye(CHUNK_SIZE, dtype=torch.float32, device=device)
    lower = torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool, device=device), diagonal=-1)
    index = torch.arange(CHUNK_SIZE, device=device)
    per_step = 256
    for first in range(0, chunks, per_step):
        count = min(per_step, chunks - first)
        entries = 0.05 * torch.randn(count, heads, CHUNK_SIZE, CHUNK_SIZE, generator=generator,
                                     device=device, dtype=torch.float32)
        valid = (first + torch.arange(count, device=device))[:, None] * CHUNK_SIZE + index < tokens
        mask = lower & valid[:, None, :, None] & valid[:, None, None, :]
        system = eye + torch.where(mask, entries, torch.zeros_like(entries))
        inverse = torch.linalg.solve_triangular(system, eye.expand_as(system), upper=False,
                                                unitriangular=True)
        out[first * CHUNK_SIZE:(first + count) * CHUNK_SIZE] = (
            inverse.permute(0, 2, 1, 3).reshape(count * CHUNK_SIZE, heads, CHUNK_SIZE).bfloat16())
    return out[:tokens].contiguous()


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    tokens, heads = axes["n_tokens"], axes["num_heads"]
    return {
        "A": _inverses(tokens, heads, generator, device),
        "gk": _gates(tokens, heads, axes["head_dim_k"], generator, device),
    }
