"""High-precision reference of (I + L)^-1, L the strict lower triangle of each 64x64 chunk block of A.

The diagonal of A and everything above it are not part of the matrix: torch.tril(..., diagonal=-1)
drops them before the solve."""
import torch

CHUNK_SIZE = 64


def reference(*, A):
    tokens, heads, _ = A.shape
    out = torch.zeros(tokens, heads, CHUNK_SIZE, dtype=torch.float64, device=A.device)
    full = tokens // CHUNK_SIZE
    step = 2048  # chunks per step, to bound float64 temporaries
    eye = torch.eye(CHUNK_SIZE, dtype=torch.float64, device=A.device)
    for start in range(0, full, step):
        stop = min(start + step, full)
        blocks = A[start * CHUNK_SIZE:stop * CHUNK_SIZE].double().view(stop - start, CHUNK_SIZE, heads, CHUNK_SIZE)
        blocks = blocks.permute(0, 2, 1, 3)  # [chunk, head, row, col]
        inverse = torch.linalg.solve_triangular(eye + torch.tril(blocks, diagonal=-1), eye.expand_as(blocks),
                                                upper=False)
        out[start * CHUNK_SIZE:stop * CHUNK_SIZE] = inverse.permute(0, 2, 1, 3).reshape(-1, heads, CHUNK_SIZE)
    rest = tokens - full * CHUNK_SIZE
    if rest:  # the last chunk is partial: invert only its rest x rest block
        blocks = A[full * CHUNK_SIZE:].double().permute(1, 0, 2)[:, :, :rest]  # [head, row, col]
        matrix = torch.eye(rest, dtype=torch.float64, device=A.device) + torch.tril(blocks, diagonal=-1)
        inverse = torch.linalg.solve_triangular(matrix, torch.eye(rest, dtype=torch.float64, device=A.device).expand_as(matrix),
                                                upper=False)
        out[full * CHUNK_SIZE:, :, :rest] = inverse.permute(1, 0, 2)
    return out
