"""High-precision reference of the Mamba2 SSD intra-chunk C B^T Gram matrices."""
import torch

CHUNK_SIZE = 256


def reference(*, C, B, cu_chunk_seqlens):
    ngroups = C.shape[1]
    bounds = cu_chunk_seqlens.tolist()
    nchunks = len(bounds) - 1
    out = torch.zeros(nchunks, ngroups, CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float64, device=C.device)
    for c in range(nchunks):
        start, stop = bounds[c], bounds[c + 1]
        n = stop - start
        cc = C[start:stop].double().permute(1, 0, 2)  # [group, token, k]
        bb = B[start:stop].double().permute(1, 0, 2)
        out[c, :, :n, :n] = cc @ bb.transpose(1, 2)
    return out
