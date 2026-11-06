import torch

_CHUNK_ELEMENTS = 1 << 27


def reference(*, a, b):
    """out = a @ b.T accumulated in float64; rows of a are processed in chunks to bound memory."""
    m, k = a.shape
    b64 = b.double()
    out = torch.empty((m, b.shape[0]), dtype=torch.float64, device=a.device)
    rows = max(1, _CHUNK_ELEMENTS // k)
    for start in range(0, m, rows):
        out[start:start + rows] = a[start:start + rows].double() @ b64.T
    return out
