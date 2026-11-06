"""Float64 k-nearest-neighbour reference used only for correctness."""
import torch

K = 5  # the scenario's constant k
GRID = 128.0 ** 2  # squared distances of the task's grid points are multiples of 1 / GRID
CHUNK_ELEMENTS = 1 << 24


def reference(*, xyz, center_xyz):
    """The K nearest points of xyz[b] to every centre, nearest first; equal distances resolve to
    the smaller point index. Distances are formed in float64 and ordered by the exact integer key
    (distance * GRID) * n_points + index, which is exact for the task's grid-valued coordinates."""
    points = xyz.double()
    centers = center_xyz.double()
    batch, n_points, _ = points.shape
    n_query = centers.shape[1]
    positions = torch.arange(n_points, dtype=torch.int64, device=points.device)
    step = max(1, CHUNK_ELEMENTS // (batch * n_points))
    idx_parts, dist_parts = [], []
    for start in range(0, n_query, step):
        chunk = centers[:, start:start + step]
        d2 = (chunk.unsqueeze(2) - points.unsqueeze(1)).square().sum(dim=-1)
        key = torch.round(d2 * GRID).to(torch.int64) * n_points + positions
        best = torch.topk(key, K, dim=-1, largest=False, sorted=True).values
        idx_parts.append((best % n_points).to(torch.int32))
        dist_parts.append(torch.div(best, n_points, rounding_mode="floor").double() / GRID)
    return torch.cat(idx_parts, dim=1), torch.cat(dist_parts, dim=1).to(torch.float32)
