"""Float64 furthest point sampling reference used only for correctness."""
import torch


def reference(*, points_xyz):
    """Greedy FPS from point 0: each step picks the point whose distance to the selected set is
    largest; equal distances resolve to the smallest point index (torch.argmax returns the first
    maximum). The number of samples is max(1, n_points // 4), the task's derived n_samples axis."""
    points = points_xyz.double()
    batch, n_points, _ = points.shape
    n_samples = max(1, n_points // 4)
    rows = torch.arange(batch, device=points.device)
    idx = torch.zeros(batch, n_samples, dtype=torch.int64, device=points.device)
    min_dist = torch.full((batch, n_points), 1.0e10, dtype=torch.float64, device=points.device)
    last = torch.zeros(batch, dtype=torch.int64, device=points.device)
    for step in range(1, n_samples):
        center = points[rows, last].unsqueeze(1)
        dist = (points - center).square().sum(dim=-1)
        min_dist = torch.minimum(min_dist, dist)
        last = min_dist.argmax(dim=1)
        idx[:, step] = last
    return idx.to(torch.int32)
