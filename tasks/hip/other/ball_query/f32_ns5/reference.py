"""float64 reference of MMCV's ball query (exact index order and padding)."""
import torch

NSAMPLE = 5
_CHUNK_CENTERS = 128


def _squared(radius):
    r = torch.tensor(radius, dtype=torch.float32)
    return float((r * r).item())


def reference(*, xyz, center_xyz, min_radius, max_radius):
    """For each center, the indices of the first NSAMPLE points (in index order) with squared distance
    d2 == 0 or min_radius^2 <= d2 < max_radius^2. Remaining slots repeat the first match; a center
    without any match yields zeros."""
    batch, n_points, _ = xyz.shape
    n_centers = center_xyz.shape[1]
    lower2, upper2 = _squared(min_radius), _squared(max_radius)
    points = xyz.double()
    centers = center_xyz.double()
    position = torch.arange(n_points, device=xyz.device)
    out = torch.zeros((batch, n_centers, NSAMPLE), dtype=torch.int32, device=xyz.device)
    slots = torch.arange(NSAMPLE, device=xyz.device)
    for b in range(batch):
        for start in range(0, n_centers, _CHUNK_CENTERS):
            c = centers[b, start:start + _CHUNK_CENTERS]
            d2 = (c[:, None, :] - points[b][None, :, :]).square().sum(-1)
            match = (d2 == 0) | ((d2 >= lower2) & (d2 < upper2))
            key = torch.where(match, position[None, :].expand_as(match),
                              torch.full_like(match, n_points, dtype=torch.long))
            first = torch.topk(key, min(NSAMPLE, n_points), dim=1, largest=False).values
            if first.shape[1] < NSAMPLE:
                first = torch.cat([first, torch.full((first.shape[0], NSAMPLE - first.shape[1]), n_points,
                                                     dtype=first.dtype, device=first.device)], dim=1)
            count = match.sum(1, keepdim=True)
            padded = torch.where(slots[None, :] < count, first, first[:, :1])
            padded = torch.where(count > 0, padded, torch.zeros_like(padded))
            out[b, start:start + c.shape[0]] = padded.to(torch.int32)
    return out
