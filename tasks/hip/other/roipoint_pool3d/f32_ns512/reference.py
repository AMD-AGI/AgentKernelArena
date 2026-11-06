"""float64 reference of RoI point pooling (OpenPCDet / MMCV roipoint_pool3d)."""
import torch

NUM_SAMPLED = 512
_CHUNK = 1024


def _membership(points, boxes):
    """(M, N) bool: point n lies in box m (cz <= z <= cz + dz, |local_x| < dx/2, |local_y| < dy/2)."""
    p, bx = points.double(), boxes.double()
    sx, sy, sz = p[None, :, 0] - bx[:, None, 0], p[None, :, 1] - bx[:, None, 1], p[None, :, 2] - bx[:, None, 2]
    cos_r, sin_r = torch.cos(bx[:, None, 6]), torch.sin(bx[:, None, 6])
    lx = sx * cos_r + sy * sin_r
    ly = -sx * sin_r + sy * cos_r
    return ((sz >= 0) & (sz <= bx[:, None, 5]) & (lx.abs() < bx[:, None, 3] / 2)
            & (ly.abs() < bx[:, None, 4] / 2))


def reference(*, points, point_features, boxes3d):
    """For each box, the first NUM_SAMPLED points (in index order) inside it; fewer than that are
    repeated cyclically (slot k takes the point k % count). Returns their coordinates and features
    concatenated, (B, M, NUM_SAMPLED, 3 + C), and an int32 flag that is 1 for a box with no point
    (its pooled features are zero)."""
    batch, n_points, _ = points.shape
    n_boxes = boxes3d.shape[1]
    channels = point_features.shape[2]
    pooled = torch.zeros((batch, n_boxes, NUM_SAMPLED, 3 + channels), dtype=torch.float32, device=points.device)
    empty = torch.zeros((batch, n_boxes), dtype=torch.int32, device=points.device)
    position = torch.arange(n_points, device=points.device)
    slots = torch.arange(NUM_SAMPLED, device=points.device)
    width = min(NUM_SAMPLED, n_points)
    for b in range(batch):
        for start in range(0, n_boxes, _CHUNK):
            inside = _membership(points[b], boxes3d[b, start:start + _CHUNK])
            key = torch.where(inside, position[None, :], torch.full_like(position, n_points)[None, :])
            first = torch.topk(key, width, dim=1, largest=False).values
            count = inside.sum(1)
            source = torch.gather(first, 1, (slots[None, :] % count.clamp(min=1)[:, None]).clamp(max=width - 1))
            source = source.clamp(max=n_points - 1)  # the sentinel of an empty box
            rows = torch.cat([points[b][source].float(), point_features[b][source].float()], dim=-1)
            rows = torch.where((count > 0)[:, None, None], rows, torch.zeros_like(rows))
            pooled[b, start:start + rows.shape[0]] = rows
            empty[b, start:start + rows.shape[0]] = (count == 0).to(torch.int32)
    return {"pooled_features": pooled, "pooled_empty_flag": empty}
