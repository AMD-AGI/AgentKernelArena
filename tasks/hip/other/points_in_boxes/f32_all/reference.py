"""float64 reference of MMCV's points_in_boxes_all: the flag of every (point, box) pair."""
import torch

_CHUNK_POINTS = 2048


def reference(*, points, boxes):
    """Box format (cx, cy, cz, dx, dy, dz, heading) with cz at the bottom. A point is inside when
    cz <= z <= cz + dz and, rotated by -heading about the box center, |local_x| < dx/2 and
    |local_y| < dy/2 (faces in x and y are outside). Returns int32 [batch, n_points, n_boxes]:
    1 where the point is inside the box, else 0."""
    batch, n_points, _ = points.shape
    n_boxes = boxes.shape[1]
    p = points.double()
    bx = boxes.double()
    out = torch.zeros((batch, n_points, n_boxes), dtype=torch.int32, device=points.device)
    for b in range(batch):
        box = bx[b]
        cos_r, sin_r = torch.cos(box[:, 6]), torch.sin(box[:, 6])
        for start in range(0, n_points, _CHUNK_POINTS):
            q = p[b, start:start + _CHUNK_POINTS]
            sx = q[:, None, 0] - box[None, :, 0]
            sy = q[:, None, 1] - box[None, :, 1]
            sz = q[:, None, 2] - box[None, :, 2]
            local_x = sx * cos_r + sy * sin_r
            local_y = -sx * sin_r + sy * cos_r
            inside = ((sz >= 0) & (sz <= box[None, :, 5]) & (local_x.abs() < box[None, :, 3] / 2)
                      & (local_y.abs() < box[None, :, 4] / 2))
            out[b, start:start + q.shape[0]] = inside.to(torch.int32)
    return out
