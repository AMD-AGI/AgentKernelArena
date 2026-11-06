"""A LiDAR scene: oriented 3D boxes (RoIs) and a point cloud with per-point features.

Boxes (cx, cy, cz, dx, dy, dz, heading) are drawn like vehicle annotations of a driving dataset
(forward range 0..70 m, lateral -40..40 m, bottom height -3..1 m, length 3..5 m, width 1.5..2.2 m,
height 1.4..2 m, heading uniform in [-pi, pi)). Half of the points fill the same volume uniformly
(background); the other half lie inside randomly chosen boxes among the first three quarters of the
box list, so some boxes hold hundreds of points (more than the 512 that are sampled), some few
(their points are repeated) and the remaining boxes are empty unless a background point lands in them.

The kernel decides membership in float32 while the reference works in float64, so a point lying within
float32 rounding error of a box face would make an exact comparison depend on rounding. The generator
therefore redraws every point that lies within MARGIN of a face of a box that it could otherwise be inside
of (and finally moves stubborn ones far away), leaving no ambiguous decision.
"""
import torch

MARGIN = 1e-3
_CHUNK = 1024
_FAR = 1.0e4
_FOREGROUND = 0.5


def _ambiguous(points, boxes):
    """Points (N, 3) within MARGIN of a face of some box (N-sized bool mask)."""
    p, bx = points.double(), boxes.double()
    cos_r, sin_r = torch.cos(bx[:, 6]), torch.sin(bx[:, 6])
    flags = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    for start in range(0, points.shape[0], _CHUNK):
        q = p[start:start + _CHUNK]
        sx, sy, sz = (q[:, None, 0] - bx[None, :, 0], q[:, None, 1] - bx[None, :, 1],
                      q[:, None, 2] - bx[None, :, 2])
        lx = sx * cos_r + sy * sin_r
        ly = -sx * sin_r + sy * cos_r
        half_x, half_y, height = bx[None, :, 3] / 2, bx[None, :, 4] / 2, bx[None, :, 5]
        near_x, near_y = (lx.abs() - half_x).abs() < MARGIN, (ly.abs() - half_y).abs() < MARGIN
        near_z = ((sz.abs() < MARGIN) | ((sz - height).abs() < MARGIN))
        in_x, in_y = lx.abs() < half_x + MARGIN, ly.abs() < half_y + MARGIN
        in_z = (sz > -MARGIN) & (sz < height + MARGIN)
        bad = (near_x & in_y & in_z) | (near_y & in_x & in_z) | (near_z & in_x & in_y)
        flags[start:start + q.shape[0]] = bad.any(dim=1)
    return flags


def _uniform(generator, device, shape, low, high):
    return low + (high - low) * torch.rand(shape, generator=generator, device=device, dtype=torch.float32)


def _background(generator, device, count):
    return torch.stack([_uniform(generator, device, (count,), 0.0, 70.0),
                        _uniform(generator, device, (count,), -40.0, 40.0),
                        _uniform(generator, device, (count,), -3.0, 1.0)], dim=-1)


def _inside(generator, device, boxes, count):
    """``count`` points uniformly inside boxes chosen among the first three quarters of ``boxes``."""
    active = max(1, (3 * boxes.shape[0]) // 4)
    chosen = boxes[torch.randint(0, active, (count,), generator=generator, device=device)]
    lx = (_uniform(generator, device, (count,), -1.0, 1.0)) * chosen[:, 3] / 2
    ly = (_uniform(generator, device, (count,), -1.0, 1.0)) * chosen[:, 4] / 2
    lz = _uniform(generator, device, (count,), 0.0, 1.0) * chosen[:, 5]
    cos_r, sin_r = torch.cos(chosen[:, 6]), torch.sin(chosen[:, 6])
    return torch.stack([chosen[:, 0] + lx * cos_r - ly * sin_r,
                        chosen[:, 1] + lx * sin_r + ly * cos_r,
                        chosen[:, 2] + lz], dim=-1)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, n_points, n_boxes = axes["batch"], axes["n_points"], axes["n_boxes"]
    boxes = torch.stack([_uniform(generator, device, (batch, n_boxes), 0.0, 70.0),
                         _uniform(generator, device, (batch, n_boxes), -40.0, 40.0),
                         _uniform(generator, device, (batch, n_boxes), -3.0, 1.0),
                         _uniform(generator, device, (batch, n_boxes), 3.0, 5.0),
                         _uniform(generator, device, (batch, n_boxes), 1.5, 2.2),
                         _uniform(generator, device, (batch, n_boxes), 1.4, 2.0),
                         _uniform(generator, device, (batch, n_boxes), -3.14159265, 3.14159265)], dim=-1)
    points = torch.empty((batch, n_points, 3), dtype=torch.float32, device=device)
    n_foreground = int(n_points * _FOREGROUND)
    for b in range(batch):
        cloud = torch.cat([_inside(generator, device, boxes[b], n_foreground),
                           _background(generator, device, n_points - n_foreground)])
        cloud = cloud[torch.randperm(n_points, generator=generator, device=device)]
        for _ in range(8):
            bad = _ambiguous(cloud, boxes[b])
            count = int(bad.sum())
            if count == 0:
                break
            cloud[bad] = _background(generator, device, count)
        else:
            bad = _ambiguous(cloud, boxes[b])
            cloud[bad] = _FAR
        points[b] = cloud
    return {"points": points, "boxes3d": boxes}
