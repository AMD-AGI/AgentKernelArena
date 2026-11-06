"""A LiDAR scene: oriented 3D boxes (RoIs) and a point cloud with per-point features.

Boxes (cx, cy, cz, dx, dy, dz, heading) are drawn like vehicle annotations of a driving dataset
(forward range 0..70 m, lateral -40..40 m, bottom height -3..1 m, length 3..5 m, width 1.5..2.2 m,
height 1.4..2 m, heading uniform in [-pi, pi)). A quarter of the points lie inside randomly chosen boxes
and the rest fill the same volume uniformly, so a box holds a few points up to hundreds (a voxel then
exceeds the 127 points the operator keeps per voxel when only a few boxes share the foreground points).

The kernel decides box membership and voxel index in float32 while the reference works in float64, so a
point lying within float32 rounding error of a box face or of a voxel boundary would make an exact
comparison depend on rounding. The generator therefore redraws every point that lies within MARGIN of
a face or voxel boundary of a box that it could otherwise be inside of (and finally moves stubborn ones
far away), leaving no ambiguous decision.
"""
import torch

OUT_SIZE = 4
MARGIN = 1e-3
_CHUNK = 1024
_FAR = 1.0e4
_FOREGROUND = 0.25


def _near_plane(coordinate, size):
    """Distance of ``coordinate`` (relative to the box face at 0) to the nearest voxel boundary."""
    cell = size / OUT_SIZE
    scaled = coordinate / cell
    return (scaled - scaled.round()).abs() * cell


def _ambiguous(points, rois):
    """Points (P, 3) within MARGIN of a face or voxel boundary of some box (P-sized bool mask)."""
    p, bx = points.double(), rois.double()
    cos_r, sin_r = torch.cos(bx[:, 6]), torch.sin(bx[:, 6])
    flags = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    for start in range(0, points.shape[0], _CHUNK):
        q = p[start:start + _CHUNK]
        sx, sy, sz = (q[:, None, 0] - bx[None, :, 0], q[:, None, 1] - bx[None, :, 1],
                      q[:, None, 2] - bx[None, :, 2])
        lx = sx * cos_r + sy * sin_r
        ly = -sx * sin_r + sy * cos_r
        size_x, size_y, size_z = bx[None, :, 3], bx[None, :, 4], bx[None, :, 5]
        ux, uy = lx + size_x / 2, ly + size_y / 2
        near_x, near_y = _near_plane(ux, size_x) < MARGIN, _near_plane(uy, size_y) < MARGIN
        near_z = _near_plane(sz, size_z) < MARGIN
        in_x, in_y = (ux > -MARGIN) & (ux < size_x + MARGIN), (uy > -MARGIN) & (uy < size_y + MARGIN)
        in_z = (sz > -MARGIN) & (sz < size_z + MARGIN)
        bad = (near_x & in_y & in_z) | (near_y & in_x & in_z) | (near_z & in_x & in_y)
        flags[start:start + q.shape[0]] = bad.any(dim=1)
    return flags


def _uniform(generator, device, shape, low, high):
    return low + (high - low) * torch.rand(shape, generator=generator, device=device, dtype=torch.float32)


def _background(generator, device, count):
    return torch.stack([_uniform(generator, device, (count,), 0.0, 70.0),
                        _uniform(generator, device, (count,), -40.0, 40.0),
                        _uniform(generator, device, (count,), -3.0, 1.0)], dim=-1)


def _inside(generator, device, rois, count):
    chosen = rois[torch.randint(0, rois.shape[0], (count,), generator=generator, device=device)]
    lx = _uniform(generator, device, (count,), -1.0, 1.0) * chosen[:, 3] / 2
    ly = _uniform(generator, device, (count,), -1.0, 1.0) * chosen[:, 4] / 2
    lz = _uniform(generator, device, (count,), 0.0, 1.0) * chosen[:, 5]
    cos_r, sin_r = torch.cos(chosen[:, 6]), torch.sin(chosen[:, 6])
    return torch.stack([chosen[:, 0] + lx * cos_r - ly * sin_r,
                        chosen[:, 1] + lx * sin_r + ly * cos_r,
                        chosen[:, 2] + lz], dim=-1)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_rois, n_points = axes["n_rois"], axes["n_points"]
    rois = torch.stack([_uniform(generator, device, (n_rois,), 0.0, 70.0),
                        _uniform(generator, device, (n_rois,), -40.0, 40.0),
                        _uniform(generator, device, (n_rois,), -3.0, 1.0),
                        _uniform(generator, device, (n_rois,), 3.0, 5.0),
                        _uniform(generator, device, (n_rois,), 1.5, 2.2),
                        _uniform(generator, device, (n_rois,), 1.4, 2.0),
                        _uniform(generator, device, (n_rois,), -3.14159265, 3.14159265)], dim=-1)
    n_foreground = int(n_points * _FOREGROUND)
    cloud = torch.cat([_inside(generator, device, rois, n_foreground),
                       _background(generator, device, n_points - n_foreground)])
    pts = cloud[torch.randperm(n_points, generator=generator, device=device)]
    for _ in range(8):
        bad = _ambiguous(pts, rois)
        count = int(bad.sum())
        if count == 0:
            break
        pts[bad] = _background(generator, device, count)
    else:
        pts[_ambiguous(pts, rois)] = _FAR
    return {"rois": rois, "pts": pts}
