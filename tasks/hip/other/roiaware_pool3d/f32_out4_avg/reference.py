"""float64 reference of RoI-aware 3D max pooling (OpenPCDet / MMCV RoIAwarePool3d, mode 'avg')."""
import torch

OUT_SIZE = 4
MAX_PTS_PER_VOXEL = 128


def _voxels(pts, roi):
    """(P,) voxel index of every point in the box (flat x, y, z) or OUT_SIZE**3 for points outside."""
    p, bx = pts.double(), roi.double()
    sx, sy, sz = p[:, 0] - bx[0], p[:, 1] - bx[1], p[:, 2] - bx[2]
    cos_r, sin_r = torch.cos(bx[6]), torch.sin(bx[6])
    lx = sx * cos_r + sy * sin_r
    ly = -sx * sin_r + sy * cos_r
    inside = (sz >= 0) & (sz <= bx[5]) & (lx.abs() < bx[3] / 2) & (ly.abs() < bx[4] / 2)
    ix = ((lx + bx[3] / 2) / (bx[3] / OUT_SIZE)).floor().clamp(0, OUT_SIZE - 1).long()
    iy = ((ly + bx[4] / 2) / (bx[4] / OUT_SIZE)).floor().clamp(0, OUT_SIZE - 1).long()
    iz = (sz / (bx[5] / OUT_SIZE)).floor().clamp(0, OUT_SIZE - 1).long()
    voxel = (ix * OUT_SIZE + iy) * OUT_SIZE + iz
    return torch.where(inside, voxel, torch.full_like(voxel, OUT_SIZE ** 3))


def reference(*, rois, pts, pts_feature):
    """pooled_features[r, x, y, z, c]: mean of feature c over the points of box r that fall into voxel
    (x, y, z) of its OUT_SIZE^3 grid; a voxel keeps only its first MAX_PTS_PER_VOXEL - 1 points in index
    order and an empty voxel is 0."""
    n_rois = rois.shape[0]
    n_points, channels = pts_feature.shape
    n_voxels = OUT_SIZE ** 3
    features = pts_feature.double()
    pooled = torch.zeros((n_rois, n_voxels, channels), dtype=torch.float64, device=pts.device)
    kept_per_voxel = MAX_PTS_PER_VOXEL - 1
    for r in range(n_rois):
        voxel = _voxels(pts, rois[r])
        order = torch.sort(voxel, stable=True).indices
        sorted_voxel = voxel[order]
        first = torch.searchsorted(sorted_voxel, torch.arange(n_voxels + 1, device=pts.device))
        rank = torch.arange(n_points, device=pts.device) - first[sorted_voxel]
        keep = (sorted_voxel < n_voxels) & (rank < kept_per_voxel)
        target = sorted_voxel[keep]
        if target.numel():
            total = torch.zeros((n_voxels, channels), dtype=torch.float64, device=pts.device)
            total.index_add_(0, target, features[order[keep]])
            count = torch.bincount(target, minlength=n_voxels).to(torch.float64)
            pooled[r] = total / count.clamp(min=1)[:, None]
    return pooled.reshape(n_rois, OUT_SIZE, OUT_SIZE, OUT_SIZE, channels).float()
