"""float64 reference of three_nn (MMCV / PointNet++ feature propagation)."""
import torch

_CHUNK = 256


def reference(*, unknown, known):
    """For each target point, the Euclidean distances to and indices of its 3 nearest source points of
    the same batch element, nearest first; points at equal distance are ordered by lowest index."""
    batch, n_target, _ = unknown.shape
    n_source = known.shape[1]
    targets, sources = unknown.double(), known.double()
    dist = torch.empty((batch, n_target, 3), dtype=torch.float64, device=unknown.device)
    idx = torch.empty((batch, n_target, 3), dtype=torch.int32, device=unknown.device)
    for b in range(batch):
        for start in range(0, n_target, _CHUNK):
            t = targets[b, start:start + _CHUNK]
            d2 = (t[:, None, :] - sources[b][None, :, :]).square().sum(-1)
            order = torch.sort(d2, dim=1, stable=True).indices[:, :3]
            dist[b, start:start + t.shape[0]] = torch.gather(d2, 1, order).sqrt()
            idx[b, start:start + t.shape[0]] = order.to(torch.int32)
    return {"dist": dist, "idx": idx}
