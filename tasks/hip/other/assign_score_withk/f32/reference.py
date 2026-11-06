"""float64 reference of PAConv assign_score_withk (sum aggregation, neighbor kernel input)."""
import torch

_CHUNK_CENTERS = 256


def reference(*, scores, point_features, center_features, knn_idx):
    """out[b, o, n, k] = sum_m scores[b, n, k, m] * (point_features[b, i, m, o] - center_features[b, c, m, o])
    with i = knn_idx[b, n, k] and c = knn_idx[b, n, 0]; zero where i is outside [0, n_points)."""
    batch, n_points = point_features.shape[:2]
    n_centers, k = scores.shape[1:3]
    out_dim = point_features.shape[3]
    points = point_features.double()
    centers = center_features.double()
    weights = scores.double()
    out = torch.zeros((batch, out_dim, n_centers, k), dtype=torch.float64, device=scores.device)
    for b in range(batch):
        for start in range(0, n_centers, _CHUNK_CENTERS):
            idx = knn_idx[b, start:start + _CHUNK_CENTERS]                    # (c, K)
            valid = (idx >= 0) & (idx < n_points)
            nb = idx.clamp(0, n_points - 1)
            ct = knn_idx[b, start:start + _CHUNK_CENTERS, :1].clamp(0, n_points - 1)
            diff = points[b][nb] - centers[b][ct.expand_as(nb)]               # (c, K, M, O)
            value = torch.einsum("ckm,ckmo->ock", weights[b, start:start + _CHUNK_CENTERS], diff)
            out[b, :, start:start + idx.shape[0]] = value * valid.unsqueeze(0).double()
    return out
