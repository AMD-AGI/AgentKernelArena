import torch


def reference(*, src_pool, src_offsets, sizes, dst_offsets):
    """dst[dst_offsets[i] : dst_offsets[i] + sizes[i]] = src[src_offsets[i] : ... + sizes[i]]."""
    src = src_pool.reshape(-1)
    dst = torch.zeros_like(src_pool).reshape(-1)
    for i in range(src_offsets.shape[0]):
        s, d, n = int(src_offsets[i]), int(dst_offsets[i]), int(sizes[i])
        dst[d:d + n] = src[s:s + n]
    return dst.reshape(src_pool.shape)
