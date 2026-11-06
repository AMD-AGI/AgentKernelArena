"""AWQ dequantization reference used only for correctness."""
import torch

# Output column j of a packed int32 holds the nibble at bit offset 4 * _AWQ_ORDER[j].
_AWQ_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
_CHUNK_ROWS = 8192


def reference(*, qweight, scales, zeros):
    k, n_packed = qweight.shape
    n = n_packed * 8
    group_size = k // scales.shape[0]
    shifts = torch.tensor([4 * i for i in _AWQ_ORDER], dtype=torch.int64, device=qweight.device)
    out = torch.empty((k, n), dtype=torch.float32, device=qweight.device)
    for start in range(0, k, _CHUNK_ROWS):
        stop = min(start + _CHUNK_ROWS, k)
        rows = torch.arange(start, stop, device=qweight.device)
        groups = rows // group_size
        q = qweight[start:stop].to(torch.int64)[:, :, None]
        z = zeros[groups].to(torch.int64)[:, :, None]
        codes = ((q >> shifts) & 0xF).reshape(stop - start, n)
        zero_points = ((z >> shifts) & 0xF).reshape(stop - start, n)
        # (code - zero) is an integer in [-15, 15] and a float16 scale has 11 significant bits,
        # so the float64 product is exactly representable in float32.
        out[start:stop] = ((codes - zero_points).double() * scales[groups].double()).float()
    return out
