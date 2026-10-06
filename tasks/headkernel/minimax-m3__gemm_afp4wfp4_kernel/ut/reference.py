"""Independent CPU E2M1/E8M0 reference; no native or candidate imports.

These small pure-Python routines validate the format/control contract. They are
not a claim that an uncaptured workload has been numerically qualified.
"""
import math


FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
       -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def decode_e8m0(byte):
    if type(byte) is not int or not 0 <= byte <= 255:
        raise ValueError("E8M0 storage must be an unsigned byte")
    # 0 is the finite subnormal scale 2**-127, not zero. 255 is NaN.
    return math.nan if byte == 255 else math.ldexp(1.0, byte - 127)


def unpack_row(packed, scales):
    if len(packed) % 16 or len(scales) != len(packed) // 16:
        raise ValueError("one E8M0 scale is required for every 32 logical K values")
    result = []
    for index, byte in enumerate(packed):
        if type(byte) is not int or not 0 <= byte <= 255:
            raise ValueError("FP4 storage must be unsigned bytes")
        scale = decode_e8m0(scales[index // 16])
        if not math.isfinite(scale):
            raise ValueError("live E8M0 scale 255 is NaN; capture requires explicit handling")
        result.extend((FP4[byte & 15] * scale, FP4[byte >> 4] * scale))
    return result


def storage_rows(storage, shape, strides, offset=0):
    """Read logical rows from raw byte storage without assuming contiguity."""
    if len(shape) != 2 or len(strides) != 2 or any(type(n) is not int or n <= 0 for n in (*shape, *strides)):
        raise ValueError("positive two-dimensional shape/strides are required")
    if type(offset) is not int or offset < 0:
        raise ValueError("invalid storage offset")
    end = offset + (shape[0] - 1) * strides[0] + (shape[1] - 1) * strides[1]
    if end >= len(storage):
        raise ValueError("view exceeds captured storage")
    return [[storage[offset + i * strides[0] + j * strides[1]] for j in range(shape[1])]
            for i in range(shape[0])]


def matmul(x, w, x_scales, w_scales, *, splitk_block_size=None):
    """Return FP64 sums or unreduced split-K sums; output casting is separate.

    splitk_block_size counts logical values, whereas x/w columns count bytes.
    Actual GPU accumulation/rounding and final dtype need captured calibration.
    """
    if not x or not w or len(x) != len(x_scales) or len(w) != len(w_scales):
        raise ValueError("matrix/scale row counts differ")
    a = [unpack_row(row, scale) for row, scale in zip(x, x_scales)]
    b = [unpack_row(row, scale) for row, scale in zip(w, w_scales)]
    k = len(a[0])
    if any(len(row) != k for row in a + b):
        raise ValueError("packed activation/weight K dimensions differ")
    block = k if splitk_block_size is None else splitk_block_size
    if type(block) is not int or block <= 0 or block % 32:
        raise ValueError("split-K block must be a positive multiple of 32 logical values")
    pieces = [[[math.fsum(aa[t] * bb[t] for t in range(start, min(start + block, k)))
                for bb in b] for aa in a] for start in range(0, k, block)]
    return pieces[0] if splitk_block_size is None else pieces
