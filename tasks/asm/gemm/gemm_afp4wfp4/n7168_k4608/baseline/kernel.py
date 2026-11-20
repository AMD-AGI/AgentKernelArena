# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER's MXFP4 GEMM assembly kernels (f4gemm_bf16_per1x32Fp4_BpreShuffle_*) with AITER's host dispatch.

Python translation of the host side of AITER's tuned dispatcher for N = 7168, K = 4608 on gfx950 with 256 CUs:
`aiter/ops/gemm_op_a4w4.py::gemm_a4w4` (tuned-table lookup by M and padded M) and the gfx950 launcher
`csrc/py_itfs_cu/asm_gemm_a4w4.cu::gemm_a4w4_asm` (KernelArgs layout, grid, split-K). The kernels are AITER's code
objects, vendored unchanged.
"""
import functools

import aka_asm

_BLOCK_SIZE_X = 256     # 4 waves of 64

# hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4.csv: kernel -> (tile_m, tile_n, kernel symbol). Every row the tuned
# table below selects; all of them expect the 16x16-preshuffled weight (bpreshuffle 1).
_TILES = {
    "32x128": (32, 128, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"),
    "64x128": (64, 128, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E"),
    "128x128": (128, 128, "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x128E"),
    "256x128": (256, 128, "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x128E"),
    "256x256": (256, 256, "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E"),
    "128x512": (128, 512, "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E"),
}
_KERNELS = {name: aka_asm.load(f"f4gemm_bf16_per1x32Fp4_BpreShuffle_{name}.co", symbol)
            for name, (_, _, symbol) in _TILES.items()}

# aiter/configs/model_configs/dsv3_a4w4_blockscale_tuned_gemm.csv (merged into AITER's a4w4 tuned table), rows gfx950,
# cu_num 256, N 7168, K 4608: M -> kernel. Every row is an asm kernel with splitK 0 (log2_k_split 0: no split-K).
_TUNED = {
    1: "32x128", 2: "32x128", 4: "32x128", 8: "32x128", 16: "32x128", 32: "64x128", 64: "32x128",
    128: "32x128", 256: "64x128", 512: "128x128", 1024: "256x128", 2048: "256x256", 4096: "256x256",
    8192: "128x512", 16384: "256x256",
}


def _next_pow2(x):
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _padded_m(m, n, gl):
    """csrc/py_itfs_cu/gemm_common.cu::getPaddedM."""
    if gl == 0:
        if m <= 256:
            return (m + 15) // 16 * 16
        if m <= 1024:
            return (m + 31) // 32 * 32
        if m <= 4096:
            return (m + 63) // 64 * 64
        return (m + 127) // 128 * 128
    if m > 8192 and n > 4096:
        return 8192
    return _next_pow2(m)


@functools.lru_cache(maxsize=None)
def _select(m, n):
    """gemm_op_a4w4.py::get_GEMM_config: exact M, then fine (gl 0) and coarse (gl 1) padded M."""
    for gl in (None, 0, 1):
        key = m if gl is None else _padded_m(m, n, gl)
        if key in _TUNED:
            return _TUNED[key]
    # Unreachable for this N: gl 1 maps every M to a power of two up to 8192, all of which are rows.
    raise ValueError(f"no tuned row for M={m}, N={n}")


def _pointer(tensor):
    # KernelArgs field `void* p; p2 _p;`: 8 bytes of pointer, 8 bytes of padding.
    return [tensor, aka_asm.u64(0)]


def _null_pointer():
    return [aka_asm.u64(0), aka_asm.u64(0)]


def _u32(value):
    # KernelArgs field `unsigned int x; p3 _p;`: 4 bytes of value, 12 bytes of padding.
    return [aka_asm.u32(value), aka_asm.u32(0), aka_asm.u32(0), aka_asm.u32(0)]


def _f32(value):
    return [aka_asm.f32(value), aka_asm.u32(0), aka_asm.u32(0), aka_asm.u32(0)]


def run(*, a, a_scales, w, w_scales, out):
    m = a.shape[0]
    n = w.shape[0]
    k = a.shape[1] * 2                  # two fp4 values per byte
    tile_m, tile_n, _ = _TILES[_select(m, n)]
    kernel = _KERNELS[_select(m, n)]

    # The tuned splitK is 0 for every row: log2_k_split 0, gdz 1 and no output zeroing (asm_gemm_a4w4.cu only
    # memsets the output when 1 << log2_k_split > 1).
    args = (_pointer(out)               # ptr_D
            + _null_pointer()           # ptr_C (no bias)
            + _pointer(a)               # ptr_A
            + _pointer(w)               # ptr_B
            + _f32(1.0) + _f32(0.0)     # alpha, beta
            + _u32(0) + _u32(0)         # stride_D0, stride_D1 (not set by the launcher)
            + _u32(out.stride(0))       # stride_C0: output row stride in elements
            + _u32(0)                   # stride_C1
            + _u32(a.stride(0) * 2)     # stride_A0, in fp4 elements
            + _u32(0)
            + _u32(w.stride(0) * 2)     # stride_B0, in fp4 elements
            + _u32(0)
            + _u32(m) + _u32(n) + _u32(k)
            + _pointer(a_scales) + _pointer(w_scales)
            + _u32(a_scales.stride(0))  # stride_ScaleA0, bytes
            + _u32(0)
            + _u32(w_scales.stride(0))  # stride_ScaleB0, bytes
            + _u32(0)
            + [aka_asm.i32(0), aka_asm.u32(0), aka_asm.u32(0), aka_asm.u32(0)])   # log2_k_split
    grid = ((n + tile_n - 1) // tile_n, (m + tile_m - 1) // tile_m, 1)
    kernel.launch(grid=grid, block=(_BLOCK_SIZE_X, 1, 1), args=args, shared_bytes=0)
