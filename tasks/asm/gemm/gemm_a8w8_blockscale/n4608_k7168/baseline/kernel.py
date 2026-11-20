# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER's FP8 block-scale GEMM assembly kernels (fp8gemm_bf16_blockscale_BpreShuffle_*) with AITER's host dispatch.

Python translation of the host side of AITER's tuned dispatcher for N = 4608, K = 7168 on gfx950 with 256 CUs:
`aiter/ops/gemm_op_a8w8.py::gemm_a8w8_blockscale_bpreshuffle` (tuned-table lookup by M, padded M),
`gemm_a8w8_blockscale_bpreshuffle_asm` (zero bias buffer) and `csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu`
(kernel heuristic, split-K check and output zeroing, KernelArgs layout, grid). The kernels are AITER's code objects,
vendored unchanged.
"""
import functools

import torch

import aka_asm

_N, _K = 4608, 7168
_TILE_N, _TILE_K = 128, 128
_BLOCK_SIZE_X = 256
_NUM_CU = 256           # MI355X; the tuned rows below are AITER's gfx950 / cu_num 256 rows

# hsa/gfx950/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv: tile_m -> code object, kernel symbol. Every kernel
# supports split-K and expects the 16x16-preshuffled weight. The C++ heuristic scans these configs in the iteration
# order of a std::unordered_map; on the pinned image 128x128 comes last (see README.md), which is the order below.
_KERNELS = {
    32: aka_asm.load("fp8gemm_bf16_blockscale_BpreShuffle_32x128.co",
                     "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_32x128E"),
    48: aka_asm.load("fp8gemm_bf16_blockscale_BpreShuffle_48x128.co",
                     "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_48x128E"),
    64: aka_asm.load("fp8gemm_bf16_blockscale_BpreShuffle_64x128.co",
                     "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_64x128E"),
    80: aka_asm.load("fp8gemm_bf16_blockscale_BpreShuffle_80x128.co",
                     "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_80x128E"),
    96: aka_asm.load("fp8gemm_bf16_blockscale_BpreShuffle_96x128.co",
                     "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_96x128E"),
    128: aka_asm.load("fp8gemm_bf16_blockscale_BpreShuffle_128x128.co",
                      "_ZN5aiter43fp8gemm_bf16_blockscale_BpreShuffle_128x128E"),
}

# aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_dsv3.csv, rows gfx950, cu_num 256,
# N 4608, K 7168, M <= 864: M -> (tile_m of kernelName, splitK). None: the row's libtype is ck (a Composable Kernel
# instance, not vendorable); those M use the asm launcher's own heuristic instead (see README.md).
_TUNED = {
    1: (32, 7), 2: (64, 6), 4: (48, 7), 8: (64, 6), 16: (64, 7), 32: (80, 6), 48: (48, 6), 64: (80, 6),
    80: (96, 6), 96: (96, 6), 112: (64, 3), 128: (64, 3), 144: (80, 3), 160: (80, 3), 176: (96, 3),
    192: None, 208: (128, 3), 224: None, 240: (80, 2), 256: (96, 2), 288: (96, 2), 320: (48, 1),
    352: (128, 2), 384: (64, 1), 416: (64, 1), 448: (64, 1), 480: (80, 1), 512: (80, 1), 544: (80, 1),
    576: (96, 1), 608: (96, 1), 640: (96, 1), 672: (96, 1), 704: (128, 1), 736: (128, 1), 768: (128, 1),
    800: (128, 1), 832: None, 864: (128, 1),
}
_MISSING = object()


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


def _tuned_config(m, n):
    """gemm_op_a8w8.py::get_CKGEMM_config: exact M, then fine (gl 0) and coarse (gl 1) padded M."""
    for gl in (None, 0, 1):
        key = m if gl is None else _padded_m(m, n, gl)
        config = _TUNED.get(key, _MISSING)
        if config is not _MISSING:
            return config
    return None


def _heuristic(m, n):
    """asm_a8w8_blockscale_bpreshuffle.cu::get_heuristic_fp8_kernel without a tuned kernel name (splitK 1)."""
    round_, empty_cu, effi, selected = 0xFFFFFFFF, _NUM_CU, 1.0, None
    for tile_m in _KERNELS:
        if n % _TILE_N:
            continue
        tg_num = ((m + tile_m - 1) // tile_m) * ((n + _TILE_N - 1) // _TILE_N)
        local_round = (tg_num + _NUM_CU - 1) // _NUM_CU
        local_effi = (tile_m * _TILE_N) // (tile_m + _TILE_N)    # integer division in the C++ source
        if local_round < round_ or (local_round == round_ and
                                    (empty_cu > local_round * _NUM_CU - tg_num or local_effi > effi)):
            round_, empty_cu, effi, selected = local_round, local_round * _NUM_CU - tg_num, local_effi, tile_m
    return selected, 1


@functools.lru_cache(maxsize=None)
def _select(m, n):
    config = _tuned_config(m, n)
    return config if config is not None else _heuristic(m, n)


@functools.lru_cache(maxsize=16)
def _zero_bias(device_index, n):
    """gemm_op_a8w8.py::get_zero_bias_buf: a cached float32 [1, N] zero bias."""
    return torch.zeros(1, n, dtype=torch.float32, device=torch.device("cuda", device_index))


def _u32_padded(value):
    # KernelArgs field `unsigned int x; p3 _p;`: 4 bytes of value, 12 bytes of padding.
    return [aka_asm.u32(value), aka_asm.u32(0), aka_asm.u32(0), aka_asm.u32(0)]


def run(*, a, a_scale, b, b_scale, out):
    m, k = a.shape
    n = out.shape[1]
    tile_m, split_k = _select(m, n)
    kernel = _KERNELS[tile_m]

    # Split-K alignment (explicit splitK from the tuned table must already be 128-aligned per split).
    k_per_split = (k + split_k - 1) // split_k
    k_per_split = (k_per_split + _TILE_K - 1) // _TILE_K * _TILE_K
    if (k + k_per_split - 1) // k_per_split != split_k:
        raise ValueError(f"splitK {split_k} does not split K={k} into 128-aligned parts")
    if split_k > 1:
        out.zero_()                       # hipMemsetAsync: the splits accumulate atomically into out

    bias = _zero_bias(a.device.index, n)
    pad = aka_asm.u64(0)                  # KernelArgs field `void* p; p2 _p;`: 8 bytes of padding
    args = [out, pad, a, pad, b, pad, a_scale, pad, b_scale, pad, bias, pad]
    for value in (m, n, k,
                  a.stride(0),            # lda, elements (bytes for fp8)
                  b.stride(0),            # ldb
                  out.stride(0) * 2,      # ldc, bytes of bfloat16
                  split_k,
                  m,                      # scale_m (1-row groups)
                  (n + 127) // 128,       # scale_n
                  (k + 127) // 128):      # scale_k
        args += _u32_padded(value)
    grid_x = (n // _TILE_N) * split_k
    grid_y = (m + tile_m - 1) // tile_m
    kernel.launch(grid=(grid_x, grid_y, 1), block=(_BLOCK_SIZE_X, 1, 1), args=args, shared_bytes=0)
