# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 GEMM out = a @ b.T with AITER's FlyDSL split-K HGEMM kernel and a per-M tile dispatch table (N=6144, K=16384).

AITER's checkout has no tuned bf16 row for N=6144, K=16384, so its gemm_a16w16 sends every M to the
hipBLASLt/torch default, which cannot be vendored. This table adapts the FlyDSL rows that AITER tunes for
neighbouring shapes (N=6144 with K=4096, 6144, 2048 and 3072, aiter/configs/model_configs, gfx950, bf16 out) to
the 4x deeper reduction: split-K factors are raised where the 6144 output columns would otherwise leave the
GPU idle (the kernel's 256 tile-counter limit is respected), and 128x128x64 and 256x128x64 tiles carry large M.
"""
import functools

import flydsl.expr as fx
import torch

from .splitk_hgemm import compile_hgemm_kernel
from .tensor_shim import _run_compiled, ptr_arg

SPLIT_K_SEMAPHORE_MAX_LEN = 256

# (largest M served, TILE_M, TILE_N, TILE_K, STAGES, SPLIT_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS)
_TUNED_ROWS = (
    (4, 16, 128, 128, 3, 8, 1, 4, 1),
    (8, 16, 64, 64, 3, 8, 1, 2, 1),
    (16, 16, 64, 256, 3, 4, 1, 2, 1),
    (32, 32, 64, 256, 3, 4, 1, 1, 2),
    (48, 16, 128, 64, 6, 2, 1, 2, 1),
    (64, 32, 64, 256, 3, 2, 2, 1, 2),
    (128, 64, 64, 128, 5, 2, 2, 2, 2),
    (256, 128, 64, 128, 3, 2, 2, 4, 2),
    (512, 128, 128, 64, 4, 2, 4, 4, 1),
    (1024, 128, 128, 64, 4, 1, 2, 4, 1),
    (131072, 256, 128, 64, 3, 1, 4, 4, 1),
)


def _select_row(m):
    for row in _TUNED_ROWS:
        if m <= row[0]:
            return row
    return _TUNED_ROWS[-1]


@functools.lru_cache(maxsize=None)
def _build(device_index, n, k, tile_m, tile_n, tile_k, stages, split_k, block_m_warps, block_n_warps,
           block_k_warps):
    kernel = compile_hgemm_kernel(
        "bf16",
        n,
        k,
        TILE_M=tile_m,
        TILE_N=tile_n,
        TILE_K=tile_k,
        STAGES=stages,
        SPLIT_K=split_k,
        BLOCK_M_WARPS=block_m_warps,
        BLOCK_N_WARPS=block_n_warps,
        BLOCK_K_WARPS=block_k_warps,
        B_TO_LDS=True,
        HAS_BIAS=False,
    )
    # Split-K work counters; the kernel zeroes them again after every launch.
    semaphore = torch.zeros((SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int32, device=f"cuda:{device_index}")
    signal = torch.zeros((SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int32, device=f"cuda:{device_index}")
    return kernel, semaphore, signal


def run(*, a, b, out):
    m, k = a.shape
    n = b.shape[0]
    _, tile_m, tile_n, tile_k, stages, split_k, bmw, bnw, bkw = _select_row(m)
    kernel, semaphore, signal = _build(a.device.index, n, k, tile_m, tile_n, tile_k, stages, split_k, bmw,
                                       bnw, bkw)
    if split_k > 1 and -(-m // tile_m) * (n // tile_n) > SPLIT_K_SEMAPHORE_MAX_LEN:
        raise ValueError("split-K needs more tile counters than the semaphore holds")
    stream = torch.cuda.current_stream(a.device)
    # The bias slot is unused (HAS_BIAS=False); the kernel still takes a pointer for it.
    _run_compiled(kernel, ptr_arg(out), ptr_arg(a), ptr_arg(b), ptr_arg(b), m, ptr_arg(semaphore),
                  ptr_arg(signal), fx.Stream(stream))
