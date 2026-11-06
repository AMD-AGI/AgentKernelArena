# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 GEMM out = a @ b.T with AITER's FlyDSL split-K HGEMM kernel and a per-M tile dispatch table (N=512, K=4096).

The dispatch table holds the FlyDSL rows that AITER's tuned bf16 GEMM configuration lists for gfx950,
N=512, K=4096 (aiter/configs/model_configs/dsv4_bf16_tuned_gemm.csv and qwen3_5_397b_bf16_tuned_gemm.csv; the
faster row where both list an M). AITER pads M up to a tuned row. Where that row names a library kernel
(skinny, assembly, opus, hipBLASLt/torch), which cannot be vendored, the next FlyDSL row at or above M is used,
and every M above 6144 uses the M = 6144 configuration.
"""
import functools

import flydsl.expr as fx
import torch

from .splitk_hgemm import compile_hgemm_kernel
from .tensor_shim import _run_compiled, ptr_arg

SPLIT_K_SEMAPHORE_MAX_LEN = 256

# (largest M served, TILE_M, TILE_N, TILE_K, STAGES, SPLIT_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS)
_TUNED_ROWS = (
    (2, 16, 64, 64, 8, 8, 1, 2, 1),
    (4, 16, 64, 64, 6, 8, 1, 2, 1),
    (8, 16, 64, 64, 7, 8, 1, 2, 1),
    (16, 16, 64, 64, 8, 8, 1, 2, 1),
    (32, 32, 64, 64, 5, 8, 1, 4, 1),
    (48, 16, 64, 128, 4, 8, 1, 2, 1),
    (64, 16, 64, 128, 7, 4, 1, 1, 2),
    (80, 16, 64, 128, 8, 4, 1, 1, 2),
    (96, 16, 64, 128, 5, 4, 1, 1, 2),
    (112, 16, 64, 128, 5, 4, 1, 1, 2),
    (128, 16, 64, 128, 5, 4, 1, 1, 2),
    (144, 32, 64, 128, 4, 4, 1, 4, 1),
    (160, 32, 64, 256, 3, 4, 1, 2, 2),
    (176, 32, 64, 128, 5, 4, 1, 2, 2),
    (192, 32, 64, 128, 4, 4, 1, 4, 1),
    (208, 32, 64, 64, 8, 4, 1, 4, 1),
    (224, 32, 64, 128, 4, 4, 1, 2, 2),
    (240, 32, 64, 128, 4, 4, 2, 2, 1),
    (256, 32, 64, 128, 5, 4, 1, 2, 2),
    (288, 64, 64, 128, 4, 4, 4, 1, 2),
    (320, 48, 64, 128, 4, 4, 1, 2, 1),
    (352, 32, 64, 128, 4, 2, 1, 2, 2),
    (384, 32, 64, 128, 4, 2, 1, 4, 1),
    (416, 32, 64, 256, 3, 2, 1, 4, 1),
    (448, 32, 64, 128, 5, 2, 1, 4, 1),
    (512, 32, 64, 64, 7, 2, 1, 2, 2),
    (832, 32, 64, 128, 4, 1, 1, 4, 1),
    (896, 32, 64, 128, 6, 1, 1, 2, 2),
    (928, 32, 64, 128, 5, 1, 1, 2, 2),
    (960, 32, 64, 128, 4, 1, 1, 1, 4),
    (992, 32, 64, 128, 6, 1, 1, 2, 2),
    (1024, 32, 64, 128, 5, 1, 1, 2, 2),
    (1280, 64, 64, 128, 5, 1, 2, 2, 2),
    (1344, 64, 64, 128, 3, 1, 2, 2, 2),
    (1408, 64, 64, 128, 5, 1, 2, 2, 2),
    (1472, 64, 64, 128, 5, 1, 2, 2, 2),
    (1536, 64, 64, 128, 3, 1, 2, 4, 1),
    (1600, 64, 64, 128, 5, 1, 2, 2, 2),
    (1664, 64, 64, 128, 3, 1, 2, 4, 1),
    (1728, 64, 64, 128, 3, 1, 2, 2, 2),
    (1792, 64, 64, 128, 5, 1, 2, 2, 2),
    (1856, 64, 64, 128, 3, 1, 2, 2, 2),
    (1920, 64, 64, 128, 4, 1, 4, 2, 1),
    (1984, 64, 64, 128, 3, 1, 2, 4, 1),
    (2048, 64, 64, 128, 5, 1, 2, 2, 2),
    (4096, 64, 128, 64, 4, 1, 2, 4, 1),
    (6144, 96, 128, 64, 4, 1, 1, 4, 1),
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
