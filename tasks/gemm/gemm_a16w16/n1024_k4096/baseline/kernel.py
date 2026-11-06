# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 GEMM out = a @ b.T with AITER's FlyDSL split-K HGEMM kernel and its tuned per-M dispatch.

The dispatch table holds the FlyDSL rows that AITER's tuned bf16 GEMM configuration lists for gfx950,
N=1024, K=4096 (aiter/configs/model_configs/dsv4_bf16_tuned_gemm.csv and qwen3_5_397b_bf16_tuned_gemm.csv).
AITER pads M up to a tuned row. Where that row names a library kernel (assembly, opus, hipBLASLt/torch),
which cannot be vendored, the next FlyDSL row at or above M is used, and rows above M = 2048 use the
M = 2048 configuration.
"""
import functools

import flydsl.expr as fx
import torch

from .splitk_hgemm import compile_hgemm_kernel
from .tensor_shim import _run_compiled, ptr_arg

SPLIT_K_SEMAPHORE_MAX_LEN = 256

# (largest M served, TILE_M, TILE_N, TILE_K, STAGES, SPLIT_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS)
_TUNED_ROWS = (
    (1, 16, 64, 64, 6, 8, 1, 2, 1),
    (2, 16, 64, 64, 7, 8, 1, 2, 1),
    (4, 16, 64, 64, 7, 8, 1, 2, 1),
    (8, 16, 64, 64, 7, 8, 1, 2, 1),
    (16, 16, 64, 64, 7, 8, 1, 1, 2),
    (32, 16, 64, 128, 7, 4, 1, 1, 2),
    (48, 16, 64, 256, 3, 4, 1, 1, 2),
    (64, 16, 64, 128, 6, 4, 1, 1, 2),
    (80, 32, 64, 128, 4, 4, 1, 4, 1),
    (96, 32, 64, 64, 6, 4, 1, 4, 1),
    (112, 32, 64, 128, 5, 4, 1, 4, 1),
    (128, 32, 64, 128, 5, 4, 1, 4, 1),
    (144, 64, 64, 64, 7, 4, 4, 1, 2),
    (160, 64, 64, 64, 8, 4, 2, 2, 1),
    (176, 32, 64, 256, 3, 2, 1, 4, 1),
    (192, 32, 64, 256, 3, 2, 1, 2, 2),
    (208, 32, 64, 64, 8, 2, 2, 2, 1),
    (224, 32, 64, 256, 3, 2, 1, 1, 4),
    (240, 32, 64, 128, 4, 2, 1, 4, 1),
    (256, 32, 64, 128, 6, 2, 2, 1, 2),
    (416, 32, 64, 128, 4, 1, 1, 4, 1),
    (448, 32, 64, 128, 5, 1, 1, 2, 2),
    (480, 32, 64, 128, 5, 1, 1, 2, 2),
    (512, 32, 64, 128, 4, 1, 1, 4, 1),
    (544, 64, 64, 128, 4, 1, 4, 2, 1),
    (608, 64, 64, 128, 3, 1, 2, 2, 2),
    (640, 64, 64, 128, 5, 1, 2, 2, 2),
    (672, 64, 64, 128, 5, 1, 2, 2, 2),
    (704, 64, 64, 128, 5, 1, 2, 2, 2),
    (736, 64, 64, 128, 5, 1, 2, 2, 2),
    (768, 64, 64, 128, 5, 1, 4, 1, 2),
    (800, 64, 64, 128, 5, 1, 2, 2, 2),
    (832, 64, 64, 128, 3, 1, 2, 4, 1),
    (864, 64, 64, 128, 5, 1, 2, 2, 2),
    (896, 64, 64, 128, 3, 1, 2, 2, 2),
    (928, 64, 64, 128, 5, 1, 2, 2, 2),
    (960, 64, 64, 128, 3, 1, 2, 2, 2),
    (992, 64, 64, 128, 4, 1, 4, 2, 1),
    (1024, 64, 64, 128, 5, 1, 2, 2, 2),
    (1088, 96, 64, 128, 4, 1, 1, 4, 1),
    (1152, 96, 64, 64, 4, 1, 2, 2, 1),
    (1216, 96, 64, 64, 4, 1, 2, 2, 1),
    (1280, 96, 64, 64, 4, 1, 2, 2, 1),
    (1344, 96, 64, 64, 4, 1, 2, 2, 1),
    (1408, 96, 64, 128, 4, 1, 2, 1, 2),
    (1472, 96, 64, 128, 4, 1, 1, 2, 2),
    (1536, 96, 64, 64, 8, 1, 1, 2, 2),
    (1600, 128, 64, 128, 3, 1, 4, 2, 2),
    (1664, 128, 64, 64, 4, 1, 2, 4, 1),
    (1728, 128, 64, 64, 4, 1, 2, 4, 1),
    (1792, 64, 128, 128, 3, 1, 2, 4, 2),
    (1856, 128, 64, 64, 4, 1, 2, 4, 1),
    (1920, 128, 64, 128, 3, 1, 4, 2, 2),
    (1984, 128, 64, 64, 4, 1, 4, 2, 1),
    (2048, 128, 64, 128, 3, 1, 2, 4, 2),
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
