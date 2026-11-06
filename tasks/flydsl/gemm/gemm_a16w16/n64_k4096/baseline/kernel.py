# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""bf16 out = a @ b.T with the AITER FlyDSL split-K HGEMM kernel and a per-M launch table.

`run` picks one launch configuration for the number of rows `m`, compiles the FlyDSL kernel for it
once (per n, k and configuration) and launches it on the current stream. The table keeps AITER's tuned FlyDSL rows where they exist and a hand-derived choice elsewhere.
"""
import functools

import torch

import flydsl.expr as fx

from .splitk_hgemm import SPLIT_K_SEMAPHORE_MAX_LEN, compile_hgemm_kernel
from .tensor_shim import _run_compiled, ptr_arg

N = 64
K = 4096

# (largest m served, tile_m, tile_n, tile_k, stages, split_k, block_m_warps, block_n_warps, block_k_warps)
# Rows follow AITER's tuned FlyDSL rows for gfx950, N=64, K=4096, bfloat16 output (DeepSeek-V4 and Qwen3.5 tuning files)
# for m = 8 .. 2048 and 8192 .. 16384; m = 1 .. 4 are tuned to the skinny kernel and use the m = 8 row. m = 2049 .. 6144 (assembly in
# AITER) use hand-derived split-K rows; larger sizes use the m = 16384 row.
CONFIGS = (
    (8, 16, 64, 128, 7, 4, 1, 1, 2),
    (16, 16, 64, 64, 6, 8, 1, 2, 1),
    (32, 16, 64, 64, 7, 8, 1, 2, 1),
    (48, 16, 64, 64, 6, 8, 1, 1, 2),
    (64, 16, 64, 128, 4, 8, 1, 2, 1),
    (80, 16, 64, 64, 7, 8, 1, 2, 1),
    (96, 16, 64, 64, 6, 8, 1, 2, 1),
    (112, 16, 64, 64, 7, 8, 1, 2, 1),
    (128, 32, 64, 64, 5, 8, 2, 2, 1),
    (144, 16, 64, 64, 7, 8, 1, 2, 1),
    (160, 32, 64, 64, 5, 8, 2, 2, 1),
    (176, 16, 64, 64, 6, 8, 1, 1, 2),
    (192, 32, 64, 64, 5, 8, 2, 2, 1),
    (208, 32, 64, 64, 6, 8, 1, 4, 1),
    (224, 32, 64, 64, 6, 8, 1, 4, 1),
    (240, 16, 64, 64, 7, 8, 1, 2, 1),
    (256, 16, 64, 64, 6, 8, 1, 2, 1),
    (288, 16, 64, 64, 8, 8, 1, 2, 1),
    (320, 16, 64, 64, 8, 8, 1, 2, 1),
    (352, 32, 64, 64, 5, 8, 2, 2, 1),
    (384, 32, 64, 64, 6, 8, 1, 4, 1),
    (416, 32, 64, 64, 6, 8, 2, 2, 1),
    (448, 32, 64, 64, 8, 8, 1, 4, 1),
    (480, 32, 64, 64, 7, 8, 1, 4, 1),
    (512, 16, 64, 64, 6, 8, 1, 2, 1),
    (544, 32, 64, 64, 7, 8, 1, 4, 1),
    (576, 16, 64, 256, 3, 4, 1, 1, 2),
    (608, 16, 64, 64, 8, 4, 1, 2, 1),
    (640, 32, 64, 64, 5, 8, 1, 4, 1),
    (672, 32, 64, 64, 5, 8, 1, 4, 1),
    (704, 16, 64, 64, 8, 4, 1, 1, 2),
    (736, 16, 64, 64, 8, 4, 1, 2, 1),
    (768, 16, 64, 256, 4, 4, 1, 1, 2),
    (800, 16, 64, 64, 8, 4, 1, 2, 1),
    (832, 16, 64, 128, 4, 4, 1, 1, 2),
    (864, 16, 64, 128, 4, 4, 1, 2, 1),
    (896, 32, 64, 128, 5, 4, 1, 4, 1),
    (928, 16, 64, 64, 8, 4, 1, 2, 1),
    (960, 16, 64, 256, 4, 4, 1, 1, 2),
    (992, 16, 64, 64, 8, 4, 1, 2, 1),
    (1024, 16, 64, 256, 4, 4, 1, 1, 2),
    (1088, 32, 64, 128, 5, 4, 1, 4, 1),
    (1152, 32, 64, 256, 3, 4, 1, 2, 2),
    (1216, 32, 64, 64, 6, 4, 1, 4, 1),
    (1280, 32, 64, 64, 8, 4, 1, 2, 2),
    (1344, 32, 64, 64, 6, 4, 1, 4, 1),
    (1408, 32, 64, 128, 4, 4, 1, 2, 2),
    (1472, 32, 64, 64, 8, 4, 2, 2, 1),
    (1536, 32, 64, 64, 8, 4, 1, 4, 1),
    (1600, 32, 64, 64, 8, 4, 1, 4, 1),
    (1664, 32, 64, 128, 5, 4, 1, 4, 1),
    (1728, 32, 64, 128, 5, 4, 1, 4, 1),
    (1792, 16, 64, 128, 8, 2, 1, 1, 2),
    (1856, 32, 64, 64, 5, 4, 1, 4, 1),
    (1920, 32, 64, 128, 5, 4, 1, 2, 2),
    (1984, 32, 64, 64, 5, 4, 1, 4, 1),
    (2048, 16, 64, 64, 8, 2, 1, 1, 2),
    (4096, 32, 64, 64, 5, 4, 1, 4, 1),
    (6144, 48, 64, 64, 8, 2, 1, 2, 1),
    (12288, 48, 64, 64, 8, 1, 1, 2, 1),
    (14336, 64, 64, 128, 4, 1, 4, 2, 1),
    (1 << 62, 64, 64, 128, 5, 1, 1, 4, 2),
)


def _select(m: int) -> tuple:
    for row in CONFIGS:
        if m <= row[0]:
            return row[1:]
    raise AssertionError("unreachable")


@functools.lru_cache(maxsize=None)
def _kernel(n: int, k: int, tile_m: int, tile_n: int, tile_k: int, stages: int, split_k: int,
            block_m_warps: int, block_n_warps: int, block_k_warps: int):
    return compile_hgemm_kernel(
        "bf16", n, k,
        TILE_M=tile_m, TILE_N=tile_n, TILE_K=tile_k, STAGES=stages, SPLIT_K=split_k,
        BLOCK_M_WARPS=block_m_warps, BLOCK_N_WARPS=block_n_warps, BLOCK_K_WARPS=block_k_warps,
        B_TO_LDS=True, HAS_BIAS=False,
    )


@functools.lru_cache(maxsize=None)
def _split_k_state(device: torch.device):
    # Counters of the in-kernel split-K reduction; the kernel resets them when a tile group is done.
    semaphore = torch.zeros((SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int32, device=device)
    signal = torch.zeros((SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int32, device=device)
    return semaphore, signal


def run(*, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    m = a.shape[0]
    tile_m, tile_n, tile_k, stages, split_k, bmw, bnw, bkw = _select(m)
    if split_k > 1 and ((m + tile_m - 1) // tile_m) * (N // tile_n) > SPLIT_K_SEMAPHORE_MAX_LEN:
        raise ValueError("split-K needs more tile counters than the kernel provides")
    kernel = _kernel(N, K, tile_m, tile_n, tile_k, stages, split_k, bmw, bnw, bkw)
    semaphore, signal = _split_k_state(a.device)
    stream = torch.cuda.current_stream(a.device)
    _run_compiled(kernel, ptr_arg(out), ptr_arg(a), ptr_arg(b), ptr_arg(b), m,
                  ptr_arg(semaphore), ptr_arg(signal), fx.Stream(stream))
