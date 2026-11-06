# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""bf16 out = a @ b.T with the AITER FlyDSL split-K HGEMM kernel and a per-M launch table.

`run` picks one launch configuration for the number of rows `m`, compiles the FlyDSL kernel for it
once (per n, k and configuration) and launches it on the current stream. The tables below are
AITER's tuned FlyDSL rows for this (N, K) (the model tuning files shipped with AITER) where
AITER has them, and otherwise a hand-derived choice; they are the same kind of per-M tile and
split-K selection as AITER's get_GEMM_A16W16_config.
"""
import functools

import torch

import flydsl.expr as fx

from .splitk_hgemm import SPLIT_K_SEMAPHORE_MAX_LEN, compile_hgemm_kernel
from .tensor_shim import _run_compiled, ptr_arg

N = 128
K = 6144

# (largest m served, tile_m, tile_n, tile_k, stages, split_k, block_m_warps, block_n_warps, block_k_warps)
# Rows follow AITER's tuned FlyDSL rows for gfx950, N=128, K=6144, bf16 output (GLM-5 and
# MiniMax-M3-Eagle tuning files): m <= 2 .. 128 from the GLM-5 rows, m = 196 .. 1024 from the
# MiniMax-M3-Eagle rows, m = 4096 and 16384 from the rows of the same names. Sizes AITER tunes to
# a non-FlyDSL library (m = 1 skinny, m = 2048 assembly, m = 8192 and 32768 torch) use the
# configuration of the next tuned FlyDSL row at or above m, or the largest one.
CONFIGS = (
    (2, 16, 64, 64, 7, 12, 1, 1, 2),
    (8, 16, 64, 64, 6, 12, 1, 2, 1),
    (16, 16, 64, 64, 8, 12, 1, 1, 2),
    (32, 16, 64, 128, 4, 12, 1, 2, 1),
    (48, 16, 64, 128, 5, 8, 1, 2, 1),
    (64, 32, 64, 128, 3, 12, 1, 2, 2),
    (128, 16, 64, 128, 6, 8, 1, 2, 1),
    (196, 16, 64, 64, 8, 8, 1, 2, 1),
    (256, 16, 64, 128, 5, 8, 1, 2, 1),
    (320, 16, 64, 128, 4, 6, 1, 1, 2),
    (512, 16, 64, 128, 4, 4, 1, 1, 2),
    (1024, 32, 64, 128, 3, 4, 2, 1, 2),
    (4096, 32, 64, 128, 5, 1, 1, 2, 2),
    (1 << 62, 64, 128, 128, 3, 1, 2, 4, 1),
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
