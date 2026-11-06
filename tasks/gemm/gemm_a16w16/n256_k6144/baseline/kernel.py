# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""bf16 out = a @ b.T with the AITER FlyDSL split-K HGEMM kernel and a per-M launch table.

`run` picks one launch configuration for the number of rows `m`, compiles the FlyDSL kernel for it
once (per n, k and configuration) and launches it on the current stream. The table keeps AITER's tuned rows where they exist and a hand-derived choice elsewhere.
"""
import functools

import torch

import flydsl.expr as fx

from .splitk_hgemm import SPLIT_K_SEMAPHORE_MAX_LEN, compile_hgemm_kernel
from .tensor_shim import _run_compiled, ptr_arg

N = 256
K = 6144

# (largest m served, tile_m, tile_n, tile_k, stages, split_k, block_m_warps, block_n_warps, block_k_warps)
# Rows follow AITER's tuned FlyDSL rows for gfx950, N=256, K=6144, bfloat16 output (GLM-5 model tuning file) for
# m = 4 .. 128; m = 1 and 2 are tuned to the skinny kernel and use the m <= 4 row. m = 129 .. 1024 reuse
# the m = 256 row's tile shape (split-K reduced to 6, tile counters within the kernel's limit); m = 1025 .. 8192
# are untuned (the tuning file's m = 16384 row is the only large one) and use a 64x64x128 single-pass tile, and
# larger sizes use the tuned m = 16384 row (128x128x64).
CONFIGS = (
    (4, 16, 64, 64, 6, 12, 1, 2, 1),
    (16, 16, 64, 64, 7, 12, 1, 2, 1),
    (32, 32, 64, 128, 3, 12, 2, 2, 1),
    (48, 16, 64, 256, 3, 8, 1, 2, 1),
    (64, 32, 64, 128, 5, 8, 1, 4, 1),
    (128, 16, 64, 128, 6, 8, 1, 1, 2),
    (1024, 32, 64, 128, 4, 6, 2, 2, 1),
    (8192, 64, 64, 128, 3, 1, 4, 2, 1),
    (1 << 62, 128, 128, 64, 4, 1, 4, 4, 1),
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
