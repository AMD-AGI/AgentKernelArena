# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""bf16 out = a @ b.T with the AITER FlyDSL split-K HGEMM kernel and a per-M launch table.

`run` picks one launch configuration for the number of rows `m`, compiles the FlyDSL kernel for it
once (per n, k and configuration) and launches it on the current stream. AITER's tuned table for
this (N, K) (GLM-5 tuning file, gfx950) names FlyDSL kernels for m <= 256 and an assembly kernel at
m = 16384; the rows above m = 256 below are the FlyDSL rows AITER tuned for the GPT-OSS 5120x2880
projection at the same m, whose tiles divide this N and K.
"""
import functools

import torch

import flydsl.expr as fx

from .splitk_hgemm import SPLIT_K_SEMAPHORE_MAX_LEN, compile_hgemm_kernel
from .tensor_shim import _run_compiled, ptr_arg

N = 4096
K = 2048

# (largest m served, tile_m, tile_n, tile_k, stages, split_k, block_m_warps, block_n_warps, block_k_warps)
# m <= 256: the FlyDSL rows of the tuned file for N=4096, K=2048 (bucket bounds are the tuned M values,
# AITER pads m up to the next tuned row). m > 256: single-pass tiles from the tuned FlyDSL rows of
# N=5120, K=2880 (m = 512, 1024, 2048 and 8192); the m = 8192 row also serves every larger m.

CONFIGS = (
    (1, 16, 64, 64, 4, 4, 1, 2, 1),
    (2, 16, 64, 64, 7, 4, 1, 2, 1),
    (4, 16, 64, 128, 3, 4, 1, 2, 1),
    (8, 16, 64, 64, 7, 4, 1, 2, 1),
    (16, 16, 64, 256, 3, 2, 1, 1, 2),
    (32, 16, 64, 128, 4, 2, 1, 2, 1),
    (48, 16, 64, 128, 7, 1, 1, 2, 1),
    (64, 16, 64, 128, 4, 1, 1, 1, 2),
    (128, 32, 64, 128, 4, 1, 1, 4, 1),
    (256, 64, 64, 128, 4, 1, 2, 4, 1),
    (512, 128, 64, 64, 3, 1, 4, 1, 1),
    (1024, 96, 128, 64, 2, 1, 2, 2, 1),
    (2048, 128, 128, 64, 4, 1, 4, 2, 1),
    (1 << 62, 256, 128, 64, 3, 1, 4, 2, 2),
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
