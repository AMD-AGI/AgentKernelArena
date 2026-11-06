# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""bf16 out = a @ b.T with the AITER FlyDSL split-K HGEMM kernel and a per-M launch table.

`run` picks one launch configuration for the number of rows `m`, compiles the FlyDSL kernel for it
once (per n, k and configuration) and launches it on the current stream. AITER has no tuned
FlyDSL row for this (N, K), so the table below is a hand-selected choice of tile, pipeline and
split-K configurations of the same kernel family that satisfy the kernel's tiling, LDS and
split-K counter constraints (split-K for small m, single-pass tiles that grow with m above).
Rows beyond CHUNK_M are launched in chunks because the kernel uses 32-bit byte offsets.
"""
import functools

import torch

import flydsl.expr as fx

from .splitk_hgemm import SPLIT_K_SEMAPHORE_MAX_LEN, compile_hgemm_kernel
from .tensor_shim import _run_compiled, ptr_arg

N = 3584
K = 512

CHUNK_M = 1 << 17

# (largest m served, tile_m, tile_n, tile_k, stages, split_k, block_m_warps, block_n_warps, block_k_warps)
CONFIGS = (
    (64, 16, 64, 64, 4, 2, 1, 2, 1),
    (256, 16, 64, 128, 4, 1, 1, 2, 1),
    (1024, 32, 64, 128, 4, 1, 1, 4, 1),
    (4096, 128, 64, 128, 3, 1, 2, 4, 2),
    (1 << 62, 256, 256, 64, 2, 1, 2, 4, 1),
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
    semaphore, signal = _split_k_state(a.device)
    stream = torch.cuda.current_stream(a.device)
    # The kernel addresses a and out with 32-bit byte offsets, so rows are launched in chunks
    # of at most CHUNK_M (the largest chunk of a is 1.6 GB); each chunk picks its own table row.
    for start in range(0, m, CHUNK_M):
        rows = min(CHUNK_M, m - start)
        tile_m, tile_n, tile_k, stages, split_k, bmw, bnw, bkw = _select(rows)
        if split_k > 1 and ((rows + tile_m - 1) // tile_m) * (N // tile_n) > SPLIT_K_SEMAPHORE_MAX_LEN:
            raise ValueError("split-K needs more tile counters than the kernel provides")
        kernel = _kernel(N, K, tile_m, tile_n, tile_k, stages, split_k, bmw, bnw, bkw)
        _run_compiled(kernel, ptr_arg(out[start:start + rows]), ptr_arg(a[start:start + rows]),
                      ptr_arg(b), ptr_arg(b), rows, ptr_arg(semaphore), ptr_arg(signal),
                      fx.Stream(stream))
