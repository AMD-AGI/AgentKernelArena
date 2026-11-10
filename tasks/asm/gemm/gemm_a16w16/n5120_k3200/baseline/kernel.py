# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 GEMM out = a @ b.T with AITER's assembly bf16gemm kernels and AITER's per-M dispatch.

Host side of AITER's ``gemm_a16w16_asm`` (csrc/py_itfs_cu/asm_gemm_a16w16.cu, aiter/ops/gemm_op_a16w16.py)
for gfx950, N=5120, K=3200, no bias, bfloat16 output, B not preshuffled, translated to Python:

1. Kernel choice (aiter/tuned_gemm.py get_GEMM_A16W16_config): M is looked up in AITER's tuned table
   (aiter/configs/model_configs/qwen32B_bf16_tuned_gemm.csv rows for gfx950, cu_num=256, N=5120, K=3200),
   first as is, then padded with getPaddedM granularity 0, then granularity 1. A row naming an assembly
   kernel fixes the kernel and splitK. Rows that name torch/hipBLASLt (M=512, M=1024) cannot be vendored;
   for them this baseline uses AITER's own assembly heuristic (get_heuristic_kernel), which is what
   ``gemm_a16w16_asm`` selects when no kernel name is given.
2. Grid (gdx, gdy, gdz) = (ceil(N/tileN), ceil(M/tileM), splitK), 256 threads, no dynamic LDS.
3. KernelArgs: 22 fields of 16 bytes each (pointer + 8 pad bytes, or 4-byte scalar + 12 pad bytes).
4. Split-K kernels count finished K-slices in a zero-initialized 16x64 uint32 semaphore workspace, which
   the kernel resets after use; it is allocated once per device, as AITER allocates it once per stream.
"""
import functools

import torch

import aka_asm

N_CONST = 5120
K_CONST = 3200
SUB_K = 64
SEMAPHORE_SHAPE = (16, 64)       # aiter/ops/gemm_op_a16w16.py _SEMA_SHAPE
SEMAPHORE_MAX_GRID = SEMAPHORE_SHAPE[0] * SEMAPHORE_SHAPE[1]

# The non-preshuffled kernels of hsa/gfx950/bf16gemm/bf16gemm_fp32bf16.csv in the iteration order of
# AITER's cfg_bf16gemm_fp32bf16 std::unordered_map (libstdc++, as built in the pinned image); the
# heuristic breaks ties by this order. (name, tileM, tileN, supports split-K)
CANDIDATES = (
    ("bf16gemm_fp32bf16_tn_48x64_splitk_clean", 48, 64, True),
    ("bf16gemm_fp32bf16_tn_32x64_splitk_clean", 32, 64, True),
    ("bf16gemm_bf16_tn_256x256", 256, 256, False),
    ("bf16gemm_fp32bf16_tn_96x64_pf3_splitk", 96, 64, False),
    ("bf16gemm_fp32bf16_tn_80x64_splitk_clean", 80, 64, True),
    ("bf16gemm_fp32bf16_tn_32x64_pf3_splitk", 32, 64, False),
    ("bf16gemm_fp32bf16_tn_64x64_splitk_clean", 64, 64, True),
    ("bf16gemm_fp32bf16_tn_96x64_splitk_clean", 96, 64, True),
    ("bf16gemm_fp32bf16_tn_48x64_pf3_splitk", 48, 64, False),
    ("bf16gemm_fp32bf16_tn_64x64_pf3_splitk", 64, 64, False),
)
TILES = {name: (tile_m, tile_n) for name, tile_m, tile_n, _ in CANDIDATES}

# Tuned rows for gfx950, cu_num=256, N=5120, K=3200 (qwen32B_bf16_tuned_gemm.csv): M -> (kernel, splitK),
# or None where the row names torch/hipBLASLt.
TUNED_ROWS = {
    1: ("bf16gemm_fp32bf16_tn_32x64_splitk_clean", 3),
    16: ("bf16gemm_fp32bf16_tn_32x64_splitk_clean", 3),
    32: ("bf16gemm_fp32bf16_tn_32x64_splitk_clean", 3),
    64: ("bf16gemm_fp32bf16_tn_64x64_splitk_clean", 3),
    128: ("bf16gemm_fp32bf16_tn_48x64_pf3_splitk", 1),
    256: ("bf16gemm_fp32bf16_tn_96x64_pf3_splitk", 1),
    512: None,
    1024: None,
    2048: ("bf16gemm_bf16_tn_256x256", 1),
    4096: ("bf16gemm_bf16_tn_256x256", 1),
    8192: ("bf16gemm_bf16_tn_256x256", 1),
    16384: ("bf16gemm_bf16_tn_256x256", 1),
    32768: ("bf16gemm_bf16_tn_256x256", 1),
}

# The code objects vendored in this directory (file stem -> kernel symbol): every kernel the dispatch above
# selects for this N and K at any M in the task's domain.
KERNELS = {name: aka_asm.load(f"{name}.co", symbol) for name, symbol in (
    ("bf16gemm_fp32bf16_tn_32x64_splitk_clean", "_ZN5aiter39bf16gemm_fp32bf16_tn_32x64_splitk_cleanE"),
    ("bf16gemm_fp32bf16_tn_64x64_splitk_clean", "_ZN5aiter39bf16gemm_fp32bf16_tn_64x64_splitk_cleanE"),
    ("bf16gemm_fp32bf16_tn_48x64_pf3_splitk", "_ZN5aiter37bf16gemm_fp32bf16_tn_48x64_pf3_splitkE"),
    ("bf16gemm_fp32bf16_tn_96x64_pf3_splitk", "_ZN5aiter37bf16gemm_fp32bf16_tn_96x64_pf3_splitkE"),
    ("bf16gemm_bf16_tn_256x256", "_ZN5aiter24bf16gemm_bf16_tn_256x256E"),
)}


def _next_pow2(value):
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _padded_m(m, n, gl):
    """csrc/py_itfs_cu/gemm_common.cu getPaddedM."""
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


def _heuristic(m, n, k, num_cu):
    """get_heuristic_kernel in asm_gemm_a16w16.cu without a kernel name (non-preshuffled B, no bias)."""
    empty_cu, rounds, efficiency, oob = num_cu, 0xFFFFFFFF, 1.0, m
    selected, selected_split = None, 1
    for name, tile_m, tile_n, splittable in CANDIDATES:
        if n % tile_n:
            continue
        pure_tg = -(-m // tile_m) * (n // tile_n)
        split = 1
        if splittable and k // SUB_K >= 2:
            split = max(2, min(num_cu // pure_tg, 16, k // SUB_K))
        tg = pure_tg * split
        local_round = -(-tg // num_cu)
        local_efficiency = tile_m * tile_n / (tile_m + tile_n)
        idle = local_round * num_cu - tg
        less_oob = oob > 0 if m % tile_m == 0 else tile_m - m % tile_m < oob
        same_oob = tile_m - m % tile_m == oob
        if (local_round < rounds or (local_round == rounds and (empty_cu > idle or less_oob))
                or (local_round == rounds and empty_cu == idle and same_oob and local_efficiency > efficiency)):
            rounds, empty_cu, efficiency = local_round, idle, local_efficiency
            oob = 0 if m % tile_m == 0 else tile_m - m % tile_m
            selected, selected_split = name, split
    return selected, selected_split


@functools.lru_cache(maxsize=None)
def _select(m, n, k, num_cu):
    for gl in (None, 0, 1):
        padded = m if gl is None else _padded_m(m, n, gl)
        if padded in TUNED_ROWS:
            row = TUNED_ROWS[padded]
            if row is not None:
                name, split = row
                return name, min(split, 16, k // SUB_K)
            break
    return _heuristic(m, n, k, num_cu)


@functools.lru_cache(maxsize=None)
def _num_cu(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.lru_cache(maxsize=None)
def _semaphore(device_index):
    # aiter/ops/gemm_op_a16w16.py get_semaphore_workspace: zeroed once; split-K kernels reset their counters.
    return torch.zeros(SEMAPHORE_SHAPE, dtype=torch.uint32, device=torch.device("cuda", device_index))


def _pointer(tensor):
    return [tensor, aka_asm.u32(0), aka_asm.u32(0)]


def _null_pointer():
    return [aka_asm.u64(0), aka_asm.u32(0), aka_asm.u32(0)]


def _u32(value):
    return [aka_asm.u32(value), aka_asm.u32(0), aka_asm.u32(0), aka_asm.u32(0)]


def _f32(value):
    return [aka_asm.f32(value), aka_asm.u32(0), aka_asm.u32(0), aka_asm.u32(0)]


def run(*, a, b, out):
    m, k = a.shape
    n = b.shape[0]
    device_index = a.device.index if a.device.index is not None else torch.cuda.current_device()
    name, split = _select(m, n, k, _num_cu(device_index))
    if name not in KERNELS:
        raise ValueError(f"M={m} selects {name}, which this task does not vendor")
    tile_m, tile_n = TILES[name]
    gdx, gdy = -(-n // tile_n), -(-m // tile_m)
    if split > 1 and gdx * gdy > SEMAPHORE_MAX_GRID:
        raise ValueError("split-K needs more tile counters than the semaphore workspace holds")
    semaphore = _pointer(_semaphore(device_index)) if split > 1 else _null_pointer()
    stride_out = n * out.element_size()
    args = (_pointer(out)                              # ptr_D
            + _null_pointer()                          # ptr_C
            + _pointer(a)                              # ptr_A
            + _pointer(b)                              # ptr_B
            + _f32(1.0) + _f32(0.0)                    # alpha, beta
            + _u32(stride_out) + _u32(0)               # stride_D0, stride_D1 (bytes)
            + _u32(stride_out) + _u32(0)               # stride_C0, stride_C1
            + _u32(a.stride(0) * a.element_size()) + _u32(0)    # stride_A0, stride_A1
            + _u32(b.stride(0) * b.element_size()) + _u32(0)    # stride_B0, stride_B1
            + _u32(m) + _u32(n) + _u32(k) + _u32(split)
            + _u32(1)                                  # is_out_b16
            + _null_pointer() + _u32(0)                # ptr_Bias, add_bias
            + semaphore)                               # ptr_semaphore
    KERNELS[name].launch(grid=(gdx, gdy, split), block=(256, 1, 1), args=args, shared_bytes=0)
