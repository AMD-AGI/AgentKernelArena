# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gated-delta-rule chunk state pass (K5) with AITER's FlyDSL kernel and its BV-tile heuristic.

The BV selection below is AITER's rule-based grid/CU heuristic (aiter/ops/flydsl/linear_attention_prefill_kernels.py,
`_legal_bv_candidates` .. `_heuristic_bv`). The task is one sequence (no `cu_seqlens`), where the heuristic
picks BV = 16 (a 256-program grid) for 32 value heads and V = 128.
"""
import functools

import flydsl.expr as fx
import torch

from .chunk_gated_delta_h import compile_chunk_gated_delta_h
from .tensor_shim import _run_compiled

# The kernel is compiled with G_IS_LOG2_SCALED: `g` is the cumulative gate already scaled by log2(e), as the
# production prefill passes it (use_exp2=True), so the kernel evaluates exp2(g) directly.

_BV_CANDIDATES = [16, 32, 64]
_DEFAULT_BV = 16


def _legal_bv_candidates(V: int) -> list[int]:
    return [c for c in _BV_CANDIDATES if c <= V and V % c == 0]


def _grid_ctas(*, H: int, V: int, N: int, BV: int) -> int:
    return max(1, N) * H * ((V + BV - 1) // BV)


def _select_bv_for_grid(*, H: int, V: int, N: int, target_ctas: int) -> int:
    """Choose the largest legal BV whose grid still covers target_ctas."""
    legal = sorted(_legal_bv_candidates(V), reverse=True)
    if not legal:
        return _DEFAULT_BV
    for bv in legal:
        if _grid_ctas(H=H, V=V, N=N, BV=bv) >= target_ctas:
            return bv
    # If even BV=16 cannot reach the target, use it to maximize grid size.
    return legal[-1]


def _target_bv_for_shape(*, H: int, Hg: int, T_flat: int, N: int, is_varlen: bool) -> int | None:
    """Return the calibrated BV regime before legality/grid adjustment."""
    if is_varlen and H == 32 and Hg == 16:
        if N == 2 and 11000 <= T_flat < 15000:
            return 16
        if N == 3 and not (10000 <= T_flat < 12000 or 20000 <= T_flat < 25000):
            return 64
    if is_varlen and H == 16 and T_flat >= 32768 and N >= 7:
        return 64
    return None


def _heuristic_bv(*, H: int, Hg: int, V: int, T_flat: int, N: int, is_varlen: bool) -> int:
    target_bv = _target_bv_for_shape(H=H, Hg=Hg, T_flat=T_flat, N=N, is_varlen=is_varlen)
    target_ctas = _grid_ctas(H=H, V=V, N=N, BV=target_bv) if target_bv is not None else 256
    return _select_bv_for_grid(H=H, V=V, N=N, target_ctas=target_ctas)


@functools.lru_cache(maxsize=None)
def _build(K, V, BT, BV, H, Hg):
    return compile_chunk_gated_delta_h(
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        H=H,
        Hg=Hg,
        USE_G=True,
        USE_GK=False,
        USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=True,
        SAVE_NEW_VALUE=True,
        IS_VARLEN=False,
        WU_CONTIGUOUS=True,
        STATE_DTYPE_BF16=False,
        G_IS_LOG2_SCALED=True,
    )


@functools.lru_cache(maxsize=None)
def _unused_operands(device_index):
    """Operands of the options this scenario does not use (per-K gate, sequence offsets)."""
    device = f"cuda:{device_index}"
    return (torch.empty(1, device=device, dtype=torch.float32),
            torch.empty(1, device=device, dtype=torch.int32))


def run(*, k, w, u, g, initial_state, h, v_new, final_state):
    _, T, Hg, K = k.shape
    H, T_flat, V = w.shape[1], w.shape[2], u.shape[-1]
    BT = 64
    BV = _heuristic_bv(H=H, Hg=Hg, V=V, T_flat=T_flat, N=1, is_varlen=False)
    launch_fn = _build(K, V, BT, BV, H, Hg)
    dummy_f32, dummy_i32 = _unused_operands(k.device.index)
    grid_v = (V + BV - 1) // BV
    _run_compiled(launch_fn, k, u, w, v_new, g, dummy_f32, h, initial_state, final_state, dummy_i32, dummy_i32,
                  T, T_flat, 1, grid_v, 1 * H, fx.Stream(torch.cuda.current_stream(k.device)))
