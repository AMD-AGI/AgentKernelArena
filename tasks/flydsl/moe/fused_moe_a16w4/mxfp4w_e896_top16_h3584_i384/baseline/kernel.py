# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Kimi-K3 routed MoE (bf16 tokens x MXFP4 experts, SiTUv2), the AITER FlyDSL two-stage path.
# Host dispatch vendored from ROCm/aiter (commit 652f2d4b768239aefd515c929e84817681ff598d):
#   aiter/fused_moe.py                      (get_padded_M bucketing, the tuned-config lookup, moe_sorting with
#                                            the FlyDSL backend, fused_moe_2stages, _flydsl_stage1_wrapper,
#                                            _flydsl_stage2_wrapper)
#   aiter/ops/flydsl/moe_kernels.py         (kernel-name registry, tile resolution, _flydsl_moe_stage1_impl and
#                                            _flydsl_moe_stage2_impl for a16w4, _run_moe_reduction)
#   aiter/ops/flydsl/moe_sorting.py         (flydsl_moe_sorting_fwd)
#   aiter/configs/model_configs/kimik3_fp4_tuned_fmoe.csv (the gfx950 rows with inter_dim 384)
# Kernels: moe_sorting_kernel.py (sorting), moe_gemm_a16w4.py (both GEMMs), moe_reduction.py (top-k sum).
import functools

import torch

from .moe_gemm_a16w4 import GateMode, compile_mixed_moe_gemm1_a16w4, compile_mixed_moe_gemm2_a16w4
from .moe_reduction import compile_moe_reduction
from .moe_sorting_kernel import moe_sorting_flydsl, moe_sorting_get_workspace_size
from .tensor_shim import ptr_arg

# Kimi-K3 text_config: activation_situ_beta / activation_situ_linear_beta.
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0

# aiter/configs/model_configs/kimik3_fp4_tuned_fmoe.csv, gfx950 / 256 CUs, model_dim 3584, inter_dim 384,
# 896 experts, top-16, ActivationType.Situv2, bf16 x fp4x2, per_1x32, g1u1, doweight_stage1 0, ksplit 0:
# token bucket -> (block_m, kernelName1, kernelName2). The token-3 row is unreachable (get_padded_M rounds 3 to 4).
TUNED_FMOE = {
    1: (32, "flydsl_moe1_abf16_wfp4_bf16_t32x64x256_w3_kw4",
        "flydsl_moe2_abf16_wfp4_bf16_t32x128x128_reduce_bnt2_persist"),
    2: (32, "flydsl_moe1_abf16_wfp4_bf16_t32x64x256_w4_bnt0_xcd4",
        "flydsl_moe2_abf16_wfp4_bf16_t32x256x256_reduce_bnt2"),
    3: (32, "flydsl_moe1_abf16_wfp4_bf16_t32x64x256_w4_xcd4_kw2",
        "flydsl_moe2_abf16_wfp4_bf16_t32x256x256_reduce_bnt2"),
    4: (32, "flydsl_moe1_abf16_wfp4_bf16_t32x64x256_w2_bnt0_xcd4_kw2",
        "flydsl_moe2_abf16_wfp4_bf16_t32x256x128_reduce"),
    8: (32, "flydsl_moe1_abf16_wfp4_bf16_t32x64x256_w3_xcd4",
        "flydsl_moe2_abf16_wfp4_bf16_t32x256x256_reduce"),
    16: (32, "flydsl_moe1_abf16_wfp4_bf16_t32x64x256_w3_bnt0_xcd4",
         "flydsl_moe2_abf16_wfp4_bf16_t32x256x256_atomic_bnt2"),
    32: (32, "flydsl_moe1_abf16_wfp4_bf16_t32x64x256_w3_bnt0_xcd4_kw4",
         "flydsl_moe2_abf16_wfp4_bf16_t32x256x256_atomic_persist"),
    64: (32, "flydsl_moe1_abf16_wfp4_bf16_t32x64x256_w3_xcd4_kw2",
         "flydsl_moe2_abf16_wfp4_bf16_t32x256x128_atomic_bnt2_persist"),
    128: (32, "flydsl_moe1_abf16_wfp4_bf16_t32x64x256_w3_bnt0_xcd4_kw2",
          "flydsl_moe2_abf16_wfp4_bf16_t32x256x256_atomic_bnt2"),
    256: (32, "flydsl_moe1_abf16_wfp4_bf16_t32x64x256_w3",
          "flydsl_moe2_abf16_wfp4_bf16_t32x256x128_atomic_bnt2"),
    512: (32, "flydsl_moe1_abf16_wfp4_bf16_t32x64x256_w3_bnt0_kw4",
          "flydsl_moe2_abf16_wfp4_bf16_t32x256x128_atomic_bnt2_persist"),
    1024: (32, "flydsl_moe1_abf16_wfp4_bf16_t32x128x256_w3_bnt0_xcd4",
           "flydsl_moe2_abf16_wfp4_bf16_t32x256x128_atomic_bnt2_persist"),
    2048: (32, "flydsl_moe1_abf16_wfp4_bf16_t32x128x256_w3_xcd4",
           "flydsl_moe2_abf16_wfp4_bf16_t32x256x256_atomic"),
    4096: (32, "flydsl_moe1_abf16_wfp4_bf16_t32x128x256_w3_xcd4",
           "flydsl_moe2_abf16_wfp4_bf16_t32x256x128_atomic_xcd4"),
    8192: (32, "flydsl_moe1_abf16_wfp4_bf16_t32x128x256_w2",
           "flydsl_moe2_abf16_wfp4_bf16_t32x256x256_atomic_bnt2_xcd4_persist"),
    16384: (32, "flydsl_moe1_abf16_wfp4_bf16_t32x128x256_w2_xcd4",
            "flydsl_moe2_abf16_wfp4_bf16_t32x256x128_atomic_bnt2_xcd4"),
}

# HIP limits grid.y/grid.z to 65535.
_HIP_MAX_GRID_DIM_Y = 65535


# ----------------------------------------------------------------------------- kernel-name registry
def flydsl_kernel_name(
    stage: int,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    mode: str = "",
    sort_block_m: int = 0,
) -> str:
    """Construct kernel name: ``flydsl_moe{stage}_a{a}_w{b}_{out}_t{M}x{N}x{K}[_{mode}][_sbm{S}]``."""
    name = f"flydsl_moe{stage}_a{a_dtype}_w{b_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
    if mode:
        name += f"_{mode}"
    if sort_block_m > 0 and sort_block_m != tile_m:
        name += f"_sbm{sort_block_m}"
    return name


def get_flydsl_stage1_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str
) -> dict[str, dict]:
    """Return {kernelName: params} for all supported stage1 configs."""
    kernels = {}
    is_fp4_a = a_dtype == "fp4"
    is_fp4_b = b_dtype == "fp4"

    tile_ns = [32, 64, 128] if is_fp4_b else [128]
    tile_ks = [256]
    tile_ms = [16, 32, 64, 128] if a_dtype == "fp8" and is_fp4_b else [32, 64, 128]

    waves_per_eus = [1, 2, 3, 4]
    k_batches = [1, 2, 4, 7, 14]
    b_nts = [0, 2]
    xcd_swizzles = [0, 4]

    for tm in tile_ms:
        if tm == 32:
            tile_ns = [32, 64, 128]
        else:
            tile_ns = [64, 128] if is_fp4_a else [128, 256]
        for tn in tile_ns:
            for tk in tile_ks:
                for wpe in waves_per_eus:
                    for kb in k_batches if wpe == 3 and tm == 32 and is_fp4_a else [1]:
                        for bnt in b_nts:
                            gate_onlys = (
                                [False, True] if kb > 1 and is_fp4_a else [False]
                            )
                            for go in gate_onlys:
                                for xcd in xcd_swizzles:
                                    base = flydsl_kernel_name(
                                        1, a_dtype, b_dtype, out_dtype, tm, tn, tk
                                    )
                                    if wpe != 1:
                                        base += f"_w{wpe}"
                                    if kb != 1:
                                        base += f"_kb{kb}"
                                    if bnt != 2:
                                        base += f"_bnt{bnt}"
                                    if go:
                                        base += "_go"
                                    if a_dtype == "fp8":
                                        base += "_gui"
                                    if xcd > 0:
                                        base += f"_xcd{xcd}"
                                    # k_wave (intra-block K-slice): only for the
                                    # small-M tile (tile_m==32), no split-K/mock,
                                    # and capped to <=8 total waves (<=512 threads).
                                    num_n_waves = min(4, tn // 32)
                                    k_waves = (
                                        [1, 2, 4]
                                        if (tm == 32 and kb == 1 and not go)
                                        else [1]
                                    )
                                    for kw in k_waves:
                                        if num_n_waves * kw > 8:
                                            continue
                                        if kw > 1 and 4 * tn > tk:
                                            continue
                                        if (
                                            kw > 1
                                            and a_dtype == "fp8"
                                            and num_n_waves < 2
                                        ):
                                            continue
                                        name = base + (f"_kw{kw}" if kw > 1 else "")
                                        kernels[name] = {
                                            "stage": 1,
                                            "a_dtype": a_dtype,
                                            "b_dtype": b_dtype,
                                            "out_dtype": out_dtype,
                                            "tile_m": tm,
                                            "tile_n": tn,
                                            "tile_k": tk,
                                            "MPerBlock": tm,
                                            "waves_per_eu": wpe,
                                            "k_batch": kb,
                                            "b_nt": bnt,
                                            "gate_mode": (
                                                "mock_gate_only"
                                                if go
                                                else (
                                                    "interleave"
                                                    if a_dtype == "fp8"
                                                    else "separated"
                                                )
                                            ),
                                            "xcd_swizzle": xcd,
                                            "k_wave": kw,
                                        }
    return kernels


def get_flydsl_stage2_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str
) -> dict[str, dict]:
    """Return {kernelName: params} for all supported stage2 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"
    is_fp8 = b_dtype == "fp8"
    tile_ns = [128, 256] if is_fp4 else [128]
    # fp4 stage2 supports tile_k=128 (pack_K=1 scale sub-group shift path) as
    # well as 256.  tile_k=128 cleanly tiles K=inter_dim for TP-sharded shapes
    # whose inter_dim is a multiple of 128 but not 256 (e.g. MiniMax TP4=384).
    tile_ks = [128, 256] if (is_fp4 or is_fp8) else [128]
    tile_ms = [16, 32, 64, 128] if is_fp4 else [32, 64, 128]
    modes = ["atomic", "reduce"]

    b_nts = [0, 2]

    xcd_swizzles = [0, 4]

    for tm in tile_ms:
        for tn in tile_ns:
            for tk in tile_ks:
                for mode in modes:
                    for bnt in b_nts:
                        for xcd in xcd_swizzles:
                            base_name = flydsl_kernel_name(
                                2, a_dtype, b_dtype, out_dtype, tm, tn, tk, mode
                            )
                            if bnt != 0:
                                base_name += f"_bnt{bnt}"
                            if xcd > 0:
                                base_name += f"_xcd{xcd}"
                            base_params = {
                                "stage": 2,
                                "a_dtype": a_dtype,
                                "b_dtype": b_dtype,
                                "out_dtype": out_dtype,
                                "tile_m": tm,
                                "tile_n": tn,
                                "tile_k": tk,
                                "mode": mode,
                                "MPerBlock": tm,
                                "b_nt": bnt,
                                "xcd_swizzle": xcd,
                            }
                            kernels[base_name] = base_params
                            kernels[base_name + "_persist"] = {
                                **base_params,
                                "persist": True,
                            }
    return kernels


_KERNEL_PARAMS = {**get_flydsl_stage1_kernels("bf16", "fp4", "bf16"),
                  **get_flydsl_stage2_kernels("bf16", "fp4", "bf16")}


def pick_flydsl_stage2_tile_k(inter_dim: int) -> int:
    inter_dim = int(inter_dim)
    return 256 if (inter_dim % 256 == 0) else 128


def pick_flydsl_stage1_tile_n(inter_dim: int) -> int:
    inter_dim = int(inter_dim)
    return 256 if (inter_dim % 256 == 0) else 128


def resolve_flydsl_grid_y_persist_m(
    num_m_blocks: int, requested_persist_m: int = 0
) -> int:
    """Increase persist_m as needed to keep grid.y within HIP's limit."""
    num_m_blocks = max(int(num_m_blocks), 0)
    requested_persist_m = max(int(requested_persist_m), 1)
    required_persist_m = max(
        1, (num_m_blocks + _HIP_MAX_GRID_DIM_Y - 1) // _HIP_MAX_GRID_DIM_Y
    )
    return max(requested_persist_m, required_persist_m)


def resolve_flydsl_stage2_tile_k(inter_dim: int, tile_k: int) -> int:
    """A tile_k that divides inter_dim (AITER logs a one-time warning when it downgrades)."""
    inter_dim = int(inter_dim)
    tile_k = int(tile_k)
    if inter_dim % tile_k == 0:
        return tile_k
    auto = pick_flydsl_stage2_tile_k(inter_dim)
    if inter_dim % auto == 0:
        return auto
    return tile_k


def resolve_flydsl_stage1_tile_n(inter_dim: int, tile_n: int) -> int:
    """A tile_n that divides inter_dim (AITER logs a one-time warning when it downgrades)."""
    inter_dim = int(inter_dim)
    tile_n = int(tile_n)
    if inter_dim % tile_n == 0:
        return tile_n
    auto = pick_flydsl_stage1_tile_n(inter_dim)
    if inter_dim % auto == 0:
        return auto
    return tile_n


def get_padded_M(M):
    """aiter.fused_moe.get_padded_M for M below its first tier (32768): the next power of two."""
    return 1 << (int(M) - 1).bit_length()


def tuned_config(token_num):
    bucket = get_padded_M(token_num)
    if bucket not in TUNED_FMOE:
        raise ValueError(f"no tuned Kimi-K3 FlyDSL configuration for {token_num} tokens")
    block_m, name1, name2 = TUNED_FMOE[bucket]
    return block_m, _KERNEL_PARAMS[name1], _KERNEL_PARAMS[name2]


# ----------------------------------------------------------------------------- sorting
def moe_sorting(topk_ids, topk_weights, num_experts, block_size, moe_buf):
    """aiter _flydsl_moe_sorting: expert-sorted (token, slot) ids padded to block_size per expert.

    When ``moe_buf`` is given (atomic stage 2) the sorting kernel also zeroes it.
    """
    device = topk_ids.device
    M, topk = topk_ids.shape
    max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(max_num_tokens_padded, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    if moe_buf is None:
        moe_buf = torch.empty((0, 0), dtype=torch.bfloat16, device=device)
    # flydsl_moe_sorting_fwd caches this workspace per device; here it is allocated per call.
    ws_size = moe_sorting_get_workspace_size(M, num_experts, topk, block_size)
    workspace = torch.empty(ws_size, dtype=torch.int32, device=device) if ws_size > 0 else None
    moe_sorting_flydsl(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        int(block_size),
        None,
        None,
        workspace,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids


# ----------------------------------------------------------------------------- stage 1
def flydsl_moe_stage1_a16w4(a, w1, w1_scale, sorted_token_ids, sorted_expert_ids, num_valid_ids, out, topk,
                            params):
    """_flydsl_stage1_wrapper + _flydsl_moe_stage1_impl for bf16 x mxfp4, no split-K, no bias, no fused quant."""
    token_num = a.shape[0]
    E = w1.shape[0]
    inter_dim = w1.shape[1] // 2
    model_dim = a.shape[1]
    dev = a.device
    tile_m = params["tile_m"]
    tile_n = resolve_flydsl_stage1_tile_n(inter_dim, params["tile_n"])
    tile_k = params["tile_k"]

    flat_a_scale = torch.empty(0, device=dev)
    sw = torch.empty(0, device=dev, dtype=torch.float32)
    out_scale_sorted_flat = torch.empty(0, dtype=torch.uint8, device=dev)
    bias = torch.empty(0, device=dev, dtype=torch.float32)

    _sort_block_m = tile_m
    _all_blks = sorted_expert_ids.shape[0]
    _dense_blks = (
        min(token_num * topk * _sort_block_m, sorted_token_ids.shape[0])
        // _sort_block_m
    )
    _grid_y = min(_dense_blks, _all_blks)
    _persist_m = resolve_flydsl_grid_y_persist_m(_grid_y, 0)

    exe = compile_mixed_moe_gemm1_a16w4(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=False,
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype=params["out_dtype"],
        act="situv2",
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
        persist_m=_persist_m,
        use_async_copy=True,
        k_batch=params.get("k_batch", 1),
        waves_per_eu=params.get("waves_per_eu", 3),
        b_nt=params.get("b_nt", 2),
        gate_mode=GateMode(params.get("gate_mode", "separated")),
        model_dim_pad=0,
        inter_dim_pad=0,
        enable_bias=False,
        a_scale_one=False,
        xcd_swizzle=params.get("xcd_swizzle", 0),
    )
    exe(
        ptr_arg(out.view(-1)),
        ptr_arg(a.view(-1)),
        ptr_arg(w1.view(-1)),
        ptr_arg(flat_a_scale),
        ptr_arg(w1_scale.view(-1)),
        ptr_arg(sorted_token_ids),
        ptr_arg(sorted_expert_ids),
        ptr_arg(sw),
        ptr_arg(num_valid_ids),
        ptr_arg(bias),
        ptr_arg(out_scale_sorted_flat.view(-1)),
        token_num,
        inter_dim * 2,
        model_dim,
        _grid_y,
        torch.cuda.current_stream(),
    )
    return out


# ----------------------------------------------------------------------------- stage 2
@functools.lru_cache(maxsize=None)
def _reduction(topk, model_dim):
    return compile_moe_reduction(topk=topk, model_dim=model_dim, dtype_str="bf16", use_mask=False, num_experts=0)


def flydsl_moe_stage2_a16w4(inter_states, w2, w2_scale, sorted_token_ids, sorted_weights, sorted_expert_ids,
                            num_valid_ids, out, topk, params):
    """_flydsl_stage2_wrapper + _flydsl_moe_stage2_impl for bf16 x mxfp4 (atomic or reduce epilogue)."""
    token_num = inter_states.shape[0]
    E = w2.shape[0]
    model_dim = w2.shape[1]
    inter_dim = inter_states.shape[2]
    dev = inter_states.device
    mode = params.get("mode", "atomic")
    accumulate = mode != "reduce"
    tile_m = params["tile_m"]
    tile_n = params["tile_n"]
    tile_k = resolve_flydsl_stage2_tile_k(inter_dim, params["tile_k"])

    flat_a_scale = torch.empty(0, device=dev)
    bias = torch.empty(0, device=dev, dtype=torch.float32)
    m_blocks = min(sorted_expert_ids.shape[0], token_num * topk)

    # In atomic mode ``out`` was zeroed by the sorting kernel; reduce mode writes per-slot rows first.
    target = out if accumulate else torch.empty(
        (token_num * topk * model_dim,), device=dev, dtype=out.dtype)

    exe = compile_mixed_moe_gemm2_a16w4(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=True,
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype=params["out_dtype"],
        accumulate=accumulate,
        sort_block_m=params.get("sort_block_m", 0),
        waves_per_eu=params.get("waves_per_eu", 0),
        b_nt=params.get("b_nt", 0),
        xcd_swizzle=params.get("xcd_swizzle", 0),
        model_dim_pad=0,
        inter_dim_pad=0,
        enable_bias=False,
    )
    stream = torch.cuda.current_stream()
    exe(
        ptr_arg(target),
        ptr_arg(inter_states),
        ptr_arg(w2),
        ptr_arg(flat_a_scale),
        ptr_arg(w2_scale.view(-1)),
        ptr_arg(sorted_token_ids),
        ptr_arg(sorted_expert_ids),
        ptr_arg(sorted_weights),
        ptr_arg(num_valid_ids),
        ptr_arg(bias),
        token_num,
        model_dim,
        inter_dim,
        m_blocks,
        stream,
    )
    if not accumulate:
        em = torch.empty(0, device=dev, dtype=torch.int32)
        tk = torch.empty(0, device=dev, dtype=torch.int32)
        _reduction(topk, model_dim)(
            ptr_arg(target.view(token_num, topk, model_dim)),
            ptr_arg(out),
            ptr_arg(em),
            ptr_arg(tk),
            token_num,
            stream,
        )
    return out


# ----------------------------------------------------------------------------- entry point
def run(*, hidden_states, w1, w1_scale, w2, w2_scale, topk_weights, topk_ids, out):
    token_num = hidden_states.shape[0]
    E = w1.shape[0]
    topk = topk_ids.shape[1]
    inter_dim = w1.shape[1] // 2
    block_m, params1, params2 = tuned_config(token_num)
    accumulate = params2.get("mode", "atomic") != "reduce"
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = moe_sorting(
        topk_ids, topk_weights, E, block_m, out if accumulate else None)
    intermediate = torch.empty((token_num, topk, inter_dim), dtype=hidden_states.dtype,
                               device=hidden_states.device)
    flydsl_moe_stage1_a16w4(hidden_states, w1, w1_scale, sorted_ids, sorted_expert_ids, num_valid_ids,
                            intermediate, topk, params1)
    flydsl_moe_stage2_a16w4(intermediate, w2, w2_scale, sorted_ids, sorted_weights, sorted_expert_ids,
                            num_valid_ids, out, topk, params2)
