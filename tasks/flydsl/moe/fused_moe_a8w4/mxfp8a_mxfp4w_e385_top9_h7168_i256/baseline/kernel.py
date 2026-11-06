# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL a8w4 two-stage MoE (MXFP8 activations, MXFP4 weights) in the arena run() interface.

Every step of the layer is a FlyDSL kernel of this package:

1. moe_sorting_flydsl (moe_sorting.py): sort the (token, slot) pairs by expert in blocks of 32
   rows, with the routing weight of each sorted row;
2. mxfp8_quant_rows + mx_scale_sort (mx_ops.py): MXFP8 (e4m3fn codes, e8m0 per 1 x 32) hidden
   states, with their scales gathered into the sorted tiled layout the GEMM reads;
3. stage 1 (moe_gemm.py, compile_mixed_moe_gemm1): fp8 x fp4 gate/up GEMM of every sorted row
   with silu(gate) * up fused into the epilogue, bfloat16 [tokens, top_k, intermediate];
4. mxfp8_quant_rows + mx_scale_sort: the stage-1 result requantized to MXFP8;
5. stage 2 (compile_mixed_moe_gemm2, per-slot "reduce" output): fp8 x fp4 down GEMM, multiplied
   by the routing weight, bfloat16 [tokens * top_k, hidden];
6. topk_sum (mx_ops.py): the top-k rows of every token summed in float32.

The weights arrive in the preshuffled layouts of ``inputs.to_kernel_layout`` (AITER's
shuffle_weight / e8m0_shuffle for w1 and shuffle_weight_a16w4 / shuffle_scale_a16w4 for w2), the
layouts the GEMM kernels consume; that preparation happens once when weights are loaded.
"""
import torch

from .moe_gemm import _run_compiled, _s1_args_fp4, _s2_args_fp4, build_moe_stage1_module, build_moe_stage2_module
from .moe_sorting import moe_sorting_flydsl
from .mx_ops import mx_scale_sort, mxfp8_quant_rows, topk_sum

# Tile shape of the task's a8w4 FlyDSL implementation: block_m is also the sorting unit.
BLOCK_M = 32
TILE_N = 256
TILE_K = 256


def _moe_stage1(
    a,
    w1,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    topk,
    *,
    tile_m,
    tile_n,
    tile_k,
    out_dtype,
    w1_scale,
    a1_scale,
    sorted_weights=None,
    a_dtype="fp8",
    act="silu",
    swiglu_limit=0.0,
):
    """Host runner for the inline a8w4 stage1 (fp8 act / fp4 weight -> bf16, k_batch=1)."""
    token_num = a.shape[0]
    E = w1.shape[0]
    inter_dim = w1.shape[1] // 2
    # fp8 activations store one element per byte; fp4 packs two per byte.
    model_dim = a.shape[1] * 2 if a_dtype == "fp4" else a.shape[1]
    dev = a.device
    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    out = torch.empty((token_num, topk, inter_dim), dtype=torch_out_dtype, device=dev)

    flat_a_scale = (
        a1_scale.view(-1) if a1_scale is not None else torch.empty(0, device=dev)
    )
    flat_w_scale = (
        w1_scale.view(-1) if w1_scale is not None else torch.empty(0, device=dev)
    )
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.empty(0, device=dev, dtype=torch.float32)
    )

    _sort_block_m = tile_m
    _all_blks = sorted_expert_ids.shape[0]
    _dense_blks = (
        min(token_num * topk * _sort_block_m, sorted_token_ids.shape[0])
        // _sort_block_m
    )
    _grid_y = min(_dense_blks, _all_blks)

    out_scale_sorted_flat = torch.empty(0, dtype=torch.uint8, device=dev)
    _n_in = inter_dim * 2
    _k_in = model_dim

    args = _s1_args_fp4(
        out.view(-1),
        a.view(-1),
        w1.view(-1),
        flat_a_scale,
        flat_w_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sw,
        num_valid_ids,
        out_scale_sorted_flat.view(-1),
        token_num,
        _n_in,
        _k_in,
        _grid_y,
        dev,
        bias=torch.empty(0, device=dev),
    )

    exe = build_moe_stage1_module(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=(sorted_weights is not None),
        a_dtype=a_dtype,
        b_dtype="fp4",
        out_dtype=out_dtype,
        act=act,
        persist_m=1,
        k_batch=1,
        waves_per_eu=3,
        b_nt=0,
        gate_mode="separated",
        swiglu_limit=swiglu_limit,
    )
    _run_compiled(exe, args)
    return out


def _moe_stage2_per_slot(
    inter_states,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    topk,
    *,
    tile_m,
    tile_n,
    tile_k,
    out_dtype,
    w2_scale,
    a2_scale,
    sorted_weights,
    a_dtype="fp8",
):
    """Host runner for the inline a8w4 stage2 in its "reduce" mode (accumulate=False).

    Writes one weighted row per (token, slot), [tokens * topk, model_dim]; the top-k sum is a
    separate kernel. As in the source task's launcher, the grid is persistent (one CTA per
    compute unit, persist_m=-1) above 256 M blocks and one M block per CTA otherwise.
    """
    token_num = inter_states.shape[0]
    E = w2.shape[0]
    model_dim = w2.shape[1]
    # fp8 activations store one element per byte; fp4 packs two per byte.
    inter_dim = inter_states.shape[2] * 2 if a_dtype == "fp4" else inter_states.shape[2]
    dev = inter_states.device
    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16

    flat_a_scale = (
        a2_scale.view(-1) if a2_scale is not None else torch.empty(0, device=dev)
    )
    flat_w_scale = (
        w2_scale.view(-1) if w2_scale is not None else torch.empty(0, device=dev)
    )

    m_blocks = min(sorted_expert_ids.shape[0], token_num * topk)
    _persist_m = -1 if m_blocks > 256 else 1

    _n_in = model_dim
    _k_in = inter_dim
    target = torch.empty(
        (token_num * topk * model_dim,), device=dev, dtype=torch_out_dtype
    )

    args = _s2_args_fp4(
        target,
        inter_states,
        w2,
        flat_a_scale,
        flat_w_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        token_num,
        _n_in,
        _k_in,
        m_blocks,
        dev,
        bias=None,
    )

    exe = build_moe_stage2_module(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=True,
        a_dtype=a_dtype,
        b_dtype="fp4",
        out_dtype=out_dtype,
        accumulate=False,
        persist_m=_persist_m,
        sort_block_m=0,
        b_nt=0,
    )
    _run_compiled(exe, args)
    return target


def run(*, hidden_states, w1, w1_scale, w2, w2_scale, topk_weights, topk_ids, out):
    token_num, model_dim = hidden_states.shape
    experts = w1.shape[0]
    inter_dim = w1.shape[1] // 2
    topk = topk_ids.shape[1]
    dev = hidden_states.device

    # 1. Sorted (token, slot) dispatch, as AITER allocates it for FlyDSL sorting.
    max_num_tokens_padded = token_num * topk + experts * BLOCK_M - topk
    max_num_m_blocks = (max_num_tokens_padded + BLOCK_M - 1) // BLOCK_M
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=dev)
    sorted_weights = torch.empty(max_num_tokens_padded, dtype=torch.float32, device=dev)
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device=dev)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=dev)
    # The sorting kernel also zero-fills a buffer for atomic stage-2 accumulation; the per-slot
    # stage 2 used here does not need one, so it gets a minimal placeholder.
    moe_buf = torch.empty(4, dtype=torch.int32, device=dev)
    moe_sorting_flydsl(
        topk_ids, topk_weights, sorted_ids, sorted_weights, sorted_expert_ids,
        num_valid_ids, moe_buf, experts, BLOCK_M,
    )
    # Sorted rows that can belong to a non-empty block (every such block holds a routed slot).
    sorted_rows = (max_num_tokens_padded + 31) // 32 * 32
    scale_rows = min(sorted_rows, token_num * topk * BLOCK_M)

    # 2. MXFP8 hidden states and their scales in sorted order.
    a1 = torch.empty((token_num, model_dim), dtype=torch.uint8, device=dev)
    a1_scale_rows = torch.empty((token_num, model_dim // 32), dtype=torch.int32, device=dev)
    mxfp8_quant_rows(hidden_states, a1, a1_scale_rows, block_threads=256, vec=4)
    a1_scale = torch.empty((sorted_rows, model_dim // 32), dtype=torch.uint8, device=dev)
    mx_scale_sort(a1_scale_rows, sorted_ids, num_valid_ids, a1_scale, token_num, scale_rows,
                  src_topk=1, topk=topk, block_threads=256)

    # 3. Stage 1: silu(gate) * up per sorted row, bfloat16 [tokens, topk, inter_dim].
    stage1_out = _moe_stage1(
        a1, w1, sorted_ids, sorted_expert_ids, num_valid_ids, topk,
        tile_m=BLOCK_M, tile_n=TILE_N, tile_k=TILE_K, out_dtype="bf16",
        w1_scale=w1_scale, a1_scale=a1_scale, sorted_weights=None, a_dtype="fp8", act="silu",
    )

    # 4. MXFP8 requantization of the stage-1 rows (row = token * topk + slot).
    a2 = torch.empty((token_num * topk, inter_dim), dtype=torch.uint8, device=dev)
    a2_scale_rows = torch.empty((token_num * topk, inter_dim // 32), dtype=torch.int32, device=dev)
    mxfp8_quant_rows(stage1_out.view(token_num * topk, inter_dim), a2, a2_scale_rows,
                     block_threads=64, vec=4)
    a2_scale = torch.empty((sorted_rows, inter_dim // 32), dtype=torch.uint8, device=dev)
    mx_scale_sort(a2_scale_rows, sorted_ids, num_valid_ids, a2_scale, token_num, scale_rows,
                  src_topk=topk, topk=topk, block_threads=64)

    # 5. Stage 2: routing-weighted down GEMM, one row per (token, slot).
    partial = _moe_stage2_per_slot(
        a2.view(token_num, topk, inter_dim), w2, sorted_ids, sorted_expert_ids, num_valid_ids, topk,
        tile_m=BLOCK_M, tile_n=TILE_N, tile_k=TILE_K, out_dtype="bf16",
        w2_scale=w2_scale, a2_scale=a2_scale, sorted_weights=sorted_weights, a_dtype="fp8",
    )

    # 6. Sum over the top-k slots.
    topk_sum(partial.view(token_num * topk, model_dim), out, topk, block_threads=256, vec=4)
