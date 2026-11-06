# SPDX-License-Identifier: Apache-2.0
"""Native MXFP8 (1x32 block, E8M0 scale) fused MoE for AMD CDNA4 (gfx950), from SGLang's MiniMax-M3 path.

Vendored from sgl-project/sglang python/sglang/kernels/ops/moe/mxfp8_moe_amd_gfx95.py: the grouped
``tl.dot_scaled`` GEMM kernel with its launcher, the top-k route combine kernel, and the sequence of
``fused_moe_mxfp8_native`` wrapped in ``run`` (align, quantize, GEMM1, SwiGLU-OAI, quantize, GEMM2, combine).
"""
import torch
import triton
import triton.language as tl

from .align_kernels import moe_align_block_size
from .quant_kernels import mxfp8_e4m3_quantize, swiglu_oai_split


@triton.jit
def _mxfp8_grouped_gemm_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    E,
    N,
    K,
    num_valid_tokens,
    top_k,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    stride_be,
    stride_bn,
    stride_bk,
    stride_bse,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    A_DIV: tl.constexpr,
    MUL_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_post = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_post:
        return

    offs_tid = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_tid).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    valid_expert = (off_e >= 0) & (off_e < E)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    a_row = offs_token // A_DIV

    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    as_ptrs = a_scale_ptr + a_row[:, None] * stride_asm + offs_sk[None, :] * stride_ask
    b_ptrs = (
        b_ptr
        + off_e * stride_be
        + offs_n[:, None] * stride_bn
        + offs_k[None, :] * stride_bk
    )
    bs_ptrs = (
        b_scale_ptr
        + off_e * stride_bse
        + offs_n[:, None] * stride_bsn
        + offs_sk[None, :] * stride_bsk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_mask = offs_n < N
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=valid_expert & n_mask[:, None], other=0.0)
        asc = tl.load(as_ptrs, mask=token_mask[:, None], other=0)
        bsc = tl.load(bs_ptrs, mask=valid_expert & n_mask[:, None], other=0)
        acc += tl.dot_scaled(a, asc, "e4m3", b.T, bsc, "e4m3")

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        as_ptrs += (BLOCK_K // 32) * stride_ask
        bs_ptrs += (BLOCK_K // 32) * stride_bsk

    if MUL_WEIGHT:
        w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc = acc * w[:, None]

    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=token_mask[:, None] & n_mask[None, :],
    )


def _grouped_gemm_mxfp8(
    a_q,  # [M, K] fp8 e4m3
    a_scale,  # [M, K//32] uint8 (E8M0)
    w,  # [E, N, K] fp8 e4m3
    w_scale,  # [E, N, K//32] uint8 (E8M0)
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    num_valid_tokens: int,
    top_k: int,
    block_m: int,
    out_dtype,
    a_div: int,
    mul_weight_by=None,
):
    M_routed = num_valid_tokens
    E, N, K = w.shape
    assert K % 128 == 0, f"MXFP8 native MoE requires K%128==0, got K={K}"
    # Keep zero-fill: moe_align_block_size reserves an extra expert bucket for
    # filtered routes, which should contribute zeros if present.
    out = torch.zeros((M_routed, N), dtype=out_dtype, device=a_q.device)
    if a_div == top_k and M_routed <= 32 and K >= 3072:
        BLOCK_N = 64
        num_warps = 4
    else:
        BLOCK_N = 128
        num_warps = 8
    BLOCK_K = 128
    grid = (triton.cdiv(sorted_token_ids.shape[0], block_m), triton.cdiv(N, BLOCK_N))
    _mxfp8_grouped_gemm_kernel[grid](
        a_q,
        a_scale,
        w,
        w_scale,
        out,
        mul_weight_by if mul_weight_by is not None else a_q,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        E,
        N,
        K,
        num_valid_tokens,
        top_k,
        a_q.stride(0),
        a_q.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w_scale.stride(0),
        w_scale.stride(1),
        w_scale.stride(2),
        out.stride(0),
        out.stride(1),
        A_DIV=a_div,
        MUL_WEIGHT=mul_weight_by is not None,
        BLOCK_M=block_m,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return out


@triton.jit
def _combine_topk_routes_kernel(
    route_ptr,
    out_ptr,
    H,
    top_k: tl.constexpr,
    stride_rm,
    stride_rh,
    stride_ot,
    stride_oh,
    BLOCK_H: tl.constexpr,
):
    token_id = tl.program_id(0)
    h_block = tl.program_id(1)
    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for route_idx in range(0, top_k):
        route_row = token_id * top_k + route_idx
        vals = tl.load(
            route_ptr + route_row * stride_rm + offs_h * stride_rh,
            mask=h_mask,
            other=0.0,
        )
        acc += vals.to(tl.float32)

    tl.store(out_ptr + token_id * stride_ot + offs_h * stride_oh, acc, mask=h_mask)


def _combine_topk_routes(route_outputs: torch.Tensor, out: torch.Tensor, num_tokens: int, top_k: int, hidden_size: int):
    block_h = 1024
    grid = (num_tokens, triton.cdiv(hidden_size, block_h))
    _combine_topk_routes_kernel[grid](
        route_outputs,
        out,
        hidden_size,
        top_k,
        route_outputs.stride(0),
        route_outputs.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        num_warps=8,
    )


def run(*, hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, alpha, beta, limit, out):
    T, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    M = T * top_k
    local_num_experts = w13.shape[0]
    block_m = 64
    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, block_m, local_num_experts)

    # GEMM1: x (mxfp8) @ w13^T -> [M, 2I]. The activation is quantized once over the T hidden rows; the
    # kernel gathers per route via a_row = token // top_k.
    a_q, a_s = mxfp8_e4m3_quantize(hidden_states)
    g1 = _grouped_gemm_mxfp8(
        a_q, a_s, w13, w13_scale, sorted_ids, expert_ids, num_post, M, top_k, block_m,
        hidden_states.dtype, a_div=top_k,
    )

    # Default (accuracy) path of upstream: SwiGLU-OAI to bf16, then a separate MXFP8 quantization.
    act = swiglu_oai_split(g1, alpha=alpha, beta=beta, limit=limit)
    act_q, act_s = mxfp8_e4m3_quantize(act)

    # GEMM2: act (mxfp8) @ w2^T -> [M, H] fp32, weighted by topk_weights, then the top-k sum.
    g2 = _grouped_gemm_mxfp8(
        act_q, act_s, w2, w2_scale, sorted_ids, expert_ids, num_post, M, top_k, block_m,
        torch.float32, a_div=1, mul_weight_by=topk_weights.reshape(-1),
    )
    _combine_topk_routes(g2, out, T, top_k, H)
