"""Standalone extraction of sglang's Triton fused MoE grouped-GEMM kernel.

Source: sglang python/sglang/srt/layers/moe/moe_runner/triton_utils/
        fused_moe_triton_kernels.py  (``fused_moe_kernel``)
plus the host launcher ``invoke_fused_moe_kernel`` and the two-GEMM sequence
from ``fused_moe.py`` (``_fused_moe_kernel_sequence``).

This is the editable Triton kernel that the MoE expert FFN lowers to on the
Qwen3.5-35B-A3B serving path when aiter is OFF (``SGLANG_USE_AITER=0`` +
``--moe-runner-backend triton``). With aiter ON the same compute runs on the
non-editable CK kernels (``ck::kernel_moe_gemm`` / ``aiter::ck_moe_stage*``).

The extraction keeps ONLY the bf16/fp16 (unquantized) path used by
Qwen3.5-35B-A3B (dtype=bfloat16, gated SiLU). The fp8/int8/int4 quantization,
TMA-descriptor, bias, and EP-filter branches of the upstream kernel are present
in ``fused_moe_kernel`` (selected by constexpr flags) but are driven to their
no-op settings by the trimmed ``invoke_fused_moe_kernel`` below, so the only
runtime dependencies are ``torch`` and ``triton``.

``moe_align_block_size`` (a sgl_kernel C++ op upstream) is reimplemented in pure
torch here; it is host-side setup, not the kernel under test.

Entry point: ``fused_moe(hidden_states, w1, w2, topk_weights, topk_ids)``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    a_desc,
    b_ptr,
    b_desc,
    bias_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    add_mask_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension.
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bias_e,
    stride_bias_n,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    even_Ks: tl.constexpr,
    c_sorted: tl.constexpr,
    filter_expert: tl.constexpr,
    swap_ab: tl.constexpr,
    FUSE_ADD_TO_OUTPUT: tl.constexpr,
    FUSE_SUM_ALL_REDUCE: tl.constexpr,
    ROUTER_TOPK: tl.constexpr,
):
    """Fused grouped GEMM over the MoE expert weight stack (see source docstring)."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts_i32 = tl.load(expert_ids_ptr + pid_m)
    off_experts = off_experts_i32.to(tl.int64)

    if filter_expert and off_experts == -1:
        if not FUSE_ADD_TO_OUTPUT:
            write_zeros_to_output(
                c_ptr,
                stride_cm,
                stride_cn,
                pid_n,
                N,
                offs_token,
                token_mask,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                compute_type,
            )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if a_desc is not None:
        assert use_fp8_w8a8 and group_n > 0 and group_k > 0
        start_offs_m = pid_m * BLOCK_SIZE_M
    else:
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )

    if b_desc is not None:
        start_offs_n = pid_n * BLOCK_SIZE_N
    else:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )

    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + off_experts * stride_bias_e + offs_bn[None, :] * stride_bias_n
        )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            if a_desc is not None:
                a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
            else:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            if BLOCK_SIZE_N > group_n:
                offs_bsn = offs_bn // group_n
            else:
                offs_bsn = pid_n * BLOCK_SIZE_N // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    if swap_ab:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        if a_desc is not None:
            a = a_desc.load([start_offs_m, k_start])
        elif even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )

        if b_desc is not None:
            b = (
                b_desc.load([off_experts_i32, start_offs_n, k_start])
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .T
            )
        elif even_Ks:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                if swap_ab:
                    a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                    a_scale, b_scale = b_scale, a_scale
                if BLOCK_SIZE_N > group_n:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                else:
                    accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
            else:
                if use_fp8_w8a8:
                    if swap_ab:
                        a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        if a_desc is None:
            a_ptrs += BLOCK_SIZE_K * stride_ak
        if b_desc is None:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if swap_ab:
        accumulator = tl.trans(accumulator, (1, 0))

    if use_int8_w8a16:
        accumulator *= b_scale
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k == 0 or group_n == 0:
            accumulator *= a_scale * b_scale

    if bias_ptr is not None:
        accumulator += bias

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if FUSE_ADD_TO_OUTPUT:
        offs_token_out = offs_token // ROUTER_TOPK
        add_mask = tl.load(add_mask_ptr + offs_token_out, mask=token_mask, other=False)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & add_mask[:, None] & (offs_cn[None, :] < N)
        existing = tl.load(c_ptrs, mask=c_mask, other=0.0)
        tl.store(c_ptrs, existing + accumulator, mask=c_mask)
    elif FUSE_SUM_ALL_REDUCE:
        offs_token_out = offs_token // ROUTER_TOPK
        c_ptrs = (
            c_ptr + stride_cm * offs_token_out[:, None] + stride_cn * offs_cn[None, :]
        )
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)
    else:
        if c_sorted:
            c_ptrs = (
                c_ptr
                + stride_cm * offs_token_id[:, None]
                + stride_cn * offs_cn[None, :]
            )
        else:
            c_ptrs = (
                c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
            )
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


def moe_align_block_size(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    """Pure-torch reimplementation of sgl_kernel's moe_align_block_size.

    Groups the ``topk_ids.numel()`` (token, expert-slot) pairs by expert,
    padding each expert's token run up to a multiple of ``block_size`` so the
    grouped GEMM tiles cleanly. Padding slots use the sentinel value
    ``num_valid_tokens = topk_ids.numel()`` so the kernel masks them out
    (``offs_token < num_valid_tokens``).

    Returns (sorted_token_ids, expert_ids, num_tokens_post_padded).
    """
    device = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int64)
    num_valid = flat.numel()

    # Sort the (token, expert-slot) pairs by expert id; tokens for a given expert
    # then occupy a contiguous run in `order`.
    order = torch.argsort(flat, stable=True)
    sorted_experts = flat[order]

    counts = torch.bincount(flat, minlength=num_experts)            # [E]
    padded = ((counts + block_size - 1) // block_size) * block_size  # [E]

    # Exclusive-cumsum start offsets, in the padded output and in `order`.
    pad_cum = torch.cumsum(padded, 0)
    expert_start = pad_cum - padded                                 # [E]
    cnt_cum = torch.cumsum(counts, 0)
    count_start = cnt_cum - counts                                  # [E]

    total = int(pad_cum[-1].item()) if num_experts > 0 else 0
    sorted_ids = torch.full((total,), num_valid, dtype=torch.int32, device=device)

    pos = torch.arange(num_valid, device=device)
    within = pos - count_start[sorted_experts]                      # rank within expert
    dest = expert_start[sorted_experts] + within
    sorted_ids[dest] = order.to(torch.int32)

    nblocks = padded // block_size                                  # [E]
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=device), nblocks
    )

    num_tokens_post_padded = torch.tensor(
        [total], dtype=torch.int32, device=device
    )
    return sorted_ids, expert_ids, num_tokens_post_padded


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
) -> None:
    """Trimmed bf16/fp16 launcher for ``fused_moe_kernel`` (no quant/TMA/bias)."""
    # Accept the backend-neutral task host ABI.
    if compute_type is torch.bfloat16:
        compute_type = tl.bfloat16
    elif compute_type is torch.float16:
        compute_type = tl.float16
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )

    K = B.shape[2]
    even_Ks = K % config["BLOCK_SIZE_K"] == 0

    fused_moe_kernel[grid](
        A,
        None,
        B,
        None,
        None,
        C,
        None,
        None,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        None,
        B.shape[1],
        B.shape[2],
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        0,
        0,
        C.stride(-2),
        C.stride(-1),
        0,
        0,
        0,
        0,
        0,
        group_n=0,
        group_k=0,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=False,
        even_Ks=even_Ks,
        c_sorted=False,
        filter_expert=False,
        swap_ab=False,
        FUSE_ADD_TO_OUTPUT=False,
        FUSE_SUM_ALL_REDUCE=False,
        ROUTER_TOPK=top_k,
        **config,
    )


def _default_config() -> Dict[str, Any]:
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
    }


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Two-GEMM gated-SiLU MoE expert FFN driven by ``fused_moe_kernel``.

    hidden_states : [M, K]           (bf16/fp16)
    w1            : [E, 2*I, K]       gate/up stacked weights
    w2            : [E, K, I]         down weights
    topk_weights  : [M, topk]        renormalized router weights (fp32)
    topk_ids      : [M, topk]        selected expert ids (int32)
    returns        : [M, K]          combined expert output
    """
    assert hidden_states.dtype in (torch.bfloat16, torch.float16)
    assert hidden_states.is_contiguous() and w1.is_contiguous() and w2.is_contiguous()
    M, K = hidden_states.shape
    E, N, _ = w1.shape
    I = N // 2
    topk = topk_ids.shape[1]
    dtype = hidden_states.dtype
    device = hidden_states.device
    compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16
    if config is None:
        config = _default_config()

    sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E
    )

    # GEMM 1: gate/up projection -> [M*topk, 2I]
    ic1 = torch.empty((M * topk, N), device=device, dtype=dtype)
    invoke_fused_moe_kernel(
        hidden_states,
        w1,
        ic1,
        topk_weights,
        topk_ids,
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=topk,
        config=config,
        compute_type=compute_type,
    )

    # SiLU-and-mul activation (host-side; not the kernel under test) -> [M*topk, I]
    gate, up = ic1[:, :I], ic1[:, I:]
    ic2 = (F.silu(gate.float()) * up.float()).to(dtype)

    # GEMM 2: down projection (applies router weight) -> [M, topk, K]
    ic3 = torch.empty((M, topk, K), device=device, dtype=dtype)
    invoke_fused_moe_kernel(
        ic2,
        w2,
        ic3,
        topk_weights,
        topk_ids,
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
        config=config,
        compute_type=compute_type,
    )

    # Combine the top-k experts (moe_sum, fp32 accumulate -> bf16).
    out = ic3.float().sum(dim=1).to(dtype)
    return out
