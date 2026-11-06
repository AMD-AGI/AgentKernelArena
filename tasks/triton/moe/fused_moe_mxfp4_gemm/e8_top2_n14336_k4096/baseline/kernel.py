# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MoE expert GEMM with MXFP4 activations and weights: AITER's Triton fused_moe_mxfp4.

Vendored from ROCm/aiter (the previous task package carried the same code inlined):
  _fused_moe_kernel_mxfp4: aiter/ops/triton/_triton_kernels/moe/moe_op_mxfp4.py
  fused_moe_mxfp4 launcher: aiter/ops/triton/moe/moe_op_mxfp4.py
  remap_xcd, pid_grid: aiter/ops/triton/utils/_triton/pid_preprocessing.py
  _write_zeros_to_output: aiter/ops/triton/utils/_triton/moe_common.py
  tile configuration: aiter/ops/triton/configs/moe/gfx950-MOE-MX_FP4.json, selected by token count as
  aiter/ops/triton/utils/moe_config_utils.py get_optimal_moe_config does.
Changes: the logging repr= of triton.jit and the logger are dropped; the activation scale load is
masked by the token mask (upstream reads the scale row of the padding sentinel, one row past the end
of the activation scales); the launcher is reduced to MXFP4 x MXFP4 without swizzled scales.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    if tall_xcds == 0:
        tall_xcds = tl.cast(NUM_XCDS, tall_xcds.type)
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    return pid


@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
    """
    Maps 1D pid to 2D grid coords (pid_m, pid_n).

    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - GROUP_SIZE_M: tl.constexpr: default is 1
    """
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


@triton.jit
def _write_zeros_to_output(
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


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
def _fused_moe_kernel_mxfp4(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    a_mx_scale_ptr,
    b_mx_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_amxm,
    stride_amxk,
    stride_bmxe,
    stride_bmxk,
    stride_bmxn,
    # Meta-parameters
    A_DTYPE_FORMAT: tl.constexpr,
    B_DTYPE_FORMAT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    SWIZZLE_MX_A: tl.constexpr,  # TODO add swizzle support
    SWIZZLE_MX_B: tl.constexpr,  # TODO add swizzle support
    NUM_XCDS: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    is_a_microscaled_format: tl.constexpr = a_mx_scale_ptr is not None
    is_b_microscaled_format: tl.constexpr = b_mx_scale_ptr is not None
    MX_PACK_DIVISOR: tl.constexpr = 32
    if is_a_microscaled_format:
        a_type: tl.constexpr = a_ptr.dtype.element_ty
        tl.static_assert(
            a_type == tl.uint8 or (a_type == tl.float8e4nv or a_type == tl.float8e5),
            "mx_weight_ptr must be 1 byte",
        )
        tl.static_assert(
            a_mx_scale_ptr.dtype.element_ty == tl.uint8, "a_mx_scale_ptr must be uint8"
        )
        tl.static_assert(
            BLOCK_SIZE_K % MX_PACK_DIVISOR == 0,
            "BLOCK_SIZE_K must be a multiple of MX_PACK_DIVISOR",
        )
    if is_b_microscaled_format:
        b_type: tl.constexpr = b_ptr.dtype.element_ty
        tl.static_assert(
            b_type == tl.uint8 or (b_type == tl.float8e4nv or b_type == tl.float8e5),
            "mx_weight_ptr must be 1 byte",
        )
        tl.static_assert(
            b_mx_scale_ptr.dtype.element_ty == tl.uint8, "b_mx_scale_ptr must be uint8"
        )
        tl.static_assert(
            BLOCK_SIZE_K % MX_PACK_DIVISOR == 0,
            "BLOCK_SIZE_K must be a multiple of MX_PACK_DIVISOR",
        )

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return  # rest of the tiles are dummy paddings
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        _write_zeros_to_output(
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

    # Load a_scale, b_scale
    a_scale = tl.load(a_scale_ptr)
    b_scale = tl.load(b_scale_ptr + off_expert)
    # Set offsets of B on dim N
    offs_b_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_b_n = tl.max_contiguous(
        tl.multiple_of(offs_b_n % N, BLOCK_SIZE_N), BLOCK_SIZE_N
    )
    # Load a_mx_scale
    if is_a_microscaled_format:
        # We have pack 2 fp4 values in a byte
        A_PACK_DIVISOR: tl.constexpr = 2 if a_ptr.dtype.element_ty == tl.uint8 else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K // A_PACK_DIVISOR  # 64
        MX_SCALE_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K // MX_PACK_DIVISOR  # 4

        if SWIZZLE_MX_A:
            tl.static_assert(BLOCK_SIZE_M % 128 == 0)
            tl.static_assert(MX_SCALE_BLOCK_K_A % 4 == 0)
            PACKED_MX_BLOCK_A: tl.constexpr = (MX_SCALE_BLOCK_K_A // 4) * 32 * 4 * 4
            offs_inner = tl.arange(0, PACKED_MX_BLOCK_A)
            offs_scale_m = (
                pid_m * (BLOCK_SIZE_M // 128) + tl.arange(0, BLOCK_SIZE_M // 128)
            ) % N
            offs_scale_m = tl.max_contiguous(
                tl.multiple_of(offs_scale_m, BLOCK_SIZE_M // 128), BLOCK_SIZE_M // 128
            )

            a_mx_scale_ptrs = (
                a_mx_scale_ptr
                + offs_scale_m.to(tl.int64)[:, None] * stride_amxm
                + offs_inner[None, :]
            )
        else:
            offs_scale_ak = tl.arange(0, MX_SCALE_BLOCK_K_A)
            offs_scale_m = offs_token
            # K dimension must be the last dimension for the scales
            a_mx_scale_ptrs = (
                a_mx_scale_ptr
                + offs_scale_ak.to(tl.int64)[None, :] * stride_amxk
                + offs_scale_m.to(tl.int64)[:, None] // top_k * stride_amxm
            )
    else:
        a_mx_scale_ptrs = None
        A_PACK_DIVISOR: tl.constexpr = 1
        MX_SCALE_BLOCK_K_A: tl.constexpr = 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K
    # Load b_mx_scale
    if is_b_microscaled_format:
        # We have pack 2 fp4 values in a byte
        B_PACK_DIVISOR: tl.constexpr = 2 if b_ptr.dtype.element_ty == tl.uint8 else 1
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K // B_PACK_DIVISOR  # 64
        MX_SCALE_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K // MX_PACK_DIVISOR  # 4

        b_mx_scale_ptr += off_expert * stride_bmxe

        if SWIZZLE_MX_B:
            tl.static_assert(BLOCK_SIZE_N % 128 == 0)
            tl.static_assert(MX_SCALE_BLOCK_K_B % 4 == 0)
            PACKED_MX_BLOCK_B: tl.constexpr = (MX_SCALE_BLOCK_K_B // 4) * 32 * 4 * 4
            offs_inner = tl.arange(0, PACKED_MX_BLOCK_B)
            offs_scale_n = (
                pid_n * (BLOCK_SIZE_N // 128) + tl.arange(0, BLOCK_SIZE_N // 128)
            ) % N
            offs_scale_n = tl.max_contiguous(
                tl.multiple_of(offs_scale_n, BLOCK_SIZE_N // 128), BLOCK_SIZE_N // 128
            )

            b_mx_scale_ptrs = (
                b_mx_scale_ptr
                # + offs_scale_n.to(tl.int64)[:, None] * stride_bmxn
                + offs_scale_n.to(tl.int64)[:, None]
                * PACKED_MX_BLOCK_B
                * (K // MX_SCALE_BLOCK_K_B // (MX_PACK_DIVISOR // B_PACK_DIVISOR))
                + offs_inner[None, :]
            )
        else:
            offs_scale_bk = tl.arange(0, MX_SCALE_BLOCK_K_B)
            offs_scale_n = offs_b_n
            # K dimension must be the last dimension for the scales
            b_mx_scale_ptrs = (
                b_mx_scale_ptr
                + offs_scale_bk.to(tl.int64)[None, :] * stride_bmxk
                + offs_scale_n.to(tl.int64)[:, None] * stride_bmxn
            )
    else:
        b_mx_scale_ptrs = None
        B_PACK_DIVISOR: tl.constexpr = 1
        MX_SCALE_BLOCK_K_B: tl.constexpr = 1
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K

    offs_a_k = tl.arange(0, PACKED_BLOCK_K_A)
    offs_b_k = tl.arange(0, PACKED_BLOCK_K_B)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_a_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_expert * stride_be
        + (offs_b_k[:, None] * stride_bk + offs_b_n[None, :] * stride_bn)
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, PACKED_BLOCK_K_A)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if EVEN_K:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None]
                & (offs_a_k[None, :] < (K - k * PACKED_BLOCK_K_A)),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=offs_b_k[:, None] < (K - k * PACKED_BLOCK_K_B),
                other=0.0,
            )
        # We accumulate along the K dimension.
        if is_a_microscaled_format or is_b_microscaled_format:
            if is_a_microscaled_format:
                # if SWIZZLE_MX_A:
                #    a_mx_scales = _unswizzle_mx_block(tl.load(a_mx_scale_ptrs))
                # else:
                mask_ak_scale = offs_scale_ak < (K - k * PACKED_BLOCK_K_A) // (
                    MX_PACK_DIVISOR // A_PACK_DIVISOR
                )
                if SWIZZLE_MX_A:
                    a_mx_scale_mask = mask_ak_scale[None, :]
                else:
                    # Padding rows (sentinel token ids) would read past the scales.
                    a_mx_scale_mask = token_mask[:, None] & mask_ak_scale[None, :]
                a_mx_scales = tl.load(
                    a_mx_scale_ptrs, mask=a_mx_scale_mask, other=0.0
                )
            else:
                a_mx_scales = None
            # if SWIZZLE_MX_B:
            #    b_mx_scales = _unswizzle_mx_block(tl.load(b_mx_scale_ptrs))
            # else:
            mask_bk_scale = offs_scale_bk < (K - k * PACKED_BLOCK_K_B) // (
                MX_PACK_DIVISOR // B_PACK_DIVISOR
            )
            b_mx_scales = tl.load(
                b_mx_scale_ptrs, mask=mask_bk_scale[None, :], other=0.0
            )

            accumulator = tl.dot_scaled(
                a,
                a_mx_scales,
                A_DTYPE_FORMAT,
                b,
                b_mx_scales,
                B_DTYPE_FORMAT,
                acc=accumulator,
                fast_math=True,
            )

            if is_a_microscaled_format:
                if SWIZZLE_MX_A:
                    a_mx_scale_ptrs += MX_SCALE_BLOCK_K_A // 4 * stride_amxk
                else:
                    a_mx_scale_ptrs += MX_SCALE_BLOCK_K_A * stride_amxk
            if SWIZZLE_MX_B:
                b_mx_scale_ptrs += MX_SCALE_BLOCK_K_B // 4 * 512
            else:
                b_mx_scale_ptrs += MX_SCALE_BLOCK_K_B * stride_bmxk
        # Advance the ptrs to the next K block.
        a_ptrs += PACKED_BLOCK_K_A * stride_ak
        b_ptrs += PACKED_BLOCK_K_B * stride_bk

    # Multiply with the scalar weight
    accumulator *= a_scale * b_scale
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# aiter/ops/triton/configs/moe/gfx950-MOE-MX_FP4.json
_MOE_MX_FP4_CONFIGS = {
    "small_M": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 2, "num_warps": 4,
                "num_stages": 4, "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1},
    "medium_M": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4, "num_warps": 8,
                 "num_stages": 2, "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1},
    "large_M": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 8,
                "num_stages": 4, "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1},
}
# aiter/ops/triton/utils/moe_config_utils.py
M_THRESHOLD_SMALL = 256
M_THRESHOLD_MEDIUM = 1024


def get_optimal_moe_config(M):
    if M < M_THRESHOLD_SMALL:
        return dict(_MOE_MX_FP4_CONFIGS["small_M"])
    if M < M_THRESHOLD_MEDIUM:
        return dict(_MOE_MX_FP4_CONFIGS["medium_M"])
    return dict(_MOE_MX_FP4_CONFIGS["large_M"])


def get_num_xcds():
    # Currently, you can't query this programmatically. For gfx942/gfx950 it's 8, so we hardcode that here.
    return 8


def fused_moe_mxfp4(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    A_mx_scale: torch.Tensor,
    B_mx_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
    compute_type,
) -> None:
    """
    Fused MoE computation with MXFP4 (microscale FP4) quantization.

    A [num_tokens, K // 2] and B [E, N, K // 2] hold packed E2M1 codes (uint8) with E8M0 scales
    A_mx_scale [num_tokens, K // 32] and B_mx_scale [E, N, K // 32]; C is [num_tokens, top_k, N].
    Results are written in place to C.
    """
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    assert A_scale is not None
    assert B_scale is not None
    assert A.dtype == torch.uint8 and A_mx_scale is not None, "A_mx_scale should exist when A is mxfp4"
    A_mx_scale_strid_m, A_mx_scale_strid_k = A_mx_scale.stride()
    # NOTE: Only supports B_mx_scale
    assert B_mx_scale is not None

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    _fused_moe_kernel_mxfp4[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        A_mx_scale,
        B_mx_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        A.shape[1],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_mx_scale_strid_m,
        A_mx_scale_strid_k,
        B_mx_scale.stride(0),
        B_mx_scale.stride(2),
        B_mx_scale.stride(1),
        A_DTYPE_FORMAT="e2m1",
        B_DTYPE_FORMAT="e2m1",
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        SWIZZLE_MX_A=False,  # TODO add swizzle support
        SWIZZLE_MX_B=False,  # TODO add swizzle support
        NUM_XCDS=get_num_xcds(),
        **config,
    )


def run(*, a, a_scale, b, b_scale, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, c) -> None:
    num_tokens = a.shape[0]
    num_experts = b.shape[0]
    top_k = topk_ids.shape[1]
    device = a.device
    config = get_optimal_moe_config(num_tokens)
    unit_scale = torch.ones((1,), dtype=torch.float32, device=device)
    unit_expert_scale = torch.ones((num_experts,), dtype=torch.float32, device=device)
    # Unused: the routing weight is not applied (MUL_ROUTED_WEIGHT=False).
    topk_weights = torch.empty((num_tokens, top_k), dtype=torch.float32, device=device)
    fused_moe_mxfp4(a, b, c, unit_scale, unit_expert_scale, a_scale, b_scale, topk_weights, topk_ids,
                    sorted_token_ids, expert_ids, num_tokens_post_padded, False, top_k, config, tl.bfloat16)
