# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA shrink (A) kernel from vLLM (vllm/lora/ops/triton_ops/lora_shrink_op.py, kernel_utils.py).

Based on: Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-tenant LoRA serving. https://arxiv.org/abs/2310.18547
"""
import torch
import triton
import triton.language as tl


@triton.jit
def mm_k(
    a_ptr,
    b_ptr,
    ak_stride,
    bk_stride,
    offset_k,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    b_dtype: tl.constexpr,
    USE_GDC: tl.constexpr,
    base_k,
):
    """
    Given a_ptr and b_ptr, iterate through the K dimension to compute
    the partial/complete matrix block product.
    """
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    STEP_K = BLOCK_K * SPLIT_K
    num_iters = tl.cdiv(K, STEP_K)

    for k in range(num_iters):
        iter_k = k * STEP_K + base_k
        block_end = iter_k + BLOCK_K

        if EVEN_K:
            tiled_b = tl.load(b_ptr)
            tiled_a = tl.load(a_ptr)
            if CAST_TYPE:
                tiled_a = tiled_a.to(b_dtype)
            accumulator += tl.dot(tiled_a, tiled_b)
        else:
            if iter_k >= K:
                pass
            elif block_end <= K:
                tiled_b = tl.load(b_ptr)
                tiled_a = tl.load(a_ptr)
                if CAST_TYPE:
                    tiled_a = tiled_a.to(b_dtype)
                accumulator += tl.dot(tiled_a, tiled_b)
            else:
                k_offsets = tl.arange(0, BLOCK_K)
                mask = iter_k + k_offsets < K
                tiled_b = tl.load(b_ptr, mask=mask[:, None], other=0.0)
                tiled_a = tl.load(a_ptr, mask=mask[None, :], other=0.0)
                if CAST_TYPE:
                    tiled_a = tiled_a.to(b_dtype)
                accumulator += tl.dot(tiled_a, tiled_b)

        a_ptr += STEP_K * ak_stride
        b_ptr += STEP_K * bk_stride

    return accumulator


@triton.jit
def do_shrink_kernel(
    pid_n,
    pid_sk,
    slice_id,
    lora_index,
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
    ram,
    input_d0_stride,
    input_d1_stride,
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    if SLICE_NUM == 1:
        cur_lora_ptr = lora_ptr
    else:
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
            tl.pointer_type(input_ptr.dtype.element_ty)
        )

    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptr = (
        input_ptr + ram[:, None] * input_d0_stride + offset_k[None, :] * input_d1_stride
    )
    b_ptr = (
        cur_lora_ptr
        + lora_d0_stride * lora_index
        + rbn[None, :] * lora_d1_stride
        + offset_k[:, None] * lora_d2_stride
    )

    accumulator = mm_k(
        a_ptr,
        b_ptr,
        input_d1_stride,
        lora_d2_stride,
        offset_k,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        False,
        cur_lora_ptr.dtype.element_ty,
        False,
        base_k=pid_sk * BLOCK_K,
    )

    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_cm = tl.arange(0, BLOCK_M)
    cur_out_ptr = out_ptr if SLICE_NUM == 1 else out_ptr + slice_id * output_d0_stride
    c_ptr = (
        cur_out_ptr
        + ram[:, None] * output_d1_stride
        + offset_cn[None, :] * output_d2_stride
    )
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < N)
    accumulator *= scaling

    if SPLIT_K == 1:
        tl.store(c_ptr, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, accumulator, mask=c_mask, sem="relaxed")


@triton.jit
def _lora_shrink_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    M,
    N,
    K,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    scaling,
    input_d0_stride,
    input_d1_stride,
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)

    pid_sk_m_n = tl.program_id(axis=0)
    pid_sk = pid_sk_m_n % SPLIT_K

    pid_m_n = pid_sk_m_n // SPLIT_K
    num_pid_in_group = GROUP_SIZE_M * cta_n_num
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(cta_m_num - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m

    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        return

    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)

    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (
        token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
    )
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)

    do_shrink_kernel(
        pid_n,
        pid_sk,
        slice_id,
        lora_id,
        input_ptr,
        lora_ptr,
        out_ptr,
        N,
        K,
        cta_m_len,
        ram,
        input_d0_stride,
        input_d1_stride,
        lora_d0_stride,
        lora_d1_stride,
        lora_d2_stride,
        output_d0_stride,
        output_d1_stride,
        output_d2_stride,
        scaling,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        SLICE_NUM,
        USE_GDC,
    )


@triton.jit
def _zero_kernel(ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offsets, tl.zeros([BLOCK], dtype=tl.float32), mask=offsets < n_elements)


def run(*, x, lora_a, token_indices_sorted, num_tokens_per_lora, lora_token_start_loc, lora_ids, scaling, out):
    M, K = x.shape
    N = lora_a.shape[1]
    num_active_loras = lora_ids.shape[0]

    # vLLM zeroes the output before the split-K atomic accumulation; here it is a kernel so that it is
    # captured with the launch.
    _zero_kernel[(triton.cdiv(out.numel(), 1024),)](out, out.numel(), BLOCK=1024)

    # vLLM's default shrink configuration (vllm/lora/ops/triton_ops/utils.py, get_lora_op_configs).
    SPLIT_K = 64 if M < 128 else 8
    BLOCK_M = 32
    BLOCK_N = 16
    BLOCK_K = 256 if M < 128 else 32
    NUM_WARPS = 4
    NUM_STAGES = 2
    GROUP_SIZE_M = 8
    EVEN_K = K % (BLOCK_K * SPLIT_K) == 0

    grid = (SPLIT_K * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, num_active_loras)
    _lora_shrink_kernel[grid](
        x,
        lora_a,
        out,
        M,
        N,
        K,
        token_indices_sorted,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        scaling,
        x.stride(0),
        x.stride(1),
        lora_a.stride(0),
        lora_a.stride(1),
        lora_a.stride(2),
        0,  # output slice stride (a single slice)
        out.stride(0),
        out.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        GROUP_SIZE_M,
        1,  # SLICE_NUM
        False,  # USE_GDC
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        launch_pdl=False,
    )
