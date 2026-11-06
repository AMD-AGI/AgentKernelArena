# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA expand (B) kernel from vLLM (vllm/lora/ops/triton_ops/lora_expand_op.py, kernel_utils.py).

Based on: Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-tenant LoRA serving. https://arxiv.org/abs/2310.18547
"""
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
def do_expand_kernel(
    pid_n,
    lora_index,
    slice_id,
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
    ram,
    slice_start_loc,
    input_d0_stride,
    input_d1_stride,
    input_d2_stride,
    ls_d0_ptr,
    ls_d1_ptr,
    ls_d2_ptr,
    output_d0_stride,
    output_d1_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    EVEN_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    if SAME_STRIDE:
        cur_lora_d0_stride = ls_d0_ptr
        cur_lora_d1_stride = ls_d1_ptr
        cur_lora_d2_stride = ls_d2_ptr
    else:
        cur_lora_d0_stride = tl.load(ls_d0_ptr + slice_id)
        cur_lora_d1_stride = tl.load(ls_d1_ptr + slice_id)
        cur_lora_d2_stride = tl.load(ls_d2_ptr + slice_id)

    if SLICE_NUM == 1:
        cur_input_ptr = input_ptr
        cur_lora_ptr = lora_ptr
    else:
        cur_input_ptr = input_ptr + slice_id * input_d0_stride
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
            tl.pointer_type(out_ptr.dtype.element_ty)
        )

    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    offset_k = tl.arange(0, BLOCK_K)
    a_ptr = (
        cur_input_ptr
        + ram[:, None] * input_d1_stride
        + offset_k[None, :] * input_d2_stride
    )
    b_ptr = (
        cur_lora_ptr
        + cur_lora_d0_stride * lora_index
        + offset_k[:, None] * cur_lora_d2_stride
        + rbn[None, :] * cur_lora_d1_stride
    )

    SPLIT_K = 1
    accumulator = mm_k(
        a_ptr,
        b_ptr,
        input_d2_stride,
        cur_lora_d2_stride,
        offset_k,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        CAST_TYPE,
        cur_lora_ptr.dtype.element_ty,
        USE_GDC,
        base_k=0,
    )

    tiled_c = accumulator.to(cur_lora_ptr.dtype.element_ty)
    if SLICE_NUM == 1:
        cur_slice_start = slice_start_loc
    else:
        cur_slice_start = tl.load(slice_start_loc + slice_id)

    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + cur_slice_start
    offset_cm = tl.arange(0, BLOCK_M)
    c_ptr = (
        out_ptr
        + ram[:, None] * output_d0_stride
        + offset_cn[None, :] * output_d1_stride
    )
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < (cur_slice_start + N))

    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)


@triton.jit
def _lora_expand_kernel(
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
    slice_start_loc,
    input_d0_stride,
    input_d1_stride,
    input_d2_stride,
    ls_d0_ptr,
    ls_d1_ptr,
    ls_d2_ptr,
    output_d0_stride,
    output_d1_stride,
    output_hs_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)

    pid_mn = tl.program_id(axis=0)
    pid_m = pid_mn % cta_m_num
    pid_n = (pid_mn // cta_m_num) % cta_n_num

    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        return

    curr_N = N if SAME_STRIDE else tl.load(output_hs_ptr + slice_id)
    if pid_n * BLOCK_N >= curr_N:
        return

    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)

    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (
        token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
    )

    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)

    do_expand_kernel(
        pid_n,
        lora_id,
        slice_id,
        input_ptr,
        lora_ptr,
        out_ptr,
        curr_N,
        K,
        cta_m_len,
        ram,
        slice_start_loc,
        input_d0_stride,
        input_d1_stride,
        input_d2_stride,
        ls_d0_ptr,
        ls_d1_ptr,
        ls_d2_ptr,
        output_d0_stride,
        output_d1_stride,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        SAME_STRIDE,
        SLICE_NUM,
        EVEN_K,
        CAST_TYPE,
        ADD_INPUTS,
        USE_GDC,
    )


def run(*, x, lora_b, token_indices_sorted, num_tokens_per_lora, lora_token_start_loc, lora_ids, out):
    M = x.shape[0]
    N, K = lora_b.shape[1], lora_b.shape[2]
    num_active_loras = lora_ids.shape[0]
    inputs = x.unsqueeze(0)  # [num_slices = 1, M, rank]

    # vLLM's default expand configuration (vllm/lora/ops/triton_ops/utils.py, get_lora_op_configs) for one slice,
    # with BLOCK_K equal to the rank instead of the default 32.
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 16
    NUM_WARPS = 4
    NUM_STAGES = 2
    EVEN_K = K % BLOCK_K == 0

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, num_active_loras)
    _lora_expand_kernel[grid](
        inputs,
        lora_b,
        out,
        M,
        N,
        K,
        token_indices_sorted,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        0,  # slice_start_loc: offset of the single slice in the output
        inputs.stride(0),
        inputs.stride(1),
        inputs.stride(2),
        lora_b.stride(0),
        lora_b.stride(1),
        lora_b.stride(2),
        out.stride(0),
        out.stride(1),
        N,  # output_hs_ptr: unused when all slices share one width
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        False,  # ADD_INPUTS
        False,  # CAST_TYPE
        1,  # SLICE_NUM
        True,  # SAME_STRIDE
        False,  # USE_GDC
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        launch_pdl=False,
    )
