# SPDX-License-Identifier: Apache-2.0
"""Activation and top-k reduction kernels of SGLang's Triton fused-MoE path.

``act_and_mul_kernel`` and ``_moe_sum_reduce_kernel`` come from
python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py, reduced to the SiLU activation without expert filter,
SwiGLU clamp or PDL support. ``_moe_sum_reduce_kernel`` was itself modified from the lightllm project
(https://github.com/ModelTC/lightllm, lightllm/common/fused_moe/moe_sum_reduce.py, Apache-2.0).
"""
import triton
import triton.language as tl


@triton.jit
def _silu(x):
    x = x.to(tl.float32)
    return x * tl.sigmoid(x)


@triton.jit
def act_and_mul_kernel(
    gateup_output,
    down_input,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """silu(gate) * up of a row holding [gate | up]."""
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2
    pid = tl.program_id(0)

    gateup_output_ptr = gateup_output + pid * hidden_size
    down_input_ptr = down_input + pid * half_hidden_size
    gate_output_ptr = gateup_output_ptr
    up_output_ptr = gateup_output_ptr + half_hidden_size

    for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < half_hidden_size

        gate_output = tl.load(gate_output_ptr + offset, mask=mask)
        up_output = tl.load(up_output_ptr + offset, mask=mask)

        gate_output_activated = _silu(gate_output)
        gate_output_activated = gate_output_activated.to(InDtype)

        act_mul_output = gate_output_activated * up_output
        act_mul_output = act_mul_output.to(OutDtype)
        tl.store(down_input_ptr + offset, act_mul_output, mask=mask)


@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)

    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(
            base_ptrs + i * input_stride_1,
            mask=mask_token[:, None] & mask_dim[None, :],
            other=0.0,
        )
        accumulator += tile.to(tl.float32)
    accumulator *= routed_scaling_factor

    # -------- Write back --------
    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
    tl.store(
        store_ptrs,
        accumulator.to(input_ptr.dtype.element_ty),
        mask=mask_token[:, None] & mask_dim[None, :],
    )


def act_and_mul_triton(gateup_output, down_input):
    grid = (down_input.shape[0],)
    hidden_size = gateup_output.shape[1]
    act_and_mul_kernel[grid](gateup_output, down_input, hidden_size, BLOCK_SIZE=512)


def moe_sum_reduce_triton(x, output, routed_scaling_factor):
    token_num, topk_num, hidden_dim = x.shape

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 16

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _moe_sum_reduce_kernel[grid](
        x,
        *x.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )
