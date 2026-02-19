# Triton SSD chunk state varlen kernel
# Adapted from vLLM mamba ops

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=2),
    ],
    key=["hdim", "dstate", "chunk_size"],
)
@triton.jit
def _chunk_state_varlen_kernel(
    x_ptr, b_ptr, dt_ptr, dA_cumsum_ptr, chunk_states_ptr, cu_seqlens_ptr,
    states_ptr, initstates_ptr,
    hdim: tl.constexpr, dstate: tl.constexpr, chunk_size: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    stride_x_seqlen: tl.int64, stride_x_head: tl.int64, stride_x_hdim: tl.constexpr,
    stride_b_seqlen: tl.int64, stride_b_head: tl.int64, stride_b_dstate: tl.constexpr,
    stride_dt_head: tl.int64, stride_dt_chunk: tl.int64, stride_dt_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64, stride_dA_cs_chunk: tl.int64, stride_dA_cs_csize: tl.constexpr,
    stride_chunk_states_chunk: tl.int64, stride_chunk_states_head: tl.int64,
    stride_chunk_states_hdim: tl.int64, stride_chunk_states_dstate: tl.constexpr,
    stride_states_batch: tl.int64, stride_states_head: tl.int64,
    stride_states_hdim: tl.int64, stride_states_dstate: tl.constexpr,
    stride_init_states_batch: tl.int64, stride_init_states_head: tl.int64,
    stride_init_states_hdim: tl.int64, stride_init_states_dstate: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    end_idx = tl.load(cu_seqlens_ptr + pid_b + 1)
    pid_c = (end_idx - 1) // chunk_size
    b_ptr += pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_ptr += pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    chunk_states_ptr += pid_c * stride_chunk_states_chunk + pid_h * stride_chunk_states_head

    if HAS_INITSTATES:
        initstates_ptr += pid_h * stride_init_states_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (end_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = end_idx - pid_c * chunk_size
    start_idx = tl.load(cu_seqlens_ptr + pid_b)
    start_idx_cur = tl.maximum(start_idx - pid_c * chunk_size, 0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k) & (offs_k[None, :] >= start_idx_cur - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate) & (offs_k[:, None] >= start_idx_cur - k), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        scale = tl.where((offs_k >= start_idx_cur - k) & (offs_k < chunk_size_limit - k), tl.exp(dA_cs_last - dA_cs_k) * dt_k, 0.0)
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    if ((start_idx < pid_c * chunk_size) or (HAS_INITSTATES)):
        dA_cs_boundary = 0.0
        if not HAS_INITSTATES:
            past_states_ptrs = chunk_states_ptr + (offs_m[:, None] * stride_chunk_states_hdim + offs_n[None, :] * stride_chunk_states_dstate)
        else:
            if start_idx < pid_c * chunk_size:
                past_states_ptrs = chunk_states_ptr + (offs_m[:, None] * stride_chunk_states_hdim + offs_n[None, :] * stride_chunk_states_dstate)
            else:
                past_states_ptrs = initstates_ptr + (pid_b * stride_init_states_batch + offs_m[:, None] * stride_init_states_hdim + offs_n[None, :] * stride_init_states_dstate)
                if start_idx > pid_c * chunk_size:
                    dA_cs_boundary = tl.load(dA_cumsum_ptr + (start_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
        past_states = tl.load(past_states_ptrs, mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        scale = tl.exp(dA_cs_last - dA_cs_boundary)
        acc += past_states * scale

    states = acc.to(states_ptr.dtype.element_ty)
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


def chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states, initial_states=None):
    total_seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    assert nheads % ngroups == 0
    assert B.shape == (total_seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert chunk_states.shape == (nchunks, nheads, headdim, dstate)

    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)

    states = torch.empty(batch, nheads, headdim, dstate, dtype=chunk_states.dtype, device=chunk_states.device)
    initial_states_strides = (
        (initial_states.stride(0), initial_states.stride(1), initial_states.stride(2), initial_states.stride(3))
        if initial_states is not None else (0, 0, 0, 0)
    )
    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"]) * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        batch, nheads,
    )
    with torch.cuda.device(x.device.index):
        _chunk_state_varlen_kernel[grid](
            x, B, dt, dA_cumsum, chunk_states, cu_seqlens, states, initial_states,
            headdim, dstate, chunk_size, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2),
            B.stride(0), B.stride(1), B.stride(2),
            dt.stride(0), dt.stride(1), dt.stride(2),
            dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2),
            chunk_states.stride(0), chunk_states.stride(1), chunk_states.stride(2), chunk_states.stride(3),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3),
            initial_states_strides[0], initial_states_strides[1], initial_states_strides[2], initial_states_strides[3],
            HAS_INITSTATES=initial_states is not None,
        )
    return states
