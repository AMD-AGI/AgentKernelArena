# Triton SSD chunk state kernel
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
def _chunk_state_fwd_kernel(
    x_ptr, b_ptr, states_ptr, dt_ptr, dA_cumsum_ptr, cu_chunk_seqlens_ptr,
    hdim: tl.constexpr, dstate: tl.constexpr, chunk_size: tl.constexpr, seqlen,
    nheads_ngroups_ratio: tl.constexpr,
    stride_x_seqlen: tl.int64, stride_x_head: tl.int64, stride_x_hdim: tl.constexpr,
    stride_b_seqlen: tl.int64, stride_b_head: tl.int64, stride_b_dstate: tl.constexpr,
    stride_states_chunk: tl.int64, stride_states_head: tl.int64, stride_states_hdim: tl.int64, stride_states_dstate: tl.constexpr,
    stride_dt_head: tl.int64, stride_dt_chunk: tl.int64, stride_dt_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64, stride_dA_cs_chunk: tl.int64, stride_dA_cs_csize: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_c = tl.program_id(axis=1).to(tl.int64)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)
    b_ptr += chunk_seqlen_start * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_ptr += chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        scale = tl.exp(dA_cs_last - dA_cs_k) * dt_k
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    states = acc.to(states_ptr.dtype.element_ty)
    states_ptr += pid_c * stride_states_chunk + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


def chunk_state_fwd(B, x, dt, dA_cumsum, cu_chunk_seqlens, states=None, states_in_fp32=True):
    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape

    if states is not None:
        assert states.shape == (nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty((nchunks, nheads, headdim, dstate), device=x.device, dtype=states_dtype)

    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"]) * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        nchunks, nheads,
    )
    with torch.cuda.device(x.device.index):
        _chunk_state_fwd_kernel[grid](
            x, B, states, dt, dA_cumsum, cu_chunk_seqlens,
            headdim, dstate, chunk_size, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2),
            B.stride(0), B.stride(1), B.stride(2),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3),
            dt.stride(0), dt.stride(1), dt.stride(2),
            dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2),
        )
    return states
