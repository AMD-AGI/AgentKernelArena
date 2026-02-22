# Triton SSD chunk cumsum kernel
# Adapted from vLLM mamba ops

import torch
import triton
import triton.language as tl


@triton.jit
def softplus(dt):
    dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
    return dt


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_H": 2}),
        triton.Config({"BLOCK_SIZE_H": 4}),
        triton.Config({"BLOCK_SIZE_H": 8}),
        triton.Config({"BLOCK_SIZE_H": 16}),
        triton.Config({"BLOCK_SIZE_H": 32}),
        triton.Config({"BLOCK_SIZE_H": 64}),
    ],
    key=["chunk_size", "nheads"],
)
@triton.jit
def _chunk_cumsum_fwd_kernel(
    dt_ptr, A_ptr, dt_bias_ptr, dt_out_ptr, dA_cumsum_ptr, cu_chunk_seqlens_ptr,
    seqlen, nheads: tl.constexpr, chunk_size: tl.constexpr,
    dt_min: tl.constexpr, dt_max: tl.constexpr,
    stride_dt_seqlen: tl.int64, stride_dt_head: tl.constexpr,
    stride_A_head: tl.constexpr, stride_dt_bias_head: tl.constexpr,
    stride_dt_out_head: tl.int64, stride_dt_out_chunk: tl.int64, stride_dt_out_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64, stride_dA_cs_chunk: tl.int64, stride_dA_cs_csize: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr, HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_c = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1)

    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)

    dt_ptr += chunk_seqlen_start * stride_dt_seqlen
    dt_out_ptr += pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize)
    dA_cs_ptrs = dA_cumsum_ptr + (offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize)
    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)

    dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)
    tl.store(dt_out_ptrs, dt, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))


def chunk_cumsum_fwd(dt, A, chunk_size, cu_chunk_seqlens, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = cu_chunk_seqlens.shape[0] - 1
    dt_out = torch.empty(nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    dA_cumsum = torch.empty(nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    grid_chunk_cs = lambda META: (nchunks, triton.cdiv(nheads, META["BLOCK_SIZE_H"]))
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_fwd_kernel[grid_chunk_cs](
            dt, A, dt_bias, dt_out, dA_cumsum, cu_chunk_seqlens,
            seqlen, nheads, chunk_size, dt_limit[0], dt_limit[1],
            dt.stride(0), dt.stride(1), A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_out.stride(0), dt_out.stride(1), dt_out.stride(2),
            dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2),
            dt_softplus, dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return dA_cumsum, dt_out
