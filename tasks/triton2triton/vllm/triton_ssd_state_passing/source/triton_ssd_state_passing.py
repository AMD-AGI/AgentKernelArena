# Triton SSD state passing kernel
# Adapted from vLLM mamba ops

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["dim"],
)
@triton.jit
def _state_passing_fwd_kernel(
    states_ptr, out_ptr, dA_cs_ptr, initstates_ptr, seq_idx_ptr, cu_chunk_seqlens_ptr,
    dim: tl.constexpr, nchunks, seqlen, chunk_size: tl.constexpr,
    stride_states_chunk: tl.int64, stride_states_head: tl.int64, stride_states_dim: tl.constexpr,
    stride_out_chunk: tl.int64, stride_out_head: tl.int64, stride_out_dim: tl.constexpr,
    stride_dA_cs_head: tl.int64, stride_dA_cs_chunk: tl.int64, stride_dA_cs_csize: tl.constexpr,
    stride_initstates_batch: tl.int64, stride_initstates_head: tl.int64, stride_initstates_dim: tl.constexpr,
    stride_seq_idx_chunk: tl.constexpr,
    HAS_INITSTATES: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid_h = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=0)

    states_ptr += pid_h * stride_states_head
    dA_cs_ptr += pid_h * stride_dA_cs_head + (chunk_size - 1) * stride_dA_cs_csize
    out_ptr += pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

    if HAS_INITSTATES:
        initstates_ptrs = initstates_ptr + pid_h * stride_initstates_head + offs_m * stride_initstates_dim
        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    else:
        states = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    prev_seq_idx = 0
    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        seq_idx = tl.load(seq_idx_ptr + c * stride_seq_idx_chunk)
        if prev_seq_idx != seq_idx:
            if HAS_INITSTATES:
                initstates_ptrs = (
                    initstates_ptr + seq_idx * stride_initstates_batch
                    + pid_h * stride_initstates_head + offs_m * stride_initstates_dim
                )
                states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            else:
                states = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        prev_seq_idx = seq_idx
        states = tl.exp(dA_cs) * states + new_states
        tl.store(out_ptrs, states, mask=offs_m < dim)

        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


def state_passing_fwd(states, dA_cumsum, cu_chunk_seqlens, seq_idx, initial_states=None, out_dtype=None):
    nchunks, nheads, dim = states.shape
    chunk_size = dA_cumsum.shape[-1]
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    seqlen = seq_idx.shape[-1]
    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((nchunks, nheads, dim), device=states.device, dtype=out_dtype)

    initial_states_strides = (
        (initial_states.stride(0), initial_states.stride(1), initial_states.stride(2))
        if initial_states is not None else (0, 0, 0)
    )
    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE"]), nheads)
    with torch.cuda.device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states, out, dA_cumsum, initial_states, seq_idx, cu_chunk_seqlens,
            dim, nchunks, seqlen if seq_idx is not None else 0,
            chunk_size if seq_idx is not None else 0,
            states.stride(0), states.stride(1), states.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2),
            initial_states_strides[0], initial_states_strides[1], initial_states_strides[2],
            seq_idx.stride(0),
            HAS_INITSTATES=initial_states is not None,
        )
    return out
