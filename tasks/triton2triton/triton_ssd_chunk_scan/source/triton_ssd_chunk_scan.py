# Triton SSD chunk scan kernel
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
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=2),
    ],
    key=["chunk_size", "hdim", "dstate", "IS_CAUSAL"],
)
@triton.jit
def _chunk_scan_fwd_kernel(
    cb_ptr, x_ptr, z_ptr, out_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    C_ptr, states_ptr, D_ptr, initstates_ptr, cu_chunk_seqlens_ptr,
    chunk_size: tl.constexpr, hdim: tl.constexpr, dstate: tl.constexpr, seqlen,
    nheads_ngroups_ratio: tl.constexpr,
    stride_cb_chunk: tl.int64, stride_cb_head: tl.int64,
    stride_cb_csize_m: tl.int64, stride_cb_csize_k: tl.constexpr,
    stride_x_seqlen: tl.int64, stride_x_head: tl.int64, stride_x_hdim: tl.constexpr,
    stride_z_seqlen: tl.int64, stride_z_head: tl.int64, stride_z_hdim: tl.constexpr,
    stride_out_seqlen: tl.int64, stride_out_head: tl.int64, stride_out_hdim: tl.constexpr,
    stride_dt_chunk: tl.int64, stride_dt_head: tl.int64, stride_dt_csize: tl.constexpr,
    stride_dA_cs_chunk: tl.int64, stride_dA_cs_head: tl.int64, stride_dA_cs_csize: tl.constexpr,
    stride_seq_idx_chunk: tl.constexpr,
    stride_C_seqlen: tl.int64, stride_C_head: tl.int64, stride_C_dstate: tl.constexpr,
    stride_states_chunk: tl.int64, stride_states_head: tl.int64,
    stride_states_hdim: tl.int64, stride_states_dstate: tl.constexpr,
    stride_init_states_batch: tl.int64, stride_init_states_head: tl.int64,
    stride_init_states_hdim: tl.int64, stride_init_states_dstate: tl.constexpr,
    stride_D_head: tl.constexpr,
    IS_CAUSAL: tl.constexpr, HAS_D: tl.constexpr, D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr, IS_TRITON_22: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
):
    pid_c = tl.program_id(axis=1).to(tl.int64)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    cb_ptr += pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)
    x_ptr += chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_ptr += chunk_seqlen_start * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    seq_idx_ptr += pid_c * stride_seq_idx_chunk
    seq_idx = tl.load(seq_idx_ptr)
    seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_chunk, mask=pid_c >= 1, other=-1)

    if HAS_INITSTATES and (seq_idx != seq_idx_prev):
        prev_states_ptr = initstates_ptr + seq_idx * stride_init_states_batch + pid_h * stride_init_states_head
        prev_states_hdim = stride_init_states_hdim
        prev_states_dstate = stride_init_states_dstate
    else:
        prev_states_ptr = states_ptr + (pid_c - 1) * stride_states_chunk + pid_h * stride_states_head
        prev_states_hdim = stride_states_hdim
        prev_states_dstate = stride_states_dstate

    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)

    scale_m = tl.exp(dA_cs_m)
    if BLOCK_SIZE_DSTATE <= 128:
        C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate), other=0.0)
        if not HAS_INITSTATES and (seq_idx != seq_idx_prev):
            prev_states = tl.zeros((BLOCK_SIZE_DSTATE, BLOCK_SIZE_N), dtype=C_ptr.dtype.element_ty)
        else:
            prev_states_ptrs = prev_states_ptr + offs_n[None, :] * prev_states_hdim + offs_k_dstate[:, None] * prev_states_dstate
            prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
            prev_states = prev_states.to(C_ptr.dtype.element_ty)
        acc = tl.dot(C, prev_states) * scale_m[:, None]
    else:
        prev_states_ptrs = prev_states_ptr + offs_n[None, :] * prev_states_hdim + offs_k_dstate[:, None] * prev_states_dstate
        for k in range(0, dstate, BLOCK_SIZE_K):
            C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate - k), other=0.0)
            if not HAS_INITSTATES and (seq_idx != seq_idx_prev):
                prev_states = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=C_ptr.dtype.element_ty)
            else:
                prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
                prev_states = prev_states.to(C_ptr.dtype.element_ty)
            acc += tl.dot(C, prev_states)
            C_ptrs += BLOCK_SIZE_K
            prev_states_ptrs += BLOCK_SIZE_K
        acc *= scale_m[:, None]

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    x_ptrs = x_ptr + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit if not IS_CAUSAL else min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= tl.exp(dA_cs_m[:, None] - dA_cs_k[None, :])
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= dt_k
        if IS_CAUSAL:
            mask = offs_m[:, None] >= k + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)
        cb = cb.to(x_ptr.dtype.element_ty)
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim), other=0.0)
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = tl.load(
            x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0,
        ).to(tl.float32)
        acc += x_residual * D

    if HAS_Z:
        z_ptr += chunk_seqlen_start * stride_z_seqlen + pid_h * stride_z_head
        z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :])
        z = tl.load(z_ptrs, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim), other=0.0).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    out_ptr += chunk_seqlen_start * stride_out_seqlen + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim)
    tl.store(out_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))


def chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, out, seq_idx, D=None, z=None, initial_states=None):
    assert seq_idx is not None
    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (seqlen, ngroups, dstate)
    assert cb.shape == (nchunks, ngroups, chunk_size, chunk_size)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if z is not None:
        assert z.shape == x.shape
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    assert states.shape == (nchunks, nheads, headdim, dstate)
    assert seq_idx.shape == (nchunks,)

    grid = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"]) * triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
        nchunks, nheads,
    )
    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    initial_states_strides = (
        (initial_states.stride(0), initial_states.stride(1), initial_states.stride(2), initial_states.stride(3))
        if initial_states is not None else (0, 0, 0, 0)
    )

    _chunk_scan_fwd_kernel[grid](
        cb, x, z, out, dt, dA_cumsum, seq_idx, C, states, D, initial_states, cu_chunk_seqlens,
        chunk_size, headdim, dstate, seqlen, nheads // ngroups,
        cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(3),
        x.stride(0), x.stride(1), x.stride(2),
        z_strides[0], z_strides[1], z_strides[2],
        out.stride(0), out.stride(1), out.stride(2),
        dt.stride(1), dt.stride(0), dt.stride(2),
        dA_cumsum.stride(1), dA_cumsum.stride(0), dA_cumsum.stride(2),
        seq_idx.stride(0),
        C.stride(0), C.stride(1), C.stride(2),
        states.stride(0), states.stride(1), states.stride(2), states.stride(3),
        initial_states_strides[0], initial_states_strides[1], initial_states_strides[2], initial_states_strides[3],
        D.stride(0) if D is not None else 0,
        True,  # IS_CAUSAL
        D is not None,  # HAS_D
        D.dim() == 2 if D is not None else True,  # D_HAS_HDIM
        z is not None,  # HAS_Z
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        IS_TRITON_22=True,
        HAS_INITSTATES=initial_states is not None,
    )
