# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_state.py
"""Mamba2 SSD final state of every sequence of a packed batch, in Triton.

Vendored from SGLang's `_chunk_state_varlen_kernel` (itself adapted from vLLM). The kernel keeps the
production default tile shape (16 x 16 x 16); the launch wrapper is specialised to the scenario (initial
states always present) and wrapped in the run() interface. The carried-state load moves into the branches
that choose its source, so the kernel compiles with Triton 3.8.0 on gfx950 when chunk_states exceeds 2 GiB
(see the comment in the kernel).
"""
import triton
import triton.language as tl


@triton.jit
def _chunk_state_varlen_kernel(
    # Pointers to matrices
    x_ptr,
    b_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    chunk_states_ptr,
    cu_seqlens_ptr,
    states_ptr,
    initstates_ptr,
    # Matrix dimensions
    hdim,
    dstate,
    chunk_size,
    seqlen,
    nheads_ngroups_ratio,
    # Strides
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_chunk_states_chunk,
    stride_chunk_states_head,
    stride_chunk_states_hdim,
    stride_chunk_states_dstate,
    stride_states_batch,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_init_states_batch,
    stride_init_states_head,
    stride_init_states_hdim,
    stride_init_states_dstate,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 16,
    BLOCK_SIZE_N: tl.constexpr = 16,
    BLOCK_SIZE_K: tl.constexpr = 16,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    end_idx = tl.load(cu_seqlens_ptr + pid_b + 1)
    pid_c = (end_idx - 1) // chunk_size
    b_ptr += (
        pid_c * chunk_size * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_head
    )
    x_ptr += pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    chunk_states_ptr += (
        pid_c * stride_chunk_states_chunk + pid_h * stride_chunk_states_head
    )

    if HAS_INITSTATES:
        # if there are init states provided, we differentiate between states (which
        # are boundary conditions at a chunk boundary) and initstates (which are boundary
        # conditions when a new example in a cont batch starts)
        initstates_ptr += pid_h * stride_init_states_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen
    )
    b_ptrs = b_ptr + (
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(
        dA_cumsum_ptr + (end_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize
    ).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = end_idx - pid_c * chunk_size
    start_idx = tl.load(cu_seqlens_ptr + pid_b)
    start_idx_cur = tl.maximum(start_idx - pid_c * chunk_size, 0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim)
            & (offs_k[None, :] < chunk_size_limit - k)
            & (offs_k[None, :] >= start_idx_cur - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k)
            & (offs_n[None, :] < dstate)
            & (offs_k[:, None] >= start_idx_cur - k),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(
            dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
        ).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
            tl.float32
        )
        scale = tl.where(
            (offs_k >= start_idx_cur - k) & (offs_k < chunk_size_limit - k),
            tl.exp(dA_cs_last - dA_cs_k) * dt_k,
            0.0,
        )
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # If the sequence starts after the last chunk idx, we don't need to add the contribution from the last chunk
    # If HAS_INITSTATES==True need to consider two possibilities
    # - if start_idx < pid_c * chunk_size, then we need to take the past_states_ptrs
    # - if state_idx >= pid * chunk_size, then we need to insert initstates
    if (start_idx < pid_c * chunk_size) or (HAS_INITSTATES):  # first chunk
        dA_cs_boundary = 0.0  # default

        # Task adaptation (Triton 3.8.0 on gfx950): upstream selects past_states_ptrs in the
        # branches below and loads once after them, so the branches yield a pointer tensor. When
        # chunk_states and initstates fall on different sides of the 2 GiB buffer-ops range
        # (chunk_states reaches 2 GiB at 512 chunks), TritonAMDGPUCanonicalizePointers rewrites the
        # two yielded pointers differently and the compile fails. Each branch therefore performs
        # the same masked load itself and yields the loaded values; the arithmetic is unchanged.
        past_states_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
        if not HAS_INITSTATES:
            past_states_ptrs = chunk_states_ptr + (
                offs_m[:, None] * stride_chunk_states_hdim
                + offs_n[None, :] * stride_chunk_states_dstate
            )
            past_states = tl.load(
                past_states_ptrs, mask=past_states_mask, other=0.0
            ).to(tl.float32)
        else:
            # - this seems repetitive, buts its to help the compiler
            if start_idx < pid_c * chunk_size:
                past_states_ptrs = chunk_states_ptr + (
                    offs_m[:, None] * stride_chunk_states_hdim
                    + offs_n[None, :] * stride_chunk_states_dstate
                )
                past_states = tl.load(
                    past_states_ptrs, mask=past_states_mask, other=0.0
                ).to(tl.float32)
            else:
                past_states_ptrs = initstates_ptr + (
                    pid_b * stride_init_states_batch
                    + offs_m[:, None] * stride_init_states_hdim
                    + offs_n[None, :] * stride_init_states_dstate
                )
                past_states = tl.load(
                    past_states_ptrs, mask=past_states_mask, other=0.0
                ).to(tl.float32)

                # need to adjust the boundary
                if start_idx > pid_c * chunk_size:
                    dA_cs_boundary = tl.load(
                        dA_cumsum_ptr
                        + (start_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize
                    ).to(tl.float32)

        scale = tl.exp(dA_cs_last - dA_cs_boundary)
        acc += past_states * scale

    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)



def run(*, x, B, dt, dA_cumsum, cu_seqlens, chunk_states, initial_states, states):
    total_seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.shape[0] - 1
    grid = (
        triton.cdiv(headdim, 16) * triton.cdiv(dstate, 16),
        batch,
        nheads,
    )
    _chunk_state_varlen_kernel[grid](
        x,
        B,
        dt,
        dA_cumsum,
        chunk_states,
        cu_seqlens,
        states,
        initial_states,
        headdim,
        dstate,
        chunk_size,
        total_seqlen,
        nheads // ngroups,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        dt.stride(1),
        dt.stride(0),
        dt.stride(2),
        dA_cumsum.stride(1),
        dA_cumsum.stride(0),
        dA_cumsum.stride(2),
        chunk_states.stride(0),
        chunk_states.stride(1),
        chunk_states.stride(2),
        chunk_states.stride(3),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        initial_states.stride(0),
        initial_states.stride(1),
        initial_states.stride(2),
        initial_states.stride(3),
        HAS_INITSTATES=True,
    )
