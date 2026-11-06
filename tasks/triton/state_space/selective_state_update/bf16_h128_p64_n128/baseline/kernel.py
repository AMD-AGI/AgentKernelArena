# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/selective_state_update.py
# and vllm-project/vllm, vllm/model_executor/layers/mamba/ops/mamba_ssm.py
"""Mamba2 single-step selective state update (decode) in Triton.

Per sequence b, head h, channel p and state index n:

    dt  = softplus(dt[b, h] + dt_bias[h])
    state[s, h, p, n] = state[s, h, p, n] * exp(A[h] * dt) + B[b, g, n] * dt * x[b, h, p]
    out[b, h, p] = (sum_n state[s, h, p, n] * C[b, g, n] + D[h] * x[b, h, p]) * silu(z[b, h, p])

with s = state_batch_indices[b] and g = h // (nheads / ngroups). As in Mamba2 decode, A, dt,
dt_bias and D are per head and enter the kernel as stride-0 views over the head dimension
(the kernel's TIE_HDIM path); the state is read and written in place.
"""
import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1


@triton.jit
def fast_exp(x):
    # vllm/model_executor/layers/mamba/ops/triton_helpers.py: exp(x) = exp2(x * log2(e))
    LOG2E = tl.constexpr(1.4426950408889634)
    return tl.math.exp2(LOG2E * x)


@triton.jit
def softplus(dt):
    dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
    return dt


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"] is not None})
@triton.heuristics({"IS_SPEC_DECODING": lambda args: args["num_accepted_tokens_ptr"] is not None})
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit(do_not_specialize=["N"])
def _selective_scan_update_kernel(
    state_ptr, x_ptr, dt_ptr, dt_bias_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr,
    out_ptr, state_batch_indices_ptr, dst_state_batch_indices_ptr,
    pad_slot_id, num_accepted_tokens_ptr, cu_seqlens_ptr,
    N, nheads, dim, dstate, nheads_ngroups_ratio,
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    stride_x_batch, stride_x_head, stride_x_dim,
    stride_dt_batch, stride_dt_head, stride_dt_dim,
    stride_dt_bias_head, stride_dt_bias_dim,
    stride_A_head, stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_group, stride_B_dstate,
    stride_C_batch, stride_C_group, stride_C_dstate,
    stride_D_head, stride_D_dim,
    stride_z_batch, stride_z_head, stride_z_dim,
    stride_out_batch, stride_out_head, stride_out_dim,
    stride_state_indices_batch, stride_state_indices_T,
    stride_dst_state_indices_batch, stride_dst_state_indices_T,
    DT_SOFTPLUS: tl.constexpr, TIE_HDIM: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr, HAS_D: tl.constexpr, HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr, IS_SPEC_DECODING: tl.constexpr,
    IS_VARLEN: tl.constexpr, BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
        seq_len = eos - bos
        if seq_len == 0:
            return
    else:
        bos = pid_b
        seq_len = 1

    state_ptr_base = state_ptr

    if HAS_STATE_BATCH_INDICES:
        if IS_SPEC_DECODING:
            num_accepted = tl.load(num_accepted_tokens_ptr + pid_b).to(tl.int64)
            init_token_idx = tl.maximum(num_accepted - 1, 0)
        else:
            init_token_idx = 0

        dst_state_batch_indices_ptr += pid_b * stride_dst_state_indices_batch
        if not IS_SPEC_DECODING:
            dst_state_batch_idx = tl.load(dst_state_batch_indices_ptr + init_token_idx * stride_dst_state_indices_T).to(tl.int64)
            dst_state_ptr = state_ptr + (dst_state_batch_idx * stride_state_batch + pid_h * stride_state_head)

        state_batch_indices_ptr += (pid_b * stride_state_indices_batch + init_token_idx * stride_state_indices_T)
        state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    else:
        dst_state_ptr = state_ptr + pid_b * stride_state_batch + pid_h * stride_state_head
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    x_ptr += bos * stride_x_batch + pid_h * stride_x_head
    dt_ptr += bos * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += bos * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += bos * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += bos * stride_z_batch + pid_h * stride_z_head
    out_ptr += bos * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate)
    if not IS_SPEC_DECODING:
        dst_state_ptrs = dst_state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate)

    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
    if HAS_STATE_BATCH_INDICES:
        mask &= state_batch_idx != pad_slot_id
    state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)

    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
        D_ptrs = D_ptr + offs_m * stride_D_dim
    A_ptrs = A_ptr + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate

    for i_t in range(seq_len):
        x_ptrs = x_ptr + offs_m * stride_x_dim
        dt_ptrs = dt_ptr + offs_m * stride_dt_dim
        B_ptrs = B_ptr + offs_n * stride_B_dstate
        C_ptrs = C_ptr + offs_n * stride_C_dstate
        if HAS_Z:
            z_ptrs = z_ptr + offs_m * stride_z_dim
        out_ptrs = out_ptr + offs_m * stride_out_dim

        x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if not TIE_HDIM:
            dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
            dA = fast_exp(A * dt[:, None])
        else:
            dt = tl.load(dt_ptr).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(A_ptr).to(tl.float32)
            dA = fast_exp(A * dt)

        B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        if HAS_D:
            D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_Z:
            z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

        dB = B[None, :] * dt[:, None] if not TIE_HDIM else B * dt
        state = state * dA + dB * x[:, None]

        if IS_SPEC_DECODING:
            dst_idx_ptr = dst_state_batch_indices_ptr + i_t * stride_dst_state_indices_T
            token_dst_idx = tl.load(dst_idx_ptr).to(tl.int64)
            if token_dst_idx != pad_slot_id:
                token_dst_ptrs = (
                    state_ptr_base + token_dst_idx * stride_state_batch
                    + pid_h * stride_state_head
                    + offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
                )
                tl.store(token_dst_ptrs, state.to(token_dst_ptrs.dtype.element_ty), mask=mask)

        out = tl.sum(state * C[None, :], axis=1)
        if HAS_D:
            out += x * D
        if HAS_Z:
            out *= z * tl.sigmoid(z)
        tl.store(out_ptrs, out, mask=offs_m < dim)

        x_ptr += stride_x_batch
        dt_ptr += stride_dt_batch
        B_ptr += stride_B_batch
        C_ptr += stride_C_batch
        out_ptr += stride_out_batch
        if HAS_Z:
            z_ptr += stride_z_batch

    if not IS_SPEC_DECODING:
        tl.store(dst_state_ptrs, state.to(dst_state_ptrs.dtype.element_ty), mask=mask)


def run(*, state, x, dt, dt_bias, A, B, C, D, z, state_batch_indices, out) -> None:
    _, nheads, dim, dstate = state.shape
    batch = x.shape[0]
    ngroups = B.shape[1]
    # Per-head parameters broadcast over the head dimension (stride 0), as vLLM's Mamba2 mixer does.
    A_t = A[:, None, None].expand(nheads, dim, dstate)
    dt_t = dt[:, :, None].expand(batch, nheads, dim)
    dt_bias_t = dt_bias[:, None].expand(nheads, dim)
    D_t = D[:, None].expand(nheads, dim)
    sbi = state_batch_indices.unsqueeze(1)

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)
    BLOCK_SIZE_M, num_warps = 4, 8
    if dstate <= 16:
        BLOCK_SIZE_M, num_warps = 32, 4
    elif dstate <= 32:
        BLOCK_SIZE_M, num_warps = 16, 4
    elif dstate <= 64:
        BLOCK_SIZE_M, num_warps = 8, 4
    elif dstate <= 128:
        BLOCK_SIZE_M, num_warps = 4, 4

    _selective_scan_update_kernel[grid](
        state, x, dt_t, dt_bias_t, A_t, B, C, D_t, z, out,
        sbi, sbi, PAD_SLOT_ID, None, None,
        batch, nheads, dim, dstate, nheads // ngroups,
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        x.stride(0), x.stride(1), x.stride(2),
        dt_t.stride(0), dt_t.stride(1), dt_t.stride(2),
        dt_bias_t.stride(0), dt_bias_t.stride(1),
        A_t.stride(0), A_t.stride(1), A_t.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        D_t.stride(0), D_t.stride(1),
        z.stride(0), z.stride(1), z.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        sbi.stride(0), sbi.stride(1),
        sbi.stride(0), sbi.stride(1),
        True, True, BLOCK_SIZE_M,
        num_warps=num_warps,
    )
