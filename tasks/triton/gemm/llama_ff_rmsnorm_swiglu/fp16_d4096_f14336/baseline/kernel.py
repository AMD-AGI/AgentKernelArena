# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

# The kernel in this file is adapted from TritonBench's llama_ff_triton:
# https://github.com/thunlp/TritonBench - Apache License 2.0

#  LLaMA Feed-Forward: fused RMSNorm + SiLU-gated linear projections Triton kernel.
import triton
import triton.language as tl


@triton.jit
def ff_llama(
    a_ptr, w1_ptr, w3_ptr, out_ptr, rms_w_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_outm, stride_outn,
    stride_rms_w,
    USE_FP8: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_bn[None, :] * stride_w1n)
    w3_ptrs = w3_ptr + (offs_k[:, None] * stride_w3k + offs_bn[None, :] * stride_w3n)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    rms_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        a_f32 = a.to(tl.float32)
        rms_acc += tl.sum(a_f32 * a_f32, axis=1)
        rms_w = tl.load(rms_w_ptr + (kb * BLOCK_SIZE_K + offs_k) * stride_rms_w)[None, :]
        if USE_FP8:
            rms_w = rms_w.to(tl.float8e5, bitcast=True)
            rms_w = rms_w.to(tl.float16)
        a = a * rms_w
        b = tl.load(w1_ptrs)
        if USE_FP8:
            b = b.to(tl.float8e5, bitcast=True).to(tl.float32).to(tl.float16)
        acc1 += tl.dot(a, b)
        c = tl.load(w3_ptrs)
        if USE_FP8:
            c = c.to(tl.float8e5, bitcast=True).to(tl.float32).to(tl.float16)
        acc2 += tl.dot(a, c)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w1_ptrs += BLOCK_SIZE_K * stride_w1k
        w3_ptrs += BLOCK_SIZE_K * stride_w3k
    a_mean = rms_acc / K + EPS
    a_norm = tl.math.rsqrt(a_mean)
    acc1 = acc1 * a_norm[:, None]
    acc2 = acc2 * a_norm[:, None]
    accumulator = (acc1 * tl.sigmoid(acc1)) * acc2
    offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (stride_outm * offs_outm[:, None] + stride_outn * offs_outn[None, :])
    out_mask = (offs_outm[:, None] < M) & (offs_outn[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=out_mask)


def run(*, x, w1, w3, rms_w, epsilon, out):
    M, K = x.shape
    N = w1.shape[0]
    # The kernel does not mask the N and K tails: both are multiples of the tile.
    block_m = 16 if M <= 16 else 32 if M <= 64 else 64 if M <= 256 else 128
    block_n, block_k = 64, 64
    ff_llama[(triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)](
        x, w1, w3, out, rms_w,
        M, N, K,
        x.stride(0), x.stride(1),
        w1.stride(1), w1.stride(0),
        w3.stride(1), w3.stride(0),
        out.stride(0), out.stride(1),
        rms_w.stride(0),
        USE_FP8=False,
        EPS=epsilon,
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
        num_stages=2, num_warps=4,
    )
