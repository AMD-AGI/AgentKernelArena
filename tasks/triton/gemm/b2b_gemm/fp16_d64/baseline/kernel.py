# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Back-to-back GEMM E = (A @ B^T) @ C in Triton: one program per BLOCK_M rows of A keeps A resident and
streams B and C in BLOCK_N-row tiles; the float32 intermediate is rounded to float16 before the second dot."""
import triton
import triton.language as tl

_BLOCK_M = 128
_BLOCK_N = 128


@triton.jit
def gemm_fusion_kernel(A, B, C, E,  #
                       M, N, K,  #
                       stride_am, stride_ak, stride_bn, stride_bk, stride_cn, stride_ck, stride_em, stride_ek,  #
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m[:, None] < M

    a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=m_mask, other=0.0)
    acc_e = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n[:, None] < N
        b = tl.load(B + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk, mask=n_mask, other=0.0)
        c = tl.load(C + offs_n[:, None] * stride_cn + offs_k[None, :] * stride_ck, mask=n_mask, other=0.0)
        o_ab = tl.dot(a, tl.trans(b))
        o_ab = o_ab.to(tl.float16)
        acc_e += tl.dot(o_ab, c)

    acc_e = acc_e.to(tl.float16)
    tl.store(E + offs_m[:, None] * stride_em + offs_k[None, :] * stride_ek, acc_e, mask=m_mask)


def run(*, a, b, c, e):
    n_rows, head_dim = a.shape
    n_keys = b.shape[0]
    gemm_fusion_kernel[(triton.cdiv(n_rows, _BLOCK_M),)](
        a, b, c, e, n_rows, n_keys, head_dim,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), e.stride(0), e.stride(1),
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=head_dim, num_warps=4)
