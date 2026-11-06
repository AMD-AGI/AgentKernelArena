# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Jagged x dense batched matmul with a per-group broadcast bias add (Triton).

Out[s_b:e_b, :] = Jagged[s_b:e_b, :] @ Dense[b] + Bias[b][None, :], where the group boundaries
s_b, e_b come from seq_offsets. Adapted from the forward Triton kernel of Meta's generative-recommenders
(generative_recommenders/ops/triton/triton_jagged.py).
"""

import triton
import triton.language as tl

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
NUM_STAGES = 3
NUM_WARPS = 8


@triton.jit
def jagged_dense_bmm_broadcast_add_kernel(
    seq_offsets,
    Jagged,
    Dense,
    Bias,
    Out,
    N,
    K,
    stride_jm,
    stride_db,
    stride_dk,
    stride_dn,
    stride_bias_b,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computing bmm Out = Jagged x Dense + Bias
    M is the jagged dimension
    Jagged has shape (sum_B(M_i), K), Dense has shape (B, K, N), Bias has shape (B, N), and Out has shape (sum_B(M_i), N)
    """

    off_n = tl.program_id(0)
    off_m = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N
    if start_m >= seq_len:
        return

    Jagged += (seq_start + start_m) * stride_jm
    Dense += off_b.to(tl.int64) * stride_db
    Out += seq_start * stride_om

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    jg_ptrs = Jagged + offs_m[:, None] * stride_jm + offs_k[None, :]
    dn_ptrs = Dense + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        jg = tl.load(
            jg_ptrs,
            mask=(offs_m[:, None] < (seq_len - start_m)) & ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        dn = tl.load(
            dn_ptrs,
            mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(jg, dn)
        jg_ptrs += BLOCK_K
        dn_ptrs += BLOCK_K * stride_dk

    bias_ptrs = Bias + off_b.to(tl.int64) * stride_bias_b + offs_n
    bias = tl.load(bias_ptrs, mask=offs_n < N)
    accumulator += bias[None, :].to(tl.float32)

    out = accumulator.to(Out.dtype.element_ty)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    Out += start_m * stride_om
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < (seq_len - start_m)) & (offs_n[None, :] < N),
    )


def run(*, seq_offsets, jagged, dense, bias, max_seq_len, out) -> None:
    n_groups = dense.shape[0]
    n = dense.shape[2]
    k = jagged.shape[1]
    grid = (
        triton.cdiv(n, BLOCK_N),
        triton.cdiv(max_seq_len, BLOCK_M),
        n_groups,
    )
    jagged_dense_bmm_broadcast_add_kernel[grid](
        seq_offsets,
        jagged,
        dense,
        bias,
        out,
        n,
        k,
        jagged.stride(0),
        dense.stride(0),
        dense.stride(1),
        dense.stride(2),
        bias.stride(0),
        out.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
