# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
"""Batched vector-matrix product out[m, n] = sum_k a[m, k] * b[m, n, k], the batched_vecmat kernel of the
rocmbench test_batched_vecmat task (after the test_vecmat unit test of the Triton project), with its
float32 reduction and tail masks. The kernel is copied unchanged; run() is the old timed wrapper
(batched_vecmat_triton_wrapper) writing into the preallocated output. See LICENSE (MIT).
"""
import triton
import triton.language as tl


@triton.jit
def batched_vecmat(
        # inputs
        A,  # shape: [dim_m, dim_k]
        B,  # shape: [dim_m, dim_n, dim_k]
        # dimensions
    dim_m, dim_n, dim_k,
        # outputs
        output,
        # block information
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
    m_index = tl.program_id(0)
    n_index = tl.program_id(1)
    # Output tile
    output_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_n \
        + (n_index * block_n + tl.arange(0, block_n))[None, :]

    vecmat = tl.zeros([block_m, block_n], dtype=tl.float32)
    k_blocks = tl.cdiv(dim_k, block_k)
    for k_index in range(k_blocks):
        # Load A tile
        a_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_k \
            + (k_index * block_k + tl.arange(0, block_k))[None, :]
        a = tl.load(A + a_tile,
                    mask=((m_index * block_m + tl.arange(0, block_m))[:, None] < dim_m) &
                         ((k_index * block_k + tl.arange(0, block_k))[None, :] < dim_k), other=0)

        # Load B tile, transposed to [n, m, k] in order to broadcast A on a
        # leading dimension.
        b_tile = (m_index * block_m + tl.arange(0, block_m))[None, :, None] * dim_n * dim_k \
            + (n_index * block_n + tl.arange(0, block_n))[:, None, None] * dim_k \
            + (k_index * block_k + tl.arange(0, block_k))[None, None, :]
        b = tl.load(B + b_tile,
                    mask=((m_index * block_m + tl.arange(0, block_m))[None, :, None] < dim_m) &
                         ((n_index * block_n + tl.arange(0, block_n))[:, None, None] < dim_n) &
                         ((k_index * block_k + tl.arange(0, block_k))[None, None, :] < dim_k), other=0)

        expanded_a, _ = tl.broadcast(a, b)
        vecmat += tl.trans(tl.sum(expanded_a * b, axis=2, dtype=tl.float32))

    tl.store(output + output_tile, vecmat,
             mask=((m_index * block_m + tl.arange(0, block_m))[:, None] < dim_m) &
                  ((n_index * block_n + tl.arange(0, block_n))[None, :] < dim_n))


# The block sizes of the original test_vecmat case and of the timed wrapper; num_warps and num_stages are
# Triton's defaults, as in that wrapper.
BLOCK_M = 16
BLOCK_N = 32
BLOCK_K = 64


def run(*, a, b, out):
    dim_m, dim_k = a.shape
    dim_n = b.shape[1]
    grid = (triton.cdiv(dim_m, BLOCK_M), triton.cdiv(dim_n, BLOCK_N))
    batched_vecmat[grid](
        a, b,
        dim_m, dim_n, dim_k,
        out,
        block_m=BLOCK_M, block_n=BLOCK_N, block_k=BLOCK_K)
