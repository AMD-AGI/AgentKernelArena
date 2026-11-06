"""Split-K fp32 matrix-vector product y = A @ x in Triton (task-original).

Each program reduces one row segment of A against the matching segment of x with a vector accumulator; with few
rows the K axis is split across programs so the GPU is filled, and a second kernel adds the partial sums.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_partial_kernel(a_ptr, x_ptr, out_ptr, K, stride_am, K_PER_SPLIT: tl.constexpr, BLOCK_K: tl.constexpr,
                         SPLIT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    k_start = split * K_PER_SPLIT
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k_off in range(0, K_PER_SPLIT, BLOCK_K):
        idx = k_start + k_off + tl.arange(0, BLOCK_K)
        mask = idx < K
        a = tl.load(a_ptr + row * stride_am + idx, mask=mask, other=0.0)
        xv = tl.load(x_ptr + idx, mask=mask, other=0.0)
        acc += a * xv
    tl.store(out_ptr + row * SPLIT + split, tl.sum(acc, axis=0))


@triton.jit
def _gemv_reduce_kernel(partial_ptr, out_ptr, M, SPLIT: tl.constexpr, BLOCK_M: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, SPLIT)
    mask = rows < M
    p = tl.load(partial_ptr + rows[:, None] * SPLIT + cols[None, :], mask=mask[:, None], other=0.0)
    tl.store(out_ptr + rows, tl.sum(p, axis=1), mask=mask)


def run(*, a, x, y):
    M, K = a.shape
    BLOCK_K = 2048
    split = 1
    while M * split < 2048 and split < 32:
        split *= 2
    k_per_split = triton.cdiv(triton.cdiv(K, split), BLOCK_K) * BLOCK_K
    if split == 1:
        _gemv_partial_kernel[(M, 1)](a, x, y, K, a.stride(0), K_PER_SPLIT=k_per_split, BLOCK_K=BLOCK_K, SPLIT=1)
        return
    partial = torch.empty((M, split), dtype=torch.float32, device=a.device)
    _gemv_partial_kernel[(M, split)](a, x, partial, K, a.stride(0), K_PER_SPLIT=k_per_split, BLOCK_K=BLOCK_K,
                                     SPLIT=split)
    BLOCK_M = 128
    _gemv_reduce_kernel[(triton.cdiv(M, BLOCK_M),)](partial, y, M, SPLIT=split, BLOCK_M=BLOCK_M)
