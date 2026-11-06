# Copyright (c) 2016- Facebook, Inc and the PyTorch contributors. BSD-3-Clause, see LICENSE.
"""Batched float32 GEMM out[q] = a[q] @ b[q], a standalone Triton rewrite of the kernel PyTorch
Inductor generates for torch.bmm under max-autotune (pytorch/pytorch v2.15.0):

- the kernel body is torch/_inductor/kernel/templates/triton_bmm.py.jinja rendered for contiguous
  [batch, m, k] @ [batch, k, n] float32 operands (strides and M, N, K become compile-time constants,
  the batch stays a runtime argument, the store is Inductor's generated suffix for a contiguous
  [batch, m, n] output);
- the launch grid is bmm_grid from torch/_inductor/kernel/bmm.py (batch split across grid y/z);
- the block configuration is one entry of ROCmConfigHeuristic.mm_configs in
  torch/_inductor/heuristics/template/triton.py, ROCmGemmConfig(128, 128, 32, num_stages=2,
  num_warps=8, group_m=16) with matrix_instr_nonkdim=16, num_stages being Inductor's ROCm default;
- ACC_TYPE is float32 and ALLOW_TF32 is False (torch.backends.cuda.matmul.allow_tf32 defaults to
  False), so the dot runs in IEEE float32 precision.
"""
import triton
import triton.language as tl

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
GROUP_M = 16
NUM_STAGES = 2
NUM_WARPS = 8
MATRIX_INSTR_NONKDIM = 16
MAX_Y_GRID = 65535  # torch/_inductor/runtime/runtime_utils.py get_max_y_grid()


@triton.jit
def _triton_tem_fused_bmm(A, B, out_ptr0, BATCH,
                          M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                          GROUP_M: tl.constexpr, EVEN_K: tl.constexpr, ACC_TYPE: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          INDEX_DTYPE: tl.constexpr):
    stride_aq: tl.constexpr = M * K
    stride_am: tl.constexpr = K
    stride_ak: tl.constexpr = 1

    stride_bq: tl.constexpr = K * N
    stride_bk: tl.constexpr = N
    stride_bn: tl.constexpr = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(INDEX_DTYPE)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(INDEX_DTYPE)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K).to(INDEX_DTYPE)

    # Reconstruct batch index from grid_y/grid_z split (handles batch > 65535)
    idx_q = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)).to(INDEX_DTYPE)
    # Clamp to valid range for safe pointer arithmetic; out-of-bounds CTAs are
    # masked off at the store below.
    idx_q_clamped = tl.minimum(idx_q, BATCH - 1)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q_clamped * stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q_clamped * stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k_remaining in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.)
        acc += tl.dot(a, b, input_precision="ieee")
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(INDEX_DTYPE)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(INDEX_DTYPE)
    idx_q = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)).to(INDEX_DTYPE)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N) & (idx_q < BATCH)

    # inductor generates a suffix
    xindex = idx_n + N * idx_m + (M * N) * idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def run(*, a, b, out):
    batch, m, k = a.shape
    n = b.shape[2]
    # bmm_grid: tiles on x, the batch split across y and z so y stays within the grid limit.
    grid_z = max(triton.cdiv(batch, MAX_Y_GRID), 1)
    grid_y = triton.cdiv(batch, grid_z)
    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N), grid_y, grid_z)
    # Inductor indexes in int32 when every operand fits, otherwise in int64.
    index_dtype = tl.int32 if max(a.numel(), b.numel(), out.numel()) < 2 ** 31 else tl.int64
    _triton_tem_fused_bmm[grid](
        a, b, out, batch,
        M=m, N=n, K=k, GROUP_M=GROUP_M, EVEN_K=k % BLOCK_K == 0, ACC_TYPE=tl.float32,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, INDEX_DTYPE=index_dtype,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES, matrix_instr_nonkdim=MATRIX_INSTR_NONKDIM)
