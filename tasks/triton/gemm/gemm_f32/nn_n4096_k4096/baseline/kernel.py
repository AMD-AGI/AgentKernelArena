# Copyright (c) 2016- Facebook, Inc and the PyTorch contributors. BSD-3-Clause, see LICENSE.
"""Float32 GEMM out = a @ b, a standalone Triton rewrite of the kernel PyTorch Inductor generates for
torch.mm under max-autotune (pytorch/pytorch v2.15.0):

- the kernel body is torch/_inductor/kernel/templates/triton_mm.py.jinja (the template Inductor uses
  on ROCm with Triton >= 3.3) rendered for a contiguous [m, k] @ [k, n] float32 product with a
  dynamic M: M is a runtime argument, N, K and the strides are compile-time constants, the loads are
  Inductor's generated index expressions and the store is its suffix for a contiguous [m, n] output;
- the launch grid is mm_grid from torch/_inductor/kernel/mm_common.py;
- the block configurations are entries of ROCmConfigHeuristic.mm_configs in
  torch/_inductor/heuristics/template/triton.py, scaled to the problem size as
  BaseConfigHeuristic._scale_mm_configs and ROCmConfigHeuristic._finalize_mm_configs do (block
  sizes clamped to max(next_power_of_2(dim), 16), num_warps capped at block_m * block_n // 256,
  waves_per_eu, where an entry sets it, becoming 8 // num_warps), num_stages being Inductor's ROCm
  default of 2. Inductor picks among them by autotuning; this baseline instead fixes one entry per
  range of M (_CONFIGS below);
- ACC_TYPE is float32 and ALLOW_TF32 is False (torch.backends.cuda.matmul.fp32_precision defaults to
  "ieee"), so the dot runs in IEEE float32 precision.
"""
import triton
import triton.language as tl

NUM_STAGES = 2
MATRIX_INSTR_NONKDIM = 16

# (largest M, ROCmGemmConfig(block_m, block_n, block_k, num_warps, group_m, waves_per_eu)), from
# ROCmConfigHeuristic.mm_configs; the last entry covers every larger M.
_CONFIGS = (
    (64, (64, 16, 128, 4, 8, 2)),
    (256, (64, 32, 64, 4, 8, 0)),
    (1024, (128, 64, 64, 4, 16, 0)),
    (None, (128, 128, 32, 8, 16, 0)),
)


@triton.jit
def _triton_tem_fused_mm(arg_A, arg_B, out_ptr0, M,
                         N: tl.constexpr, K: tl.constexpr,
                         GROUP_M: tl.constexpr, EVEN_K: tl.constexpr, ACC_TYPE: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                         INDEX_DTYPE: tl.constexpr):
    A = arg_A
    B = arg_B

    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am: tl.constexpr = K
    stride_ak: tl.constexpr = 1
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
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K).to(INDEX_DTYPE)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + K * idx_m
        if EVEN_K:
            a = tl.load(A + (xindex))
        else:
            a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
            a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + N * idx_m
        if EVEN_K:
            b = tl.load(B + (xindex))
        else:
            b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
            b = tl.load(B + (xindex), mask=b_mask, other=0.0)

        acc += tl.dot(a, b, input_precision="ieee", out_dtype=ACC_TYPE)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(INDEX_DTYPE)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(INDEX_DTYPE)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + N * idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)


def _next_power_of_2(value):
    return 1 << (value - 1).bit_length()


def _config(m, n, k):
    """The configuration of the range holding m, scaled as Inductor scales its configs."""
    for limit, config in _CONFIGS:
        if limit is None or m <= limit:
            break
    block_m, block_n, block_k, num_warps, group_m, waves_per_eu = config
    block_m = max(min(block_m, max(_next_power_of_2(m), 16)), 16)
    block_n = max(min(block_n, max(_next_power_of_2(n), 16)), 16)
    block_k = max(min(block_k, max(_next_power_of_2(k), 16)), 16)
    num_warps = min(num_warps, block_m * block_n // 256)
    # ROCmConfigHeuristic._finalize_mm_configs: an entry that sets waves_per_eu gets 8 // num_warps.
    if waves_per_eu:
        waves_per_eu = 8 // num_warps
    return block_m, block_n, block_k, num_warps, group_m, waves_per_eu


def run(*, a, b, out):
    m, k = a.shape
    n = b.shape[1]
    block_m, block_n, block_k, num_warps, group_m, waves_per_eu = _config(m, n, k)
    # mm_grid: one program per output tile.
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), 1, 1)
    # Inductor indexes in int32 when every operand fits, otherwise in int64.
    index_dtype = tl.int32 if max(a.numel(), b.numel(), out.numel()) < 2 ** 31 else tl.int64
    options = {"waves_per_eu": waves_per_eu} if waves_per_eu else {}
    _triton_tem_fused_mm[grid](
        a, b, out, m,
        N=n, K=k, GROUP_M=group_m, EVEN_K=k % block_k == 0, ACC_TYPE=tl.float32,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, INDEX_DTYPE=index_dtype,
        num_warps=num_warps, num_stages=NUM_STAGES, matrix_instr_nonkdim=MATRIX_INSTR_NONKDIM, **options)
