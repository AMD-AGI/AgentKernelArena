# Copyright (c) 2016- Facebook, Inc and the PyTorch contributors. BSD-3-Clause, see LICENSE.
"""out = max_j(gelu(avg_pool1d(x @ weight.T + bias, 16))[:, j] * scale_factor) in float32, a standalone
Triton rewrite of the two kernels PyTorch Inductor generates for nn.Linear -> AvgPool1d(16) -> exact
GELU -> scale -> max(dim=1) under torch.compile(mode="max-autotune") (pytorch/pytorch v2.15.0):

1. _triton_tem_fused_addmm: torch/_inductor/kernel/templates/triton_mm.py.jinja (the mm template that
   tuned_addmm in torch/_inductor/kernel/mm.py autotunes) rendered for A = x [M, K] row-major and
   B = weight.T, i.e. weight [N, K] read with strides (1, K); N, K and the strides are compile-time
   constants and M is a runtime size, as in Inductor's dynamic-batch kernel. Its store suffix is the
   addmm bias add (mm_common.addmm_epilogue). The pooling reads 16 neighbouring outputs, so Inductor
   cannot fuse it into the template epilogue; the linear output is materialized in float32.
2. _triton_per_fused_avg_pool2d_gelu_max_mul: the persistent reduction Inductor generates for the
   rest. avg_pool1d decomposes to avg_pool2d with a (1, 16) window, which _avg_poolnd in
   torch/_inductor/lowering.py unrolls (window <= 25) into a left-to-right sum of the 16 loads
   multiplied by 1/16 (div_prim); then the exact GELU decomposition
   x * 0.5 * (1 + erf(x * M_SQRT1_2)) (torch/_refs/nn/functional gelu), the multiplication by
   scale_factor and the NaN-propagating row maximum (triton_helpers.max2 / maximum), one row per
   XBLOCK slot over the 512 pooled values.

The mm launch grid is mm_grid from torch/_inductor/kernel/mm_common.py. The block configuration is an
entry of ROCmConfigHeuristic.mm_configs in torch/_inductor/heuristics/template/triton.py chosen by batch
size instead of autotuning, with BLOCK_M clamped to the batch as _scale_mm_configs does; num_stages is
Inductor's ROCm default (2) and matrix_instr_nonkdim the ROCmGemmConfig default (16). ACC_TYPE is
float32 and ALLOW_TF32 is False (the PyTorch default), written as input_precision="ieee".
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

IN_FEATURES = 8192   # K
OUT_FEATURES = 8192  # N
POOL = 16            # AvgPool1d kernel size (and stride)
POOLED = OUT_FEATURES // POOL
NUM_STAGES = 2
MATRIX_INSTR_NONKDIM = 16
# Persistent reduction: XBLOCK rows of 512 pooled values per program (rnumel * XBLOCK <= 4096).
REDUCE_XBLOCK = 8
REDUCE_NUM_WARPS = 8

# (largest batch, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, GROUP_M), each an entry of
# ROCmConfigHeuristic.mm_configs: ROCmGemmConfig(32, 32, 128, 2, 4, group_m=8),
# ROCmGemmConfig(64, 64, 64, 2, 4, group_m=4) and ROCmGemmConfig(128, 128, 32, 2, 8, group_m=16).
# The smaller tiles keep at least 256 programs (one per CU) at small batches.
CONFIGS = (
    (64, 32, 32, 128, 4, 8),
    (256, 64, 64, 64, 4, 4),
    (None, 128, 128, 32, 8, 16),
)


def _config(m):
    for limit, block_m, block_n, block_k, num_warps, group_m in CONFIGS:
        if limit is None or m <= limit:
            # _scale_mm_configs: BLOCK_M = max(min(BLOCK_M, next_power_of_2(M)), 16).
            return max(min(block_m, triton.next_power_of_2(m)), 16), block_n, block_k, num_warps, group_m
    raise AssertionError("unreachable")


@triton.jit
def _maximum(a, b):
    # torch/_inductor/runtime/triton_helpers.py maximum
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _max2(a, dim):
    # torch/_inductor/runtime/triton_helpers.py max2
    return tl.reduce(a, dim, _maximum)


@triton.jit
def _triton_tem_fused_addmm(arg_A, arg_B, in_ptr2, out_ptr0, M,
                            N: tl.constexpr, K: tl.constexpr,
                            A_ROWS_FULL: tl.constexpr, B_COLS_FULL: tl.constexpr,
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
    stride_bk: tl.constexpr = 1
    stride_bn: tl.constexpr = K

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
    # The template's ((stride_am == K and stride_ak == 1) ...) and (M >= BLOCK_M and K > 1)
    # tests, evaluated on the host because M is a runtime size here.
    if A_ROWS_FULL:
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if B_COLS_FULL:
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
        xindex = idx_n * stride_ak + idx_m * stride_am
        if EVEN_K:
            a = tl.load(A + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_K])))
        else:
            a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
            a = tl.load(A + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_K])), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_m * stride_bk + idx_n * stride_bn
        if EVEN_K:
            b = tl.load(B + (tl.broadcast_to(xindex, [BLOCK_K, BLOCK_N])))
        else:
            b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
            b = tl.load(B + (tl.broadcast_to(xindex, [BLOCK_K, BLOCK_N])), mask=b_mask, other=0.0)

        acc += tl.dot(a, b, input_precision="ieee", out_dtype=ACC_TYPE)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(INDEX_DTYPE)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(INDEX_DTYPE)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + N * idx_m
    tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)


@triton.jit
def _triton_per_fused_avg_pool2d_gelu_max_mul(in_ptr0, out_ptr0, xnumel, scale_factor,
                                              ROW: tl.constexpr, POOL_SIZE: tl.constexpr,
                                              R0_BLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_1 = r0_index
    x0 = xindex
    # The unrolled (1, 16) avg_pool2d window: tmp1 + tmp0, then tmp3 + tmp2, ... in load order.
    tmp_sum = tl.load(in_ptr0 + (POOL_SIZE * r0_1 + ROW * x0), xmask, other=0.0)
    for t in tl.static_range(1, POOL_SIZE):
        tmp_t = tl.load(in_ptr0 + (t + POOL_SIZE * r0_1 + ROW * x0), xmask, other=0.0)
        tmp_sum = tmp_t + tmp_sum
    tmp31 = 0.0625  # 1 / POOL_SIZE, the reciprocal div_prim multiplies by
    tmp32 = tmp_sum * tmp31
    tmp33 = 0.5
    tmp34 = tmp32 * tmp33
    tmp35 = 0.7071067811865476
    tmp36 = tmp32 * tmp35
    tmp37 = libdevice.erf(tmp36)
    tmp38 = 1.0
    tmp39 = tmp37 + tmp38
    tmp40 = tmp34 * tmp39
    tmp41 = scale_factor
    tmp42 = tmp40 * tmp41
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, R0_BLOCK])
    tmp45 = tl.where(xmask, tmp43, float("-inf"))
    tmp46 = _max2(tmp45, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp46, xmask)


def run(*, x, weight, bias, scale_factor, out):
    m = x.shape[0]
    block_m, block_n, block_k, num_warps, group_m = _config(m)
    linear = torch.empty((m, OUT_FEATURES), device=x.device, dtype=x.dtype)
    # mm_grid: one program per output tile.
    grid = (triton.cdiv(m, block_m) * triton.cdiv(OUT_FEATURES, block_n), 1, 1)
    # Inductor indexes in int32 when every operand fits, otherwise in int64.
    index_dtype = tl.int32 if max(x.numel(), weight.numel(), linear.numel()) < 2 ** 31 else tl.int64
    _triton_tem_fused_addmm[grid](
        x, weight, bias, linear, m,
        N=OUT_FEATURES, K=IN_FEATURES, A_ROWS_FULL=m >= block_m, B_COLS_FULL=OUT_FEATURES >= block_n,
        GROUP_M=group_m, EVEN_K=IN_FEATURES % block_k == 0, ACC_TYPE=tl.float32,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, INDEX_DTYPE=index_dtype,
        num_warps=num_warps, num_stages=NUM_STAGES, matrix_instr_nonkdim=MATRIX_INSTR_NONKDIM)
    _triton_per_fused_avg_pool2d_gelu_max_mul[(triton.cdiv(m, REDUCE_XBLOCK), 1, 1)](
        linear, out, m, scale_factor,
        ROW=OUT_FEATURES, POOL_SIZE=POOL, R0_BLOCK=POOLED, XBLOCK=REDUCE_XBLOCK,
        num_warps=REDUCE_NUM_WARPS)
