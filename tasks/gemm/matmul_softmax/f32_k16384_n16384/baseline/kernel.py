# Copyright (c) 2016- Facebook, Inc and the PyTorch contributors. BSD-3-Clause, see LICENSE.
"""out = softmax(x @ weight.T + bias, dim=1), a standalone Triton rewrite of the two kernels PyTorch
Inductor generates for this module (nn.Linear, nn.Dropout in eval mode, softmax) under max-autotune
(pytorch/pytorch v2.15.0). Dropout in eval mode is the identity and leaves no trace in the graph.

- Kernel 1, the GEMM. The addmm feeds the softmax's amax reduction, which is not a pointwise use, so the
  post-grad pass keeps the addmm (torch/_inductor/fx_passes/post_grad.py, should_prefer_unfused_addmm)
  and tuned_addmm (torch/_inductor/kernel/mm.py) lowers it to the Triton mm template
  (torch/_inductor/kernel/templates/triton_mm.py.jinja) with addmm_epilogue (alpha = beta = 1, so
  acc + bias) as its store suffix, writing the logits to an intermediate buffer. The template is
  rendered for a contiguous [M, K] activation and B = weight.T (strides (1, K)), with N, K and the
  strides as compile-time constants and the batch M as a runtime size, as in Inductor's dynamic-batch
  kernel (the template's M >= BLOCK_M contiguity test becomes the constexpr A_M_HINT); the grid is
  mm_grid (torch/_inductor/kernel/mm_common.py). The block configuration is chosen from three entries
  of ROCmConfigHeuristic.mm_configs (torch/_inductor/heuristics/template/triton.py) instead of
  autotuning: the largest tile that still launches at least one tile per MI355X compute unit, its
  BLOCK_M scaled to the batch and its warp count capped as BaseConfigHeuristic._scale_mm_configs and
  _finalize_mm_configs do; num_stages=2 is Inductor's ROCm default and matrix_instr_nonkdim=16 the
  ROCmGemmConfig default. ACC_TYPE is float32 and ALLOW_TF32 is False, so the dot is IEEE float32.
- Kernel 2, the softmax. Inductor decomposes softmax into amax, sub, exp, sum and div and, with
  config.online_softmax, generates one looped reduction kernel per row block: a first loop folds each
  block into per-lane running (max, sum) pairs with triton_helpers.online_softmax_combine, a final
  online_softmax_reduce combines the lanes, and a second loop writes exp(y - max) / sum
  (torch/_inductor/codegen/triton.py; ROCm keeps per-lane, not scalar, accumulators). The helpers are
  copied from torch/_inductor/runtime/triton_helpers.py with use_fast_math and strict_signed_zero False
  (their defaults), so the exponentials are libdevice.exp. The launch configuration is the contiguous
  config of torch/_inductor/heuristics/triton_codegen/reduction.py for this size: XBLOCK 1,
  R0_BLOCK 2048, r // 128 warps capped at 8 on wave64.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, GROUP_M) entries of ROCmConfigHeuristic.mm_configs, largest first.
MM_CONFIGS = (
    (128, 128, 32, 8, 16),
    (64, 64, 64, 4, 4),
    (32, 32, 128, 4, 8),
)
NUM_STAGES = 2
MATRIX_INSTR_NONKDIM = 16
MIN_BLOCK = 16  # _scale_mm_configs min_block_size and min_block_size_k
TARGET_TILES = 256  # compute units of an MI355X
SOFTMAX_XBLOCK = 1
SOFTMAX_R0_BLOCK = 2048
SOFTMAX_NUM_WARPS = 8


@triton.jit
def _triton_tem_fused_addmm(in_ptr0, arg_A, arg_B, out_ptr0, M,
                            N: tl.constexpr, K: tl.constexpr, A_M_HINT: tl.constexpr,
                            GROUP_M: tl.constexpr, EVEN_K: tl.constexpr, ACC_TYPE: tl.constexpr,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                            INDEX_DTYPE: tl.constexpr):
    A = arg_A
    B = arg_B

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
    if A_M_HINT:
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
        xindex = idx_n + stride_am * idx_m
        if EVEN_K:
            a = tl.load(A + (xindex))
        else:
            a = tl.load(A + (xindex), mask=offs_k[None, :] < (K - k_idx * BLOCK_K), other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_m + stride_bn * idx_n
        if EVEN_K:
            b = tl.load(B + (xindex))
        else:
            b = tl.load(B + (xindex), mask=offs_k[:, None] < (K - k_idx * BLOCK_K), other=0.0)

        acc += tl.dot(a, b, input_precision="ieee", out_dtype=ACC_TYPE)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(INDEX_DTYPE)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(INDEX_DTYPE)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + N * idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _max2(a, dim):
    return tl.reduce(a, dim, _maximum)


@triton.jit
def _online_softmax_reduce(lhs_max, lhs_sum, dim):
    out_max = _max2(lhs_max, dim)
    out_max_keepdim = tl.expand_dims(out_max, dim)
    delta = tl.where(out_max_keepdim == float("-inf"), 0, lhs_max - out_max_keepdim)
    out_sum = tl.sum(lhs_sum * libdevice.exp(delta), dim)
    return out_max, out_sum


@triton.jit
def _online_softmax_combine(lhs_max, lhs_sum, rhs_max):
    # lhs is the accumulator and rhs the next block of data, whose sum is 1.
    out_max = _maximum(lhs_max, rhs_max)
    lhs_scale = tl.where(out_max == float("-inf"), 1.0, libdevice.exp(lhs_max - out_max))
    rhs_scale = tl.where(out_max == float("-inf"), 1.0, libdevice.exp(rhs_max - out_max))
    out_sum = lhs_sum * lhs_scale + rhs_scale
    return out_max, out_sum


@triton.jit
def _triton_red_fused__softmax(in_ptr0, out_ptr2, xnumel, r0_numel: tl.constexpr,
                               XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp1_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp1_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + r0_numel * x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])

        _tmp1_max_next, _tmp1_sum_next = _online_softmax_combine(_tmp1_max, _tmp1_sum, tmp1)

        _tmp1_max = tl.where(r0_mask & xmask, _tmp1_max_next, _tmp1_max)
        _tmp1_sum = tl.where(r0_mask & xmask, _tmp1_sum_next, _tmp1_sum)

    tmp1, tmp2 = _online_softmax_reduce(_tmp1_max, _tmp1_sum, 1)
    tmp1 = tmp1[:, None]
    tmp2 = tmp2[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp3 = tl.load(in_ptr0 + (r0_1 + r0_numel * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tmp3 - tmp1
        tmp5 = libdevice.exp(tmp4)
        tmp6 = (tmp5 / tmp2)
        tl.store(out_ptr2 + (r0_1 + r0_numel * x0), tmp6, r0_mask & xmask)


def _scaled(block: int, size: int) -> int:
    # _scale_mm_configs: max(min(block, next_power_of_2(size)), min_block_size)
    return max(min(block, triton.next_power_of_2(size)), MIN_BLOCK)


def _config(m: int, n: int, k: int):
    for block_m, block_n, block_k, num_warps, group_m in MM_CONFIGS:
        block_m, block_n, block_k = _scaled(block_m, m), _scaled(block_n, n), _scaled(block_k, k)
        if triton.cdiv(m, block_m) * triton.cdiv(n, block_n) >= TARGET_TILES:
            break
    # _finalize_mm_configs: each warp computes at least a 16x16 tile.
    return block_m, block_n, block_k, min(num_warps, block_m * block_n // 256), group_m


def run(*, x, weight, bias, out):
    m, k = x.shape
    n = weight.shape[0]
    # The logits buffer Inductor allocates between the two kernels.
    logits = torch.empty((m, n), dtype=out.dtype, device=out.device)
    block_m, block_n, block_k, num_warps, group_m = _config(m, n, k)
    # Inductor indexes in int32 when every operand fits, otherwise in int64.
    index_dtype = tl.int32 if max(x.numel(), weight.numel(), out.numel()) < 2 ** 31 else tl.int64
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), 1, 1)
    _triton_tem_fused_addmm[grid](
        bias, x, weight, logits, m,
        N=n, K=k, A_M_HINT=m >= block_m, GROUP_M=group_m, EVEN_K=k % block_k == 0, ACC_TYPE=tl.float32,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, INDEX_DTYPE=index_dtype,
        num_warps=num_warps, num_stages=NUM_STAGES, matrix_instr_nonkdim=MATRIX_INSTR_NONKDIM)
    _triton_red_fused__softmax[(triton.cdiv(m, SOFTMAX_XBLOCK), 1, 1)](
        logits, out, m, r0_numel=n, XBLOCK=SOFTMAX_XBLOCK, R0_BLOCK=SOFTMAX_R0_BLOCK,
        num_warps=SOFTMAX_NUM_WARPS)
