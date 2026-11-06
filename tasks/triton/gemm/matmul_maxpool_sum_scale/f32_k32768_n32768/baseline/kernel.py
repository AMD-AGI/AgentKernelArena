# Copyright (c) 2016- Facebook, Inc and the PyTorch contributors. BSD-3-Clause, see LICENSE.
"""out = scale_factor * sum(max_pool1d(x @ weight.T + bias, 2), dim=1) in float32: a standalone Triton
rewrite of the two kernels PyTorch Inductor generates for this module under max-autotune on ROCm
(pytorch/pytorch v2.15.0).

1. ``_triton_tem_fused_addmm_0`` is the mm template, torch/_inductor/kernel/templates/triton_mm.py.jinja,
   rendered for x [M, K] row-major and weight.T [K, N] (the transposed view of the row-major [N, K]
   weight, strides (1, K)) with the addmm epilogue: tuned_addmm passes the bias as the prefix input
   and addmm_epilogue (torch/_inductor/kernel/mm_common.py, alpha = beta = 1) adds it in the generated
   store suffix. The bias stays in the template because the addmm's user is a max pool, not a
   pointwise op, so the post-grad pattern unfuse_bias_add_to_pointwise does not apply. The launch grid
   is mm_grid (torch/_inductor/kernel/mm_common.py). torch.backends.cuda.matmul.allow_tf32 defaults to
   False, so the dot runs in IEEE float32 (ALLOW_TF32=False is written as input_precision="ieee").
2. ``_triton_red_fused_max_pool2d_with_indices_mul_sum_1`` is the looped inner reduction Inductor
   generates for the pooled sum: max_pool1d lowers through max_pool2d_with_indices to the low-memory
   max pool (torch/_inductor/lowering.py), whose two-element window is unrolled into
   tl.maximum(..., PropagateNan.ALL) of the two loads (the unused offsets are dropped), fused into the
   sum over the 16384 windows; the multiply by scale_factor runs on the reduced value before the store.
   Inductor bakes the module's scale_factor = 0.5 in as a constant; here it is the scalar argument.

Inductor autotunes the mm template over ROCmConfigHeuristic.mm_configs
(torch/_inductor/heuristics/template/triton.py). Without a GPU to autotune on, the baseline picks
one entry of that list per M bucket (MM_CONFIGS); num_stages=2 is get_backend_num_stages() on ROCm
and matrix_instr_nonkdim=16 is the ROCmGemmConfig default. The reduction uses Inductor's contiguous
inner-reduction config (torch/_inductor/heuristics/triton_codegen/reduction.py): XBLOCK=1,
R0_BLOCK=min(r0_numel, 2048), num_warps = R0_BLOCK // 128 capped at 8 on a wave64 GPU.
"""
import torch
import triton
import triton.language as tl

# (largest M, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, GROUP_M), each an entry of the ROCm mm_configs.
MM_CONFIGS = (
    (32, 32, 32, 128, 4, 8),
    (256, 64, 64, 64, 4, 4),
    (None, 128, 128, 32, 8, 16),
)
MM_NUM_STAGES = 2
MATRIX_INSTR_NONKDIM = 16
POOL_KERNEL = 2
RED_XBLOCK = 1
RED_R0_BLOCK = 2048
RED_NUM_WARPS = 8


@triton.jit
def _triton_tem_fused_addmm_0(in_ptr0, arg_A, arg_B, out_ptr0, M,
                              N: tl.constexpr, K: tl.constexpr, A_CONTIGUOUS_HINT: tl.constexpr,
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
    # A is row-major (stride_am == K, stride_ak == 1); the template's M >= BLOCK_M test arrives as
    # A_CONTIGUOUS_HINT because M is a runtime argument here.
    if A_CONTIGUOUS_HINT:
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
            a = tl.load(A + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_K])), None, eviction_policy='evict_last')
        else:
            a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
            a = tl.load(A + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_K])), a_mask, eviction_policy='evict_last',
                        other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_m + stride_bn * idx_n
        if EVEN_K:
            b = tl.load(B + (tl.broadcast_to(xindex, [BLOCK_K, BLOCK_N])), None, eviction_policy='evict_last')
        else:
            b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
            b = tl.load(B + (tl.broadcast_to(xindex, [BLOCK_K, BLOCK_N])), b_mask, eviction_policy='evict_last',
                        other=0.0)

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
def _triton_red_fused_max_pool2d_with_indices_mul_sum_1(out_ptr1, in_ptr0, scale_factor, xnumel,
                                                         r0_numel: tl.constexpr, XBLOCK: tl.constexpr,
                                                         R0_BLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (2 * r0_1 + 2 * r0_numel * x0), r0_mask & xmask, eviction_policy='evict_first',
                       other=0.0)
        tmp1 = tl.load(in_ptr0 + (1 + 2 * r0_1 + 2 * r0_numel * x0), r0_mask & xmask,
                       eviction_policy='evict_first', other=0.0)
        tmp2 = tl.maximum(tmp1, tmp0, tl.PropagateNan.ALL)
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = tmp4 * scale_factor
    tl.store(out_ptr1 + (x0), tmp6, xmask)


def _mm_config(m):
    for limit, block_m, block_n, block_k, num_warps, group_m in MM_CONFIGS:
        if limit is None or m <= limit:
            return block_m, block_n, block_k, num_warps, group_m
    raise ValueError(m)


def run(*, x, weight, bias, scale_factor, out):
    m, k = x.shape
    n = weight.shape[0]
    block_m, block_n, block_k, num_warps, group_m = _mm_config(m)
    # Inductor indexes in int32 when every operand has fewer than 2**31 elements.
    index_dtype = tl.int32 if max(x.numel(), weight.numel(), m * n) < 2 ** 31 else tl.int64
    buf0 = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _triton_tem_fused_addmm_0[(triton.cdiv(m, block_m) * triton.cdiv(n, block_n), 1, 1)](
        bias, x, weight, buf0, m,
        N=n, K=k, A_CONTIGUOUS_HINT=(m >= block_m and k > 1), GROUP_M=group_m, EVEN_K=k % block_k == 0,
        ACC_TYPE=tl.float32, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, INDEX_DTYPE=index_dtype,
        num_warps=num_warps, num_stages=MM_NUM_STAGES, matrix_instr_nonkdim=MATRIX_INSTR_NONKDIM)
    _triton_red_fused_max_pool2d_with_indices_mul_sum_1[(triton.cdiv(m, RED_XBLOCK), 1, 1)](
        out, buf0, scale_factor, m, r0_numel=n // POOL_KERNEL, XBLOCK=RED_XBLOCK, R0_BLOCK=RED_R0_BLOCK,
        num_warps=RED_NUM_WARPS, num_stages=1)
