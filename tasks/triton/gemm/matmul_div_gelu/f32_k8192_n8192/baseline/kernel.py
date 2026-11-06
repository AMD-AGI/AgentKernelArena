# Copyright (c) 2016- Facebook, Inc and the PyTorch contributors. BSD-3-Clause, see LICENSE.
"""out = gelu((x @ weight.T + bias) / divisor) in float32, a standalone Triton rewrite of the kernel
PyTorch Inductor generates for nn.Linear -> divide -> exact GELU under
torch.compile(mode="max-autotune") (pytorch/pytorch v2.15.0):

- the kernel body is torch/_inductor/kernel/templates/triton_mm.py.jinja (the mm template that
  tuned_mm in torch/_inductor/kernel/mm.py autotunes) rendered for A = x [M, K] row-major and
  B = weight.T, i.e. weight [N, K] read with strides (1, K); N, K and the strides are
  compile-time constants and M is a runtime size, as in Inductor's dynamic-batch kernel;
- the store is the epilogue Inductor fuses into the template: the bias add (the post-grad pass
  splits an addmm whose users are all pointwise into mm + add, should_prefer_unfused_addmm in
  torch/_inductor/fx_passes/post_grad.py), the division by the constant divisor lowered to a
  multiplication by its reciprocal (lowering.div_prim), and the exact GELU decomposition
  x * 0.5 * (1 + erf(x * M_SQRT1_2)) (torch/_refs/nn/functional gelu, approximate="none");
- the launch grid is mm_grid from torch/_inductor/kernel/mm_common.py;
- the block configuration is an entry of ROCmConfigHeuristic.mm_configs in
  torch/_inductor/heuristics/template/triton.py chosen by batch size instead of autotuning, with
  BLOCK_M clamped to the batch as _scale_mm_configs does; num_stages is Inductor's ROCm default
  (2) and matrix_instr_nonkdim the ROCmGemmConfig default (16);
- ACC_TYPE is float32 and ALLOW_TF32 is False (the PyTorch default), written as
  input_precision="ieee": the dot runs in IEEE float32.
"""
import triton
import triton.language as tl
from triton.language.extra import libdevice

IN_FEATURES = 8192   # K
OUT_FEATURES = 8192  # N
NUM_STAGES = 2
MATRIX_INSTR_NONKDIM = 16

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
def _triton_tem_fused_mm_add_div_gelu(arg_A, arg_B, in_ptr2, out_ptr1, M, inv_divisor,
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
    tmp2 = inv_divisor
    tmp3 = tmp1 * tmp2
    tmp4 = 0.5
    tmp5 = tmp3 * tmp4
    tmp6 = 0.7071067811865476
    tmp7 = tmp3 * tmp6
    tmp8 = libdevice.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = tmp5 * tmp10
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp11, mask)


def run(*, x, weight, bias, divisor, out):
    m = x.shape[0]
    block_m, block_n, block_k, num_warps, group_m = _config(m)
    # mm_grid: one program per output tile.
    grid = (triton.cdiv(m, block_m) * triton.cdiv(OUT_FEATURES, block_n), 1, 1)
    # Inductor indexes in int32 when every operand fits, otherwise in int64.
    index_dtype = tl.int32 if max(x.numel(), weight.numel(), out.numel()) < 2 ** 31 else tl.int64
    # div_prim replaces the division by a constant with a multiplication by its reciprocal.
    _triton_tem_fused_mm_add_div_gelu[grid](
        x, weight, bias, out, m, 1.0 / divisor,
        N=OUT_FEATURES, K=IN_FEATURES, A_ROWS_FULL=m >= block_m, B_COLS_FULL=OUT_FEATURES >= block_n,
        GROUP_M=group_m, EVEN_K=IN_FEATURES % block_k == 0, ACC_TYPE=tl.float32,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, INDEX_DTYPE=index_dtype,
        num_warps=num_warps, num_stages=NUM_STAGES, matrix_instr_nonkdim=MATRIX_INSTR_NONKDIM)
