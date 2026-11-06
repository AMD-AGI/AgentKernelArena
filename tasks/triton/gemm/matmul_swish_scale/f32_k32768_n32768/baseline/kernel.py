# Copyright (c) 2016- Facebook, Inc and the PyTorch contributors. BSD-3-Clause, see LICENSE.
"""swish(x @ weight.T + bias) * scaling_factor in float32, a standalone Triton rewrite of the kernel
PyTorch Inductor generates for this module under torch.compile(mode="max-autotune")
(pytorch/pytorch v2.15.0):

- the kernel body is the mm template torch/_inductor/kernel/templates/triton_mm.py.jinja rendered for
  addmm(bias, x, weight.t()) (A = x [M, K] row-major, B = weight.t(), a [K, N] view with
  stride_bk = 1 and stride_bn = K; N, K and the strides are compile-time constants, M is a runtime
  argument), with Inductor's addmm_epilogue (acc + bias) and the fused pointwise epilogue
  t * sigmoid(t) * scaling_factor in its store suffix, launched on mm_grid
  (torch/_inductor/kernel/mm_common.py);
- the block configuration is one entry of ROCmConfigHeuristic.mm_configs
  (torch/_inductor/heuristics/template/triton.py) per batch-size bucket, with Inductor's ROCm
  default num_stages = 2 and matrix_instr_nonkdim = 16, instead of autotuning;
- ACC_TYPE is float32 and ALLOW_TF32 is False (Inductor sets it only when
  torch.backends.cuda.matmul.fp32_precision is "tf32", which is not the default), so the dot runs in
  IEEE float32 precision; the k loop uses the template's USE_FAST_ACCUM
  form acc = tl.dot(a, b, acc), which fixes the accumulation to one k-sequential float32 chain.
"""
import triton
import triton.language as tl

# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, GROUP_M) of ROCmConfigHeuristic.mm_configs, by batch size.
MM_CONFIGS = (
    (32, (32, 32, 128, 4, 8)),
    (64, (64, 32, 128, 4, 8)),
    (128, (64, 64, 128, 8, 16)),
    (None, (128, 128, 32, 8, 16)),
)
NUM_STAGES = 2
MATRIX_INSTR_NONKDIM = 16


@triton.jit
def _triton_tem_fused_addmm_mul_sigmoid_0(in_ptr0, A, B, out_ptr1, scaling_factor, M,
                                          N: tl.constexpr, K: tl.constexpr,
                                          GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
                                          ACC_TYPE: tl.constexpr, BLOCK_M: tl.constexpr,
                                          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                          INDEX_DTYPE: tl.constexpr):
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
            a = tl.load(A + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_K])))
        else:
            a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
            a = tl.load(A + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_K])), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_m + K * idx_n
        if EVEN_K:
            b = tl.load(B + (tl.broadcast_to(xindex, [BLOCK_K, BLOCK_N])))
        else:
            b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
            b = tl.load(B + (tl.broadcast_to(xindex, [BLOCK_K, BLOCK_N])), mask=b_mask, other=0.0)

        # The template's USE_FAST_ACCUM form: the MFMA chain accumulates straight into acc, one
        # k-sequential float32 accumulation over all 32768 terms.
        acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=ACC_TYPE)

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
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3 * scaling_factor
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp4, mask)


def _mm_config(m):
    for limit, config in MM_CONFIGS:
        if limit is None or m <= limit:
            return config
    raise ValueError(m)


def run(*, x, weight, bias, scaling_factor, out):
    m, k = x.shape
    n = weight.shape[0]
    block_m, block_n, block_k, num_warps, group_m = _mm_config(m)
    # Inductor indexes in int32 when every operand fits, otherwise in int64.
    index_dtype = tl.int32 if max(x.numel(), weight.numel(), out.numel()) < 2 ** 31 else tl.int64
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), 1, 1)
    _triton_tem_fused_addmm_mul_sigmoid_0[grid](
        bias, x, weight, out, scaling_factor, m,
        N=n, K=k, GROUP_M=group_m, EVEN_K=k % block_k == 0, ACC_TYPE=tl.float32,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, INDEX_DTYPE=index_dtype,
        num_warps=num_warps, num_stages=NUM_STAGES, matrix_instr_nonkdim=MATRIX_INSTR_NONKDIM)
