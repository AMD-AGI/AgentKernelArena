"""8-bit GEMM with per-token / per-channel scales, written in Gluon, from ROCm/aiter:
the device kernel `_gemm_a8w8_kernel` of aiter/ops/triton/gluon/gemm_a8w8.py, its launcher `gemm_a8w8` and the
gfx950 tuned configuration aiter/ops/triton/configs/gemm/gluon/gfx950-GEMM-A8W8.json (inlined).

The upstream Python file carries no SPDX or copyright header; the repository LICENSE (MIT, Copyright (c)
Advanced Micro Devices, Inc.) applies and is included as baseline/LICENSE. The pid helpers of
aiter/ops/triton/utils/_triton/pid_preprocessing.py are in pid_preprocessing.py.
"""
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .pid_preprocessing import pid_grid, remap_xcd


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@gluon.jit
def _gemm_a8w8_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    HAS_BIAS: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    EVEN_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    FP8_FORMAT: gl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call gemm_a8w8 function
    below

    Computes the 8 bit matmul C = A x B, applies a conversion scale and optionally adds a bias to
    the result.
    The conversion scale is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B: Matrix B with shape (K, N).
    - C: Matrix C with shape (M, N).
    - A_scale: First scale tensor with shape (M, 1).
    - B_scale: Second scale tensor with shape (1, N).
    - Bias: Bias tensor with shape (1, N).
    """

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

    pid = remap_xcd(pid, GRID_MN, NUM_XCDS)

    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[4, 16],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[16, 4],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )

    # FP8 uses dot_scaled: mfma_scale_f32_16x16x128_f8f6f4 (K=128, K_WIDTH=32)
    # INT8 uses dot: mfma_i32_16x16x64_i8 (K=64, K_WIDTH=16)
    MFMA_K: gl.constexpr = 128 if FP8_FORMAT is not None else 64
    MFMA_K_WIDTH: gl.constexpr = 32 if FP8_FORMAT is not None else 16
    MFMA_INSTR_SHAPE: gl.constexpr = [16, 16, MFMA_K]

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=MFMA_INSTR_SHAPE,
        transposed=True,
        warps_per_cta=[2, NUM_WARPS // 2],
    )

    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=MFMA_K_WIDTH, per_phase=1, max_phase=16, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=MFMA_K_WIDTH, per_phase=1, max_phase=16, order=[0, 1]
    )

    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=MFMA_K_WIDTH
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=MFMA_K_WIDTH
    )

    # Load first blocks of A and B input matrices
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_mk))
    offs_am = (
        pid_m * BLOCK_SIZE_M
        + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk))
    ) % M
    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    if EVEN_K:
        a = gl.amd.cdna4.buffer_load(
            ptr=a_ptr,
            offsets=offs_a,
        )
    else:
        a = gl.amd.cdna4.buffer_load(
            ptr=a_ptr,
            offsets=offs_a,
            mask=(offs_ak[None, :] < K),
        )

    offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, blocked_kn))
    offs_bn = (
        pid_n * BLOCK_SIZE_N
        + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_kn))
    ) % N
    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    if EVEN_K:
        b = gl.amd.cdna4.buffer_load(
            ptr=b_ptr,
            offsets=offs_b,
        )
    else:
        b = gl.amd.cdna4.buffer_load(
            ptr=b_ptr,
            offsets=offs_b,
            mask=(offs_bk[:, None] < K),
        )

    # Load scales
    offs_a_scale = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout)
    )
    offs_b_scale = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)
    )
    a_scale = gl.amd.cdna4.buffer_load(
        ptr=a_scale_ptr,
        offsets=offs_a_scale,
        mask=offs_a_scale < M,
    )
    b_scale = gl.amd.cdna4.buffer_load(
        ptr=b_scale_ptr,
        offsets=offs_b_scale,
        mask=offs_b_scale < N,
    )

    # Create shared memories
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )
    smem_b = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [BLOCK_SIZE_K, BLOCK_SIZE_N], layout=shared_b
    )

    # LDS write first block of A
    smem_a.store(a)

    acc_dtype = gl.float32 if a_ptr.type.element_ty != gl.int8 else gl.int32
    acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=mfma_layout)

    # num_stages:2
    for k in range(gl.cdiv(K, BLOCK_SIZE_K) - 1):

        # advance pointers for block A and B
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

        # load next block of A
        if EVEN_K:
            a = gl.amd.cdna4.buffer_load(
                ptr=a_ptr,
                offsets=offs_a,
            )
        else:
            a = gl.amd.cdna4.buffer_load(
                ptr=a_ptr,
                offsets=offs_a,
                mask=(offs_ak[None, :] < K - (k + 1) * BLOCK_SIZE_K),
            )

        # LDS write current block of B
        smem_b.store(b)

        # read current block of A from LDS
        cur_a = smem_a.load(layout=dot_a_layout)

        # load next block of B
        if EVEN_K:
            b = gl.amd.cdna4.buffer_load(
                ptr=b_ptr,
                offsets=offs_b,
            )
        else:
            b = gl.amd.cdna4.buffer_load(
                ptr=b_ptr,
                offsets=offs_b,
                mask=(offs_bk[:, None] < K - (k + 1) * BLOCK_SIZE_K),
            )

        # read current block of B from LDS
        cur_b = smem_b.load(layout=dot_b_layout)

        if FP8_FORMAT is None:  # in_dtype is int8
            acc = gl.amd.cdna4.mfma(cur_a, cur_b, acc)
        else:
            acc = gl.amd.cdna4.mfma_scaled(
                a=cur_a,
                a_scale=None,
                a_format=FP8_FORMAT,
                b=cur_b,
                b_scale=None,
                b_format=FP8_FORMAT,
                acc=acc,
            )

        # write next block of A to LDS
        smem_a.store(a)

    # ======= Epilogue ========

    # write last block of B to LDS
    smem_b.store(b)

    # read last blocks of A and B from LDS
    cur_a = smem_a.load(layout=dot_a_layout)
    cur_b = smem_b.load(layout=dot_b_layout)

    if FP8_FORMAT is None:  # in_dtype is int8
        acc = gl.amd.cdna4.mfma(cur_a, cur_b, acc)
    else:
        acc = gl.amd.cdna4.mfma_scaled(
            a=cur_a,
            a_scale=None,
            a_format=FP8_FORMAT,
            b=cur_b,
            b_scale=None,
            b_format=FP8_FORMAT,
            acc=acc,
        )

    # apply scales to accumulator
    acc *= a_scale[:, None] * b_scale[None, :]

    # add bias
    if HAS_BIAS:
        bias = gl.amd.cdna4.buffer_load(
            ptr=bias_ptr,
            offsets=offs_b_scale,
            mask=offs_b_scale < N,
        )
        acc = acc.to(bias_ptr.type.element_ty) + bias[None, :]

    c = acc.to(c_ptr.type.element_ty)

    # store block C back to global memory with masks
    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout)
    )
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)
    )
    c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    gl.amd.cdna4.buffer_store(stored_value=c, ptr=c_ptr, offsets=c_offs, mask=c_mask)


# gfx950-GEMM-A8W8.json: the Gluon table has a single entry, used for every shape.
_CONFIGS = {
    "any": {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 4,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    }
}

# Upstream aiter.ops.triton.utils.device_info.get_num_xcds(): hardcoded for gfx942 / gfx950.
_NUM_XCDS = 8


def _get_config(M, N, K):
    return _CONFIGS["any"]


def run(*, x, w, x_scale, w_scale, out):
    """out = (x @ w^T) * (x_scale * w_scale), no bias (upstream gemm_a8w8 with y=out).

    x [M, K] float8_e4m3fn; w [N, K] float8_e4m3fn; x_scale [M, 1] and w_scale [1, N] float32; out [M, N] bf16.
    """
    M, K = x.shape
    N, K = w.shape

    # Transpose w (kernel expects (K, N)); a view.
    w = w.T

    config = _get_config(M, N, K)

    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )
    _gemm_a8w8_kernel[grid](
        x,
        w,
        x_scale,
        w_scale,
        None,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        False,
        NUM_XCDS=_NUM_XCDS,
        NUM_WARPS=config["num_warps"],
        **config,
        FP8_FORMAT="e4m3",
    )
