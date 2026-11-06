# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""a8w8 block-scaled GEMM Triton kernel with a preshuffled weight (fp8 e4m3fn activations with 1x128 scales,
fp8 weights in the 16x16 shuffled layout with 128x128 scales, bfloat16 output).

Provenance: ROCm/aiter, aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py (launcher
`gemm_a8w8_blockscale_preshuffle`) and aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py
(device kernel `_gemm_a8w8_blockscale_preshuffle_kernel`, unchanged). The plain (non-preshuffled) kernel, the gluon
backend, the split-K reduce kernel and the on-disk config lookup are dropped: the tuned-config table is the N=7168, K=2048 specialization configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=7168-K=2048.json,
inlined with the upstream M_LEQ / any lookup order (every row has NUM_KSPLIT == 1, so no split-K
workspace or reduce kernel is needed). The pid helpers (`remap_xcd`, `pid_grid`) and the constexpr-aware kernel
naming helper are inlined, so the module depends only on `triton` + `torch`.

Op:
    Y = X @ W^T with X [M, K] fp8 and per-(row, 128-column) scales, W [N, K] fp8 stored preshuffled with
    per-(128 x 128 block) scales; accumulation in float32, output rounded to bfloat16. The activation scale
    is passed as the column-major payload of the upstream activation-quantization kernel: a buffer of shape
    [K/128, M] read as x_scale_logical[m, j] = payload_flat[j * M + m].
"""
import triton
import triton.language as tl


def _sanitize_constexpr_value(value):
    if value is None:
        return "NONE"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    cleaned_value = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return cleaned_value.upper() if cleaned_value else "NONE"


def make_kernel_repr(base_name, config_keys):
    def _repr(specialization):
        constants = specialization.constants
        name_parts = []
        for key in config_keys:
            symbol = _sanitize_constexpr_value(constants.get(key, None))
            name_parts.append(f"{key}_{symbol}")
        if not name_parts:
            return base_name
        return f"{base_name}_{'_'.join(name_parts)}"

    return _repr


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    if tall_xcds == 0:
        tall_xcds = tl.cast(NUM_XCDS, tall_xcds.type)
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    return pid


@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
    """
    Maps 1D pid to 2D grid coords (pid_m, pid_n).

    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - GROUP_SIZE_M: tl.constexpr: default is 1
    """
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


_gemm_a8w8_blockscale_preshuffle_repr = make_kernel_repr(
    "_gemm_a8w8_blockscale_preshuffle_kernel",
    [
        "GROUP_K",
        "GROUP_N",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "EVEN_K",
        "GRID_MN",
        "cache_modifier",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_gemm_a8w8_blockscale_preshuffle_repr)
def _gemm_a8w8_blockscale_preshuffle_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
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
    stride_bn,
    stride_bk,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    # Meta-parameters
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call gemm_a8w8_blockscale function
    below

    Computes the 8 bit matmul C = A x B using the block-scale quantization approach.

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B: Matrix B with shape (K, N).
    - C: Matrix C with shape (M, N).
    - A_scale: Scale tensor for A with shape (M, *scale_k).
    - B_scale: Scale tensor for B with shape (*scale_k, **scale_n).

    *scale_k = (K + GROUP_K - 1) // GROUP_K
    **scale_n = (N + GROUP_N - 1) // GROUP_N

    For this kernel implementation, GROUP_K must equal BLOCK_K.
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_ck > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ascale_m > 0)
    tl.assume(stride_ascale_k > 0)
    tl.assume(stride_bscale_k > 0)
    tl.assume(stride_bscale_n > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        remap_xcd(pid, GRID_MN)

        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE) < K:

        # SPLITK_BLOCK_SIZE = tl.cdiv(K, NUM_KSPLIT)
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)
        # ^ Number of K blocks within our split-K partition

        # Create pointers for first block of A and B input matrices
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_shuffle_arr = tl.arange(0, BLOCK_SIZE_K * 16)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE + offs_k
        offs_k_shuffle = pid_k * SPLITK_BLOCK_SIZE * 16 + offs_k_shuffle_arr

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * (BLOCK_SIZE_N // 16) + tl.arange(0, BLOCK_SIZE_N // 16)) % (
            N // 16
        )
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_bn[:, None] * stride_bn + offs_k_shuffle[None, :] * stride_bk
        )

        # Create pointers for the scales
        offs_bsn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k_scale = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
        a_scale_ptrs = (
            a_scale_ptr + offs_am * stride_ascale_m + offs_k_scale * stride_ascale_k
        )
        # When the scale block covers the whole tile (GROUP_N >= BLOCK_SIZE_N and
        # GROUP_K >= BLOCK_SIZE_K) every element shares one scalar b_scale.
        if GROUP_N >= BLOCK_SIZE_N and GROUP_K >= BLOCK_SIZE_K:
            offs_b_scale_n = (pid_n * BLOCK_SIZE_N) // GROUP_N
        else:
            offs_b_scale_n = offs_bsn // GROUP_N
        b_scale_ptrs = (
            b_scale_ptr
            + offs_k_scale * stride_bscale_k
            + offs_b_scale_n * stride_bscale_n
        )

        acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            # Load the per-K-tile block scales up-front
            a_scale = tl.load(a_scale_ptrs)
            b_scale = tl.load(b_scale_ptrs)

            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
                )
                b = tl.load(
                    b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                )

            b = (
                b.reshape(
                    1,
                    BLOCK_SIZE_N // 16,
                    BLOCK_SIZE_K // 32,
                    2,
                    16,
                    16,
                )
                .permute(0, 1, 4, 2, 3, 5)
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .trans(1, 0)
            )

            if GROUP_N >= BLOCK_SIZE_N and GROUP_K >= BLOCK_SIZE_K:
                accumulator += tl.dot(a, b) * (a_scale * b_scale)[:, None]
            else:
                accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale[None, :])

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * 16 * stride_bk

            offs_ks_step = (
                k + 1
            ) * BLOCK_SIZE_K // GROUP_K - k * BLOCK_SIZE_K // GROUP_K
            a_scale_ptrs += offs_ks_step * stride_ascale_k
            b_scale_ptrs += offs_ks_step * stride_bscale_k

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


_CONFIGS = {
    "M_LEQ_8": dict(BLOCK_SIZE_M=4, BLOCK_SIZE_N=16, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_warps=1, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "M_LEQ_16": dict(BLOCK_SIZE_M=8, BLOCK_SIZE_N=16, BLOCK_SIZE_K=128, GROUP_SIZE_M=4, num_warps=2, num_stages=3, waves_per_eu=6, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "M_LEQ_32": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=4, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_64": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=16, BLOCK_SIZE_K=128, GROUP_SIZE_M=4, num_warps=2, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "M_LEQ_128": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_warps=4, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "M_LEQ_256": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_warps=4, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "M_LEQ_512": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "any": dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=4, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
}

_STANDARD_M_BOUNDS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)

# Triton releases before 3.7 fail the AMD buffer-ops conversion of gfx950 preshuffle kernels that use three
# pipeline stages; upstream keeps the tuned tile and split-K and lowers the stage count to two there.
_TRITON_BEFORE_3_7 = tuple(int(p) for p in triton.__version__.split("+")[0].split(".")[:2]) < (3, 7)


def _get_config(M):
    for bound in _STANDARD_M_BOUNDS:
        key = f"M_LEQ_{bound}"
        if M <= bound and key in _CONFIGS:
            return dict(_CONFIGS[key])
    return dict(_CONFIGS["any"])


def run(*, a, b, a_scale, b_scale, out):
    """out = (a * a_scale) @ (b_logical * b_scale)^T.

    a [M, K] fp8; b [N, K] fp8 in the aiter shuffle_weight(16, 16) layout; a_scale [M, K/128] fp32 holding the
    column-major payload (a buffer of shape [K/128, M]); b_scale [N/128, K/128] fp32; out [M, N] bf16.
    """
    M, K = a.shape
    N = b.shape[0]

    # The preshuffled weight is read as [N/16, 16*K] (same bytes), the scale of the weight transposed to
    # [K/128, N/128] (a view).
    w = b.view(N // 16, K * 16)
    w_scale = b_scale.T

    config = _get_config(M)
    if _TRITON_BEFORE_3_7 and config.get("num_stages", 1) > 2:
        config["num_stages"] = 2

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])
    # If block size is greater than split k size, shrink the block size
    if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 4
    config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 16)  # minimum block size is 16 for perf

    # Scale block sizes
    config["GROUP_K"] = triton.next_power_of_2(triton.cdiv(K, w_scale.shape[0]))  # scale_block_size_k
    config["GROUP_N"] = triton.next_power_of_2(triton.cdiv(N, w_scale.shape[1]))  # scale_block_size_n

    assert config["GROUP_K"] == config["BLOCK_SIZE_K"], "GROUP_K must equal BLOCK_SIZE_K"

    grid = lambda META: (  # noqa: E731
        META["NUM_KSPLIT"]
        * triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _gemm_a8w8_blockscale_preshuffle_kernel[grid](
        a,
        w,
        out,
        a_scale,
        w_scale,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        w.stride(0),
        w.stride(1),
        0,
        out.stride(0),
        out.stride(1),
        1,  # stride of the activation scale along M (column-major payload)
        M,  # stride of the activation scale along K blocks
        w_scale.stride(0),
        w_scale.stride(1),
        **config,
    )
