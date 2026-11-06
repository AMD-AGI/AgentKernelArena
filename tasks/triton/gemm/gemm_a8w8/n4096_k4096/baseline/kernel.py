# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""8-bit (fp8) scaled GEMM with per-token / per-channel scales from ROCm/aiter:
the device kernel aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8.py and its launcher
aiter/ops/triton/gemm/basic/gemm_a8w8.py, with the pid helpers of utils/_triton/pid_preprocessing.py and
the kernel-name helper of utils/_triton/kernel_repr.py inlined.

The tuned configuration table is the generic gfx950 file configs/gemm/gfx950-GEMM-A8W8.json (no N=4096, K=4096 specialization exists for gfx950), inlined with the upstream
M_LEQ / M_GEQ / any lookup order. Every entry uses NUM_KSPLIT == 1, so the split-K reduction kernel is not
included.
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


@triton.jit
def pid_grid_3d(pid: int, num_pid_m: int, num_pid_n: int, num_pid_k):
    """
    Maps 1D pid to 3D grid coords (pid_m, pid_n, pid_k).
    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - num_pid_k: grid k size

    Returns:
        - pid_m, pid_n, pid_k: 3D grid coordinates
    """
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n
    pid_k = pid // (num_pid_m * num_pid_n) % num_pid_k

    return pid_m, pid_n, pid_k


@triton.jit
def remap_workgroup_spatial(
    wid,
    NUM_Q_HEADS,
    NUM_BLOCKS,
    BATCH,
    NUM_QUERIES_PER_KV: tl.constexpr,
    NUM_XCDS: tl.constexpr = 8,
):
    """
    XCD-aware workgroup mapping for multi-head attention on AMD CDNA3/3.5 GPUs.

    Hardware assigns workgroup `wid` to XCD `wid % NUM_XCDS` (round-robin).
    This function remaps (wid) -> (off_q_head, start_m, off_z) so that workgroups
    sharing KV data land on the same XCD, keeping that data hot in its 4 MB L2 slice.

    MHA path  (NUM_QUERIES_PER_KV == 1):
        Groups ceil(NUM_Q_HEADS / NUM_XCDS) consecutive Q heads onto each XCD so each
        head's full KV tensor is processed by one XCD only.  Within each XCD the
        ordering is head-first: all sequence blocks for head i complete before head i+1.

    GQA path  (NUM_QUERIES_PER_KV > 1):
        Assigns each KV head exclusively to one XCD so all Q heads that share a KV head
        run on the same XCD.  Within each XCD the ordering is block-first: all Q heads
        in the group process sequence block b together before advancing to block b+1.
        Block-first avoids the causal load-imbalance that head-first ordering creates.

        Three regimes depending on NUM_KV_HEADS vs NUM_XCDS:
          HK == NUM_XCDS : one KV head per XCD (aligned).
          HK >  NUM_XCDS : ceil(HK / NXCD) KV heads per XCD, each processed fully
                           before the next, so only one KV head's data is live at a time.
          HK <  NUM_XCDS : NXCD / HK XCDs share each KV head, splitting the per-group
                           work evenly across the cluster of co-owners.

    Args:
        wid               : flat workgroup ID (tl.program_id(0))
        NUM_Q_HEADS       : total number of Q heads
        NUM_BLOCKS        : number of sequence blocks along the Q dimension
        BATCH             : batch size
        NUM_QUERIES_PER_KV: Q heads per KV head (1 for MHA, >1 for GQA)
        NUM_XCDS          : number of XCDs on the device (8 for MI3xx)

    Returns:
        off_q_head : Q head index for this workgroup
        start_m    : sequence block index for this workgroup
        off_z      : batch index for this workgroup
    """
    xcd = wid % NUM_XCDS
    pos = wid // NUM_XCDS

    # ── MHA path (spatial_mha) ────────────────────────────────────────────────
    if NUM_QUERIES_PER_KV == 1:
        wgs_per_head = NUM_BLOCKS * BATCH
        local_head_idx = pos // wgs_per_head
        remainder = pos % wgs_per_head
        off_z = remainder // NUM_BLOCKS
        start_m = remainder % NUM_BLOCKS
        off_q_head = local_head_idx * NUM_XCDS + xcd
        # Bounds check for non-divisible NUM_Q_HEADS / NUM_XCDS.
        if off_q_head >= NUM_Q_HEADS:
            off_q_head = wid % NUM_Q_HEADS
            start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
            off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH
        return off_q_head, start_m, off_z

    # ── GQA path (spatial_gqa) ────────────────────────────────────────────────
    NUM_KV_HEADS: tl.constexpr = NUM_Q_HEADS // NUM_QUERIES_PER_KV

    if NUM_KV_HEADS >= NUM_XCDS:
        NUM_KV_REPLICAS: tl.constexpr = (NUM_KV_HEADS + NUM_XCDS - 1) // NUM_XCDS
        wgs_per_block = NUM_QUERIES_PER_KV * BATCH
        if NUM_KV_REPLICAS == 1:
            # Aligned: one KV head per XCD.
            kv_head = xcd % NUM_KV_HEADS
            start_m = (pos // wgs_per_block) % NUM_BLOCKS
            remainder = pos % wgs_per_block
            off_z = remainder // NUM_QUERIES_PER_KV
            local_q_in_group = remainder % NUM_QUERIES_PER_KV
            off_q_head = kv_head * NUM_QUERIES_PER_KV + local_q_in_group
        else:
            # HK > NXCD: each XCD owns ceil(HK/NXCD) KV heads processed one at a time.
            wgs_per_kv_group = NUM_QUERIES_PER_KV * NUM_BLOCKS * BATCH
            kv_slot = pos // wgs_per_kv_group
            kv_head = xcd + kv_slot * NUM_XCDS
            local = pos % wgs_per_kv_group
            start_m = (local // (NUM_QUERIES_PER_KV * BATCH)) % NUM_BLOCKS
            remainder = local % (NUM_QUERIES_PER_KV * BATCH)
            off_z = remainder // NUM_QUERIES_PER_KV
            local_q_in_group = remainder % NUM_QUERIES_PER_KV
            off_q_head = kv_head * NUM_QUERIES_PER_KV + local_q_in_group
    else:
        # HK < NXCD: NXCD/HK XCDs share each KV head, splitting the per-group work.
        NXCD_PER_KV: tl.constexpr = NUM_XCDS // NUM_KV_HEADS
        xcd_in_cluster = xcd // NUM_KV_HEADS
        kv_head = xcd % NUM_KV_HEADS
        wgs_per_kv_group = NUM_QUERIES_PER_KV * NUM_BLOCKS * BATCH
        wgs_per_xcd_in_cluster = wgs_per_kv_group // NXCD_PER_KV
        local = (pos % wgs_per_xcd_in_cluster) + xcd_in_cluster * wgs_per_xcd_in_cluster
        start_m = (local // (NUM_QUERIES_PER_KV * BATCH)) % NUM_BLOCKS
        remainder = local % (NUM_QUERIES_PER_KV * BATCH)
        off_z = remainder // NUM_QUERIES_PER_KV
        local_q_in_group = remainder % NUM_QUERIES_PER_KV
        off_q_head = kv_head * NUM_QUERIES_PER_KV + local_q_in_group

    # Bounds check for non-divisible edge cases.
    if kv_head >= NUM_KV_HEADS or off_q_head >= NUM_Q_HEADS:
        off_q_head = wid % NUM_Q_HEADS
        start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
        off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH

    return off_q_head, start_m, off_z


_gemm_a8w8_repr = make_kernel_repr(
    "_gemm_a8w8_kernel",
    [
        "HAS_BIAS",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "EVEN_K",
        "GRID_MN",
        "cache_modifier",
        "HAS_BIAS",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_SIZE_K"] == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % args["SPLITK_BLOCK_SIZE"] == 0),
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_gemm_a8w8_repr)
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
    stride_ck,
    stride_cm,
    stride_cn,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
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
    Note: this is Triton jited function and not meant to be called directly. Call gemm_a8w8 instead.

    Computes the 8 bit matmul C = A x B, applies a conversion scale and optionally adds a bias to
    the result.
    The conversion scale is received in the form of two 1D tensors that are multiplied and form a
    2D one before being applied.

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B: Matrix B with shape (K, N).
    - C: Matrix C with shape (M, N) if NUM_KSPLIT==1, (NUM_KSPLIT, M, N) otherwise.
    - A_scale: First scale tensor with shape (M, 1).
    - B_scale: Second scale tensor with shape (1, N).
    - Bias: Bias tensor with shape (1, N).
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    # remap so that XCDs get continous chunks of pids (of CHUNK_SIZE).
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)

    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    if split_k_start < K:

        # Create pointers for first block of A and B input matrices
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = split_k_start + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )

        # Create pointers for the scale tensors and load them
        a_scale = tl.load(a_scale_ptr + offs_am)
        b_scale = tl.load(b_scale_ptr + offs_bn)

        acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
        k_span = split_k_end - split_k_start
        num_k_iter = tl.cdiv(k_span, BLOCK_SIZE_K)

        for k in range(num_k_iter):
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < k_span - k * BLOCK_SIZE_K, other=0.0
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < k_span - k * BLOCK_SIZE_K,
                    other=0.0,
                    cache_modifier=cache_modifier,
                )

            accumulator += tl.dot(a, b)

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # Apply scale
        accumulator *= a_scale[:, None] * b_scale[None, :]

        # Add bias (only when not splitting K; bias is added after reduce)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_bn)
            accumulator = accumulator.to(bias_ptr.type.element_ty) + bias[None, :]

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


_CONFIGS = {
    "M_LEQ_8": dict(BLOCK_SIZE_M=8, BLOCK_SIZE_N=32, BLOCK_SIZE_K=512, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_16": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=512, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_32": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=512, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_64": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=512, GROUP_SIZE_M=4, num_warps=4, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_128": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "M_LEQ_256": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=8, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "M_LEQ_512": dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=8, num_stages=3, waves_per_eu=4, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "any": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=4, num_warps=8, num_stages=2, waves_per_eu=4, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
}

_STANDARD_M_BOUNDS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def _get_config(M, K):
    for bound in _STANDARD_M_BOUNDS:
        key = f"M_LEQ_{bound}"
        if M <= bound and key in _CONFIGS:
            config = dict(_CONFIGS[key])
            break
    else:
        for bound in reversed(_STANDARD_M_BOUNDS):
            key = f"M_GEQ_{bound}"
            if M >= bound and key in _CONFIGS:
                config = dict(_CONFIGS[key])
                break
        else:
            config = dict(_CONFIGS["any"])
    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])
    if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 4
    config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 16)
    return config


def run(*, x, w, x_scale, w_scale, out):
    M, K = x.shape
    N = w.shape[0]
    wt = w.T  # (N, K) -> (K, N), a view
    config = _get_config(M, K)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _gemm_a8w8_kernel[grid](
        x, wt, x_scale, w_scale, None, out,
        M, N, K,
        x.stride(0), x.stride(1),
        wt.stride(0), wt.stride(1),
        0, out.stride(0), out.stride(1),
        False,
        **config,
    )
