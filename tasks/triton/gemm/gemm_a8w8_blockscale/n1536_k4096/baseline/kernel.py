# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Preshuffled a8w8 block-scale GEMM (fp8 e4m3fn, 1x128 activation scales, 128x128 weight scales) in Triton.

Provenance: ROCm/aiter, aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py (launcher
`gemm_a8w8_blockscale_preshuffle`), aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py
(device kernel `_gemm_a8w8_blockscale_preshuffle_kernel`), aiter/ops/triton/_triton_kernels/common/splitk_reduce.py
(`_gemm_splitk_reduce_kernel`), and the tuned configuration table of the gfx950 file
aiter/ops/triton/configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=1536-K=4096.json.

The weight is the (16, 16) preshuffled payload viewed as [N // 16, K * 16]; the activation scales are stored
transposed ([K // 128, M]). The kernel, the split-K reduce kernel, the XCD/pid remapping helpers and the
constexpr-aware kernel naming helper are copied from upstream; the on-disk config lookup is replaced by the
M_LEQ table below (first row whose bound is >= M, else the last row), and the launcher writes into the
preallocated output. The module depends only on `triton` + `torch`.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Inlined helper utils (kernel repr, pid remapping)
# ---------------------------------------------------------------------------
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

    # for lists, tuples, sets - recursively join each
    if isinstance(value, (list, tuple, set)):
        items = sorted(value, key=str) if isinstance(value, set) else value
        sanitized_items = [_sanitize_constexpr_value(item) for item in items]
        joined = "_".join(sanitized_items)
        return joined if joined else "NONE"

    if isinstance(value, str):
        cleaned_value = "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")
        return cleaned_value.upper() if cleaned_value else "NONE"

    cleaned_value = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return cleaned_value.upper() if cleaned_value else "NONE"


def make_kernel_repr(base_name, config_keys, name_key=None):
    # When name_key is set, the base name is taken from the matching constexpr
    # kwarg at call time (falling back to base_name if missing/empty). Lets a
    # single shared kernel produce caller-specific names in compiled artifacts.
    def _repr(specialization):
        constants = specialization.constants

        name = base_name
        if name_key is not None:
            override = constants.get(name_key, None)
            if override:
                cleaned = "".join(
                    ch if ch.isalnum() or ch == "_" else "_" for ch in str(override)
                )
                if cleaned:
                    name = cleaned

        name_parts = []
        for key in config_keys:
            value = constants.get(key, None)
            symbol = _sanitize_constexpr_value(value)
            name_parts.append(f"{key}_{symbol}")

        if not name_parts:
            return name

        suffix = "_".join(name_parts)
        return f"{name}_{suffix}"

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


_gemm_splitk_reduce_repr = make_kernel_repr(
    "_gemm_splitk_reduce_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "ACTUAL_KSPLIT",
        "MAX_KSPLIT",
        "ADD_BIAS",
        "activation",
        "use_activation",
    ],
    name_key="KERNEL_NAME",
)


@triton.jit(repr=_gemm_splitk_reduce_repr)
def _gemm_splitk_reduce_kernel(
    c_in_ptr,
    c_out_ptr,
    bias_ptr,
    M,
    N,
    stride_c_in_k,
    stride_c_in_m,
    stride_c_in_n,
    stride_c_out_m,
    stride_c_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    activation: tl.constexpr,
    use_activation: tl.constexpr,
    KERNEL_NAME: tl.constexpr = "_gemm_splitk_reduce_kernel",
):
    tl.assume(stride_c_in_k > 0)
    tl.assume(stride_c_in_m > 0)
    tl.assume(stride_c_in_n > 0)
    tl.assume(stride_c_out_m > 0)
    tl.assume(stride_c_out_n > 0)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Tell the AMD backend pid * stride stays non-negative so it can lower
    # the loads/stores to buffer ops instead of generic global ops.
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, MAX_KSPLIT)
    c_in_ptrs = (
        c_in_ptr
        + (offs_k[:, None, None] * stride_c_in_k)
        + (offs_m[None, :, None] * stride_c_in_m)
        + (offs_n[None, None, :] * stride_c_in_n)
    )

    if ACTUAL_KSPLIT == MAX_KSPLIT:
        c = tl.load(c_in_ptrs)
    else:
        c = tl.load(c_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT, other=0.0)
    c = tl.sum(c, axis=0)

    if ADD_BIAS:
        acc_dtype = tl.float32 if c_in_ptr.type.element_ty != tl.int8 else tl.int32
        bias = tl.load(bias_ptr + offs_n).to(dtype=acc_dtype)
        bias = tl.broadcast_to(bias[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
        c += bias

    if use_activation:
        c = activation(c)

    c = c.to(c_out_ptr.type.element_ty)

    c_out_ptrs = (
        c_out_ptr
        + (offs_m[:, None] * stride_c_out_m)
        + (offs_n[None, :] * stride_c_out_n)
    )

    tl.store(c_out_ptrs, c)


# ---------------------------------------------------------------------------
# Tuned configuration (M_LEQ_<bound> rows, then "any"), copied from the upstream gfx950 JSON table
# ---------------------------------------------------------------------------
_CONFIGS = [
    (4, {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 1, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None, "NUM_KSPLIT": 8}),
    (8, {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 1, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": '.cg', "NUM_KSPLIT": 8}),
    (16, {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 1, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": '.cg', "NUM_KSPLIT": 8}),
    (32, {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 6, "matrix_instr_nonkdim": 16, "cache_modifier": '.cg', "NUM_KSPLIT": 8}),
    (64, {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 1, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": '.cg', "NUM_KSPLIT": 8}),
    (128, {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": '.cg', "NUM_KSPLIT": 4}),
    (256, {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None, "NUM_KSPLIT": 1}),
    (None, {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 1, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": '.cg', "NUM_KSPLIT": 1}),
]


def _get_config(M: int) -> dict:
    for bound, config in _CONFIGS:
        if bound is None or M <= bound:
            return dict(config)
    raise KeyError(M)


def run(*, a: torch.Tensor, a_scale: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor, out: torch.Tensor) -> None:
    """out = (a * a_scale) @ (b * b_scale)^T.

    a [M, K] fp8, a_scale [K // 128, M] fp32 (column-major scale payload), b [N // 16, K * 16] fp8 (preshuffled
    16x16), b_scale [N // 128, K // 128] fp32, out [M, N] bf16.
    """
    M, K = a.shape
    N = b.shape[0] * 16
    assert K == b.shape[1] // 16, "Incompatible dimensions!!!"

    w_scale = b_scale.T  # (scale_k, scale_n), a view

    config = _get_config(M)
    # Triton < 3.7 fails TritonAMDGPUConvertToBufferOps for gfx950 preshuffle configs with three pipeline
    # stages (upstream clamps them to two; the task requires gfx950).
    if config.get("num_stages", 1) > 2:
        config["num_stages"] = 2

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])
    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty((config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=a.device)
    else:
        y_pp = None

    # If block size is greater than split k size, shrink the block size
    if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 4
    config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 16)

    # Scale block sizes
    config["GROUP_K"] = triton.next_power_of_2(triton.cdiv(K, w_scale.shape[0]))
    config["GROUP_N"] = triton.next_power_of_2(triton.cdiv(N, w_scale.shape[1]))
    assert config["GROUP_K"] == config["BLOCK_SIZE_K"], "GROUP_K must equal BLOCK_SIZE_K"

    grid = lambda META: (  # noqa: E731
        META["NUM_KSPLIT"]
        * triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _gemm_a8w8_blockscale_preshuffle_kernel[grid](
        a,
        b,
        out if config["NUM_KSPLIT"] == 1 else y_pp,
        a_scale,
        w_scale,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        out.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        out.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        # a_scale is stored transposed: element (m, kg) lives at kg * M + m.
        a_scale.stride(1),
        a_scale.stride(0),
        w_scale.stride(0),
        w_scale.stride(1),
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_splitk_reduce_kernel[grid_reduce](
            y_pp,
            out,
            None,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            out.stride(0),
            out.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=False,
            activation="",
            use_activation=False,
            KERNEL_NAME="_gemm_a8w8_blockscale_reduce_kernel",
        )
