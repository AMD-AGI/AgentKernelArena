# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Token-to-expert routing kernels. Vendored from ROCm/aiter (commit 652f2d4b768239aefd515c929e84817681ff598d),
# which adapted them from the triton_kernels package (MIT):
#   aiter/ops/triton/_triton_kernels/moe/moe_routing/bitmatrix.py  (vpopc, _sum_bitmatrix_rows,
#                                                                  _sum_bitmatrix_rows_fused)
#   aiter/ops/triton/_triton_kernels/moe/moe_routing/expt_data.py  (_cdiv_pow2, _expt_data_compute_stage1,
#                                                                  _expt_data_compute_stage2,
#                                                                  _expt_data_compute_stage2_fused)
#   aiter/ops/triton/_triton_kernels/moe/moe_routing/routing.py    (_keyed_add, _routing_compute_indx,
#                                                                  _routing_compute_indx_fused, _combined_routing,
#                                                                  _combined_routing_fused)
# Changes: imports made local. pack_bitmatrix is in pack_bitmatrix.py.

import triton
import triton.language as tl


@triton.jit
def vpopc(x):
    """
    Vertical popcount
    Input  x : uint32[..., N]
    Output y : uint32[..., 32]
    semantics : y[..., i] = sum_j((x[..., j] >> i) & 1)
    credits: @apgoucher
    """

    tl.static_assert(
        x.dtype == tl.uint32, "x should consist of 32-bit unsigned integers"
    )

    BLOCK_N: tl.constexpr = x.shape[-1]  # summation axis
    BATCHES: tl.constexpr = x.numel // BLOCK_N  # number of batches
    if BLOCK_N >= 8:
        sa1: tl.constexpr = 8
    else:
        sa1: tl.constexpr = BLOCK_N
    # create 8-way sums in 4-bit fields:
    y = tl.reshape(x, [BATCHES, BLOCK_N // sa1, sa1, 1])
    y = (y >> tl.arange(0, 4)[None, None, None, :]) & 0x11111111
    y = tl.sum(y, 2)  # [BATCHES, BLOCK_N // sa1, 4]
    if BLOCK_N >= 128:
        sa2: tl.constexpr = 16
    else:
        sa2: tl.constexpr = BLOCK_N // sa1
    # create 128-way sums in 8-bit fields:
    y = tl.reshape(y, [BATCHES, BLOCK_N // (sa1 * sa2), sa2, 1, 4])
    y = (y >> (4 * tl.arange(0, 2))[None, None, None, :, None]) & 0x0F0F0F0F
    y = tl.sum(y, 2)  # [BATCHES, BLOCK_N // (sa1 * sa2), 2, 4]
    sa3: tl.constexpr = BLOCK_N // (sa1 * sa2)
    # create N-way sums in 32-bit fields:
    y = tl.reshape(y, [BATCHES, 1, sa3, 8])
    y = (y >> (8 * tl.arange(0, 4))[None, :, None, None]) & 0x000000FF
    y = tl.sum(y, 2)  # [BATCHES, 4, 8]
    y = tl.reshape(y, x.shape[:-1] + [32])
    return y


@triton.jit
def _sum_bitmatrix_rows(
    B,
    shape_bm,
    stride_bm,
    stride_bn,  # input bitmatrix
    Ret,
    Partials,
    stride_pm,
    stride_pn,
    shape_pn,
    num_pids_m,  # outputs
    BLOCK_MM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):

    tl.static_assert(BLOCK_MM % BLOCK_M == 0)
    TILE_SIZE: tl.constexpr = BLOCK_MM // BLOCK_M
    if isinstance(shape_bm, tl.tensor) and shape_bm.dtype.is_ptr():
        shape_bm = tl.load(shape_bm)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_MM + tl.arange(0, BLOCK_MM)
    offs_n = pid_n * 32 + tl.arange(0, 32)
    n_rows = shape_bm
    bits = tl.load(
        B + pid_n * stride_bn + offs_m * stride_bm, mask=offs_m < n_rows, other=0
    )
    bits = tl.reshape(bits, [TILE_SIZE, BLOCK_M])
    ret = vpopc(bits)  # [TILE_SIZE, 32]

    offs_t = pid_m * TILE_SIZE + tl.arange(0, TILE_SIZE)

    tl.atomic_add(Ret + offs_n, tl.sum(ret, 0), sem="relaxed")

    curr = tl.cumsum(ret, 0) - ret
    tl.atomic_add(
        Partials + offs_t[:, None] * stride_pm + offs_n[None, :] * stride_pn,
        curr,
        sem="relaxed",
    )
    curr = tl.sum(ret, 0, keep_dims=True)
    for i in range(pid_m + 1, num_pids_m):
        offs_t = i * TILE_SIZE + tl.arange(0, TILE_SIZE)
        tl.atomic_add(
            Partials + offs_t[:, None] * stride_pm + offs_n[None, :] * stride_pn,
            curr,
            sem="relaxed",
        )


@triton.jit
def _sum_bitmatrix_rows_fused(
    B,
    shape_bm,
    stride_bm,
    stride_bn,
    Ret,
    N_BLKS_BITMATRIX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    if isinstance(shape_bm, tl.tensor) and shape_bm.dtype.is_ptr():
        shape_bm = tl.load(shape_bm)
    for i in tl.static_range(N_BLKS_BITMATRIX):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = i * 32 + tl.arange(0, 32)
        n_rows = shape_bm
        if EVEN_M:
            bits = tl.load(B + i * stride_bn + offs_m * stride_bm)
        else:
            bits = tl.load(
                B + i * stride_bn + offs_m * stride_bm, mask=offs_m < n_rows, other=0
            )
        bits = tl.reshape(bits, [1, BLOCK_M])
        ret = vpopc(bits)  # [1, 32]
        ret = tl.reshape(ret, [32])

        tl.store(Ret + offs_n, ret)


@triton.jit
def _cdiv_pow2(n, log2_k):
    return (n + ((1 << log2_k) - 1)) >> log2_k


@triton.jit
def _expt_data_compute_stage1(
    pid,
    Hist,
    n_expts_tot: tl.constexpr,
    TokenStart,
    TileStart,
    MDTileInfo,
    max_num_tiles,
    n_gates,
    tile_dim_log2: tl.constexpr,
    BLOCK: tl.constexpr,
    EQUAL_BLOCK: tl.constexpr,
):
    if EQUAL_BLOCK:
        offs_n = tl.arange(0, BLOCK)
        hist_token = tl.load(Hist + offs_n)
        hist_tile = _cdiv_pow2(hist_token, tile_dim_log2)
        token_starts = tl.cumsum(hist_token, 0) - hist_token
        tile_starts = tl.cumsum(hist_tile, 0) - hist_tile
        tl.store(TokenStart + offs_n, token_starts)
        tl.store(TileStart + offs_n, tile_starts)
    else:
        token_acc = tl.zeros([BLOCK], dtype=TokenStart.dtype.element_ty)
        tile_acc = tl.zeros([BLOCK], dtype=TileStart.dtype.element_ty)
        offs_n = tl.arange(0, BLOCK)
        for i in range(0, n_expts_tot, BLOCK):
            mask_n = offs_n < n_expts_tot
            hist_token = tl.load(Hist + offs_n, mask=mask_n, other=0)
            hist_tile = _cdiv_pow2(hist_token, tile_dim_log2)
            token_starts = tl.cumsum(hist_token, 0) - hist_token + token_acc
            tile_starts = tl.cumsum(hist_tile, 0) - hist_tile + tile_acc
            token_acc += tl.sum(hist_token, 0)
            tile_acc += tl.sum(hist_tile, 0)
            tl.store(TokenStart + offs_n, token_starts)
            tl.store(TileStart + offs_n, tile_starts)
            offs_n += BLOCK

    if pid == 0:
        tl.store(TokenStart + n_expts_tot, n_gates)

        hist_tok_last = tl.load(Hist + n_expts_tot - 1)
        hist_tile_last = _cdiv_pow2(hist_tok_last, tile_dim_log2)
        tile_off_last = tl.load(TileStart + n_expts_tot - 1) + hist_tile_last
        tl.store(TileStart + n_expts_tot, tile_off_last)

        MEMSET_BLOCK: tl.constexpr = 16
        for block_off in range(tile_off_last, max_num_tiles, MEMSET_BLOCK):
            block_offs = block_off + tl.arange(0, MEMSET_BLOCK)
            tl.store(
                MDTileInfo + block_offs, 0xFFFFFFFF, mask=block_offs < max_num_tiles
            )


@triton.jit
def _expt_data_compute_stage2(
    pid, Hist, TileStart, TileInfo, tile_dim_log2: tl.constexpr
):

    expt_id = pid

    n_tokens = tl.load(Hist + expt_id)
    if n_tokens == 0:
        return
    BLOCK: tl.constexpr = 8
    n_blocks = _cdiv_pow2(n_tokens, tile_dim_log2)
    TileInfo += tl.load(TileStart + expt_id)

    n_blocks = _cdiv_pow2(n_tokens, tile_dim_log2)
    block_offs = tl.arange(0, BLOCK)
    for i in range(0, n_blocks, BLOCK):
        data = (block_offs << 16) + expt_id
        tl.store(TileInfo + block_offs, data, mask=block_offs < n_blocks)
        block_offs += BLOCK


@triton.jit
def _expt_data_compute_stage2_fused(expt_id, Hist, TileStart, TileInfo):
    n_tokens = tl.load(Hist + expt_id)
    if n_tokens == 0:
        return
    TileInfo += tl.load(TileStart + expt_id)
    tl.store(TileInfo, expt_id)


@triton.jit
def _keyed_add(x, y):

    # we keep the key in the upper 16 bits of a uint32:
    key_mask: tl.constexpr = 0xFFFF0000

    kx = x & key_mask
    ky = y & key_mask
    z = tl.where(kx == ky, x + y - kx, y)
    return z


@triton.jit
def _routing_compute_indx(
    pid_m,
    GatherIndx,
    ScatterIndx,
    GateScal,
    ExptScal,
    ExptIndx,
    PartialOffs,
    stride_pm,
    stride_pn,
    TokensStart,
    n_gates,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    N_EXPTS_ACT_PAD: tl.constexpr,
    USE_TDM: tl.constexpr,
):

    tl.static_assert(N_EXPTS_ACT_PAD * BLOCK_M <= 32768)

    LOAD_SIZE: tl.constexpr = N_EXPTS_ACT_PAD * BLOCK_M
    local_offs = tl.arange(0, LOAD_SIZE)
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + local_offs
    # TDM tensor descriptors require >=16 bytes in the last dim. The expert-index
    # load is int16 (2 bytes), so LOAD_SIZE must be >=8 elements; for tiny routing
    # tiles (e.g. decode bs=1, where BLOCK_M=1 -> LOAD_SIZE=N_EXPTS_ACT_PAD) fall
    # back to the functionally-identical plain-load branch below.
    if USE_TDM and EVEN_M and N_EXPTS_ACT == N_EXPTS_ACT_PAD and LOAD_SIZE >= 8:
        expt_desc = tl.make_tensor_descriptor(
            base=ExptIndx + pid_m * BLOCK_M * N_EXPTS_ACT,
            shape=(1, LOAD_SIZE),
            strides=(LOAD_SIZE, 1),
            block_shape=(1, LOAD_SIZE),
        )
        expert = tl.reshape(expt_desc.load([0, 0]), (LOAD_SIZE,))
        expert = tl.where(offs < n_gates, expert, -1).to(tl.uint32)
    elif EVEN_M and N_EXPTS_ACT == N_EXPTS_ACT_PAD:
        expert = tl.load(ExptIndx + offs).to(tl.uint32)
    else:
        expert = tl.load(ExptIndx + offs, mask=(offs < n_gates), other=-1).to(tl.uint32)

    # stable-sort by expert ID:
    kv_pairs = ((expert << 16) | local_offs).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    expert = kv_pairs >> 16
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + (kv_pairs & 0xFFFF)

    if EVEN_M and N_EXPTS_ACT == N_EXPTS_ACT_PAD:
        gate_scal = tl.load(ExptScal + offs)

        # compute run lengths in expert-sorted order:
        x = kv_pairs & 0xFFFF0000 | 0x00000001
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
        exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xFFFF

        gates = tl.load(PartialOffs + pid_m * stride_pm + expert * stride_pn)
        gates += tl.load(TokensStart + expert)
        gates += exclusive_run_lengths

        tl.store(ScatterIndx + offs, gates)
        tl.store(GatherIndx + gates, offs)
        tl.store(GateScal + gates, gate_scal)
    else:
        mask = expert != 0xFFFF
        gate_scal = tl.load(ExptScal + offs, mask=mask)

        # compute run lengths in expert-sorted order:
        x = kv_pairs & 0xFFFF0000 | 0x00000001
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
        exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xFFFF

        gates = tl.load(PartialOffs + pid_m * stride_pm + expert * stride_pn, mask=mask)
        gates += tl.load(TokensStart + expert, mask=mask)
        gates += exclusive_run_lengths

        tl.store(ScatterIndx + offs, gates, mask=mask)
        tl.store(GatherIndx + gates, offs, mask=mask)
        tl.store(GateScal + gates, gate_scal, mask=mask)


@triton.jit
def _routing_compute_indx_fused(
    GatherIndx,
    ScatterIndx,
    GateScal,
    ExptScal,
    ExptIndx,
    TokensStart,
    n_gates,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    N_EXPTS_ACT_PAD: tl.constexpr,
    USE_TDM: tl.constexpr,
):

    tl.static_assert(N_EXPTS_ACT_PAD * BLOCK_M <= 32768)

    LOAD_SIZE: tl.constexpr = N_EXPTS_ACT_PAD * BLOCK_M
    local_offs = tl.arange(0, LOAD_SIZE)
    offs = local_offs
    # TDM tensor descriptors require >=16 bytes in the last dim. The expert-index
    # load is int16 (2 bytes), so LOAD_SIZE must be >=8 elements; for tiny routing
    # tiles (e.g. decode bs=1, where BLOCK_M=1 -> LOAD_SIZE=N_EXPTS_ACT_PAD) fall
    # back to the functionally-identical plain-load branch below.
    if USE_TDM and EVEN_M and N_EXPTS_ACT == N_EXPTS_ACT_PAD and LOAD_SIZE >= 8:
        expt_desc = tl.make_tensor_descriptor(
            base=ExptIndx,
            shape=(1, LOAD_SIZE),
            strides=(LOAD_SIZE, 1),
            block_shape=(1, LOAD_SIZE),
        )
        expert = tl.reshape(expt_desc.load([0, 0]), (LOAD_SIZE,))
        expert = tl.where(offs < n_gates, expert, -1).to(tl.uint32)
    elif EVEN_M and N_EXPTS_ACT == N_EXPTS_ACT_PAD:
        expert = tl.load(ExptIndx + offs).to(tl.uint32)
    else:
        expert = tl.load(ExptIndx + offs, mask=(offs < n_gates), other=-1).to(tl.uint32)

    # stable-sort by expert ID:
    kv_pairs = ((expert << 16) | local_offs).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    expert = kv_pairs >> 16
    offs = kv_pairs & 0xFFFF

    if EVEN_M and N_EXPTS_ACT == N_EXPTS_ACT_PAD:
        gate_scal = tl.load(ExptScal + offs)

        # compute run lengths in expert-sorted order:
        x = kv_pairs & 0xFFFF0000 | 0x00000001
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
        exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xFFFF

        gates = tl.load(TokensStart + expert)
        gates += exclusive_run_lengths

        tl.store(ScatterIndx + offs, gates)
        tl.store(GatherIndx + gates, offs)
        tl.store(GateScal + gates, gate_scal)
    else:
        mask = expert != 0xFFFF
        gate_scal = tl.load(ExptScal + offs, mask=mask)

        # compute run lengths in expert-sorted order:
        x = kv_pairs & 0xFFFF0000 | 0x00000001
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
        exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xFFFF

        gates = tl.load(TokensStart + expert, mask=mask)
        gates += exclusive_run_lengths

        tl.store(ScatterIndx + offs, gates, mask=mask)
        tl.store(GatherIndx + gates, offs, mask=mask)
        tl.store(GateScal + gates, gate_scal, mask=mask)


@triton.jit
def _combined_routing(
    GatherIndx,
    ScatterIndx,
    GateScal,
    ExptScal,
    ExptIndx,
    PartialOffs,
    stride_pm,
    stride_pn,
    n_gates,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    N_EXPTS_ACT_PAD: tl.constexpr,
    ExpertHist,
    n_expts_tot: tl.constexpr,
    TokenStart,
    TileStart,
    blocks1a,
    MDTileInfo,
    max_num_tiles,
    tile_dim_log2: tl.constexpr,
    BLOCK_A: tl.constexpr,
    EQUAL_A: tl.constexpr,
    USE_TDM: tl.constexpr,
):

    pid = tl.program_id(0)

    _expt_data_compute_stage1(
        pid,
        ExpertHist,
        n_expts_tot,
        TokenStart,
        TileStart,
        MDTileInfo,
        max_num_tiles,
        n_gates,
        tile_dim_log2,
        BLOCK_A,
        EQUAL_A,
    )

    if pid < blocks1a:
        _expt_data_compute_stage2(pid, ExpertHist, TileStart, MDTileInfo, tile_dim_log2)
    else:
        pid -= blocks1a
        _routing_compute_indx(
            pid,
            GatherIndx,
            ScatterIndx,
            GateScal,
            ExptScal,
            ExptIndx,
            PartialOffs,
            stride_pm,
            stride_pn,
            TokenStart,
            n_gates,
            BLOCK_M,
            EVEN_M,
            N_EXPTS_ACT,
            N_EXPTS_ACT_PAD,
            USE_TDM,
        )


@triton.jit
def _combined_routing_fused(
    GatherIndx,
    ScatterIndx,
    GateScal,
    ExptScal,
    ExptIndx,
    Bitmatrix,
    shape_bm,
    stride_bm,
    stride_bn,
    N_BLKS_BITMATRIX: tl.constexpr,
    n_gates,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    N_EXPTS_ACT_PAD: tl.constexpr,
    N_EXPTS_TOT: tl.constexpr,
    ExpertHist,
    TokenStart,
    TileStart,
    blocks1a,
    MDTileInfo,
    max_num_tiles,
    tile_dim_log2: tl.constexpr,
    BLOCK_A: tl.constexpr,
    EQUAL_A: tl.constexpr,
    USE_TDM: tl.constexpr,
):

    pid = tl.program_id(0)

    _sum_bitmatrix_rows_fused(
        Bitmatrix,
        shape_bm,
        stride_bm,
        stride_bn,
        ExpertHist,
        N_BLKS_BITMATRIX,
        BLOCK_M,
        EVEN_M,
    )

    tl.debug_barrier()

    if pid != 0 and pid < blocks1a:
        n_tokens = tl.load(ExpertHist + pid)
        if n_tokens == 0:
            return

    _expt_data_compute_stage1(
        pid,
        ExpertHist,
        N_EXPTS_TOT,
        TokenStart,
        TileStart,
        MDTileInfo,
        max_num_tiles,
        n_gates,
        tile_dim_log2,
        BLOCK_A,
        EQUAL_A,
    )

    if pid < blocks1a:
        _expt_data_compute_stage2_fused(pid, ExpertHist, TileStart, MDTileInfo)
    else:
        _routing_compute_indx_fused(
            GatherIndx,
            ScatterIndx,
            GateScal,
            ExptScal,
            ExptIndx,
            TokenStart,
            n_gates,
            BLOCK_M,
            EVEN_M,
            N_EXPTS_ACT,
            N_EXPTS_ACT_PAD,
            USE_TDM,
        )
