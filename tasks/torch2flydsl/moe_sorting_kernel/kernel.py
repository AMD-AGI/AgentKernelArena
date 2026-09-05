# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL MoE token-sorting op.

Counting sort in LDS (histogram -> prefix-sum -> scatter) producing the packed
dispatch layout. Packed id ``(slot << 24) | token``; padding sentinel
``(topk << 24) | M``; per-expert runs padded to ``unit_size`` (32).

Supports the oneshot path only (the single-kernel, all-phases-in-LDS path
selected when ``M <= min(sub_tokens, ONESHOT_MAX_T)``). The 2k / 4k HBM
multiphase paths are not implemented; the host launcher raises if a shape would
require them.

Host entry points:
  * ``build_moe_sorting_module(num_experts, topk, max_tokens, unit_size)``
    -> compiled FlyDSL oneshot launcher.
  * ``flydsl_moe_sorting(topk_ids, topk_weights, num_experts, unit_size,
    model_dim)`` -> runs the kernel, returns ``(sorted_token_ids,
    sorted_weights, sorted_expert_ids, num_valid_ids)``.
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import memref as memref_ops
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import buffer_ops, gpu, range_constexpr
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.runtime.device import is_rdna_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


def _get_warp_size(arch=None):
    """Wavefront size: CDNA (gfx9xx) = 64, RDNA = 32. Inlined from kernels_common."""
    if arch is None:
        arch = get_rocm_arch()
    return 32 if is_rdna_arch(arch) else 64


BLOCK_SIZE = 256
UNIT_SIZE = 32  # GEMM tile-M, aka block_size in CK
WARP_SIZE = _get_warp_size()

# DPP constants for prefix sum
DPP_ROW_SHR_1 = 0x111
DPP_ROW_SHR_2 = 0x112
DPP_ROW_SHR_4 = 0x114
DPP_ROW_SHR_8 = 0x118
DPP_ROW_MASK = 0xF
DPP_BANK_MASK = 0xF


def _unwrap_val(v):
    """Unwrap DSL value to raw MLIR ir.Value."""
    return v.ir_value() if hasattr(v, "ir_value") else v


def _dpp_intra_wave_prefix_sum(val, lane, WARP_SIZE):
    """Inclusive prefix sum within a single wave using DPP + ds_bpermute."""
    val_raw = _unwrap_val(val)
    zero_raw = _unwrap_val(fx.Int32(0))

    for shift, dpp_op, threshold in [
        (1, DPP_ROW_SHR_1, 1),
        (2, DPP_ROW_SHR_2, 2),
        (4, DPP_ROW_SHR_4, 4),
        (8, DPP_ROW_SHR_8, 8),
    ]:
        remote = fly_rocdl.update_dpp(T.i32, zero_raw, val_raw, dpp_op, DPP_ROW_MASK, DPP_BANK_MASK, True)
        val = (lane >= fx.Int32(threshold)).select(val + fx.Int32(remote), val)
        val_raw = _unwrap_val(val)

    src_lane_16 = (lane & fx.Int32(0x30)) - fx.Int32(1)
    remote16 = fly_rocdl.ds_bpermute(T.i32, src_lane_16 * fx.Int32(4), val)
    val = (lane >= fx.Int32(16)).select(val + fx.Int32(remote16), val)

    if WARP_SIZE > 32:
        src_lane_32 = (lane & fx.Int32(0x30)) - fx.Int32(17)
        remote32 = fly_rocdl.ds_bpermute(T.i32, src_lane_32 * fx.Int32(4), val)
        val = (lane >= fx.Int32(32)).select(val + fx.Int32(remote32), val)

    return val


@flyc.jit
def _allwave_inclusive_prefix_sum(val, lane, wave, scratch_mr, NUM_WAVES, WARP_SIZE):
    """DPP intra-wave prefix sum + cross-wave LDS accumulation."""
    val = _dpp_intra_wave_prefix_sum(val, lane, WARP_SIZE)
    if lane == fx.Int32(WARP_SIZE - 1):
        _lds_store_raw(scratch_mr, val, wave)
    gpu.barrier()
    cross = fx.Int32(0)
    for _w in range_constexpr(NUM_WAVES - 1):
        wt = _lds_load_raw(scratch_mr, fx.Int32(_w))
        cross = (wave > fx.Int32(_w)).select(cross + wt, cross)
    return val, val + cross


@flyc.jit
def _zero_moe_buf_grid_stride(moe_buf_rsrc, gid_v4, stride_v4, total_v4, oob_idx):
    """Grid-stride loop zeroing moe_buf via vectorized buffer_store."""
    c_one = fx.Int32(1)
    niters = (total_v4 + stride_v4 - c_one) // stride_v4
    c_zero_v4 = fx.Vector.filled(4, 0, fx.Int32)
    c4 = fx.Int32(4)
    for _z in range(fx.Index(0), ArithValue(niters).index_cast(T.index), fx.Index(1)):
        idx = gid_v4 + fx.Int32(_z) * stride_v4
        valid = idx < total_v4
        buffer_ops.buffer_store(c_zero_v4, moe_buf_rsrc, valid.select(idx * c4, oob_idx))


def _extend_prefix_sum_serial(mr, start_block, E, load_fn, store_fn):
    """Thread-0 serial extension of prefix sum for experts >= start_block."""
    prev = load_fn(mr, fx.Int32(start_block))
    for _ext in range_constexpr(start_block, E):
        cur = load_fn(mr, fx.Int32(_ext + 1))
        new_val = prev + cur
        store_fn(mr, new_val, fx.Int32(_ext + 1))
        prev = new_val
    return prev


@flyc.jit
def _write_expert_id_blocks(sorted_e_rsrc, local_eid, blk_start, n_blks):
    """Write local_eid to sorted_expert_ids[blk_start .. blk_start+n_blks)."""
    for _jb in range(fx.Index(0), ArithValue(n_blks).index_cast(T.index), fx.Index(1)):
        blk_idx = blk_start + fx.Int32(_jb)
        buffer_ops.buffer_store(local_eid, sorted_e_rsrc, blk_idx)


@flyc.jit
def _fill_sentinel_slots(sorted_ids_rsrc, sorted_w_rsrc, start, count, sentinel, block_size, tid, oob_idx):
    """Cooperative sentinel fill: threads fill [start, start+count) with sentinels."""
    c_zero = fx.Int32(0)
    end = start + count
    niters = (count + fx.Int32(block_size) - fx.Int32(1)) // fx.Int32(block_size)
    for _p in range(fx.Index(0), ArithValue(niters).index_cast(T.index), fx.Index(1)):
        slot = start + fx.Int32(_p) * fx.Int32(block_size) + tid
        safe = (slot < end).select(slot, oob_idx)
        buffer_ops.buffer_store(sentinel, sorted_ids_rsrc, safe)
        buffer_ops.buffer_store(c_zero, sorted_w_rsrc, safe)


def _lds_load_raw(raw_mr, idx):
    """Load i32 from LDS raw memref. idx can be i32 or index."""
    raw_idx = idx.ir_value() if hasattr(idx, "ir_value") else idx
    if not isinstance(raw_idx.type, ir.IndexType):
        raw_idx = ArithValue(idx).index_cast(T.index)
        raw_idx = raw_idx.ir_value() if hasattr(raw_idx, "ir_value") else raw_idx
    return fx.Int32(memref_ops.load(raw_mr, [raw_idx]))


def _lds_store_raw(raw_mr, val, idx):
    """Store i32 to LDS raw memref. idx can be i32 or index."""
    v = val.ir_value() if hasattr(val, "ir_value") else val
    raw_idx = idx.ir_value() if hasattr(idx, "ir_value") else idx
    if not isinstance(raw_idx.type, ir.IndexType):
        raw_idx = ArithValue(idx).index_cast(T.index)
        raw_idx = raw_idx.ir_value() if hasattr(raw_idx, "ir_value") else raw_idx
    memref_ops.store(v, raw_mr, [raw_idx])


# ---------------------------------------------------------------------------
# FlyDSL GPU kernel — oneshot path (single kernel, all phases in LDS)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=256)
def _compile_moe_sorting_oneshot(
    *,
    num_experts: int,
    topk: int,
    max_tokens: int = 128,
    unit_size: int = UNIT_SIZE,
    has_mask: bool = False,
):
    """Compile the oneshot MoE sorting kernel (single kernel, all phases in LDS)."""
    arch = get_hip_arch()
    E = num_experts
    max_oneshot_block = 512 if WARP_SIZE == 64 else 256
    ONESHOT_BLOCK = 256 if E <= 256 else min(512, max_oneshot_block)
    NUM_WAVES = ONESHOT_BLOCK // WARP_SIZE
    smem_cols = E + 1

    if arch in ("gfx942",) or str(arch).startswith("gfx94"):
        lds_capacity_bytes = 65536
    elif str(arch).startswith("gfx95"):
        lds_capacity_bytes = 163840
    else:
        lds_capacity_bytes = 65536  # conservative default

    lds_capacity_ints = lds_capacity_bytes // 4
    target_occupancy = 2
    r = lds_capacity_ints // target_occupancy // smem_cols
    sub_unroll = 8
    cumsum_bufs = 2
    if r < (cumsum_bufs + sub_unroll):
        raise ValueError(f"LDS too small for E={E}: need at least {(cumsum_bufs + sub_unroll) * smem_cols * 4} bytes")
    r_for_sub = ((r - cumsum_bufs) // sub_unroll) * sub_unroll
    r_token_min = ((max_tokens + sub_unroll - 1) // sub_unroll) * sub_unroll
    r_for_sub = min(r_for_sub, r_token_min)
    sub_tokens = r_for_sub

    allocator = SmemAllocator(None, arch=arch)

    cumsum_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = cumsum_offset + smem_cols * 4

    cumdup_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = cumdup_offset + smem_cols * 4

    mesh_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = mesh_offset + sub_tokens * smem_cols * 4

    scratch_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = scratch_offset + NUM_WAVES * 4

    @flyc.kernel(known_block_size=[ONESHOT_BLOCK, 1, 1])
    def moe_sorting_oneshot_kernel(
        topk_ids_tensor: fx.Tensor,
        topk_weights_tensor: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights_out: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        moe_buf: fx.Tensor,
        expert_mask_tensor: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
    ):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE
        tokens = i32_tokens
        c_zero_i32 = fx.Int32(0)
        c_one_i32 = fx.Int32(1)
        c_oob_idx = fx.Int32(0x7FFFFFFF)
        c4_i32 = fx.Int32(4)

        moe_buf_rsrc = buffer_ops.create_buffer_resource(moe_buf, max_size=True)
        topk_ids_rsrc = buffer_ops.create_buffer_resource(topk_ids_tensor, max_size=True)
        weights_rsrc = buffer_ops.create_buffer_resource(topk_weights_tensor, max_size=True)
        sorted_ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        sorted_w_rsrc = buffer_ops.create_buffer_resource(sorted_weights_out, max_size=True)
        sorted_e_rsrc = buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=True)
        nvalid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
        mask_rsrc = buffer_ops.create_buffer_resource(expert_mask_tensor, max_size=True)

        base_ptr = allocator.get_base()
        cumsum_mr = SmemPtr(base_ptr, cumsum_offset, T.i32, shape=(smem_cols,)).get()
        cumdup_mr = SmemPtr(base_ptr, cumdup_offset, T.i32, shape=(smem_cols,)).get()
        mesh_mr = SmemPtr(base_ptr, mesh_offset, T.i32, shape=(sub_tokens * smem_cols,)).get()

        c_topk = fx.Int32(topk)
        c_E = fx.Int32(E)
        c_unit = fx.Int32(unit_size)
        c_sub_tokens = fx.Int32(sub_tokens)
        c_smem_cols = fx.Int32(smem_cols)
        c_sentinel = fx.Int32((topk << 24))

        # =================== MOE_BUF ZEROING (blocks > 0 only) ===============
        if bid != c_zero_i32:
            zero_gid_v4 = (bid - c_one_i32) * fx.Int32(ONESHOT_BLOCK) + tid
            num_zero_blocks = gpu.grid_dim.x - c_one_i32
            zero_stride_v4 = num_zero_blocks * fx.Int32(ONESHOT_BLOCK)
            _zero_moe_buf_grid_stride(
                moe_buf_rsrc, zero_gid_v4, zero_stride_v4, i32_moe_buf_elems >> fx.Int32(2), c_oob_idx
            )

        # =================== SORTING (block 0 only) ==========================
        if bid == c_zero_i32:
            # ========================= PHASE 1: Histogram =========================
            for i_clear in range_constexpr(0, sub_tokens * smem_cols, ONESHOT_BLOCK):
                idx = fx.Int32(i_clear) + tid
                is_valid = idx < fx.Int32(sub_tokens * smem_cols)
                safe_idx = is_valid.select(idx, c_zero_i32)
                safe_idx_ix = ArithValue(safe_idx).index_cast(T.index)
                _lds_store_raw(mesh_mr, c_zero_i32, safe_idx_ix)
            gpu.barrier()

            total_assignments = tokens * c_topk
            for i_assign in range_constexpr(0, max_tokens * topk, ONESHOT_BLOCK):
                flat_idx = fx.Int32(i_assign) + tid
                is_valid = flat_idx < total_assignments
                safe_flat = is_valid.select(flat_idx, c_zero_i32)

                token_id = safe_flat // c_topk
                topk_slot = safe_flat % c_topk

                global_idx = token_id * c_topk + topk_slot
                eid = buffer_ops.buffer_load(topk_ids_rsrc, global_idx, vec_width=1, dtype=T.i32)

                mesh_addr = token_id * c_smem_cols + eid
                last_mesh_idx = fx.Int32(sub_tokens * smem_cols - 1)
                safe_mesh_addr = is_valid.select(mesh_addr, last_mesh_idx)
                safe_mesh_ix = ArithValue(safe_mesh_addr).index_cast(T.index)
                val = is_valid.select(topk_slot + c_one_i32, c_zero_i32)
                _lds_store_raw(mesh_mr, val, safe_mesh_ix)
            gpu.barrier()

            # ===================== PHASE 2: Count + Prefix Sum =====================
            c_lane_group_sz = fx.Int32(8)
            lane_group_id = tid // c_lane_group_sz
            lane_group_os = tid % c_lane_group_sz
            width8_i32 = fx.Int32(8)

            is_t0 = tid == c_zero_i32

            _lds_store_raw(cumsum_mr, c_zero_i32, c_zero_i32)
            gpu.barrier()

            for i_e in range_constexpr(0, E, ONESHOT_BLOCK // 8):
                eid_local = fx.Int32(i_e) + lane_group_id
                eid_valid = eid_local < c_E

                cnt = c_zero_i32
                for i_sub in range_constexpr(0, sub_tokens, 8):
                    sub_idx = fx.Int32(i_sub) + lane_group_os
                    sub_valid = sub_idx < c_sub_tokens
                    combined_valid = eid_valid & sub_valid

                    safe_sub = combined_valid.select(sub_idx, c_zero_i32)
                    safe_eid = combined_valid.select(eid_local, c_zero_i32)
                    mesh_rd_addr = safe_sub * c_smem_cols + safe_eid
                    mesh_rd_ix = ArithValue(mesh_rd_addr).index_cast(T.index)
                    mesh_val = _lds_load_raw(mesh_mr, mesh_rd_ix)

                    has_token = combined_valid.select(
                        (mesh_val != c_zero_i32).select(c_one_i32, c_zero_i32),
                        c_zero_i32,
                    )

                    reduced = has_token
                    for sh in range_constexpr(3):
                        off = fx.Int32(1 << sh)
                        peer = reduced.shuffle_xor(off, width8_i32)
                        reduced = reduced + peer
                    cnt = cnt + reduced

                write_valid = eid_valid & (lane_group_os == c_zero_i32)
                cs_idx = write_valid.select(eid_local + c_one_i32, c_zero_i32)
                cs_ix = ArithValue(cs_idx).index_cast(T.index)
                cs_val = write_valid.select(cnt, c_zero_i32)
                _lds_store_raw(cumsum_mr, cs_val, cs_ix)
            gpu.barrier()

            # Phase 2b: Prefix sum over expert counts.
            for i_cvt in range_constexpr(0, E, ONESHOT_BLOCK):
                cvt_eid = fx.Int32(i_cvt) + tid
                cvt_valid = cvt_eid < c_E
                safe_cvt_idx = cvt_valid.select(cvt_eid + c_one_i32, c_zero_i32)
                cvt_ix = ArithValue(safe_cvt_idx).index_cast(T.index)
                raw_cnt_cvt = _lds_load_raw(cumsum_mr, cvt_ix)
                blocks_cvt = (raw_cnt_cvt + c_unit - c_one_i32) // c_unit
                padded_cvt = (raw_cnt_cvt == c_zero_i32).select(c_zero_i32, blocks_cvt * c_unit)
                _lds_store_raw(cumsum_mr, cvt_valid.select(padded_cvt, c_zero_i32), cvt_ix)
            gpu.barrier()

            if has_mask:
                for i_ep in range_constexpr(0, E, ONESHOT_BLOCK):
                    ep_eid = fx.Int32(i_ep) + tid
                    ep_valid = ep_eid < c_E
                    ep_safe_eid = ep_valid.select(ep_eid, c_zero_i32)
                    ep_m = buffer_ops.buffer_load(mask_rsrc, ep_safe_eid, vec_width=1, dtype=T.i32)
                    should_zero = ep_valid & (ep_m == c_zero_i32)
                    ep_cs_ix = ArithValue(ep_valid.select(ep_eid + c_one_i32, c_zero_i32)).index_cast(T.index)
                    _lds_store_raw(
                        cumsum_mr, should_zero.select(c_zero_i32, _lds_load_raw(cumsum_mr, ep_cs_ix)), ep_cs_ix
                    )
                gpu.barrier()

            # Step 2: All-wave parallel prefix sum (cumsum → cumdup).
            scratch_mr = SmemPtr(base_ptr, scratch_offset, T.i32, shape=(NUM_WAVES,)).get()

            for _ps_chunk in range_constexpr(0, E, ONESHOT_BLOCK):
                ps_eid = fx.Int32(_ps_chunk) + tid
                ps_valid = ps_eid < c_E
                ps_safe_ix = ArithValue(ps_valid.select(ps_eid + c_one_i32, c_zero_i32)).index_cast(T.index)
                ps_val = ps_valid.select(_lds_load_raw(cumsum_mr, ps_safe_ix), c_zero_i32)
                _lds_store_raw(cumdup_mr, ps_val, ps_safe_ix)
            _lds_store_raw(cumdup_mr, c_zero_i32, c_zero_i32)
            gpu.barrier()

            ps_tid_valid = tid < c_E
            val = ps_tid_valid.select(_lds_load_raw(cumdup_mr, tid + c_one_i32), c_zero_i32)
            _, inclusive_ps = _allwave_inclusive_prefix_sum(val, lane, wave, scratch_mr, NUM_WAVES, WARP_SIZE)
            _lds_store_raw(
                cumdup_mr,
                ps_tid_valid.select(inclusive_ps, c_zero_i32),
                ArithValue(ps_tid_valid.select(tid + c_one_i32, c_zero_i32)).index_cast(T.index),
            )
            gpu.barrier()

            if E > ONESHOT_BLOCK:
                if is_t0:
                    _extend_prefix_sum_serial(cumdup_mr, ONESHOT_BLOCK, E, _lds_load_raw, _lds_store_raw)
                gpu.barrier()

            _lds_store_raw(cumdup_mr, c_zero_i32, c_zero_i32)
            gpu.barrier()

            cs_E_ix_ps = ArithValue(c_E).index_cast(T.index)
            total_padded = _lds_load_raw(cumdup_mr, cs_E_ix_ps)
            buffer_ops.buffer_store(total_padded, nvalid_rsrc, c_zero_i32)
            buffer_ops.buffer_store(tokens, nvalid_rsrc, c_one_i32)
            gpu.barrier()

            for i_cp in range_constexpr(0, E + 1, ONESHOT_BLOCK):
                cp_idx = fx.Int32(i_cp) + tid
                cp_valid = cp_idx <= c_E
                safe_cp_idx = cp_valid.select(cp_idx, c_zero_i32)
                cp_ix = ArithValue(safe_cp_idx).index_cast(T.index)
                cp_val = _lds_load_raw(cumdup_mr, cp_ix)
                _lds_store_raw(cumsum_mr, cp_val, cp_ix)
            gpu.barrier()

            if has_mask:
                for i_ml in range_constexpr(0, E, ONESHOT_BLOCK):
                    ml_eid = fx.Int32(i_ml) + tid
                    ml_valid = ml_eid < c_E
                    safe_ml_eid = ml_valid.select(ml_eid, c_zero_i32)
                    ml_mask = buffer_ops.buffer_load(mask_rsrc, safe_ml_eid, vec_width=1, dtype=T.i32)
                    ml_val = ml_valid.select(ml_mask, c_zero_i32)
                    ml_ix = ArithValue(ml_valid.select(ml_eid + c_one_i32, c_zero_i32)).index_cast(T.index)
                    _lds_store_raw(cumdup_mr, ml_val, ml_ix)
                _lds_store_raw(cumdup_mr, c_zero_i32, c_zero_i32)
                gpu.barrier()

                m_tid_valid = tid < c_E
                mval = m_tid_valid.select(_lds_load_raw(cumdup_mr, tid + c_one_i32), c_zero_i32)
                _, inclusive_m = _allwave_inclusive_prefix_sum(mval, lane, wave, scratch_mr, NUM_WAVES, WARP_SIZE)
                _lds_store_raw(
                    cumdup_mr,
                    m_tid_valid.select(inclusive_m, c_zero_i32),
                    ArithValue(m_tid_valid.select(tid + c_one_i32, c_zero_i32)).index_cast(T.index),
                )
                gpu.barrier()

                if E > ONESHOT_BLOCK:
                    if is_t0:
                        _extend_prefix_sum_serial(cumdup_mr, ONESHOT_BLOCK, E, _lds_load_raw, _lds_store_raw)
                    gpu.barrier()

                _lds_store_raw(cumdup_mr, c_zero_i32, c_zero_i32)
                gpu.barrier()
            else:
                for i_ml in range_constexpr(0, E, ONESHOT_BLOCK):
                    ml_eid = fx.Int32(i_ml) + tid
                    ml_valid = ml_eid < c_E
                    safe_ml_eid = ml_valid.select(ml_eid, c_zero_i32)
                    ml_ix = ArithValue(safe_ml_eid).index_cast(T.index)
                    _lds_store_raw(cumdup_mr, ml_valid.select(safe_ml_eid, c_zero_i32), ml_ix)
                gpu.barrier()

            for i_eid in range_constexpr(0, E, ONESHOT_BLOCK):
                eid_wr = fx.Int32(i_eid) + tid
                eid_wr_valid = eid_wr < c_E
                safe_eid_wr = eid_wr_valid.select(eid_wr, c_zero_i32)

                cs_start_ix = ArithValue(safe_eid_wr).index_cast(T.index)
                cs_end_ix = ArithValue(safe_eid_wr + c_one_i32).index_cast(T.index)
                e_start = _lds_load_raw(cumsum_mr, cs_start_ix)
                e_end = eid_wr_valid.select(_lds_load_raw(cumsum_mr, cs_end_ix), e_start)
                local_eid = _lds_load_raw(cumdup_mr, cs_start_ix)

                _lds_store_raw(cumdup_mr, e_start, cs_start_ix)

                blk_start = e_start // c_unit
                blk_end = e_end // c_unit
                n_blks_wr = eid_wr_valid.select(blk_end - blk_start, c_zero_i32)
                _write_expert_id_blocks(sorted_e_rsrc, local_eid, blk_start, n_blks_wr)
            gpu.barrier()

            cs_E_ix = ArithValue(c_E).index_cast(T.index)
            cumE = _lds_load_raw(cumsum_mr, cs_E_ix)
            _lds_store_raw(cumdup_mr, cumE, cs_E_ix)
            gpu.barrier()

            # ====================== PRE-FILL: Sentinel fill (cooperative) ===========
            total_padded_pre = _lds_load_raw(cumdup_mr, ArithValue(c_E).index_cast(T.index))
            _fill_sentinel_slots(
                sorted_ids_rsrc,
                sorted_w_rsrc,
                c_zero_i32,
                total_padded_pre,
                c_sentinel | tokens,
                ONESHOT_BLOCK,
                tid,
                c_oob_idx,
            )
            gpu.barrier()

            # ====================== PHASE 3: Scatter ==============================
            for i_e2 in range_constexpr(0, E, ONESHOT_BLOCK // 8):
                eid_sc = fx.Int32(i_e2) + lane_group_id
                eid_sc_valid = eid_sc < c_E
                safe_eid_sc = eid_sc_valid.select(eid_sc, c_E)

                sc_expert_enabled = eid_sc_valid
                if has_mask:
                    sc_mask_val = buffer_ops.buffer_load(
                        mask_rsrc, eid_sc_valid.select(eid_sc, c_zero_i32), vec_width=1, dtype=T.i32
                    )
                    sc_expert_enabled = eid_sc_valid & (sc_mask_val != c_zero_i32)

                cs_sc_ix = ArithValue(safe_eid_sc).index_cast(T.index)
                position = _lds_load_raw(cumsum_mr, cs_sc_ix)

                for i_sub2 in range_constexpr(0, sub_tokens, 8):
                    my_sub = fx.Int32(i_sub2) + lane_group_os
                    my_sub_valid = sc_expert_enabled & (my_sub < c_sub_tokens)
                    safe_my_sub = my_sub_valid.select(my_sub, c_zero_i32)
                    my_mesh_addr = safe_my_sub * c_smem_cols + safe_eid_sc
                    my_mesh_ix = ArithValue(my_mesh_addr).index_cast(T.index)
                    my_x = _lds_load_raw(mesh_mr, my_mesh_ix)
                    my_has_token = my_sub_valid & (my_x != c_zero_i32)
                    local_cnt = my_has_token.select(c_one_i32, c_zero_i32)

                    cnt_raw = _unwrap_val(local_cnt)
                    zero_raw = _unwrap_val(c_zero_i32)

                    remote = fly_rocdl.update_dpp(
                        T.i32, zero_raw, cnt_raw, DPP_ROW_SHR_1, DPP_ROW_MASK, DPP_BANK_MASK, True
                    )
                    should_add = lane_group_os >= c_one_i32
                    local_cnt = should_add.select(local_cnt + fx.Int32(remote), local_cnt)

                    cnt_raw = _unwrap_val(local_cnt)
                    remote = fly_rocdl.update_dpp(
                        T.i32, zero_raw, cnt_raw, DPP_ROW_SHR_2, DPP_ROW_MASK, DPP_BANK_MASK, True
                    )
                    should_add = lane_group_os >= fx.Int32(2)
                    local_cnt = should_add.select(local_cnt + fx.Int32(remote), local_cnt)

                    cnt_raw = _unwrap_val(local_cnt)
                    remote = fly_rocdl.update_dpp(
                        T.i32, zero_raw, cnt_raw, DPP_ROW_SHR_4, DPP_ROW_MASK, DPP_BANK_MASK, True
                    )
                    should_add = lane_group_os >= fx.Int32(4)
                    local_cnt = should_add.select(local_cnt + fx.Int32(remote), local_cnt)

                    last_lane_of_group = tid | fx.Int32(7)
                    last_addr = last_lane_of_group * c4_i32
                    batch_total = fly_rocdl.ds_bpermute(T.i32, last_addr, local_cnt)
                    batch_total = fx.Int32(batch_total)

                    slot = position + local_cnt - c_one_i32
                    safe_x = my_has_token.select(my_x, c_one_i32)
                    topk_slot_sc = safe_x - c_one_i32
                    packed_id = (topk_slot_sc << fx.Int32(24)) | my_sub
                    safe_slot = my_has_token.select(slot, c_oob_idx)
                    buffer_ops.buffer_store(packed_id, sorted_ids_rsrc, safe_slot)

                    w_addr = my_has_token.select(my_sub * c_topk + topk_slot_sc, c_zero_i32)
                    w_val_i32 = buffer_ops.buffer_load(weights_rsrc, w_addr, vec_width=1, dtype=T.i32)
                    buffer_ops.buffer_store(w_val_i32, sorted_w_rsrc, safe_slot)

                    position = position + batch_total

                _lds_store_raw(cumsum_mr, position, cs_sc_ix)
            gpu.barrier()

    @flyc.jit
    def launch_moe_sorting_oneshot(
        topk_ids_tensor: fx.Tensor,
        topk_weights_tensor: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights_out: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids_out: fx.Tensor,
        moe_buf: fx.Tensor,
        expert_mask_tensor: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
        n_grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        launcher = moe_sorting_oneshot_kernel(
            topk_ids_tensor,
            topk_weights_tensor,
            sorted_token_ids,
            sorted_weights_out,
            sorted_expert_ids,
            num_valid_ids_out,
            moe_buf,
            expert_mask_tensor,
            i32_tokens,
            i32_moe_buf_elems,
        )
        launcher.launch(
            grid=(n_grid_blocks, 1, 1),
            block=(ONESHOT_BLOCK, 1, 1),
            stream=stream,
        )

    return launch_moe_sorting_oneshot


def _compute_sub_tokens(num_experts, arch=None):
    """LDS-capacity threshold (max T for the oneshot path)."""
    if arch is None:
        arch = get_hip_arch()
    E = num_experts
    smem_cols = E + 1
    if arch in ("gfx942",) or str(arch).startswith("gfx94"):
        lds_capacity_bytes = 65536
    elif str(arch).startswith("gfx95"):
        lds_capacity_bytes = 163840
    else:
        lds_capacity_bytes = 65536
    lds_capacity_ints = lds_capacity_bytes // 4
    target_occupancy = 2
    r = lds_capacity_ints // target_occupancy // smem_cols
    sub_unroll = 8
    cumsum_bufs = 2
    if r < (cumsum_bufs + sub_unroll):
        return 0
    return ((r - cumsum_bufs) // sub_unroll) * sub_unroll


def oneshot_max_t(num_experts, topk):
    """Max M for which the oneshot path is selected (mirrors moe_sorting_flydsl)."""
    sub_tokens = _compute_sub_tokens(num_experts)
    return min(sub_tokens, max(16, BLOCK_SIZE // max(topk, num_experts // 8)))


def build_moe_sorting_module(num_experts, topk, max_tokens=16, unit_size=UNIT_SIZE):
    """Compile + return the oneshot launcher (entry point for compile_command)."""
    mt = max(int(max_tokens), 8)
    mt = ((mt + 7) // 8) * 8
    return _compile_moe_sorting_oneshot(
        num_experts=num_experts, topk=topk, max_tokens=mt, unit_size=unit_size, has_mask=False
    )


_dummy_mask_cache = {}
_moe_buf_cache = {}


def flydsl_moe_sorting(topk_ids, topk_weights, num_experts, unit_size=UNIT_SIZE, model_dim=512):
    """Run the FlyDSL oneshot MoE sorting kernel.

    Allocates the output tensors and dispatches the oneshot kernel. Returns
    ``(sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids)``.

    Raises ``NotImplementedError`` if the shape would require the unsupported
    multiphase paths.
    """
    assert topk_ids.dtype == torch.int32, "topk_ids must be int32"
    assert topk_ids.is_cuda, "topk_ids must be on the GPU"
    M, topk = topk_ids.shape
    device = topk_ids.device
    topk_weights = topk_weights.to(torch.float32).contiguous()

    if M > oneshot_max_t(num_experts, topk):
        raise NotImplementedError(
            f"M={M} exceeds oneshot threshold {oneshot_max_t(num_experts, topk)} "
            f"for E={num_experts}, topk={topk}; multiphase path is not supported."
        )

    max_num_tokens_padded = M * topk + num_experts * unit_size - topk
    max_num_m_blocks = (max_num_tokens_padded + unit_size - 1) // unit_size

    sorted_token_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(max_num_tokens_padded, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    # ``moe_buf`` is scratch storage rather than a returned output.  Cache it so
    # CUDA Graph capture measures the sorting kernel instead of allocator work.
    stream_id = torch.cuda.current_stream(device).cuda_stream
    moe_buf_key = (device, stream_id, M, model_dim)
    moe_buf = _moe_buf_cache.get(moe_buf_key)
    if moe_buf is None:
        moe_buf = torch.empty((M, model_dim), dtype=torch.bfloat16, device=device)
        _moe_buf_cache[moe_buf_key] = moe_buf

    moe_buf_i32 = moe_buf.view(torch.int32)
    moe_buf_elems = moe_buf_i32.numel()

    mask_tensor = _dummy_mask_cache.get(device)
    if mask_tensor is None:
        mask_tensor = torch.ones(1, dtype=torch.int32, device=device)
        _dummy_mask_cache[device] = mask_tensor

    max_tokens = ((max(M, 8) + 7) // 8) * 8
    target_occupancy = 2
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    n_zero_blocks = min((moe_buf_elems + BLOCK_SIZE - 1) // BLOCK_SIZE, num_cu * target_occupancy)
    n_grid_blocks = 1 + n_zero_blocks

    launch_oneshot = _compile_moe_sorting_oneshot(
        num_experts=num_experts, topk=topk, max_tokens=max_tokens, unit_size=unit_size, has_mask=False
    )
    launch_oneshot(
        topk_ids,
        topk_weights,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf_i32,
        mask_tensor,
        M,
        moe_buf_elems,
        n_grid_blocks,
        stream=torch.cuda.current_stream(),
    )
    return sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids
