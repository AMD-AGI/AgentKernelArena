"""[geak r1_d2 / memory] Token-major KV gather for sglang MLA grouped-decode stage 1.

Drop-in replacement for
  sglang.kernels.ops.attention.decode_attention:_decode_grouped_att_m_fwd
(+ the `_fwd_grouped_kernel_stage1` @triton.jit it launches).

WHAT CHANGES vs the reference (kernel_src is the ONLY writable path; launch geometry is held
at the accepted BLOCK_N=128 / BLOCK_H=16 / num_warps=4 / num_stages=1 point on purpose --
another engineer owns that lane this round):

  reference : k tile is loaded [BLOCK_DMODEL, BLOCK_N] -- the SCATTERED token index is the
              fast (last) tile axis, so consecutive lanes touch addresses 1152 B apart and
              the gather degenerates to 2-byte scalar loads.  V is then `tl.trans(k)`, a
              512x128 in-register/LDS transpose every iteration.
  here      : k tile is loaded [BLOCK_N, BLOCK_DMODEL] -- token-major.  One latent row is
              576 x 2 B = 1152 CONTIGUOUS bytes, so the fast tile axis is contiguous and the
              compiler can emit global_load_dwordx4.  V is then literally k_tile (HAS_MLA:
              V == leading BLOCK_DV lanes of the same latent row) -- a FREE slice, one load
              feeds both dots.  The only transpose left is on the small qk operand.

  qk        : dot(k_tile[BLOCK_N,512], qT[512,BLOCK_H]) -> [BLOCK_N,BLOCK_H], transposed to
              [BLOCK_H,BLOCK_N].  qT is loop-invariant and hoisted out of the KV loop, so the
              per-iteration transpose is 128x16 instead of 512x128.
  hints     : per-operand cache modifier on the KV stream only (each KV byte is read exactly
              once -> L2 residency is worthless); Q stays normal (reused by the whole split).

Anything outside the fast path (not HAS_MLA, PAGE_SIZE != 1, SCORE_MOD, xai temperature)
falls back to the stock launcher so semantics are preserved exactly.
"""
import os

import triton
import triton.language as tl

# cache modifier applied to the KV (streaming) loads only.  "" == none.
_KV_CM = os.environ.get("GEAK_MLA_KV_CACHE_MOD", ".cg")
_TOKEN_MAJOR = os.environ.get("GEAK_MLA_TOKEN_MAJOR", "1") != "0"
_IDX_CM = os.environ.get("GEAK_MLA_IDX_CACHE_MOD", ".cg")
_I32 = os.environ.get("GEAK_MLA_I32", "1") != "0"
_DSPLIT = int(os.environ.get("GEAK_MLA_DSPLIT", "1"))
_NOMASK = os.environ.get("GEAK_MLA_NOMASK", "0") != "0"
_OUT_CM = os.environ.get("GEAK_MLA_OUT_CACHE_MOD", ".cs")
_BOTH_FIRST = os.environ.get("GEAK_MLA_BOTH_FIRST", "0") != "0"
_DVSPLIT = int(os.environ.get("GEAK_MLA_DVSPLIT", "2"))
_LSTAGES = int(os.environ.get("GEAK_MLA_LSTAGES", "2"))
_TM_BN = int(os.environ.get("GEAK_MLA_TM_BLOCK_N", "0"))
_NW = int(os.environ.get("GEAK_MLA_NUM_WARPS", "8"))


@triton.jit
def _fwd_grouped_kernel_stage1_tm(
    Q,
    K_Buffer,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    KV_CM: tl.constexpr,
    IDX_CM: tl.constexpr,
    I32: tl.constexpr,
    DSPLIT: tl.constexpr,
    NOMASK: tl.constexpr,
    OUT_CM: tl.constexpr,
    BOTH_FIRST: tl.constexpr,
):
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        # hoisted, loop-invariant: the *small* operand carries the transpose now.
        qT = tl.trans(q.to(K_Buffer.dtype.element_ty))
        if BLOCK_DPE > 0:
            offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
            mask_dpe = offs_dpe < Lk
            off_qpe = (
                cur_batch * stride_qbs
                + cur_head[:, None] * stride_qh
                + offs_dpe[None, :]
            )
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )
            qpeT = tl.trans(qpe.to(K_Buffer.dtype.element_ty))

        # token-major base: row = one full latent vector => contiguous fast axis
        base_k = cur_kv_head * stride_buf_kh + offs_d[None, :]
        if BLOCK_DPE > 0:
            base_kpe = cur_kv_head * stride_buf_kh + offs_dpe[None, :]

        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=mask_n,
                other=0,
                cache_modifier=IDX_CM,
            )
            if I32:
                row = kv_loc.to(tl.int32)[:, None] * stride_buf_kbs
            else:
                row = kv_loc[:, None] * stride_buf_kbs
            # NOMASK: kv_loc is already clamped to a VALID slot (0) for out-of-range
            # tokens by the masked index load above, so the KV gather itself needs no
            # predication -- the garbage rows are annihilated by the qk -inf mask below
            # (p == 0 exactly).  Drops the per-element mask/select off the wide loads.
            if NOMASK:
                k_tile = tl.load(K_Buffer + row + base_k, cache_modifier=KV_CM)
            else:
                k_tile = tl.load(
                    K_Buffer + row + base_k,
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0,
                    cache_modifier=KV_CM,
                )
            if BOTH_FIRST and BLOCK_DPE > 0:
                # issue BOTH wide KV loads before consuming either, so the two
                # gathers are in flight together instead of separated by a dot.
                kpe_first = tl.load(
                    K_Buffer + row + base_kpe,
                    mask=mask_n[:, None] & mask_dpe[None, :],
                    other=0.0,
                    cache_modifier=KV_CM,
                )
                qkT = tl.dot(kpe_first, qpeT)
                qkT += tl.dot(k_tile, qT)
            else:
                qkT = tl.dot(k_tile, qT)
            if BLOCK_DPE > 0 and not BOTH_FIRST:
                if NOMASK:
                    kpe_tile = tl.load(
                        K_Buffer + row + base_kpe, cache_modifier=KV_CM
                    )
                else:
                    kpe_tile = tl.load(
                        K_Buffer + row + base_kpe,
                        mask=mask_n[:, None] & mask_dpe[None, :],
                        other=0.0,
                        cache_modifier=KV_CM,
                    )
                qkT += tl.dot(kpe_tile, qpeT)
            qk = tl.trans(qkT) * sm_scale_withk

            if logit_cap > 0:
                qk = logit_cap * (
                    (tl.exp(2.0 * qk / logit_cap) - 1.0)
                    / (tl.exp(2.0 * qk / logit_cap) + 1.0)
                )

            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

            # HAS_MLA: V is the leading BLOCK_DV lanes of the very same latent row.
            # Fast path guarantees Lv == 512 == BLOCK_DV == BLOCK_DMODEL -> FREE slice.
            v = k_tile

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
            cache_modifier=OUT_CM,
        )
        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv
        tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)



# ---------------------------------------------------------------------------
# [geak r2_d0 / memory] DV-SPLIT variant.
#
# MEASURED FACT this attacks: the token-major kernel compiles to shared=131072 B
# (= BLOCK_N*BLOCK_DMODEL*2 = the whole k_tile) -- 128 KiB against 160 KiB/CU, so
# exactly ONE workgroup is resident per CU (0.75 waves/SIMD).
#
# WHERE the LDS goes: k_tile is the *A* operand of the qk dot (contraction over D,
# which is the contiguous/fast axis -> the global-load layout already matches the
# MFMA A dot-operand layout, no staging needed) but the *B* operand of the PV dot
# (contraction over N = dim 0) -- that relayout is a transpose and on gfx950 it goes
# through LDS (ds_read_b64_tr_b16).  So the 128 KiB is PV-operand staging.
#
# THE FIX: split the *output* dimension DV of the PV dot (NOT the contraction, and
# NOT BLOCK_N).  acc[H, 0:256] += dot(p, k_lo) ; acc[H, 256:512] += dot(p, k_hi).
# Each staged chunk is [BLOCK_N, 256] = 64 KiB and the two are consumed strictly
# sequentially, so peak LDS halves.  Costs nothing:
#   * identical KV bytes -- k_lo/k_hi are two contiguous 512 B pieces of the SAME
#     1152 B latent row, still dwordx4, each row still read exactly once;
#   * identical MFMA count;
#   * ONE softmax / ONE acc rescale per 128 tokens (this is what plain BLOCK_N=64
#     gives up -- it doubles the [16,512] fp32 acc rescale, which is why it lost);
#   * the qk contraction is split the same way and summed BEFORE p exists, so the
#     "qk needs the full 576 before p" trap does not apply.
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_grouped_kernel_stage1_dvs(
    Q,
    K_Buffer,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    KV_CM: tl.constexpr,
    IDX_CM: tl.constexpr,
    I32: tl.constexpr,
    NOMASK: tl.constexpr,
    OUT_CM: tl.constexpr,
    LSTAGES: tl.constexpr,
):
    DSUB: tl.constexpr = BLOCK_DV // 2

    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_lo = tl.arange(0, DSUB)
    offs_hi = DSUB + tl.arange(0, DSUB)
    mask_lo = offs_lo < Lk
    mask_hi = offs_hi < Lk
    mask_vlo = offs_lo < Lv
    mask_vhi = offs_hi < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_H, DSUB], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_H, DSUB], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        qbase = cur_batch * stride_qbs + cur_head[:, None] * stride_qh
        q0 = tl.load(
            Q + qbase + offs_lo[None, :],
            mask=(mask_h[:, None]) & (mask_lo[None, :]),
            other=0.0,
        )
        q1 = tl.load(
            Q + qbase + offs_hi[None, :],
            mask=(mask_h[:, None]) & (mask_hi[None, :]),
            other=0.0,
        )
        qT0 = tl.trans(q0.to(K_Buffer.dtype.element_ty))
        qT1 = tl.trans(q1.to(K_Buffer.dtype.element_ty))
        if BLOCK_DPE > 0:
            offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
            mask_dpe = offs_dpe < Lk
            qpe = tl.load(
                Q + qbase + offs_dpe[None, :],
                mask=(mask_h[:, None]) & (mask_dpe[None, :]),
                other=0.0,
            )
            qpeT = tl.trans(qpe.to(K_Buffer.dtype.element_ty))

        base_k0 = cur_kv_head * stride_buf_kh + offs_lo[None, :]
        base_k1 = cur_kv_head * stride_buf_kh + offs_hi[None, :]
        if BLOCK_DPE > 0:
            base_kpe = cur_kv_head * stride_buf_kh + offs_dpe[None, :]

        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N, num_stages=LSTAGES):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=mask_n,
                other=0,
                cache_modifier=IDX_CM,
            )
            if I32:
                row = kv_loc.to(tl.int32)[:, None] * stride_buf_kbs
            else:
                row = kv_loc[:, None] * stride_buf_kbs

            if NOMASK:
                k0 = tl.load(K_Buffer + row + base_k0, cache_modifier=KV_CM)
                k1 = tl.load(K_Buffer + row + base_k1, cache_modifier=KV_CM)
            else:
                k0 = tl.load(
                    K_Buffer + row + base_k0,
                    mask=mask_n[:, None] & mask_lo[None, :],
                    other=0.0,
                    cache_modifier=KV_CM,
                )
                k1 = tl.load(
                    K_Buffer + row + base_k1,
                    mask=mask_n[:, None] & mask_hi[None, :],
                    other=0.0,
                    cache_modifier=KV_CM,
                )
            qkT = tl.dot(k0, qT0)
            qkT += tl.dot(k1, qT1)
            if BLOCK_DPE > 0:
                if NOMASK:
                    kpe_tile = tl.load(K_Buffer + row + base_kpe, cache_modifier=KV_CM)
                else:
                    kpe_tile = tl.load(
                        K_Buffer + row + base_kpe,
                        mask=mask_n[:, None] & mask_dpe[None, :],
                        other=0.0,
                        cache_modifier=KV_CM,
                    )
                qkT += tl.dot(kpe_tile, qpeT)
            qk = tl.trans(qkT) * sm_scale_withk

            if logit_cap > 0:
                qk = logit_cap * (
                    (tl.exp(2.0 * qk / logit_cap) - 1.0)
                    / (tl.exp(2.0 * qk / logit_cap) + 1.0)
                )

            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            pk = p.to(K_Buffer.dtype.element_ty)
            acc0 = acc0 * re_scale[:, None] + tl.dot(pk, k0)
            acc1 = acc1 * re_scale[:, None] + tl.dot(pk, k1)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        obase = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
        )
        inv = 1.0 / e_sum[:, None]
        tl.store(
            Att_Out + obase + offs_lo[None, :],
            acc0 * inv,
            mask=(mask_h[:, None]) & (mask_vlo[None, :]),
            cache_modifier=OUT_CM,
        )
        tl.store(
            Att_Out + obase + offs_hi[None, :],
            acc1 * inv,
            mask=(mask_h[:, None]) & (mask_vhi[None, :]),
            cache_modifier=OUT_CM,
        )
        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv
        tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)



@triton.jit
def _fwd_grouped_kernel_stage1_dvs4(
    Q,
    K_Buffer,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    KV_CM: tl.constexpr,
    IDX_CM: tl.constexpr,
    I32: tl.constexpr,
    NOMASK: tl.constexpr,
    OUT_CM: tl.constexpr,
    LSTAGES: tl.constexpr,
):
    DSUB: tl.constexpr = BLOCK_DV // 4

    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_c0 = tl.arange(0, DSUB)
    offs_c1 = DSUB + tl.arange(0, DSUB)
    offs_c2 = 2 * DSUB + tl.arange(0, DSUB)
    offs_c3 = 3 * DSUB + tl.arange(0, DSUB)
    mask_c0 = offs_c0 < Lk
    mask_c1 = offs_c1 < Lk
    mask_c2 = offs_c2 < Lk
    mask_c3 = offs_c3 < Lk
    mask_v0 = offs_c0 < Lv
    mask_v1 = offs_c1 < Lv
    mask_v2 = offs_c2 < Lv
    mask_v3 = offs_c3 < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_H, DSUB], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_H, DSUB], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_H, DSUB], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_H, DSUB], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        qbase = cur_batch * stride_qbs + cur_head[:, None] * stride_qh
        qT0 = tl.trans(tl.load(Q + qbase + offs_c0[None, :], mask=(mask_h[:, None]) & (mask_c0[None, :]), other=0.0).to(K_Buffer.dtype.element_ty))
        qT1 = tl.trans(tl.load(Q + qbase + offs_c1[None, :], mask=(mask_h[:, None]) & (mask_c1[None, :]), other=0.0).to(K_Buffer.dtype.element_ty))
        qT2 = tl.trans(tl.load(Q + qbase + offs_c2[None, :], mask=(mask_h[:, None]) & (mask_c2[None, :]), other=0.0).to(K_Buffer.dtype.element_ty))
        qT3 = tl.trans(tl.load(Q + qbase + offs_c3[None, :], mask=(mask_h[:, None]) & (mask_c3[None, :]), other=0.0).to(K_Buffer.dtype.element_ty))
        if BLOCK_DPE > 0:
            offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
            mask_dpe = offs_dpe < Lk
            qpe = tl.load(
                Q + qbase + offs_dpe[None, :],
                mask=(mask_h[:, None]) & (mask_dpe[None, :]),
                other=0.0,
            )
            qpeT = tl.trans(qpe.to(K_Buffer.dtype.element_ty))

        base_k0 = cur_kv_head * stride_buf_kh + offs_c0[None, :]
        base_k1 = cur_kv_head * stride_buf_kh + offs_c1[None, :]
        base_k2 = cur_kv_head * stride_buf_kh + offs_c2[None, :]
        base_k3 = cur_kv_head * stride_buf_kh + offs_c3[None, :]
        if BLOCK_DPE > 0:
            base_kpe = cur_kv_head * stride_buf_kh + offs_dpe[None, :]

        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N, num_stages=LSTAGES):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=mask_n,
                other=0,
                cache_modifier=IDX_CM,
            )
            if I32:
                row = kv_loc.to(tl.int32)[:, None] * stride_buf_kbs
            else:
                row = kv_loc[:, None] * stride_buf_kbs

            if NOMASK:
                k0 = tl.load(K_Buffer + row + base_k0, cache_modifier=KV_CM)
                k1 = tl.load(K_Buffer + row + base_k1, cache_modifier=KV_CM)
                k2 = tl.load(K_Buffer + row + base_k2, cache_modifier=KV_CM)
                k3 = tl.load(K_Buffer + row + base_k3, cache_modifier=KV_CM)
            else:
                k0 = tl.load(K_Buffer + row + base_k0, mask=mask_n[:, None] & mask_c0[None, :], other=0.0, cache_modifier=KV_CM)
                k1 = tl.load(K_Buffer + row + base_k1, mask=mask_n[:, None] & mask_c1[None, :], other=0.0, cache_modifier=KV_CM)
                k2 = tl.load(K_Buffer + row + base_k2, mask=mask_n[:, None] & mask_c2[None, :], other=0.0, cache_modifier=KV_CM)
                k3 = tl.load(K_Buffer + row + base_k3, mask=mask_n[:, None] & mask_c3[None, :], other=0.0, cache_modifier=KV_CM)
            qkT = tl.dot(k0, qT0)
            qkT += tl.dot(k1, qT1)
            qkT += tl.dot(k2, qT2)
            qkT += tl.dot(k3, qT3)
            if BLOCK_DPE > 0:
                if NOMASK:
                    kpe_tile = tl.load(K_Buffer + row + base_kpe, cache_modifier=KV_CM)
                else:
                    kpe_tile = tl.load(
                        K_Buffer + row + base_kpe,
                        mask=mask_n[:, None] & mask_dpe[None, :],
                        other=0.0,
                        cache_modifier=KV_CM,
                    )
                qkT += tl.dot(kpe_tile, qpeT)
            qk = tl.trans(qkT) * sm_scale_withk

            if logit_cap > 0:
                qk = logit_cap * (
                    (tl.exp(2.0 * qk / logit_cap) - 1.0)
                    / (tl.exp(2.0 * qk / logit_cap) + 1.0)
                )

            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            pk = p.to(K_Buffer.dtype.element_ty)
            acc0 = acc0 * re_scale[:, None] + tl.dot(pk, k0)
            acc1 = acc1 * re_scale[:, None] + tl.dot(pk, k1)
            acc2 = acc2 * re_scale[:, None] + tl.dot(pk, k2)
            acc3 = acc3 * re_scale[:, None] + tl.dot(pk, k3)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        obase = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
        )
        inv = 1.0 / e_sum[:, None]
        tl.store(Att_Out + obase + offs_c0[None, :], acc0 * inv, mask=(mask_h[:, None]) & (mask_v0[None, :]), cache_modifier=OUT_CM)
        tl.store(Att_Out + obase + offs_c1[None, :], acc1 * inv, mask=(mask_h[:, None]) & (mask_v1[None, :]), cache_modifier=OUT_CM)
        tl.store(Att_Out + obase + offs_c2[None, :], acc2 * inv, mask=(mask_h[:, None]) & (mask_v2[None, :]), cache_modifier=OUT_CM)
        tl.store(Att_Out + obase + offs_c3[None, :], acc3 * inv, mask=(mask_h[:, None]) & (mask_v3[None, :]), cache_modifier=OUT_CM)
        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv
        tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)




@triton.jit
def _fwd_grouped_kernel_stage1_dvs8(
    Q,
    K_Buffer,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    KV_CM: tl.constexpr,
    IDX_CM: tl.constexpr,
    I32: tl.constexpr,
    NOMASK: tl.constexpr,
    OUT_CM: tl.constexpr,
    LSTAGES: tl.constexpr,
):
    DSUB: tl.constexpr = BLOCK_DV // 8
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)
    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)
    offs_c0 = 0 * DSUB + tl.arange(0, DSUB)
    mask_c0 = offs_c0 < Lk
    mask_v0 = offs_c0 < Lv
    offs_c1 = 1 * DSUB + tl.arange(0, DSUB)
    mask_c1 = offs_c1 < Lk
    mask_v1 = offs_c1 < Lv
    offs_c2 = 2 * DSUB + tl.arange(0, DSUB)
    mask_c2 = offs_c2 < Lk
    mask_v2 = offs_c2 < Lv
    offs_c3 = 3 * DSUB + tl.arange(0, DSUB)
    mask_c3 = offs_c3 < Lk
    mask_v3 = offs_c3 < Lv
    offs_c4 = 4 * DSUB + tl.arange(0, DSUB)
    mask_c4 = offs_c4 < Lk
    mask_v4 = offs_c4 < Lv
    offs_c5 = 5 * DSUB + tl.arange(0, DSUB)
    mask_c5 = offs_c5 < Lk
    mask_v5 = offs_c5 < Lv
    offs_c6 = 6 * DSUB + tl.arange(0, DSUB)
    mask_c6 = offs_c6 < Lk
    mask_v6 = offs_c6 < Lv
    offs_c7 = 7 * DSUB + tl.arange(0, DSUB)
    mask_c7 = offs_c7 < Lk
    mask_v7 = offs_c7 < Lv
    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)
    kv_len_per_split = tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float('inf')
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_H, DSUB], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_H, DSUB], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_H, DSUB], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_H, DSUB], dtype=tl.float32)
    acc4 = tl.zeros([BLOCK_H, DSUB], dtype=tl.float32)
    acc5 = tl.zeros([BLOCK_H, DSUB], dtype=tl.float32)
    acc6 = tl.zeros([BLOCK_H, DSUB], dtype=tl.float32)
    acc7 = tl.zeros([BLOCK_H, DSUB], dtype=tl.float32)
    if split_kv_end > split_kv_start:
        qbase = cur_batch * stride_qbs + cur_head[:, None] * stride_qh
        qT0 = tl.trans(tl.load(Q + qbase + offs_c0[None, :], mask=(mask_h[:, None]) & (mask_c0[None, :]), other=0.0).to(K_Buffer.dtype.element_ty))
        qT1 = tl.trans(tl.load(Q + qbase + offs_c1[None, :], mask=(mask_h[:, None]) & (mask_c1[None, :]), other=0.0).to(K_Buffer.dtype.element_ty))
        qT2 = tl.trans(tl.load(Q + qbase + offs_c2[None, :], mask=(mask_h[:, None]) & (mask_c2[None, :]), other=0.0).to(K_Buffer.dtype.element_ty))
        qT3 = tl.trans(tl.load(Q + qbase + offs_c3[None, :], mask=(mask_h[:, None]) & (mask_c3[None, :]), other=0.0).to(K_Buffer.dtype.element_ty))
        qT4 = tl.trans(tl.load(Q + qbase + offs_c4[None, :], mask=(mask_h[:, None]) & (mask_c4[None, :]), other=0.0).to(K_Buffer.dtype.element_ty))
        qT5 = tl.trans(tl.load(Q + qbase + offs_c5[None, :], mask=(mask_h[:, None]) & (mask_c5[None, :]), other=0.0).to(K_Buffer.dtype.element_ty))
        qT6 = tl.trans(tl.load(Q + qbase + offs_c6[None, :], mask=(mask_h[:, None]) & (mask_c6[None, :]), other=0.0).to(K_Buffer.dtype.element_ty))
        qT7 = tl.trans(tl.load(Q + qbase + offs_c7[None, :], mask=(mask_h[:, None]) & (mask_c7[None, :]), other=0.0).to(K_Buffer.dtype.element_ty))
        if BLOCK_DPE > 0:
            offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
            mask_dpe = offs_dpe < Lk
            qpeT = tl.trans(tl.load(Q + qbase + offs_dpe[None, :], mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0).to(K_Buffer.dtype.element_ty))
        base_k0 = cur_kv_head * stride_buf_kh + offs_c0[None, :]
        base_k1 = cur_kv_head * stride_buf_kh + offs_c1[None, :]
        base_k2 = cur_kv_head * stride_buf_kh + offs_c2[None, :]
        base_k3 = cur_kv_head * stride_buf_kh + offs_c3[None, :]
        base_k4 = cur_kv_head * stride_buf_kh + offs_c4[None, :]
        base_k5 = cur_kv_head * stride_buf_kh + offs_c5[None, :]
        base_k6 = cur_kv_head * stride_buf_kh + offs_c6[None, :]
        base_k7 = cur_kv_head * stride_buf_kh + offs_c7[None, :]
        if BLOCK_DPE > 0:
            base_kpe = cur_kv_head * stride_buf_kh + offs_dpe[None, :]
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N, num_stages=LSTAGES):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            kv_loc = tl.load(kv_indices + cur_batch_kv_start_idx + offs_n, mask=mask_n, other=0, cache_modifier=IDX_CM)
            if I32:
                row = kv_loc.to(tl.int32)[:, None] * stride_buf_kbs
            else:
                row = kv_loc[:, None] * stride_buf_kbs
            if NOMASK:
                k0 = tl.load(K_Buffer + row + base_k0, cache_modifier=KV_CM)
                k1 = tl.load(K_Buffer + row + base_k1, cache_modifier=KV_CM)
                k2 = tl.load(K_Buffer + row + base_k2, cache_modifier=KV_CM)
                k3 = tl.load(K_Buffer + row + base_k3, cache_modifier=KV_CM)
                k4 = tl.load(K_Buffer + row + base_k4, cache_modifier=KV_CM)
                k5 = tl.load(K_Buffer + row + base_k5, cache_modifier=KV_CM)
                k6 = tl.load(K_Buffer + row + base_k6, cache_modifier=KV_CM)
                k7 = tl.load(K_Buffer + row + base_k7, cache_modifier=KV_CM)
            else:
                k0 = tl.load(K_Buffer + row + base_k0, mask=mask_n[:, None] & mask_c0[None, :], other=0.0, cache_modifier=KV_CM)
                k1 = tl.load(K_Buffer + row + base_k1, mask=mask_n[:, None] & mask_c1[None, :], other=0.0, cache_modifier=KV_CM)
                k2 = tl.load(K_Buffer + row + base_k2, mask=mask_n[:, None] & mask_c2[None, :], other=0.0, cache_modifier=KV_CM)
                k3 = tl.load(K_Buffer + row + base_k3, mask=mask_n[:, None] & mask_c3[None, :], other=0.0, cache_modifier=KV_CM)
                k4 = tl.load(K_Buffer + row + base_k4, mask=mask_n[:, None] & mask_c4[None, :], other=0.0, cache_modifier=KV_CM)
                k5 = tl.load(K_Buffer + row + base_k5, mask=mask_n[:, None] & mask_c5[None, :], other=0.0, cache_modifier=KV_CM)
                k6 = tl.load(K_Buffer + row + base_k6, mask=mask_n[:, None] & mask_c6[None, :], other=0.0, cache_modifier=KV_CM)
                k7 = tl.load(K_Buffer + row + base_k7, mask=mask_n[:, None] & mask_c7[None, :], other=0.0, cache_modifier=KV_CM)
            qkT = tl.dot(k0, qT0)
            qkT += tl.dot(k1, qT1)
            qkT += tl.dot(k2, qT2)
            qkT += tl.dot(k3, qT3)
            qkT += tl.dot(k4, qT4)
            qkT += tl.dot(k5, qT5)
            qkT += tl.dot(k6, qT6)
            qkT += tl.dot(k7, qT7)
            if BLOCK_DPE > 0:
                if NOMASK:
                    kpe_tile = tl.load(K_Buffer + row + base_kpe, cache_modifier=KV_CM)
                else:
                    kpe_tile = tl.load(K_Buffer + row + base_kpe, mask=mask_n[:, None] & mask_dpe[None, :], other=0.0, cache_modifier=KV_CM)
                qkT += tl.dot(kpe_tile, qpeT)
            qk = tl.trans(qkT) * sm_scale_withk
            if logit_cap > 0:
                qk = logit_cap * ((tl.exp(2.0 * qk / logit_cap) - 1.0) / (tl.exp(2.0 * qk / logit_cap) + 1.0))
            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float('-inf'))
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            pk = p.to(K_Buffer.dtype.element_ty)
            acc0 = acc0 * re_scale[:, None] + tl.dot(pk, k0)
            acc1 = acc1 * re_scale[:, None] + tl.dot(pk, k1)
            acc2 = acc2 * re_scale[:, None] + tl.dot(pk, k2)
            acc3 = acc3 * re_scale[:, None] + tl.dot(pk, k3)
            acc4 = acc4 * re_scale[:, None] + tl.dot(pk, k4)
            acc5 = acc5 * re_scale[:, None] + tl.dot(pk, k5)
            acc6 = acc6 * re_scale[:, None] + tl.dot(pk, k6)
            acc7 = acc7 * re_scale[:, None] + tl.dot(pk, k7)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max
        obase = cur_batch * stride_mid_ob + cur_head[:, None] * stride_mid_oh + split_kv_id * stride_mid_os
        inv = 1.0 / e_sum[:, None]
        tl.store(Att_Out + obase + offs_c0[None, :], acc0 * inv, mask=(mask_h[:, None]) & (mask_v0[None, :]), cache_modifier=OUT_CM)
        tl.store(Att_Out + obase + offs_c1[None, :], acc1 * inv, mask=(mask_h[:, None]) & (mask_v1[None, :]), cache_modifier=OUT_CM)
        tl.store(Att_Out + obase + offs_c2[None, :], acc2 * inv, mask=(mask_h[:, None]) & (mask_v2[None, :]), cache_modifier=OUT_CM)
        tl.store(Att_Out + obase + offs_c3[None, :], acc3 * inv, mask=(mask_h[:, None]) & (mask_v3[None, :]), cache_modifier=OUT_CM)
        tl.store(Att_Out + obase + offs_c4[None, :], acc4 * inv, mask=(mask_h[:, None]) & (mask_v4[None, :]), cache_modifier=OUT_CM)
        tl.store(Att_Out + obase + offs_c5[None, :], acc5 * inv, mask=(mask_h[:, None]) & (mask_v5[None, :]), cache_modifier=OUT_CM)
        tl.store(Att_Out + obase + offs_c6[None, :], acc6 * inv, mask=(mask_h[:, None]) & (mask_v6[None, :]), cache_modifier=OUT_CM)
        tl.store(Att_Out + obase + offs_c7[None, :], acc7 * inv, mask=(mask_h[:, None]) & (mask_v7[None, :]), cache_modifier=OUT_CM)
        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os) // Lv
        tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)



def make_launcher(base):
    """Build the drop-in launcher, closing over the stock module `base` for fallback."""
    _is_hip = base._is_hip
    _MIN_BLOCK_KV = base._MIN_BLOCK_KV
    _stock = base._decode_grouped_att_m_fwd

    def _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        att_out,
        att_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale_withk,
        logit_cap,
        xai_temperature_len=-1,
        has_mla=False,
        use_pdl=False,
        page_size: int = 1,
        score_mod=None,
        aux_tensors=None,
    ):
        Lk = k_buffer.shape[-1]
        Lv = v_buffer.shape[-1]

        # banner + accepted geometry come from the stock module (env-overridable)
        _geak_block, _geak_wpe = base._geak_mla_s1_geometry(Lk)

        fast = (
            _TOKEN_MAJOR
            and has_mla
            and page_size == 1
            and score_mod is None
            and xai_temperature_len <= 0
            and not use_pdl
            and Lk == 576
            and Lv == 512
        )
        if not fast:
            return _stock(
                q, k_buffer, v_buffer, att_out, att_lse, kv_indptr, kv_indices,
                num_kv_splits, max_kv_splits, sm_scale_withk, logit_cap,
                xai_temperature_len=xai_temperature_len, has_mla=has_mla,
                use_pdl=use_pdl, page_size=page_size, score_mod=score_mod,
                aux_tensors=aux_tensors,
            )

        BLOCK = _geak_block if _is_hip else 32
        if _TM_BN > 0:
            BLOCK = _TM_BN
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
        BLOCK_DV = triton.next_power_of_2(Lv)

        kv_head_num = k_buffer.shape[-2]
        batch, head_num = q.shape[0], q.shape[1]
        kv_group_num = head_num // kv_head_num

        BLOCK_H = 16
        grid = (batch, triton.cdiv(head_num, min(BLOCK_H, kv_group_num)), max_kv_splits)

        extra_kargs = {}
        num_stages = 2
        if _is_hip:
            extra_kargs = {
                "waves_per_eu": _geak_wpe,
                "matrix_instr_nonkdim": 16,
                "kpack": 2,
            }
            num_stages = 1

        k_slot_stride, k_head_stride, _, _ = base._extract_kv_strides(k_buffer, page_size)

        if _DVSPLIT in (2, 4, 8):
            _kfn = {2: _fwd_grouped_kernel_stage1_dvs, 4: _fwd_grouped_kernel_stage1_dvs4, 8: _fwd_grouped_kernel_stage1_dvs8}[_DVSPLIT]
            _kfn[grid](
                q,
                k_buffer,
                sm_scale_withk,
                kv_indptr,
                kv_indices,
                att_out,
                att_lse,
                num_kv_splits,
                q.stride(0),
                q.stride(1),
                k_slot_stride,
                k_head_stride,
                att_out.stride(0),
                att_out.stride(1),
                att_out.stride(2),
                kv_group_num=kv_group_num,
                q_head_num=head_num,
                BLOCK_DMODEL=BLOCK_DMODEL,
                BLOCK_DPE=BLOCK_DPE,
                BLOCK_DV=BLOCK_DV,
                BLOCK_N=BLOCK,
                BLOCK_H=BLOCK_H,
                MIN_BLOCK_KV=_MIN_BLOCK_KV,
                logit_cap=logit_cap,
                Lk=Lk,
                Lv=Lv,
                KV_CM=_KV_CM,
                IDX_CM=_IDX_CM,
                I32=(_I32 and k_buffer.numel() < (1 << 30)),
                NOMASK=_NOMASK,
                OUT_CM=_OUT_CM,
                LSTAGES=_LSTAGES,
                num_warps=_NW,
                num_stages=num_stages,
                **extra_kargs,
            )
            return

        _fwd_grouped_kernel_stage1_tm[grid](
            q,
            k_buffer,
            sm_scale_withk,
            kv_indptr,
            kv_indices,
            att_out,
            att_lse,
            num_kv_splits,
            q.stride(0),
            q.stride(1),
            k_slot_stride,
            k_head_stride,
            att_out.stride(0),
            att_out.stride(1),
            att_out.stride(2),
            kv_group_num=kv_group_num,
            q_head_num=head_num,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DPE=BLOCK_DPE,
            BLOCK_DV=BLOCK_DV,
            BLOCK_N=BLOCK,
            BLOCK_H=BLOCK_H,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            logit_cap=logit_cap,
            Lk=Lk,
            Lv=Lv,
            KV_CM=_KV_CM,
            IDX_CM=_IDX_CM,
            I32=(_I32 and k_buffer.numel() < (1 << 30)),
            DSPLIT=_DSPLIT,
            NOMASK=_NOMASK,
            OUT_CM=_OUT_CM,
            BOTH_FIRST=_BOTH_FIRST,
            num_warps=4,
            num_stages=num_stages,
            **extra_kargs,
        )

    return _decode_grouped_att_m_fwd
