# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm at ca328e297b (Apache-2.0):
# vllm/models/glm5next/amd/ops/mono/ffn/kernel.py, itself adapted from
# ROCm/aiter#6173 at c39b56c36 (Apache-2.0 License),
# Copyright (c) 2025 FlyDSL Project Contributors:
# aiter/ops/flydsl/kernels/glm5_mono/glm/kernel.py
# Vendored into AgentKernelArena from sgl-project/sgl-amd-mono at
# 52486703fbf27e77f05b573e4d931d4eec87118d (sgl_amd_mono/glm53_flash/gfx950/kernel.py, Apache-2.0). AKA modifications:
# package-relative imports of the vendored helpers; FlashBuild a frozen dataclass; a
# fixed source digest; the sparse-attention argument table passed as kernel arguments
# (tab_*) instead of a device table; FlyDSL 0.3.2 stand-ins for the memory fence and
# the LDS atomic add of newer FlyDSL (compat.py). The kernel's algorithm is unchanged.
# ruff: noqa: B008, B023, E501, SIM102

"""GLM-5.3-Flash decoder-layer monokernel: one persistent launch per TP rank.

``grid = 256 CTAs x 512 threads`` (one CTA per MI355X CU) runs one MoE layer
for this rank's TP shard: with ``kda`` or ``dsa`` its attention first
(``build_flash_kernel``), then its FFN::

    seam: TP sum of the attention partials -> mHC post -> R' + mix partials
      -> gate: mixes -> pre / post / comb (Sinkhorn) -> RMSNorm'd FFN input row
      -> router GEMV (bf16) -> sigmoid scores + the rows' MXFP8 activation blocks
      -> top-8 of 288 (sigmoid + bias) -> 1 shared + 8 routed expert up/gate,
         clamped SiLU -> mid MXFP8 quant -> expert down + route weighting
      -> MoE TP peer reduce -> x_out

Every stage is a list of tasks; task ``t`` of a stage runs on CTA
``(stage_base + t) % 256`` and every CTA walks the stages in order. There is
no grid-wide barrier: dependencies only point to earlier stages and all CTAs
are co-resident, so every wait makes progress. Each wait runs under the guard
(``common/guard.py``): past its deadline it flags every peer and stops the
launch, so no rank runs on with words that never arrived.

Mailboxes are tagged pairs ``(value, tag)`` with this launch's epoch as the tag;
a consumer polls the payload until the tags match. Each rank pushes its MoE
partial rows into every peer's symmetric buffer and sums the TP partials in
rank order, so every rank produces bit-identical outputs.
"""

import dataclasses

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from . import buffer_ops as bo
from . import guard
from .compat import lds_atomic_add, memory_fence
from .dpp_utils import update_dpp_i32
from .epoch import kernel_symbol, spin_until
from .ops import (
    ballot,
    bpermute_i32,
    div_rn,
    fp8_pack4,
)
from .ops import fp8_to_bf16x8 as _fp8_to_bf16x8
from .ops import fresh as fresh_i32
from .ops import hw_exp as _exp
from .ops import hw_rcp as _rcp
from .ops import hw_rsq as _rsq
from .ops import (
    popcount,
    read_lane_i32,
)
from .ops import rsrc as _rsrc
from .ops import uniform as _uniform
from .ops import uniform_f32 as _uniform_f32
from .ops import (
    wave_max,
    wave_umax,
    write_lane_i32,
)
from .ops import xred as _xred
from .ops import xshfl as _xshfl
from .plan import key_tuple

from .config import (
    EPS,
    FP8_BLOCK,
    FP8_MAX,
    FWHT_SCALE,
    HC_EPS,
    HC_MULT,
    HC_POST_MULT,
    HIDDEN,
    INDEX_DIM,
    INDEX_EPS,
    INDEX_HEADS,
    KDA_DIM,
    KDA_LOWER_BOUND,
    KPOOL,
    KV_LORA,
    L2_EPS,
    MLA_SCALE,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    O_NORM_EPS,
    Q_LORA,
    ROUTE_SCALE,
    ROUTER_CANDIDATES,
    SHARED_EXPERT,
    SINKHORN_ITERS,
    SWIGLU_LIMIT,
    TOP_K,
    TOPK_POOLS,
    TOPK_TOKENS,
    V_DIM,
    VOCAB,
    FlashShard,
)
from .layout import (
    BASE_SPLITS,
    BOX_LINE,
    BOX_WORDS,
    DSA_TAB,
    DUG_TILE,
    FGB_TILE,
    GROUPED_SUM_TILE,
    HEAD_TILE,
    IN_TILE,
    N_MIX,
    N_QKV,
    N_QKV_FP8,
    N_ROUTER,
    N_ROW_TILES,
    N_SPLIT,
    O_TILE,
    PMIX,
    POOLS_PER_TASK,
    QB_TILE,
    QKV_TILE,
    ROW_BT,
    ROW_POOL_SLOT,
    ROW_POS,
    ROW_SLOT,
    ROW_TAIL_SLOT,
    ROW_WORDS,
    SEL_STRIDE,
    SPLIT_KEYS,
    STEP_COUNTS,
    STEP_TAB,
    UK_TILE,
    UNIT_ROWS,
    UV_TILE,
    XQ_GROUPS,
    box_lines,
    dn_tile,
    down_prefetch_batch,
    down_x_words,
    grouped_units_max,
    layout,
    sample_wave_batches,
    stage_tasks,
    ug_tasks,
    uv_tiles,
)
from .tiles import (
    BLOCKS,
    CM_DEV,
    CM_SYS,
    LAYER_SLOTS,
    NEG,
    POLL_MAX,
    ROUTER_TILE,
    ROW_TILE,
    THREADS,
    UG_TILE,
    WAVES,
    atom_mxfp4_scale_index,
)

# the upstream digest of the traced sources (plan.kernel_sources), fixed for the vendored copy
SOURCES = "sgl-amd-mono-52486703-aka-glm53-flash-dsa"


def dpp_f32(old, src, ctrl, bank_mask=0xF):
    """``src`` moved across lanes by the DPP control ``ctrl`` into the 4-lane banks
    of ``bank_mask``; the other lanes keep ``old``."""
    return fx.Int32(
        update_dpp_i32(
            fx.Float32(old).bitcast(fx.Int32),
            fx.Float32(src).bitcast(fx.Int32),
            ctrl,
            0xF,
            bank_mask,
            False,
        )
    ).bitcast(fx.Float32)


def dpp_quad_bcast(k):
    """DPP control: lane ``k`` of each quad to the whole quad."""
    return 0x55 * k


def dpp_row_shl(n):
    """DPP control: lane l of each 16-lane row reads lane l + n."""
    return 0x100 + n


def dpp_row_shr(n):
    """DPP control: lane l of each 16-lane row reads lane l - n."""
    return 0x110 + n


# 5-key descending sorting network (lane's router candidates)
SORT5 = ((0, 1), (3, 4), (2, 4), (2, 3), (0, 3), (0, 2), (1, 4), (1, 3), (1, 2))
# equal keys go to the lower tie rank (AITER's grouped top-k order); a pick's wave
# maximum is (RANK_BASE - rank) << ID_BITS | id
RANK_BASE = 2047
ID_BITS = 9
assert N_EXPERTS < 1 << ID_BITS and ROUTER_CANDIDATES == 5


def tie_rank(expert):
    """AITER's ``legacy_topk_tie_rank``: the lane order in which its one-wave grouped
    top-k (vec4 per lane, lane + 64 r) keeps one of two equal scores."""
    vec4 = expert >> 2
    lane = vec4 & 63
    lane_rank = (
        ((~lane) & 0x38)
        | ((((lane >> 2) ^ (lane >> 3)) & 1) << 2)
        | ((lane & 3) ^ (((lane >> 2) & 1) * 3))
    )
    return (lane_rank << 5) | (((vec4 >> 6) << 2) | (expert & 3))


class _Guard(guard.Guard):
    """The launch's guard, with the deadline and the stop words read wave-uniform, so
    the waits' stop decisions stay scalar."""

    def deadline(self):
        d = super().deadline()
        lo = _uniform(fx.Int32(d & fx.Int64(0xFFFFFFFF)))
        hi = _uniform(fx.Int32(d >> 32))
        return (fx.Int64(hi) << 32) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))

    def stopped(self):
        """``Guard.stopped`` over ``guard.ERROR_WORDS``, read wave-uniform."""
        words = [self._word(i) for i in guard.ERROR_WORDS]
        own = words[0]
        for w in words[1:]:
            own = own | w
        return _uniform(own | self.flags()) != 0


@dataclasses.dataclass(frozen=True)
class FlashBuild:
    """One launch kind at one width; every field is a scalar of the JIT cache key
    (``plan.key_tuple``)."""

    rows: int
    tp: int
    launches_per_step: int  # this kind's launches a step: the epoch's layer stride
    inter: int  # this rank's expert intermediate width
    kda: bool = False
    conv_dim_first: bool = False  # KDA conv state [slots, 3P, 3], else [slots, 3, 3P]
    dsa: bool = False
    max_pools: int = 0  # kpool pools a sparse-attention row scores at most
    # which of q_a | kv_a, q_b and o_proj are FP8 with 128 x 128 block scales
    qkv_fp8: bool = True
    qb_fp8: bool = True
    o_fp8: bool = True
    # FP8 weights in AITER's (16, 16) preshuffled layout, as SGLang's gfx950
    # block-FP8 linear method leaves them
    qkv_shuffled: bool = False
    qb_shuffled: bool = False
    o_shuffled: bool = False
    # SGLANG_AMD_MONO_FAULT_INJECT: the wait this rank stalls at (0: none), once
    # fault_after steps have run
    fault_site: int = 0
    fault_after: int = 0
    # with kda: the dense MLP of layers 0-2 instead of the MoE; its FP8 gate | up and
    # down in AITER's (16, 16) preshuffled layout with dense_shuffled
    dense: bool = False
    dense_shuffled: bool = False
    # the model's tail instead of a layer
    head: bool = False
    # with kda: the step's first launch, from the step's tokens
    tokens: bool = False
    # an MTP verify step's rows a request (its draft chain; 1: decode): request r
    # holds rows r * verify .., and rows past the width's whole requests are pads
    verify: int = 1
    # False: a failed wait keeps the launch running and leaves its error words (tests)
    stop: bool = True
    # the expert-major MoE (not with dense or head): a task an expert's 16 picks, up
    # to 64 rows
    grouped: bool = False


def build_flash_kernel(key: FlashBuild):
    """Return the ``@flyc.jit`` launcher of one rank's decoder layer for
    ``key.rows`` rows; with ``kda`` the whole KDA decoder layer (attention-side
    mHC, projections, short conv, gated delta rule, gated RMSNorm, o_proj and its
    TP reduce) runs first; with ``dsa`` the whole sparse-attention decoder layer
    (attention-side mHC, q / kv LoRA projections and norms, KV cache write, kpool
    indexer, sparse MLA, o_proj) for rows of at most ``max_pools`` kpool pools.
    The shared expert is read from ``w_sh_*`` (one expert each), not the routed
    tensors, so it may be fused into them or stored apart.

    With ``dense`` (and ``kda``) the FFN is the dense FP8 SwiGLU MLP of layers 0-2.
    With ``head`` the launch is the model's tail: the owed mHC post, the stream mean,
    the final RMSNorm (``g_norm``), this rank's LM-head rows (``w_r``) and the logits'
    all-gather into ``x_out`` (f32 [S, VOCAB]). With ``tokens`` (and ``kda``) the
    launch takes the step's tokens (``dsa_tab``: ``STEP_TAB``; ``state_idx``: SGLang's
    KDA slots of the step's rows) instead of x, sums the ranks' embedding rows, and
    writes the step's KDA slot and row tables; the width's rows past the step's are
    pads.
    """
    S, W, launches_per_step = key.rows, key.tp, key.launches_per_step
    kda, dsa, max_pools = key.kda, key.dsa, key.max_pools
    conv_dim_first = key.conv_dim_first
    fault_site, fault_after = key.fault_site, key.fault_after
    dense, head, tokens = key.dense, key.head, key.tokens
    # a verify step: VL rows a request, NREQ whole requests in the width
    VL = key.verify
    VERIFY = VL > 1
    NREQ = S // VL
    GROUPED = key.grouped
    assert 1 <= VL <= S
    assert 2 <= S <= (64 if GROUPED else 16) and 1 <= launches_per_step <= LAYER_SLOTS
    assert not GROUPED or not (dense or head)
    assert key.inter % 128 == 0
    assert not kda or S <= 16
    assert not (kda and dsa)
    assert not dense or kda
    assert not head or not (kda or dsa)
    assert not tokens or (kda and W > 1)
    assert not dsa or (S <= 16 and max_pools % POOLS_PER_TASK == 0)
    SH = FlashShard(W)
    P, H, IN_ROWS = SH.kda_proj, SH.kda_heads, SH.kda_in_rows
    MH = SH.mla_heads
    # heads of a split task: one MFMA's 16 columns (or all of them); HG groups a row
    HT = min(MH, 16)
    HG = MH // HT
    QB_ROWS = MH * NOPE_DIM
    QKV8, QB8, O8 = key.qkv_fp8, key.qb_fp8, key.o_fp8
    QKV_SH, QB_SH, O_SH = key.qkv_shuffled, key.qb_shuffled, key.o_shuffled
    DENSE_SH = key.dense_shuffled
    whole = kda or dsa
    assert W <= guard.FLAG_WORDS
    if head:
        name = kernel_symbol("glm53_flash_tail", s=S, tp=W)
    else:
        extra = {k: 1 for k, on in (("n", dense), ("e", tokens), ("g", GROUPED)) if on}
        if VERIFY:
            extra["v"] = VL
        name = kernel_symbol(
            "glm53_flash_layer",
            s=S,
            tp=W,
            k=int(kda),
            d=int(dsa),
            q=int(QKV8),
            h=int(QKV_SH or QB_SH or O_SH or DENSE_SH),
            **extra,
        )
    keyed = key_tuple(key, SOURCES)
    # the KDA and sparse-attention layers read their prologue's loads only before the
    # first stage (the dense, the token and the tail launches keep theirs)
    PRO = whole and not dense and not tokens
    EARLY = PRO and S <= 4
    expert_inter = key.inter
    SHARED = SHARED_EXPERT
    G = BLOCKS
    SC, SY = layout(
        S, W, expert_inter, kda, dsa, max_pools, dense, head, tokens, GROUPED
    )
    # producers of each box: the gates' rows and the kda heads
    BOXES = {"box_a_in": S, "box_a": S, "box_o_in": S * H}
    assert not kda or S * H % BOX_WORDS == 0
    # the dense MLP's down task: one MFMA tile of output rows at every width
    DN_TILE = HIDDEN // BLOCKS if dense else dn_tile(S)
    N_DN_TILES = HIDDEN // DN_TILE
    # a routed up/gate task: 32 intermediate values (one MXFP8 block of mid), its gate
    # and up rows as 4 MFMA row groups, for half of the rows
    UG_BLK = 32
    UG_RG = 2 * UG_BLK // 16
    UG_HALF = (S + 1) // 2
    N_UG = ug_tasks(expert_inter)
    UG_RED_WORDS = UG_RG * WAVES * 64 * 4
    # split attention LDS: bf16 q of the task's HT heads, then its 64 keys' latent
    # rows (bf16 pairs); rows are padded by 4 words to spread MFMA operand banks
    QS = KV_LORA // 2 + 4
    KT_OFF = HT * QS
    SPLIT_WORDS = KT_OFF + SPLIT_KEYS * QS
    # top-512 pool select: a 4096-bin histogram, then (key, pool) candidates
    SEL_BINS = 4096
    SEL_ITEMS = 8
    SEL_CHUNK = THREADS * SEL_ITEMS
    LDS_KEYS = max(S * MOE_SLOTS, 320 if dsa else 0, SPLIT_KEYS if dsa else 0)
    # LDS X users: one router row (bf16) and its mix sums, every row's MXFP8 FFN
    # input (ug), the seam tile (x_attn + R', R after the layer's own attention, the
    # tile's mix weights), every row's MXFP8 mid (down); with KDA every row's bf16
    # attention input (inproj) and o (oproj); with DSA the attention input, q_a, a
    # split tile, the pool select
    # (grouped: a unit's rows only, its MXFP8 input or mid)
    XQ_ROWS = UNIT_ROWS if GROUPED else S
    X_WORDS = max(
        HIDDEN // 2 + 64,
        XQ_ROWS * HIDDEN // 4,
        S * ROW_TILE * (1 + 2 * HC_MULT) + HC_MULT * N_MIX * (ROW_TILE + 4),
        (UNIT_ROWS * expert_inter // 4 if GROUPED else down_x_words(S, expert_inter)),
        S * HIDDEN // 2 if whole or head else 0,
        S * SH.dense_inter // 2 if dense else 0,
        S * P // 2 if kda else 0,
        S * KDA_DIM if kda else 0,
        S * MH * V_DIM // 2 if dsa else 0,
        SPLIT_WORDS if dsa else 0,
        SEL_BINS + 2 * 2048 if dsa else 0,
    )
    SEL_CAP = (X_WORDS - SEL_BINS) // 2
    # the routed up/gate tiles' sums go past the rows' MXFP8 input in X when they fit
    UG_RED_IN_X = X_WORDS - XQ_ROWS * HIDDEN // 4 >= UG_RED_WORDS
    RED_WORDS = WAVES * 64 * 4 if UG_RED_IN_X else UG_RED_WORDS
    MISC_OFF = X_WORDS
    if GROUPED:
        # a unit's activation or mid exponents (from word 8), the seam's mix sums
        MISC_WORDS = max(
            8 + UNIT_ROWS * XQ_GROUPS,
            8 + UNIT_ROWS * (expert_inter // 32),
            S * HC_MULT * (1 + HC_MULT),
        )
    else:
        MISC_WORDS = max(8 + S * XQ_GROUPS, S * MOE_SLOTS * (expert_inter // 32))
    KEYS_OFF = MISC_OFF + MISC_WORDS
    DNW_OFF = KEYS_OFF + LDS_KEYS
    RED_OFF = DNW_OFF + S * MOE_SLOTS
    OUT_OFF = RED_OFF + RED_WORDS
    OUT_WORDS = max(S * ROW_TILE, S * 2 * UG_TILE, S * DN_TILE)
    # the grouped MoE's route table: each expert's row mask (2 words), pick count and
    # first pick, the units (expert | rows - 1 << 9 | first pick << 13) in expert
    # order, the picks (row * MOE_SLOTS + slot) by expert then row, scan sums, units
    NE1 = N_EXPERTS + 1
    U_MAX = grouped_units_max(S)
    RT_MASK = OUT_OFF + OUT_WORDS
    RT_CNT = RT_MASK + 2 * NE1
    RT_POFF = RT_CNT + NE1
    RT_UNIT = RT_POFF + NE1
    RT_PICK = RT_UNIT + U_MAX
    RT_WSUM = RT_PICK + S * MOE_SLOTS
    RT_N = RT_WSUM + WAVES
    RT_WORDS = RT_N + 1 - RT_MASK if GROUPED else 0
    # the step's row table (DSA), staged once a launch
    ROWS_OFF = RT_MASK + RT_WORDS
    WORK_WORDS = ROWS_OFF + (S * ROW_WORDS if dsa else 0)
    assert WORK_WORDS <= 40960, "keep static LDS below the MI355X per-workgroup budget"

    order = stage_tasks(S, expert_inter, W, kda, dsa, dense, head, GROUPED)
    base, acc = {}, 0
    for stage, n in order:
        base[stage] = acc % G
        acc += n

    @fx.struct
    class Smem:
        work: fx.Array[fx.Float32, WORK_WORDS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def flash_ffn_kernel(
        attn_part: Int64,
        res_in: Int64,
        post_in: Int64,
        comb_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        g_norm: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        x_out: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        w_sh_ug: Int64,
        s_sh_ug: Int64,
        w_sh_dn: Int64,
        s_sh_dn: Int64,
        hc_a_fn: Int64,
        hc_a_scale: Int64,
        hc_a_base: Int64,
        g_in: Int64,
        w_in: Int64,
        w_fb: Int64,
        w_gb: Int64,
        conv_w: Int64,
        a_log: Int64,
        dt_bias: Int64,
        w_on: Int64,
        w_o: Int64,
        conv_state: Int64,
        rec_state: Int64,
        state_idx: Int64,
        conv_stride: Int64,
        rec_stride: Int64,
        inter_rec: Int64,
        inter_conv: Int64,
        inter_rec_stride: Int64,
        inter_conv_stride: Int64,
        dsa_tab: Int64,
        tab_w_qkv: Int64,
        tab_s_qkv: Int64,
        tab_w_idx: Int64,
        tab_w_gate: Int64,
        tab_g_qa: Int64,
        tab_g_kv: Int64,
        tab_ln_w: Int64,
        tab_ln_b: Int64,
        tab_ape: Int64,
        tab_w_qb: Int64,
        tab_s_qb: Int64,
        tab_w_iq: Int64,
        tab_w_uk: Int64,
        tab_w_kvb: Int64,
        tab_w_o: Int64,
        tab_s_o: Int64,
        tab_kv_cache: Int64,
        tab_idx_cache: Int64,
        tab_tail_k: Int64,
        tab_tail_g: Int64,
        tab_idx_bt_group: Int64,
        tab_kv_block_elems: Int64,
        tab_kv_block_tokens: Int64,
        tab_kv_row_elems: Int64,
        tab_idx_block_bytes: Int64,
        tab_idx_block_pools: Int64,
        tab_tail_block_elems: Int64,
        dsa_rows: Int64,
        page_table: Int64,
        pt_stride: Int32,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Smem).peek()
        xs = lds.work.ptr
        misc = xs + MISC_OFF
        keys = fx.recast_iter(fx.Int32, xs + KEYS_OFF)
        dnw = xs + DNW_OFF
        red = xs + RED_OFF
        outs = xs + OUT_OFF
        rows_lds = fx.recast_iter(fx.Int32, xs + ROWS_OFF)
        v4f = fx.Vector.make_type(4, fx.Float32)

        # this launch's epoch: every mailbox tag must equal it. ``step`` is the
        # guard block, whose word 0 the host bumps once a decode step; ``layer``
        # makes the tag unique per layer within the step.
        def read_epoch(step_word):
            step_value = _uniform(step_word)
            tag = step_value * launches_per_step + layer + 1
            peer_slot = (step_value * launches_per_step + layer) & 1
            return tag, peer_slot

        def read_peer(pv):
            return (fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(
                fx.Uint32(_uniform(pv[0]))
            )

        def peer_load():
            # One wave sends to one peer, so retain only that wave's destination.
            return fx.Vector(
                bo.buffer_load(
                    _rsrc(peers), fx.min(wave, W - 1) * 2, vec_width=2, dtype=T.i32
                )
            )

        step_word = bo.buffer_load(_rsrc(step), 0, vec_width=1, dtype=T.i32)
        if const_expr(PRO):
            # The prologue's loads go out together, before anything waits for one: the
            # step counter, this wave's peer and, with sparse attention, the layer's
            # argument table (entry i in lane i) and the step's row table. They are read
            # just before the first stage (the epoch below).
            pv = peer_load()
            if const_expr(dsa):
                # (AKA: the layer's argument table arrives as the tab_* kernel arguments)
                row_words = fx.Int32(
                    bo.buffer_load(
                        _rsrc(dsa_rows),
                        fx.min(tid, S * ROW_WORDS - 1),
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
        else:
            tag, peer_slot = read_epoch(step_word)
            peer_dst = read_peer(peer_load())
        _ = keyed

        # ------------------------------------------------------------ helpers
        def ld_f32(r, i):
            return fx.Float32(bo.buffer_load(r, i, vec_width=1, dtype=T.f32))

        def ld_bf16(r, i):
            return fx.Float32(
                fx.BFloat16(bo.buffer_load(r, i, vec_width=1, dtype=T.bf16))
            )

        def ld_bf16s(r, elem, n):
            """bf16 elements elem .. elem + n - 1 (n of 4 or 8, elem a multiple of n)
            of a plain tensor -> n f32, in one load."""
            w = fx.Vector(bo.buffer_load(r, elem // 2, vec_width=n // 2, dtype=T.i32))
            return [v for e in range(n // 2) for v in bf2_f32(w[e])]

        def ld_pair(r, elem):
            """bf16 elements elem, elem + 1 (elem even) of a plain tensor -> 2 f32."""
            return bf2_f32(
                fx.Int32(bo.buffer_load(r, elem // 2, vec_width=1, dtype=T.i32))
            )

        def bf16_round(a):
            return fx.Float32(fx.Float32(a).to(fx.BFloat16))

        def div(a, b):
            return div_rn(a, b, _rcp(b))

        def sigmoid(a):
            return div(fx.Float32(1.0), 1.0 + _exp(-a))

        def lds_ld(ptr, i):
            return fx.ptr_load(ptr + i)

        def lds_st(ptr, i, v):
            fx.ptr_store(v, ptr + i)

        def bf16_pair(a, b):
            """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
            return (
                fx.Vector.from_elements([a, b], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Float32)[0]
            )

        # ---- tagged-pair mailboxes
        def mb(name):
            return scratch + fx.Int64(SC[name])

        def put(base_addr, i, v, cm=CM_DEV):
            """Pair i := (v, tag); ``v`` f32 (or int32 bits)."""
            bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
            bo.buffer_store(
                fx.Vector.from_elements([bits, tag], fx.Int32),
                _rsrc(base_addr),
                i * 2,
                cache_modifier=cm,
            )

        def put2(base_addr, i, v0, v1, cm=CM_DEV):
            """Pairs i, i+1 (i even) in one 16-byte store."""
            vec = fx.Vector.from_elements(
                [
                    fx.Float32(v0).bitcast(fx.Int32),
                    tag,
                    fx.Float32(v1).bitcast(fx.Int32),
                    tag,
                ],
                fx.Int32,
            )
            bo.buffer_store(vec, _rsrc(base_addr), i * 2, cache_modifier=cm)

        def put_bf(base_addr, i, vs, cm=CM_DEV):
            """Elements i .. i + len(vs) (2 or 4, i aligned) as packed bf16 pairs."""
            words = []
            for j in range_constexpr(len(vs) // 2):
                words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), tag]
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32),
                _rsrc(base_addr),
                i,
                cache_modifier=cm,
            )

        def bf2_f32(w):
            """Packed bf16 pair word -> (f32 low, f32 high)."""
            return (w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(
                fx.Float32
            )

        def new_guard():
            # every wait of this launch runs under the guard (deadline, error words)
            return _Guard(
                step,
                own=sym if W > 1 else None,
                peers=peers if W > 1 else None,
                rank=rank,
                npes=W,
                flags_offset=SY["flags"],
                fault_site=fault_site or None,
                fault_after=fault_after,
                stop=key.stop,
            )

        if const_expr(not PRO):
            launch_guard = new_guard()

        def poll(specs, scope="agent", batch=POLL_MAX):
            """Batched guarded poll of mailbox pairs: ``specs`` = [(base_addr, pair
            index, npairs in {1, 2})]; re-load the batch while any tag is stale."""
            if const_expr(len(specs) == 0):
                return []
            if const_expr(len(specs) > batch):  # bound live registers
                return poll(specs[:batch], scope, batch) + poll(
                    specs[batch:], scope, batch
                )
            cm = CM_DEV if const_expr(scope == "agent") else CM_SYS

            def load_all():
                words = []
                for b, i, n in specs:
                    w = fx.Vector(
                        bo.buffer_load(
                            _rsrc(b),
                            fx.Int32(i) * 2,
                            vec_width=2 * n,
                            dtype=T.i32,
                            cache_modifier=cm,
                        )
                    )
                    words += [w[e] for e in range(2 * n)]
                return fx.Vector.from_elements(words, fx.Int32)

            nw = sum(2 * n for _, _, n in specs)

            def pending(v):
                bad = v[1] != tag
                for e in range_constexpr(3, nw, 2):
                    bad = bad | (v[e] != tag)
                return bad

            with guard.scope(launch_guard):
                v = spin_until(load_all, pending)
            outs_, e = [], 0
            for _, _, n in specs:
                outs_.append([v[e + 2 * q] for q in range(n)])
                e += 2 * n
            return outs_

        def get(base_addr, i):
            return poll([(base_addr, i, 1)])[0][0]

        # ---- boxes: a consumer CTA polls its own lines, a word a producer
        def sloads(loads):
            """Scalar glc loads of [(wave-uniform address, n words)], n of 1, 2, 4, 8
            or 16, issued together. They count in lgkmcnt, so they do not wait
            behind the CU's vector loads."""
            types = [T.i32 if n == 1 else T.vec(n, T.i32) for _, n in loads]
            k = len(loads)
            asm = "\n".join(
                ["s_nop 4"]
                + [
                    f"s_load_dword{'' if n == 1 else f'x{n}'} ${i}, ${k + i}, 0x0 glc"
                    for i, (_, n) in enumerate(loads)
                ]
                + ["s_waitcnt lgkmcnt(0)"]
            )
            res = llvm.inline_asm(
                types[0] if k == 1 else llvm.StructType.get_literal(types),
                [as_ir_value(fx.Int64(a)) for a, _ in loads],
                asm,
                ",".join(["=&s"] * k + ["s"] * k),
                has_side_effects=True,
            )
            words = []
            for i, (_, n) in enumerate(loads):
                v = res if k == 1 else llvm.extractvalue(types[i], res, [i])
                words += (
                    [fx.Int32(v)] if n == 1 else [fx.Vector(v)[e] for e in range(n)]
                )
            return words

        def box_line(box, slot, line):
            return mb(box) + fx.Int64((slot * box_lines(BOXES[box]) + line) * BOX_LINE)

        def signal(box, p):
            """Producer ``p``'s tag into its word of every CTA's line of ``box``; the
            calling wave stores them, a lane a CTA."""
            r = _rsrc(mb(box))
            for i in range_constexpr(G // 64):
                c = lane + 64 * i
                line = c * box_lines(BOXES[box]) + p // BOX_WORDS
                bo.buffer_store(
                    tag,
                    r,
                    line * (BOX_LINE // 4) + p % BOX_WORDS,
                    cache_modifier=CM_DEV,
                )

        def await_box(box, words):
            """Wait (guarded) until this CTA's ``words`` = [(line, first word, count)]
            of ``box`` all hold the launch's tag: one scalar load a line."""
            loads = []
            for line, w0, cnt in words:
                n = 1 << (cnt - 1).bit_length()
                assert n <= BOX_WORDS
                loads.append((box_line(box, bid, line) + fx.Int64(w0 * 4), n, cnt))

            def load_all():
                return fx.Vector.from_elements(
                    sloads([(a, n) for a, n, _ in loads]), fx.Int32
                )

            def pending(v):
                acc, e = fx.Int32(0), 0
                for _, n, cnt in loads:
                    for q in range_constexpr(cnt):
                        acc = acc | (v[e + q] ^ tag)
                    e += n
                return acc != 0

            with guard.scope(launch_guard):
                spin_until(load_all, pending)

        def getf_many(specs):
            """[(base, i)] single pairs -> list of f32."""
            return [
                v[0].bitcast(fx.Float32) for v in poll([(b, i, 1) for b, i in specs])
            ]

        def get2_many(specs):
            """[(base, i)] double pairs (i even) -> list of (f32, f32)."""
            return [
                (v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32))
                for v in poll([(b, i, 2) for b, i in specs])
            ]

        def wave_sum(v):
            for sh in range_constexpr(6):
                v = _xred(v, 32 >> sh, lambda a, b: a + b)
            return v

        def block_sum(v):
            w = wave_sum(v)
            if lane == 0:
                lds_st(red, wave, w)
            gpu.barrier()
            t = lds_ld(red, 0)
            for i in range_constexpr(1, WAVES):
                t = t + lds_ld(red, i)
            gpu.barrier()
            return t

        # ------------------------------------------------ MFMA GEMV machinery
        def mxfp4_scale_word(s_rsrc, rg, kc, K, ln=None):
            """The 4 E8M0 weight scales of this lane's row in row groups rg, rg ^ 1 and
            K units kc, kc ^ 1 (rg and kc even): byte 2 (kc % 2) + rg % 2."""
            ln = lane if ln is None else ln
            return fx.Int32(
                bo.buffer_load(
                    s_rsrc,
                    atom_mxfp4_scale_index(
                        rg * 16 + ln % 16, kc * 4 + ln // 16, K // 32
                    )
                    // 4,
                    vec_width=1,
                    dtype=T.i32,
                )
            )

        def unit_native_mxfp4(
            w_rsrc,
            s_rsrc,
            rg,
            kc,
            K,
            b_word,
            b_scale_word,
            coef=None,
            ln=None,
            scale=None,
        ):
            """One 16-byte/lane AITER-shuffled W4 tile for the scaled MFMA. ``scale``:
            (scale word, byte) from ``mxfp4_scale_word``, else the byte loads here."""
            ln = lane if ln is None else ln
            raw = fx.Vector(
                bo.buffer_load(
                    w_rsrc,
                    (rg * (K // 64) + kc * 2) * 128 + ln * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            if const_expr(scale is None):
                scale = (
                    fx.Int32(
                        bo.buffer_load(
                            s_rsrc,
                            atom_mxfp4_scale_index(
                                rg * 16 + ln % 16, kc * 4 + ln // 16, K // 32
                            ),
                            vec_width=1,
                            dtype=T.i8,
                        )
                    ),
                    0,
                )
            return (
                "native_mxfp4",
                (raw, scale[0], b_scale_word, scale[1]),
                coef,
                b_word,
            )

        def lds_mxfp8(chunk_word):
            """One scaled-MFMA B operand from a row-major 128-value MXFP8 tile."""
            words = [
                fx.Vector(
                    fx.ptr_load(
                        xs + chunk_word + 16 * half + 4 * (lane // 16),
                        result_type=v4f,
                    )
                ).bitcast(fx.Int32)
                for half in range(2)
            ]
            return fx.Vector.from_elements(
                [words[half][e] for half in range(2) for e in range(4)],
                fx.Int32,
            )

        def unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln=None):
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        (((rg * NKC + kc) * 2 + sp) * 64 + ln) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for sp in range(2)
            ]
            return ("bf16", wv, None, b_word + (lane // 16) * 4)

        def mma_units(acc, units):
            """acc[4] += coef * (W_chunk @ X_chunk) for every issued unit."""
            for fmt, wv, coef, bw in units:
                if const_expr(callable(coef)):
                    coef = coef()
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                if const_expr(fmt == "native_mxfp4"):
                    raw, weight_scale, activation_scale_word, scale_byte = wv
                    activation_scale = lds_ld(misc, activation_scale_word).bitcast(
                        fx.Int32
                    )
                    c = fx.Vector(
                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.vec(4, T.f32),
                            [
                                raw,
                                lds_mxfp8(bw),
                                c,
                                4,
                                0,
                                scale_byte,
                                weight_scale,
                                0,
                                activation_scale,
                            ],
                        )
                    )
                if const_expr(fmt == "bf16"):
                    for sp in range_constexpr(2):
                        a = wv[sp].bitcast(fx.BFloat16)
                        b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(
                            fx.BFloat16
                        )
                        c = fx.Vector(
                            rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c])
                        )
                if const_expr(fmt == "fp8rm"):
                    # lane group g holds K 32 g .. 32 g + 31 of the block; MFMA i
                    # takes its values 8 i .. 8 i + 7 (exactly widened to bf16)
                    for i in range_constexpr(4):
                        a = _fp8_to_bf16x8(
                            wv[i // 2][(i % 2) * 2], wv[i // 2][(i % 2) * 2 + 1]
                        )
                        b = fx.ptr_load(xs + (bw + i * 4), result_type=v4f).bitcast(
                            fx.BFloat16
                        )
                        c = fx.Vector(
                            rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c])
                        )
                if const_expr(coef is None):
                    acc = [acc[e] + c[e] for e in range(4)]
                elif const_expr(isinstance(coef, tuple)):
                    # (column mask, factor or None): a select, so a non-finite row
                    # stays in its own column
                    mask, factor = coef
                    if const_expr(factor is not None):
                        c = [c[e] * factor for e in range(4)]
                    acc = [
                        acc[e] + mask.select(c[e], fx.Float32(0.0)) for e in range(4)
                    ]
                else:
                    acc = [acc[e] + c[e] * coef for e in range(4)]
            return acc

        def run_units(make_unit, cpw, batch, pre=None):
            """Software pipelined: issue batch b+1's loads before computing batch b.
            ``pre`` = the already-issued first batch (prefetched before a wait)."""
            acc = [fx.Float32(0.0) for _ in range(4)]
            starts = list(range(0, cpw, batch))
            cur = (
                pre
                if pre is not None
                else [make_unit(c) for c in range(0, min(batch, cpw))]
            )
            for bi in range_constexpr(len(starts)):
                nxt = None
                if const_expr(bi + 1 < len(starts)):
                    n0 = starts[bi + 1]
                    nxt = [make_unit(c) for c in range(n0, min(n0 + batch, cpw))]
                acc = mma_units(acc, cur)
                cur = nxt
            return acc

        def reduce_rows(R, acc, emit, count=S):
            """Sum per-wave MFMA tiles; emit(row_local, local sample column, value)."""
            wpr = WAVES // R
            fx.ptr_store(
                fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4
            )
            gpu.barrier()
            n_out = R * 16 * count
            for i in range_constexpr((n_out + THREADS - 1) // THREADS):
                t = tid + i * THREADS
                if t < n_out:
                    rl = t % (R * 16)
                    n = t // (R * 16)
                    r = rl % 16
                    tot = fx.Float32(0.0)
                    for j in range_constexpr(wpr):
                        ww = (rl // 16) * wpr + j
                        tot = tot + lds_ld(
                            red, (ww * 64 + n + 16 * (r // 4)) * 4 + r % 4
                        )
                    emit(rl, n, tot)

        def emit_out(stride):
            def f(rl, n, v):
                lds_st(outs, n * stride + rl, v)

            return f

        def quant_mxfp8(a0, a1):
            """Per-32 E4M3 values and E8M0 scale used by AITER W4A8."""
            amax = fx.max(fmath.absf(a0), fmath.absf(a1))
            for offset in (1, 2, 4, 8):
                amax = _xred(amax, offset, fx.max)
            exponent, inverse = mxfp8_scale(amax)
            q0 = fx.min(fx.max(a0 * inverse, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(a1 * inverse, -FP8_MAX), FP8_MAX)
            return q0, q1, exponent

        def mxfp8_scale(amax):
            """E8M0 exponent of a 32-value block's absolute maximum, and its inverse."""
            bits = (amax * fx.Float32(1.0 / FP8_MAX)).bitcast(fx.Int32)
            exponent = (bits >> 23) & fx.Int32(0xFF)
            exponent = exponent + ((bits & fx.Int32(0x7FFFFF)) != 0).select(
                fx.Int32(1), fx.Int32(0)
            )
            exponent = fx.max(fx.min(exponent, fx.Int32(254)), fx.Int32(1))
            return exponent, _rcp((exponent << 23).bitcast(fx.Float32))

        def stage_xq(samples):
            """Poll the gate's packed FP8 activation + block scales of ``samples``
            into LDS."""
            nxw = HIDDEN // 4 // THREADS
            nscale = XQ_GROUPS
            got = poll(
                [
                    (mb("xq"), sx * (HIDDEN // 4) + tid + i * THREADS, 1)
                    for sx in samples
                    for i in range(nxw)
                ]
                + [
                    (mb("xqs"), sx * nscale + fx.min(tid, nscale - 1), 1)
                    for sx in samples
                ]
            )
            for j in range_constexpr(len(samples)):
                for i in range_constexpr(nxw):
                    wd = tid + i * THREADS
                    lds_st(
                        xs,
                        j * (HIDDEN // 4) + wd,
                        got[j * nxw + i][0].bitcast(fx.Float32),
                    )
                if tid < nscale:
                    lds_st(
                        misc,
                        8 + j * nscale + tid,
                        got[len(samples) * nxw + j][0].bitcast(fx.Float32),
                    )

        def st_mxfp8(k, q0, q1):
            """Store one row-major MXFP8 pair for the scaled MFMA."""
            w = (
                fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
                & 0xFFFF
            )
            nb = _xshfl(w, 1)
            if lane % 2 == 0:
                lds_st(xs, k // 4, (w | (nb << 16)).bitcast(fx.Float32))

        def candidate(i):
            """This lane's router candidate i: expert lane + 64 i, or None past N_EXPERTS."""
            return lane + 64 * i

        def load_bias():
            """This lane's candidate biases, bf16 as SGLang holds them (issue before the
            scores wait)."""
            return [
                bf16_round(ld_f32(_rsrc(bias), fx.min(candidate(i), N_EXPERTS - 1)))
                for i in range(ROUTER_CANDIDATES)
            ]

        def route_top8(s, bs):
            """Top-8 of sample s over 288 experts (one whole wave, after the scores).

            Every FP32 bit of (sigmoid + bias) is kept; equal keys go to the lower
            ``tie_rank``. Returns (expert id, route weight = raw score / sum of the 8 raw
            scores * ROUTE_SCALE) of pick ``lane``, valid in lanes < TOP_K."""
            raws = getf_many(
                [
                    (mb("scores"), s * N_EXPERTS + fx.min(candidate(i), N_EXPERTS - 1))
                    for i in range(ROUTER_CANDIDATES)
                ]
            )
            ks, ids, rks = [], [], []
            for i in range_constexpr(ROUTER_CANDIDATES):
                kb = (raws[i] + bs[i]).bitcast(fx.Int32)
                ok = (kb >= 0).select(kb ^ fx.Int32(-(2**31)), ~kb)
                valid = candidate(i) < N_EXPERTS
                ks.append(valid.select(fx.Uint32(ok), fx.Uint32(0)))
                ids.append(valid.select(candidate(i), fx.Int32(N_EXPERTS)))
                rks.append(valid.select(tie_rank(candidate(i)), fx.Int32(RANK_BASE)))
            for a, b in SORT5:
                first = (ks[a] > ks[b]) | ((ks[a] == ks[b]) & (rks[a] < rks[b]))
                ka, kb, ia, ib, ra, rb = ks[a], ks[b], ids[a], ids[b], rks[a], rks[b]
                ks[a], ks[b] = first.select(ka, kb), first.select(kb, ka)
                ids[a], ids[b] = first.select(ia, ib), first.select(ib, ia)
                rks[a], rks[b] = first.select(ra, rb), first.select(rb, ra)
            ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
            ids = ids + [fx.Int32(N_EXPERTS)]
            rks = rks + [fx.Int32(RANK_BASE)]
            mv = fx.Int32(0)  # lane k: the expert ID of pick k
            for k in range_constexpr(TOP_K):
                m = wave_umax(ks[0])
                winner = wave_umax(
                    (ks[0] == m).select(
                        ((RANK_BASE - rks[0]) << ID_BITS) | ids[0], fx.Int32(0)
                    )
                )
                expert = winner & fx.Int32((1 << ID_BITS) - 1)
                hit = (ks[0] == m) & (ids[0] == expert)
                n = ROUTER_CANDIDATES
                ks = [hit.select(ks[i + 1], ks[i]) for i in range(n)] + [ks[n]]
                ids = [hit.select(ids[i + 1], ids[i]) for i in range(n)] + [ids[n]]
                rks = [hit.select(rks[i + 1], rks[i]) for i in range(n)] + [rks[n]]
                mv = write_lane_i32(expert, k, mv)
            e = mv
            src = (e % 64) * 4
            got = [bpermute_i32(src, r.bitcast(fx.Int32)) for r in raws]
            raw = got[0]
            for i in range_constexpr(1, ROUTER_CANDIDATES):
                raw = (e // 64 == i).select(got[i], raw)
            raw = (lane < TOP_K).select(raw.bitcast(fx.Float32), fx.Float32(0.0))
            tot = raw
            for off in (1, 2, 4):
                tot = _xred(tot, off, lambda a, b: a + b)
            return e, raw * (_rcp(tot) * ROUTE_SCALE)

        def peer_reduce(region, t, out_fn, tile):
            """Push BF16 partials to every peer, then sum them in rank order. The
            thread's indices are its own (``fresh_i32``): at the launch's end, none is
            kept live from the attention push's."""
            tid_ = fresh_i32(tid)
            region_base = fx.Int64(SY[region]) + fx.Int64(peer_slot) * fx.Int64(
                SY["_part_stride"]
            )
            if const_expr(W > 1):
                if wave < W:
                    pair_count = S * tile // 2
                    for batch in range_constexpr((pair_count + 63) // 64):
                        pair = tid_ % 64 + batch * 64
                        if pair < pair_count:
                            si = pair // (tile // 2)
                            ri = (pair % (tile // 2)) * 2
                            put_bf(
                                peer_dst + region_base,
                                (rank * S + si) * HIDDEN + t * tile + ri,
                                [
                                    lds_ld(outs, si * tile + ri),
                                    lds_ld(outs, si * tile + ri + 1),
                                ],
                                CM_SYS,
                            )
                gpu.barrier()
            if tid_ < S * tile // 2:
                s = tid_ // (tile // 2)
                r = (tid_ % (tile // 2)) * 2
                row = t * tile + r
                v0 = lds_ld(outs, s * tile + r)
                v1 = lds_ld(outs, s * tile + r + 1)
                if const_expr(W == 1):  # no TP peers: the sum is the local value
                    parts = [(v0, v1)]
                else:
                    own = sym + region_base
                    specs = [
                        (own, ((src * S + s) * HIDDEN + row) // 2, 1)
                        for src in range(W)
                    ]
                    got = poll(specs, "one-as")
                    parts = [bf2_f32(v[0]) for v in got]
                t0 = fx.Float32(0.0)
                t1 = fx.Float32(0.0)
                for src in range_constexpr(W):
                    t0 = t0 + parts[src][0]
                    t1 = t1 + parts[src][1]
                out_fn(s, row, t0, t1)

        def start(name):
            return (bid + (G - base[name])) & (G - 1)

        def n_sel(count=S):
            """This lane's local MFMA B column; inactive columns duplicate the last one."""
            return fx.min(lane % 16, count - 1)

        # ========= 1. seam: attention TP reduce + mHC post -> R', partial mixes
        # One task a 32-column tile, every row: x = the TP sum of the attention
        # partials (bf16) or, on the attention side, the previous layer's output;
        # R'[j] = bf16(sum_i comb[i][j] R[i] + post[j] x) -> res_out and mailbox
        # ``rnew``, and the tile's share of the mix projection (fn . R') and of
        # sum(R'^2) -> mailbox ``pmix``.
        XA = 0
        RN = S * ROW_TILE
        RA = RN + S * HC_MULT * ROW_TILE
        NPAIR = S * ROW_TILE // 2
        MIX_A = HC_MULT * (1 + HC_MULT)
        # the tile's mix weights fn[k][j][col0 ..] at LDS row j * N_MIX + k; rows
        # 16-byte aligned, consecutive k on distinct banks
        HF = RA + S * HC_MULT * ROW_TILE
        HF_STRIDE = ROW_TILE + 4
        N_HF = N_MIX * HC_MULT * ROW_TILE // 4
        MIX_CHUNK = 16
        # a tile's mix partial per (row, output) is summed as MIX_SPLIT partials over
        # MIX_J streams each, then those in stream order
        MIX_PAIRS = S * PMIX
        MIX_SPLIT = max(1, min(HC_MULT, THREADS // MIX_PAIRS))
        MIX_J = HC_MULT // MIX_SPLIT
        assert MIX_J * MIX_SPLIT == HC_MULT and MIX_SPLIT * MIX_PAIRS <= RED_WORDS

        def seam_mix_loads(r_fn, col0):
            """A seam tile's mix weights: (their LDS quads, the loads)."""
            hf_q = [
                fx.min(tid + i * THREADS, N_HF - 1)
                for i in range((N_HF + THREADS - 1) // THREADS)
            ]
            hf_v = [
                fx.Vector(
                    bo.buffer_load(
                        r_fn,
                        (q // (ROW_TILE // 4)) * HIDDEN
                        + col0
                        + (q % (ROW_TILE // 4)) * 4,
                        vec_width=4,
                        dtype=T.f32,
                    )
                )
                for q in hf_q
            ]
            return hf_q, hf_v

        RN_ITEMS = S * HC_MULT * (ROW_TILE // 2)

        def seam_early():
            """Every load of this CTA's attention seam tile, none of which waits for
            anything: the tile's mix weights, this thread's x pair and its R' items' post,
            comb and R (a CTA without a tile loads the last tile's, unused)."""
            assert N_ROW_TILES <= G
            col0 = fx.min(start("aseam"), N_ROW_TILES - 1) * ROW_TILE
            q = fx.min(tid, NPAIR - 1)
            x = ld_pair(
                _rsrc(attn_part),
                (q // (ROW_TILE // 2)) * HIDDEN + col0 + (q % (ROW_TILE // 2)) * 2,
            )
            post = [
                seam_item_loads(fx.min(tid + rn_i * THREADS, RN_ITEMS - 1), col0)
                for rn_i in range((RN_ITEMS + THREADS - 1) // THREADS)
            ]
            return dict(hf=seam_mix_loads(_rsrc(hc_a_fn), col0), x=x, post=post)

        def seam_item_loads(item, col0):
            """post[j], comb[i][j] and R[i]'s column pair of one (row, stream j, column pair)
            item of a seam tile, from the previous layer's outputs."""
            s = item // (HC_MULT * (ROW_TILE // 2))
            j = (item // (ROW_TILE // 2)) % HC_MULT
            c = (item % (ROW_TILE // 2)) * 2
            r_res, r_comb = _rsrc(res_in), _rsrc(comb_in)
            pj = ld_f32(_rsrc(post_in), s * HC_MULT + j)
            cr = [
                (
                    ld_f32(r_comb, (s * HC_MULT + i) * HC_MULT + j),
                    ld_pair(r_res, (s * HC_MULT + i) * HIDDEN + col0 + c),
                )
                for i in range(HC_MULT)
            ]
            return pj, cr

        def run_seam(stage, mode, early=None):
            """``mode``: "ffn" (the caller's attention partials), "attn" (the previous
            layer's output, attention-side mHC) or "ffn_own" (this launch's o_proj
            partials, R / post / comb from its attention-side mHC). ``early``: the
            attention seam's loads, issued at launch start (``seam_early``)."""
            r_attn = _rsrc(attn_part)
            r_res = _rsrc(res_in)
            r_fn = _rsrc(hc_a_fn if mode == "attn" else hc_fn)
            r_post = _rsrc(post_in)
            r_comb = _rsrc(comb_in)
            rnew = mb("rnew_a" if mode == "attn" else "rnew")
            pmix = mb("pmix_a" if mode == "attn" else "pmix")
            # x arrives as the ranks' partials in a symmetric region: the attention
            # partials ("ffn": the caller's, pushed here; "ffn_own": this launch's), or
            # the layer input's partials (pushed here, their own region: a peer's FFN
            # reduce of the same launch and tag must not read them)
            x_region = "xin" if mode == "attn" and tokens else "attn"
            x_tok = mode == "attn" and tokens
            x_push = W > 1 and (mode == "ffn" or x_tok)
            x_plain = (mode == "attn" and not tokens) or (mode != "ffn_own" and W == 1)
            # the launch-start loads read x where a plain attention seam does
            assert early is None or (mode == "attn" and x_plain)
            if const_expr(x_tok):
                st = step_tab()
                r_ids, r_emb = _rsrc(st["ids"]), _rsrc(st["emb"])

            def x_word(si, c):
                """Bits of x's bf16 pair (row si, columns c, c + 1) as this rank holds
                them: its attention partial, or its embedding row of the row's token
                (zero for a token outside its shard, and for the width's rows past the
                step's)."""
                if const_expr(not x_tok):
                    return fx.Int32(
                        bo.buffer_load(
                            r_attn, (si * HIDDEN + c) // 2, vec_width=1, dtype=T.i32
                        )
                    )
                live = si < st["n_rows"]
                tok = ld_i32(r_ids, fx.min(si, st["n_rows"] - 1) * st["ids_words"])
                own = live & (tok >= st["vocab_lo"]) & (tok < st["vocab_hi"])
                erow = own.select(tok - st["vocab_lo"], fx.Int32(0))
                word = fx.Int32(
                    bo.buffer_load(
                        r_emb, (erow * HIDDEN + c) // 2, vec_width=1, dtype=T.i32
                    )
                )
                return own.select(word, fx.Int32(0))

            def seam_post(item, col0, pre=None):
                """R'[j] for one (row, stream, column pair) item of a seam tile."""
                s = item // (HC_MULT * (ROW_TILE // 2))
                j = (item // (ROW_TILE // 2)) % HC_MULT
                c = (item % (ROW_TILE // 2)) * 2
                if const_expr(mode == "ffn_own"):
                    pj = lds_ld(misc, s * MIX_A + j)
                elif const_expr(pre is not None):
                    pj = pre[0]
                else:
                    pj = ld_f32(r_post, s * HC_MULT + j)
                acc0 = fx.Float32(0.0)
                acc1 = fx.Float32(0.0)
                for i in range_constexpr(HC_MULT):
                    if const_expr(mode == "ffn_own"):
                        cij = lds_ld(misc, s * MIX_A + HC_MULT + i * HC_MULT + j)
                        ra = RA + (s * HC_MULT + i) * ROW_TILE + c
                        r0, r1 = lds_ld(xs, ra), lds_ld(xs, ra + 1)
                    elif const_expr(pre is not None):
                        cij, (r0, r1) = pre[1][i]
                    else:
                        cij = ld_f32(r_comb, (s * HC_MULT + i) * HC_MULT + j)
                        r0, r1 = ld_pair(r_res, (s * HC_MULT + i) * HIDDEN + col0 + c)
                    acc0 = acc0 + cij * r0
                    acc1 = acc1 + cij * r1
                n0 = bf16_round(acc0 + pj * lds_ld(xs, XA + s * ROW_TILE + c))
                n1 = bf16_round(acc1 + pj * lds_ld(xs, XA + s * ROW_TILE + c + 1))
                elem = (s * HC_MULT + j) * HIDDEN + col0 + c
                if const_expr(mode != "attn"):
                    bo.buffer_store(
                        fx.Vector.from_elements([n0, n1], fx.Float32).to(fx.BFloat16),
                        _rsrc(res_out),
                        elem,
                    )
                put(rnew, elem // 2, bf16_pair(n0, n1))
                lds_st(xs, RN + (s * HC_MULT + j) * ROW_TILE + c, n0)
                lds_st(xs, RN + (s * HC_MULT + j) * ROW_TILE + c + 1, n1)

            for t in range(start(stage), N_ROW_TILES, G):
                t = fx.Int32(t)
                col0 = t * ROW_TILE
                attn_base = fx.Int64(SY[x_region]) + fx.Int64(peer_slot) * fx.Int64(
                    SY["_part_stride"]
                )
                # the tile's mix weights, issued before the wait
                hf_q, hf_v = (
                    early["hf"] if early is not None else seam_mix_loads(r_fn, col0)
                )
                if const_expr(x_push):
                    if wave < W:
                        for batch in range_constexpr((NPAIR + 63) // 64):
                            pair = lane + batch * 64
                            if pair < NPAIR:
                                si = pair // (ROW_TILE // 2)
                                ci = (pair % (ROW_TILE // 2)) * 2
                                word = x_word(si, col0 + ci)
                                bo.buffer_store(
                                    fx.Vector.from_elements([word, tag], fx.Int32),
                                    _rsrc(peer_dst + attn_base),
                                    (rank * S + si) * HIDDEN + col0 + ci,
                                    cache_modifier=CM_SYS,
                                )
                    gpu.barrier()
                if const_expr(mode == "ffn_own"):
                    NRA = S * HC_MULT * ROW_TILE // 2
                    for ra_i in range_constexpr((NRA + THREADS - 1) // THREADS):
                        q = tid + ra_i * THREADS
                        if q < NRA:
                            sj = q // (ROW_TILE // 2)
                            c = (q % (ROW_TILE // 2)) * 2
                            a0, a1 = bf2_f32(
                                get(mb("rnew_a"), (sj * HIDDEN + col0 + c) // 2)
                            )
                            lds_st(xs, RA + sj * ROW_TILE + c, a0)
                            lds_st(xs, RA + sj * ROW_TILE + c + 1, a1)
                    if tid < S * MIX_A:
                        lds_st(misc, tid, getf_many([(mb("mix_a"), tid)])[0])
                # one x pair a thread, in passes past THREADS pairs (rows > 32)
                for xi in range_constexpr((NPAIR + THREADS - 1) // THREADS):
                    q = tid + xi * THREADS
                    if q < NPAIR:
                        s = q // (ROW_TILE // 2)
                        c = (q % (ROW_TILE // 2)) * 2
                        if const_expr(early is not None):
                            v0, v1 = early["x"]
                        elif const_expr(x_plain):
                            v0, v1 = ld_pair(r_attn, s * HIDDEN + col0 + c)
                        else:
                            own = sym + attn_base
                            got = poll(
                                [
                                    (own, ((src * S + s) * HIDDEN + col0 + c) // 2, 1)
                                    for src in range(W)
                                ],
                                "one-as",
                            )
                            v0 = fx.Float32(0.0)
                            v1 = fx.Float32(0.0)
                            for src in range_constexpr(W):
                                a0, a1 = bf2_f32(got[src][0])
                                v0 = v0 + a0
                                v1 = v1 + a1
                            v0, v1 = bf16_round(v0), bf16_round(v1)
                        lds_st(xs, XA + s * ROW_TILE + c, v0)
                        lds_st(xs, XA + s * ROW_TILE + c + 1, v1)
                for i in range_constexpr(len(hf_q)):
                    hrow = hf_q[i] // (ROW_TILE // 4)
                    fx.ptr_store(
                        hf_v[i],
                        xs
                        + HF
                        + ((hrow % HC_MULT) * N_MIX + hrow // HC_MULT) * HF_STRIDE
                        + (hf_q[i] % (ROW_TILE // 4)) * 4,
                    )
                gpu.barrier()
                for rn_i in range_constexpr((RN_ITEMS + THREADS - 1) // THREADS):
                    item = tid + rn_i * THREADS
                    if item < RN_ITEMS:
                        seam_post(
                            item, col0, None if early is None else early["post"][rn_i]
                        )
                gpu.barrier()
                for mi in range_constexpr(
                    (MIX_SPLIT * MIX_PAIRS + THREADS - 1) // THREADS
                ):
                    mt = tid + mi * THREADS
                    if mt < MIX_SPLIT * MIX_PAIRS:
                        pair = mt % MIX_PAIRS
                        split = mt // MIX_PAIRS
                        s = pair // PMIX
                        k = pair % PMIX
                        is_mix = k < N_MIX
                        krow = HF + fx.min(k, N_MIX - 1) * HF_STRIDE
                        acc = fx.Float32(0.0)
                        for mch in range(
                            fx.Int32(0),
                            fx.Int32(MIX_J * ROW_TILE // MIX_CHUNK),
                            fx.Int32(1),
                            unroll=1,
                        ):
                            mj = split * MIX_J + mch // (ROW_TILE // MIX_CHUNK)
                            mc0 = (mch % (ROW_TILE // MIX_CHUNK)) * MIX_CHUNK
                            mv = xs + RN + (s * HC_MULT + mj) * ROW_TILE + mc0
                            mw = xs + krow + mj * (N_MIX * HF_STRIDE) + mc0
                            for mq in range_constexpr(MIX_CHUNK // 4):
                                v4 = fx.Vector(
                                    fx.ptr_load(mv + 4 * mq, result_type=v4f)
                                )
                                w4 = fx.Vector(
                                    fx.ptr_load(mw + 4 * mq, result_type=v4f)
                                )
                                for me in range_constexpr(4):
                                    acc = acc + v4[me] * is_mix.select(w4[me], v4[me])
                        lds_st(red, split * MIX_PAIRS + pair, acc)
                gpu.barrier()
                for mi in range_constexpr((MIX_PAIRS + THREADS - 1) // THREADS):
                    mt = tid + mi * THREADS
                    if mt < MIX_PAIRS:
                        tot = lds_ld(red, mt)
                        for p_ in range_constexpr(1, MIX_SPLIT):
                            tot = tot + lds_ld(red, p_ * MIX_PAIRS + mt)
                        put(
                            pmix,
                            ((mt // PMIX) * N_ROW_TILES + t) * PMIX + mt % PMIX,
                            tot,
                        )
                gpu.barrier()

        # ========= 2. gate: mixes -> pre / post / comb (Sinkhorn), layer input row
        # One CTA a row. The tile partials are summed in a fixed order, so every
        # rank computes bit-identical mixes. The layer input row needs only pre, so
        # it is published first; post and comb (Sinkhorn) follow.
        NP = 16
        TPP = N_ROW_TILES // NP
        CPT = HIDDEN // THREADS
        # misc words of the mix scales (3) and biases (N_MIX), past the mix sums
        SCW = 32
        HBW = SCW + 4
        # a row's mix partials (every tile's), staged in LDS words 0 .. PM_WORDS
        PM_WORDS = N_ROW_TILES * PMIX
        PM_LOADS = PM_WORDS // 2
        assert PM_WORDS % 2 == 0 and PM_WORDS + HBW + N_MIX <= X_WORDS

        def gate_early():
            """The attention gate's constants, loaded at launch start: this thread's norm
            weights, a mix scale and a mix bias (clamped thread indices)."""
            gam = ld_bf16s(_rsrc(g_in), tid * CPT, CPT)
            sc = ld_f32(_rsrc(hc_a_scale), fx.min(tid, 2))
            bs = ld_f32(_rsrc(hc_a_base), fx.min(tid, N_MIX - 1))
            return gam, sc, bs

        def gate_row(stage, dep, s, attn, sums, early=None):
            """Row ``s``'s mixes and its RMSNorm input (``stage`` waits on the seam
            ``dep``): (bf16 pre-norm values of this thread's CPT columns, rstd, norm
            weights, the mix rinv). The mix sums, scales and biases go to LDS ``sums``.
            ``early``: the constants ``gate_early`` loaded."""
            r_scale = _rsrc(hc_a_scale if attn else hc_scale)
            r_base = _rsrc(hc_a_base if attn else hc_base)
            r_g = _rsrc(g_in if attn else g_norm)
            pmix = mb("pmix_a" if attn else "pmix")
            rnew = mb("rnew_a" if attn else "rnew")
            if const_expr(early is not None):
                gam, sc, bs = early
                if tid < 3:
                    lds_st(sums, SCW + tid, sc)
                if tid < N_MIX:
                    lds_st(sums, HBW + tid, bs)
            else:
                gam = ld_bf16s(r_g, tid * CPT, CPT)
                if tid < 3:
                    lds_st(sums, SCW + tid, ld_f32(r_scale, tid))
                if tid < N_MIX:
                    lds_st(sums, HBW + tid, ld_f32(r_base, tid))
            # each thread's streams come from one seam tile, ahead of the
            # partials: wait for them first, as the tiles finish
            got = poll(
                [
                    (
                        rnew,
                        ((s * HC_MULT + i) * HIDDEN + tid * CPT + 4 * q) // 2,
                        2,
                    )
                    for i in range(HC_MULT)
                    for q in range(CPT // 4)
                ]
            )
            # every tile's mix partials of the row -> LDS words 0 .., two pairs a load
            pm = [
                fx.min(tid + i * THREADS, PM_LOADS - 1)
                for i in range(-(-PM_LOADS // THREADS))
            ]
            got_pm = poll([(pmix, s * N_ROW_TILES * PMIX + 2 * q, 2) for q in pm])
            for i, q in enumerate(pm):
                for e in range_constexpr(2):
                    lds_st(xs, 2 * q + e, got_pm[i][e].bitcast(fx.Float32))
            gpu.barrier()
            if tid < PMIX * NP:
                k = tid // NP
                part = tid % NP
                tot = lds_ld(xs, part * TPP * PMIX + k)
                for q in range_constexpr(1, TPP):
                    tot = tot + lds_ld(xs, (part * TPP + q) * PMIX + k)
                lds_st(red, k * NP + part, tot)
            gpu.barrier()
            if tid < PMIX:
                tot = lds_ld(red, tid * NP)
                for p_ in range_constexpr(1, NP):
                    tot = tot + lds_ld(red, tid * NP + p_)
                lds_st(sums, tid, tot)
            gpu.barrier()
            rinv = _rsq(lds_ld(sums, N_MIX) * (1.0 / (HC_MULT * HIDDEN)) + EPS)
            pre = [
                sigmoid(
                    lds_ld(sums, i) * rinv * lds_ld(sums, SCW) + lds_ld(sums, HBW + i)
                )
                + HC_EPS
                for i in range(HC_MULT)
            ]
            li = []
            for q in range_constexpr(CPT // 4):
                for h in range_constexpr(2):
                    for e in range_constexpr(2):
                        v = fx.Float32(0.0)
                        for i in range_constexpr(HC_MULT):
                            word = got[i * (CPT // 4) + q][h]
                            v = v + pre[i] * bf2_f32(word)[e]
                        li.append(bf16_round(v))
            ss = fx.Float32(0.0)
            for v in li:
                ss = ss + v * v
            rstd = _rsq(block_sum(ss) * (1.0 / HIDDEN) + EPS)
            return li, rstd, gam, rinv

        def run_gate(stage, mode, early=None):
            """``mode`` "attn" writes post / comb to mailbox ``mix_a`` and the
            attention input to ``a_in``; "ffn" writes them out and the FFN input
            to ``a``. ``early``: the attention gate's constants (``gate_early``)."""
            attn = mode == "attn"
            a_out = mb("a_in" if attn else "a")
            for t in range(start(stage), S, G):
                s = fx.Int32(t)
                li, rstd, gam, rinv = gate_row(
                    stage, "aseam" if attn else "seam", s, attn, misc, early
                )
                pairs = [
                    bf16_pair(
                        li[2 * q] * rstd * gam[2 * q],
                        li[2 * q + 1] * rstd * gam[2 * q + 1],
                    )
                    for q in range(CPT // 2)
                ]
                if const_expr(attn):
                    # plain words, released by their drain before the box word
                    bo.buffer_store(
                        fx.Vector.from_elements(
                            [p.bitcast(fx.Int32) for p in pairs], fx.Int32
                        ),
                        _rsrc(a_out),
                        (s * HIDDEN + tid * CPT) // 2,
                        cache_modifier=CM_DEV,
                    )
                else:
                    for q in range_constexpr(CPT // 2):
                        put(a_out, (s * HIDDEN + tid * CPT + 2 * q) // 2, pairs[q])
                if const_expr(not attn and not dense):
                    # the row's MXFP8 blocks for the experts: a 32-value block is 4
                    # threads' values (as rounded to bf16 in the row)
                    av = [bf16_round(li[q] * rstd * gam[q]) for q in range(CPT)]
                    amax = fmath.absf(av[0])
                    for q in range_constexpr(1, CPT):
                        amax = fx.max(amax, fmath.absf(av[q]))
                    for offset in (1, 2):
                        amax = _xred(amax, offset, fx.max)
                    exponent, inverse = mxfp8_scale(amax)
                    qv = [fx.min(fx.max(v * inverse, -FP8_MAX), FP8_MAX) for v in av]
                    for h in range_constexpr(CPT // 4):
                        put(
                            mb("xq"),
                            (s * HIDDEN + tid * CPT + 4 * h) // 4,
                            fp8_pack4(*qv[4 * h : 4 * h + 4]),
                        )
                    if tid % (32 // CPT) == 0:
                        put(mb("xqs"), s * XQ_GROUPS + tid // (32 // CPT), exponent)
                # the row's word in every CTA's box, after every thread's stores of it
                if const_expr(attn):
                    rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                if _uniform(wave) == 0:
                    signal("box_a_in" if attn else "box_a", s)
                if wave == 0:
                    # lane l of each 16-lane row holds comb[l // 4][l % 4]: a row is a
                    # quad, a column one lane of each quad; sums keep the serial order
                    el = lane % (HC_MULT * HC_MULT)

                    def quad_sum(v):
                        tot = dpp_f32(v, v, dpp_quad_bcast(0))
                        for j in range_constexpr(1, HC_MULT):
                            tot = tot + dpp_f32(v, v, dpp_quad_bcast(j))
                        return tot

                    def col_sum(v):
                        # the last quad sums its column, then the other quads copy it
                        tot = dpp_f32(v, v, dpp_row_shr(12), 0x8)
                        for i in range_constexpr(1, HC_MULT - 1):
                            tot = tot + dpp_f32(v, v, dpp_row_shr(12 - 4 * i), 0x8)
                        tot = tot + v + HC_EPS
                        out = tot
                        for i in range_constexpr(HC_MULT - 1):
                            out = dpp_f32(out, tot, dpp_row_shl(12 - 4 * i), 1 << i)
                        return out

                    sc1 = lds_ld(misc, SCW + 1)
                    sc2 = lds_ld(misc, SCW + 2)
                    pi = lane % HC_MULT
                    post_v = (
                        sigmoid(
                            lds_ld(misc, HC_MULT + pi) * rinv * sc1
                            + lds_ld(misc, HBW + HC_MULT + pi)
                        )
                        * HC_POST_MULT
                    )
                    if lane < HC_MULT:
                        if const_expr(attn):
                            put(mb("mix_a"), s * MIX_A + lane, post_v)
                        else:
                            bo.buffer_store(post_v, _rsrc(post_out), s * HC_MULT + lane)
                    o = 2 * HC_MULT
                    x = lds_ld(misc, o + el) * rinv * sc2 + lds_ld(misc, HBW + o + el)
                    mx = dpp_f32(x, x, dpp_quad_bcast(0))
                    for j in range_constexpr(1, HC_MULT):
                        mx = fx.max(mx, dpp_f32(x, x, dpp_quad_bcast(j)))
                    ex = _exp(x - mx)
                    x = div(ex, quad_sum(ex)) + HC_EPS
                    x = div(x, col_sum(x))
                    for _ in range_constexpr(SINKHORN_ITERS - 1):
                        x = div(x, quad_sum(x) + HC_EPS)
                        x = div(x, col_sum(x))
                    if lane < HC_MULT * HC_MULT:
                        if const_expr(attn):
                            put(mb("mix_a"), s * MIX_A + HC_MULT + lane, x)
                        else:
                            bo.buffer_store(
                                x, _rsrc(comb_out), s * HC_MULT * HC_MULT + lane
                            )
                gpu.barrier()

        # ================================================= KDA attention stages
        KD = KDA_DIM
        XW = HIDDEN // 2
        CONV_TAPS = 4
        # 16-byte polls a thread holds at once when staging a projection's rows
        STAGE_GROUP = 8

        def unit_rm(w_rsrc, row0, kc, K, rows, b_word):
            """One 64-K bf16 MFMA unit of a row-major [rows, K] weight; rows past the
            end load the last row and are dropped by the caller."""
            r = fx.min(row0 + lane % 16, rows - 1)
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        (r * K + kc * 64 + sp * 32 + (lane // 16) * 8) // 2,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for sp in range(2)
            ]
            return ("bf16", wv, None, b_word + (lane // 16) * 4)

        def stage_rows(box, K):
            """Poll every row's bf16 [K] vector of mailbox ``box`` into LDS words
            ``n * K / 2 ..``, the B operand layout of ``unit_rm``; at most
            ``STAGE_GROUP`` loads a thread are held at once."""
            nq = S * K // 4
            full = nq // THREADS
            for g0 in range_constexpr(0, full, STAGE_GROUP):
                group = range(g0, min(g0 + STAGE_GROUP, full))
                got = poll([(box, (tid + i * THREADS) * 2, 2) for i in group])
                for k, i in enumerate(group):
                    for j in range_constexpr(2):
                        lds_st(
                            xs,
                            (tid + i * THREADS) * 2 + j,
                            got[k][j].bitcast(fx.Float32),
                        )
            if const_expr(nq % THREADS):
                q = tid + full * THREADS
                if q < nq:
                    v = poll([(box, q * 2, 2)])[0]
                    for j in range_constexpr(2):
                        lds_st(xs, q * 2 + j, v[j].bitcast(fx.Float32))

        def stage_plain(base, K):
            """``stage_rows`` for a mailbox of plain words (packed bf16 pairs) its box
            releases: one read, 16 bytes a load."""
            nl = S * K // 8 // THREADS
            assert nl * THREADS * 8 == S * K
            r = _rsrc(base)
            for g0 in range_constexpr(0, nl, STAGE_GROUP):
                group = range(g0, min(g0 + STAGE_GROUP, nl))
                got = [
                    fx.Vector(
                        bo.buffer_load(
                            r,
                            (tid + i * THREADS) * 4,
                            vec_width=4,
                            dtype=T.i32,
                            cache_modifier=CM_DEV,
                        )
                    )
                    for i in group
                ]
                for k, i in enumerate(group):
                    fx.ptr_store(
                        got[k].bitcast(fx.Float32), xs + (tid + i * THREADS) * 4
                    )

        def wait_ready(box, row=None):
            """Wave 0 waits for every row's word (or row ``row``'s) in this CTA's
            line of ``box`` (a broadcast row's word is set after its stores), so the
            payload's own poll is read once; the barrier holds the other waves."""
            if _uniform(wave) == 0:
                if const_expr(row is None):
                    words = [
                        (ln, 0, min(BOX_WORDS, S - ln * BOX_WORDS))
                        for ln in range(box_lines(S))
                    ]
                else:
                    words = [(row // BOX_WORDS, row % BOX_WORDS, 1)]
                await_box(box, words)
            gpu.barrier()

        # ===================== 3. merged q | k | v | beta | f_a | g_a projection
        # One task a 16-row tile of this rank's row-major bf16 weight, every row;
        # outputs rounded to bf16 as vLLM's GEMM stores them.
        IN_CPW = HIDDEN // 64 // WAVES

        N_IN = (IN_ROWS + IN_TILE - 1) // IN_TILE
        # the beta, f_a and g_a tiles (rows 3P ..) run first, in the first round
        IN_FIRST = 3 * P // IN_TILE

        # the tasks run on every CTA but the attention gate's S, the S slots before
        # the stage's first: those finish the gate's post / comb meanwhile
        IN_G = G - S
        # tasks a CTA runs at most. A task's first IN_AHEAD units a wave are issued
        # ahead: the first task's before the wait, a later task's before the previous
        # one reduces; all of them up to 4 rows, half at 8 (registers)
        IN_ROUNDS = (N_IN + IN_G - 1) // IN_G
        IN_AHEAD = IN_CPW if S <= 4 else IN_CPW // 2
        # units of a CTA's second task issued with its first task's, before the wait
        # (registers: all of them at 2 rows, 6 of 8 at 4, none at 8)
        IN_EARLY = IN_AHEAD if S <= 2 else (IN_AHEAD * 3 // 4 if S <= 4 else 0)

        def run_inproj():
            r_w = _rsrc(w_in)
            t0 = start("inproj")

            def row0_of(t):
                tile = t + IN_FIRST
                return (tile >= N_IN).select(tile - N_IN, tile) * IN_TILE

            def unit_of(row0, c):
                kc = wave * IN_CPW + c
                return unit_rm(r_w, row0, kc, HIDDEN, IN_ROWS, n_sel() * XW + kc * 32)

            def ahead_of(t, lo=0, hi=IN_AHEAD):
                # past the last task: the last row only (dropped)
                row0 = (t < N_IN).select(row0_of(t), fx.Int32(IN_ROWS))
                return [unit_of(row0, c) for c in range(lo, hi)]

            def task(k, t, pre, early=()):
                row0 = row0_of(t)
                # the CTA's first task stages every row; a later task reuses them
                if const_expr(k == 0):
                    wait_ready("box_a_in")
                    stage_plain(mb("a_in"), HIDDEN)
                    gpu.barrier()
                acc = run_units(lambda c: unit_of(row0, c), IN_CPW, IN_AHEAD, pre)
                nxt = (
                    list(early) + ahead_of(t + IN_G, len(early))
                    if k + 1 < IN_ROUNDS
                    else None
                )

                def emit(rl, n, v):
                    row = row0 + rl
                    if row < IN_ROWS:
                        put(mb("proj"), n * IN_ROWS + row, bf16_round(v))

                reduce_rows(1, acc, emit)
                gpu.barrier()
                if const_expr(k + 1 < IN_ROUNDS):
                    if t + IN_G < N_IN:
                        task(k + 1, t + IN_G, nxt)

            if t0 < min(N_IN, IN_G):
                pre = ahead_of(t0)
                early = []
                if const_expr(IN_ROUNDS > 1 and IN_EARLY > 0):
                    early = ahead_of(t0 + IN_G, 0, IN_EARLY)
                task(0, t0, pre, early)

        # ===================== 4. f_b / g_b: the per-channel decay and output gates
        # One task 128 rows (a 16-row tile a wave, K = 128) of f_b or of g_b.
        FA = 3 * P + H

        def run_fgb():
            for t in range(start("fgb"), 2 * P // FGB_TILE, G):
                t = fx.Int32(t)
                half = t >= P // FGB_TILE
                src = FA + half.select(fx.Int32(KD), fx.Int32(0))
                # the task's weights load before its wait
                row0 = t * FGB_TILE + wave * 16
                r_w = _rsrc(half.select(w_gb, w_fb))
                lr = row0 - half.select(fx.Int32(P), fx.Int32(0))
                units = [
                    unit_rm(r_w, lr, kc, KD, P, n_sel() * (KD // 2) + kc * 32)
                    for kc in range(KD // 64)
                ]
                for pi in range_constexpr((S * KD // 2 + THREADS - 1) // THREADS):
                    q = tid + pi * THREADS
                    if q < S * KD // 2:
                        n = q // (KD // 2)
                        w = q % (KD // 2)
                        v0, v1 = get2_many([(mb("proj"), n * IN_ROWS + src + 2 * w)])[0]
                        lds_st(xs, n * (KD // 2) + w, bf16_pair(v0, v1))
                gpu.barrier()
                acc = mma_units([fx.Float32(0.0) for _ in range(4)], units)
                # C[r][n] sits in lane n + 16 (r // 4), element r % 4
                if lane % 16 < S:
                    for e in range_constexpr(4):
                        put(
                            mb("gates"),
                            (lane % 16) * 2 * P + row0 + 4 * (lane // 16) + e,
                            bf16_round(acc[e]),
                        )
                gpu.barrier()

        # ===================== 5. short conv + gated delta rule + gated RMSNorm
        # One task a (row, head). A row whose state index is not positive (padding)
        # writes no state and outputs zeros.
        QO, KO, VO, G1O, G2O, OO, DO = 0, KD, 2 * KD, 3 * KD, 4 * KD, 5 * KD, 6 * KD
        # at 2 rows (TP4 and TP8: slower at TP2) a kda CTA runs its o_proj task right
        # after the head's output, then stores the new states; at 8 rows the states also
        # follow the output. The conv inputs wait in LDS words XO .. (with o_proj, past
        # its staged rows)
        OPROJ_IN_KDA = S <= 2 and W > 2 and not VERIFY
        STATES_LAST = OPROJ_IN_KDA or S > 4
        XO = X_WORDS - 3 * KD if OPROJ_IN_KDA else 7 * KD
        assert not OPROJ_IN_KDA or XO >= max(7 * KD, S * P // 2)

        def conv_at(ch, j):
            """Element of channel ``ch``, tap ``j`` in a conv state slot, laid out
            [3P, taps] (dim first) or [taps, 3P]."""
            if const_expr(conv_dim_first):
                return ch * (CONV_TAPS - 1) + j
            return j * (3 * P) + ch

        def kda_slot(r_idx, s):
            """SGLang's KDA state index of row ``s`` (a token launch's rows past the
            step's are pads: 0)."""
            if const_expr(tokens):
                n_rows = step_rows()
                idx = _uniform(
                    fx.Int32(
                        bo.buffer_load(
                            r_idx, fx.min(s, n_rows - 1), vec_width=1, dtype=T.i32
                        )
                    )
                )
                idx = (s < n_rows).select(idx, fx.Int32(0))
            else:
                idx = _uniform(
                    fx.Int32(bo.buffer_load(r_idx, s, vec_width=1, dtype=T.i32))
                )
            return idx

        def run_kda_verify():
            """A verify step: one task a (request, head) runs the request's VL rows
            in order from its committed state, which it leaves as is; after row j it
            writes the state and the conv window to step j of the request's verify
            scratch row (its batch position), as SGLang's verify does. A pad request
            (state index not positive) writes nothing and outputs zeros, as do the
            width's rows past its whole requests."""
            r_cw = _rsrc(conv_w)
            r_alog = _rsrc(a_log)
            r_dtb = _rsrc(dt_bias)
            r_won = _rsrc(w_on)
            r_idx = _rsrc(state_idx)
            for t in range(start("kda"), S * H, G):
                t = fx.Int32(t)
                s = t // H
                h = t % H
                if (s % VL == 0) & (s < NREQ * VL):
                    idx = kda_slot(r_idx, s)
                    valid = idx > 0
                    slot = fx.Int64(fx.max(idx, fx.Int32(0)))
                    req = fx.Int64(s // VL)
                    r_cs = _rsrc(conv_state + slot * conv_stride * fx.Int64(2))
                    ch = (fx.min(tid, 3 * KD - 1) // KD) * P + h * KD + tid % KD
                    gi = fx.min(tid, 2 * KD - 1)
                    st = [ld_bf16(r_cs, conv_at(ch, j)) for j in range(CONV_TAPS - 1)]
                    cw = [ld_f32(r_cw, ch * CONV_TAPS + j) for j in range(CONV_TAPS)]
                    kch = tid % KD
                    dtb = ld_f32(r_dtb, h * KD + kch)
                    amp = _exp(ld_f32(r_alog, h))
                    won = [ld_f32(r_won, 2 * lane + e) for e in range(2)]
                    vr = tid // 4
                    k0 = (tid % 4) * 32
                    r_rs = _rsrc(
                        rec_state
                        + (slot * rec_stride + fx.Int64(h * KD * KD)) * fx.Int64(4)
                    )
                    hv = []
                    for q in range_constexpr(8):
                        w4 = fx.Vector(
                            bo.buffer_load(
                                r_rs, vr * KD + k0 + 4 * q, vec_width=4, dtype=T.f32
                            )
                        )
                        hv += [w4[e] for e in range(4)]
                    for j in range_constexpr(VL):
                        row = s + j
                        # every thread polls (the channel and gate clamped), so the
                        # conv window carries to the next row outside a branch
                        x, g, b_raw = getf_many(
                            [
                                (mb("proj"), row * IN_ROWS + ch),
                                (
                                    mb("gates"),
                                    row * 2 * P + (gi // KD) * P + h * KD + gi % KD,
                                ),
                                (mb("proj"), row * IN_ROWS + 3 * P + h),
                            ]
                        )
                        a = cw[CONV_TAPS - 1] * x
                        for k in range_constexpr(CONV_TAPS - 1):
                            a = a + cw[k] * st[k]
                        st = st[1:] + [x]
                        if tid < 3 * KD:
                            lds_st(xs, tid, bf16_round(div(a, 1.0 + _exp(-a))))
                            if tid < 2 * KD:
                                lds_st(xs, G1O + tid, g)
                            if tid == 0:
                                lds_st(misc, 0, b_raw)
                            if valid:
                                r_icw = _rsrc(
                                    inter_conv
                                    + (
                                        req * inter_conv_stride
                                        + fx.Int64(j * (CONV_TAPS - 1) * 3 * P)
                                    )
                                    * fx.Int64(2)
                                )
                                for k in range_constexpr(CONV_TAPS - 1):
                                    bo.buffer_store(
                                        fx.Float32(st[k]).to(fx.BFloat16),
                                        r_icw,
                                        conv_at(ch, k),
                                    )
                        gpu.barrier()
                        if wave < 2:
                            a0 = lds_ld(xs, wave * KD + lane)
                            a1 = lds_ld(xs, wave * KD + lane + 64)
                            ss = wave_sum(a0 * a0 + a1 * a1)
                            if lane == 0:
                                lds_st(misc, 1 + wave, ss)
                        if (wave >= 2) & (wave < 4):
                            gk = lds_ld(xs, G1O + kch) + dtb
                            lds_st(
                                xs,
                                DO + kch,
                                _exp(div(KDA_LOWER_BOUND, 1.0 + _exp(-(amp * gk)))),
                            )
                        gpu.barrier()
                        rq = _rsq(lds_ld(misc, 1) + L2_EPS) * (KD**-0.5)
                        rk = _rsq(lds_ld(misc, 2) + L2_EPS)
                        beta = sigmoid(lds_ld(misc, 0))
                        kk = [lds_ld(xs, KO + k0 + i) * rk for i in range(32)]
                        dot = fx.Float32(0.0)
                        for i in range_constexpr(32):
                            hv[i] = hv[i] * lds_ld(xs, DO + k0 + i)
                            dot = dot + hv[i] * kk[i]
                        for off in (1, 2):
                            dot = _xred(dot, off, lambda a, b: a + b)
                        vn = beta * (lds_ld(xs, VO + vr) - dot)
                        o = fx.Float32(0.0)
                        for i in range_constexpr(32):
                            hv[i] = hv[i] + vn * kk[i]
                            o = o + hv[i] * (lds_ld(xs, QO + k0 + i) * rq)
                        for off in (1, 2):
                            o = _xred(o, off, lambda a, b: a + b)
                        if valid:
                            r_irs = _rsrc(
                                inter_rec
                                + (
                                    req * inter_rec_stride
                                    + fx.Int64((j * H + h) * KD * KD)
                                )
                                * fx.Int64(4)
                            )
                            for q in range_constexpr(8):
                                bo.buffer_store(
                                    fx.Vector.from_elements(
                                        hv[4 * q : 4 * q + 4], fx.Float32
                                    ),
                                    r_irs,
                                    vr * KD + k0 + 4 * q,
                                )
                        if tid % 4 == 0:
                            lds_st(
                                xs,
                                OO + vr,
                                valid.select(bf16_round(o), fx.Float32(0.0)),
                            )
                        gpu.barrier()
                        if wave == 0:
                            e0 = 2 * lane
                            o0 = lds_ld(xs, OO + e0)
                            o1 = lds_ld(xs, OO + e0 + 1)
                            rstd = _rsq(
                                wave_sum(o0 * o0 + o1 * o1) * (1.0 / KD) + O_NORM_EPS
                            )
                            y0 = o0 * rstd * won[0] * sigmoid(lds_ld(xs, G2O + e0))
                            y1 = o1 * rstd * won[1] * sigmoid(lds_ld(xs, G2O + e0 + 1))
                            put(
                                mb("o_in"),
                                (row * P + h * KD + e0) // 2,
                                bf16_pair(y0, y1),
                            )
                            signal("box_o_in", row * H + h)
                        gpu.barrier()
                if const_expr(NREQ * VL < S):
                    if s >= NREQ * VL:
                        if wave == 0:
                            put(
                                mb("o_in"),
                                (s * P + h * KD + 2 * lane) // 2,
                                bf16_pair(fx.Float32(0.0), fx.Float32(0.0)),
                            )
                            signal("box_o_in", t)

        def run_kda(then):
            """``then``: the CTA's o_proj task, run between the head's output and its
            state write-back with ``OPROJ_IN_KDA``."""
            r_cw = _rsrc(conv_w)
            r_alog = _rsrc(a_log)
            r_dtb = _rsrc(dt_bias)
            r_won = _rsrc(w_on)
            r_idx = _rsrc(state_idx)
            for t in range(start("kda"), S * H, G):
                t = fx.Int32(t)
                s = t // H
                h = t % H
                idx = kda_slot(r_idx, s)
                valid = idx > 0
                slot = fx.Int64(fx.max(idx, fx.Int32(0)))
                r_cs = _rsrc(conv_state + slot * conv_stride * fx.Int64(2))
                # every load that does not depend on the projections, before the wait
                ch = (fx.min(tid, 3 * KD - 1) // KD) * P + h * KD + tid % KD
                st = [ld_bf16(r_cs, conv_at(ch, j)) for j in range(CONV_TAPS - 1)]
                cw = [ld_f32(r_cw, ch * CONV_TAPS + j) for j in range(CONV_TAPS)]
                kch = tid % KD
                dtb = ld_f32(r_dtb, h * KD + kch)
                amp = _exp(ld_f32(r_alog, h))
                won = [ld_f32(r_won, 2 * lane + e) for e in range(2)]
                vr = tid // 4
                k0 = (tid % 4) * 32
                r_rs = _rsrc(
                    rec_state
                    + (slot * rec_stride + fx.Int64(h * KD * KD)) * fx.Int64(4)
                )
                hv = []
                for q in range_constexpr(8):
                    w4 = fx.Vector(
                        bo.buffer_load(
                            r_rs, vr * KD + k0 + 4 * q, vec_width=4, dtype=T.f32
                        )
                    )
                    hv += [w4[e] for e in range(4)]
                # one poll a thread: its conv channel, its gate and the row's beta
                if tid < 3 * KD:
                    gi = fx.min(tid, 2 * KD - 1)
                    x, g, b_raw = getf_many(
                        [
                            (mb("proj"), s * IN_ROWS + ch),
                            (
                                mb("gates"),
                                s * 2 * P + (gi // KD) * P + h * KD + gi % KD,
                            ),
                            (mb("proj"), s * IN_ROWS + 3 * P + h),
                        ]
                    )
                    a = cw[CONV_TAPS - 1] * x
                    for j in range_constexpr(CONV_TAPS - 1):
                        a = a + cw[j] * st[j]
                    lds_st(xs, tid, bf16_round(div(a, 1.0 + _exp(-a))))
                    if const_expr(STATES_LAST):
                        lds_st(xs, XO + tid, x)
                    if const_expr(not STATES_LAST):
                        if valid:
                            for j in range_constexpr(CONV_TAPS - 1):
                                nxt = st[j + 1] if j + 1 < CONV_TAPS - 1 else x
                                bo.buffer_store(
                                    fx.Float32(nxt).to(fx.BFloat16),
                                    r_cs,
                                    conv_at(ch, j),
                                )
                    if tid < 2 * KD:
                        lds_st(xs, G1O + tid, g)
                    if tid == 0:
                        lds_st(misc, 0, b_raw)
                gpu.barrier()
                if wave < 2:
                    a0 = lds_ld(xs, wave * KD + lane)
                    a1 = lds_ld(xs, wave * KD + lane + 64)
                    ss = wave_sum(a0 * a0 + a1 * a1)
                    if lane == 0:
                        lds_st(misc, 1 + wave, ss)
                if (wave >= 2) & (wave < 4):
                    # the channel decay, once a channel
                    gk = lds_ld(xs, G1O + kch) + dtb
                    lds_st(
                        xs,
                        DO + kch,
                        _exp(div(KDA_LOWER_BOUND, 1.0 + _exp(-(amp * gk)))),
                    )
                gpu.barrier()
                rq = _rsq(lds_ld(misc, 1) + L2_EPS) * (KD**-0.5)
                rk = _rsq(lds_ld(misc, 2) + L2_EPS)
                beta = sigmoid(lds_ld(misc, 0))
                kk = [lds_ld(xs, KO + k0 + i) * rk for i in range(32)]
                dot = fx.Float32(0.0)
                for i in range_constexpr(32):
                    hv[i] = hv[i] * lds_ld(xs, DO + k0 + i)
                    dot = dot + hv[i] * kk[i]
                for off in (1, 2):
                    dot = _xred(dot, off, lambda a, b: a + b)
                vn = beta * (lds_ld(xs, VO + vr) - dot)
                o = fx.Float32(0.0)
                for i in range_constexpr(32):
                    hv[i] = hv[i] + vn * kk[i]
                    o = o + hv[i] * (lds_ld(xs, QO + k0 + i) * rq)
                for off in (1, 2):
                    o = _xred(o, off, lambda a, b: a + b)

                def store_rec():
                    for q in range_constexpr(8):
                        bo.buffer_store(
                            fx.Vector.from_elements(hv[4 * q : 4 * q + 4], fx.Float32),
                            r_rs,
                            vr * KD + k0 + 4 * q,
                        )

                if const_expr(not STATES_LAST):
                    if valid:
                        store_rec()
                if tid % 4 == 0:
                    lds_st(xs, OO + vr, valid.select(bf16_round(o), fx.Float32(0.0)))
                gpu.barrier()
                if wave == 0:
                    e0 = 2 * lane
                    o0 = lds_ld(xs, OO + e0)
                    o1 = lds_ld(xs, OO + e0 + 1)
                    rstd = _rsq(wave_sum(o0 * o0 + o1 * o1) * (1.0 / KD) + O_NORM_EPS)
                    y0 = o0 * rstd * won[0] * sigmoid(lds_ld(xs, G2O + e0))
                    y1 = o1 * rstd * won[1] * sigmoid(lds_ld(xs, G2O + e0 + 1))
                    put(mb("o_in"), (s * P + h * KD + e0) // 2, bf16_pair(y0, y1))
                    # the head's word in every o_proj CTA's box, after its stores
                    signal("box_o_in", t)
                gpu.barrier()
                if const_expr(OPROJ_IN_KDA):
                    then()
                if const_expr(STATES_LAST):
                    if valid:
                        store_rec()
                        if tid < 3 * KD:
                            xn = lds_ld(xs, XO + tid)
                            for j in range_constexpr(CONV_TAPS - 1):
                                nxt = st[j + 1] if j + 1 < CONV_TAPS - 1 else xn
                                bo.buffer_store(
                                    fx.Float32(nxt).to(fx.BFloat16),
                                    r_cs,
                                    conv_at(ch, j),
                                )

        # ===================== 6. o_proj partial -> every peer's attention region
        # One task a 16-row tile of the row-major [HIDDEN, K] weight; the FFN seam
        # sums the TP partials as it does vLLM's. ``unit_of(row0, chunk)`` issues
        # one K chunk (``cpw`` a wave, prefetched ``ahead`` at a time) of the weight.
        def run_oproj(unit_of, K, cpw, ahead, done=None):
            """``done``: the CTA ran its task already (inside its kda task)."""
            attn_base = fx.Int64(SY["attn"]) + fx.Int64(peer_slot) * fx.Int64(
                SY["_part_stride"]
            )
            t0 = start("oproj")
            if const_expr(done is not None):
                t0 = done.select(fx.Int32(HIDDEN // O_TILE), t0)
            for t in range(t0, HIDDEN // O_TILE, G):
                t = fx.Int32(t)
                row0 = t * O_TILE

                def u_o(c):
                    return unit_of(row0, wave * cpw + c)

                pre = [u_o(c) for c in range(ahead)]
                if const_expr(kda):
                    # wave w waits for lines w, w + WAVES, .. of the CTA's box (16
                    # kda heads' words a line), then every thread reads the payload
                    O_LINES = box_lines(BOXES["box_o_in"])
                    for li in range_constexpr((O_LINES + WAVES - 1) // WAVES):
                        if _uniform(wave) + li * WAVES < O_LINES:
                            await_box(
                                "box_o_in",
                                [(_uniform(wave) + li * WAVES, 0, BOX_WORDS)],
                            )
                    gpu.barrier()
                stage_rows(mb("o_in"), K)
                gpu.barrier()
                acc = run_units(u_o, cpw, ahead, pre)
                reduce_rows(1, acc, emit_out(O_TILE))
                gpu.barrier()
                if wave < W:
                    pair_count = S * O_TILE // 2
                    for batch in range_constexpr((pair_count + 63) // 64):
                        pair = lane + batch * 64
                        if pair < pair_count:
                            si = pair // (O_TILE // 2)
                            ri = (pair % (O_TILE // 2)) * 2
                            put_bf(
                                peer_dst + attn_base,
                                (rank * S + si) * HIDDEN + row0 + ri,
                                [
                                    lds_ld(outs, si * O_TILE + ri),
                                    lds_ld(outs, si * O_TILE + ri + 1),
                                ],
                                CM_SYS,
                            )
                gpu.barrier()

        # ================================================= sparse-attention stages
        # The step table ``dsa_rows`` holds, a row, (position, MLA cache slot, slot of
        # the index-cache pool the row completes, kpool tail slot, block-table row);
        # a padding row has position -1 or MLA slot 0 and writes nothing.
        def row_word(s, k):
            return _uniform(fx.Int32(lds_ld(rows_lds, s * ROW_WORDS + k)))

        def ld_i32(r, i):
            return fx.Int32(bo.buffer_load(r, i, vec_width=1, dtype=T.i32))

        def step_tab():
            """The step-input table (``STEP_TAB``): pointers as int64, counts as int32."""
            r_tab = _rsrc(dsa_tab)
            out = {}
            for i, n in enumerate(STEP_TAB):
                w = fx.Vector(bo.buffer_load(r_tab, i * 2, vec_width=2, dtype=T.i32))
                lo = _uniform(w[0])
                if const_expr(n in STEP_COUNTS):
                    out[n] = lo
                else:
                    out[n] = (fx.Int64(_uniform(w[1])) << 32) | fx.Int64(fx.Uint32(lo))
            return out

        def step_rows():
            """The step's rows (``n_rows`` of a token launch's step table)."""
            w = bo.buffer_load(
                _rsrc(dsa_tab), STEP_TAB.index("n_rows") * 2, vec_width=1, dtype=T.i32
            )
            return _uniform(fx.Int32(w))

        def write_step_tables():
            """CTA G - 1 writes the step's KDA slot table and row table for the
            step's later launches, as the spec's own tables hold them: a row with MLA
            slot 0 (SGLang's padding) has -1 words; the width's rows past the step's
            have KDA slot 0, -1 words and page-table row 0."""
            if bid == G - 1:
                if tid < S:
                    st = step_tab()
                    live = tid < st["n_rows"]
                    i = fx.min(tid, st["n_rows"] - 1)

                    def entry(name, words):
                        return ld_i32(_rsrc(st[name]), i * st[words])

                    pos = entry("positions", "positions_words")
                    loc = entry("out_cache_loc", "loc_words")
                    rpi = entry("req_pool_indices", "rpi_words")
                    slot = ld_i32(_rsrc(state_idx), i)
                    bo.buffer_store(
                        live.select(slot, fx.Int32(0)), _rsrc(st["slots"]), tid
                    )
                    valid = live & (loc != 0)
                    r_rows = _rsrc(st["rows"])
                    for k, v in (
                        (ROW_POS, pos),
                        (ROW_SLOT, loc),
                        (ROW_POOL_SLOT, loc // KPOOL),
                        (ROW_TAIL_SLOT, rpi * KPOOL),
                    ):
                        bo.buffer_store(
                            valid.select(fx.Int32(v), fx.Int32(-1)),
                            r_rows,
                            tid * ROW_WORDS + k,
                        )
                    bo.buffer_store(
                        live.select(fx.Int32(tid), fx.Int32(0)),
                        r_rows,
                        tid * ROW_WORDS + ROW_BT,
                    )

        def chain_rows(s):
            """The rows whose cache writes row ``s`` reads (a static row): itself,
            and in a verify step its request's earlier rows."""
            first = s - s % VL if s < NREQ * VL else s
            return range(first, s + 1)

        def chain_ready(s):
            """Poll specs of ``kv_ready`` of row ``s`` (dynamic) and, in a verify
            step, of its request's earlier rows (a pad row's repeat its own)."""
            return (
                [(mb("kv_ready"), fx.max(s - d, s - s % VL), 1) for d in range(VL)]
                if VERIFY
                else [(mb("kv_ready"), s, 1)]
            )

        def start_at(b):
            """First task of a stage whose task 0 runs on CTA ``b``."""
            return (bid + (G - b % G)) & (G - 1)

        def row_keys(pos):
            """(complete pools, scored, keys attended) of a row at ``pos``: a row with
            more than TOPK_POOLS complete pools attends to the selected pools' tokens
            plus its tail, any other row to every token."""
            bound = (pos + 1) // KPOOL
            scored = bound > TOPK_POOLS
            nk = scored.select(TOPK_TOKENS + (pos + 1) - bound * KPOOL, pos + 1)
            return bound, scored, nk

        def unit_fp8_rm(
            w_rsrc, s_rsrc, row0, kb, K, rows, b_word, scale_cols, shuffled
        ):
            """One 128-K block ``kb`` of a 16-row tile of an FP8 [rows, K] weight
            with f32 [rows / 128, K / 128] block scales (``row0 % 16 == 0``), row-major
            or AITER-shuffled ([rows / 16, K / 32, 2, 16 rows, 16 bytes])."""
            r = fx.min(row0 + lane % 16, rows - 1)
            if const_expr(shuffled):
                blk = (r // 16) * (K // 32) + kb * 4 + lane // 16
                word, half = (blk * 512 + (r % 16) * 16) // 4, 64
            else:
                word, half = (r * K + kb * 128 + (lane // 16) * 32) // 4, 4
            wv = [
                fx.Vector(
                    bo.buffer_load(w_rsrc, word + half * h, vec_width=4, dtype=T.i32)
                )
                for h in range(2)
            ]
            sc = ld_f32(s_rsrc, (row0 // 128) * scale_cols + kb)
            return ("fp8rm", wv, sc, b_word + (lane // 16) * 16)

        def proj_unit(fp8, w_rsrc, s_rsrc, row0, kc, K, rows, n_words, shuffled):
            """K chunk ``kc`` (128 FP8 or 64 bf16 values) of a 16-row tile of a
            [rows, K] projection; the B row starts at LDS word n_words."""
            if const_expr(fp8):
                return unit_fp8_rm(
                    w_rsrc,
                    s_rsrc,
                    row0,
                    kc,
                    K,
                    rows,
                    n_words + kc * 64,
                    K // 128,
                    shuffled,
                )
            return unit_rm(w_rsrc, row0, kc, K, rows, n_words + kc * 32)

        def fwht128(x0, x1):
            """Hadamard-128 of one wave's vector (lane holds d = lane and lane + 64):
            butterflies of stride 1 .. 64 in vLLM's order, then the 1/sqrt(128)."""
            for sh in range_constexpr(6):
                st = 1 << sh
                upper = (lane & st) != 0
                p0, p1 = _xshfl(x0, st), _xshfl(x1, st)
                x0 = upper.select(p0 - x0, x0 + p0)
                x1 = upper.select(p1 - x1, x1 + p1)
            x0, x1 = x0 + x1, x0 - x1
            return x0 * FWHT_SCALE, x1 * FWHT_SCALE

        def ue8m0_quant(x0, x1):
            """Per-vector FP8 (vLLM's) of one wave's 128 values: scale =
            2^ceil(log2(max(absmax, 1e-4) / 448)); returns (q0, q1, scale)."""
            amax = wave_max(fx.max(fmath.absf(x0), fmath.absf(x1)))
            y = fx.max(amax, fx.Float32(1.0e-4)) * fx.Float32(1.0 / FP8_MAX)
            bits = y.bitcast(fx.Int32)
            e = (bits >> 23) & fx.Int32(0xFF)
            e = e + ((bits & fx.Int32(0x7FFFFF)) != 0).select(fx.Int32(1), fx.Int32(0))
            scale = (e << 23).bitcast(fx.Float32)
            inv = ((fx.Int32(254) - e) << 23).bitcast(fx.Float32)
            q0 = fx.min(fx.max(x0 * inv, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(x1 * inv, -FP8_MAX), FP8_MAX)
            return q0, q1, scale

        # ========= q_a | kv_a (FP8 or bf16) and wk | weights_proj | kpool gate (bf16)
        # One task a 16-row tile, every row. Outputs are bf16-rounded as vLLM's
        # GEMMs store them, except the indexer head weights (fp32 in vLLM).
        def dsa_qkv_fp8(tv, t32):
            r_wqkv, r_sqkv = _rsrc(tv["w_qkv"]), _rsrc(tv["s_qkv"])
            QKV_CPW = HIDDEN // (128 if QKV8 else 64) // WAVES
            for t in range(start("qkv_a"), N_QKV_FP8, G):
                t = fx.Int32(t)
                row0 = t * QKV_TILE

                def u8(c):
                    return proj_unit(
                        QKV8,
                        r_wqkv,
                        r_sqkv,
                        row0,
                        wave * QKV_CPW + c,
                        HIDDEN,
                        Q_LORA + KV_LORA,
                        n_sel() * XW,
                        QKV_SH,
                    )

                pre = [u8(c) for c in range(4)]
                wait_ready("box_a_in")
                stage_plain(mb("a_in"), HIDDEN)
                gpu.barrier()
                acc = run_units(u8, QKV_CPW, 4, pre)

                def emit_qkv(rl, n, v):
                    row = row0 + rl
                    v = bf16_round(v)
                    if row < Q_LORA:
                        put(mb("qa"), n * Q_LORA + row, v)
                    else:
                        put(mb("kva"), n * KV_LORA + row - Q_LORA, v)

                reduce_rows(1, acc, emit_qkv)
                gpu.barrier()

        def dsa_qkv_bf16(tv, t32):
            N_IK = (INDEX_DIM + INDEX_HEADS) // QKV_TILE
            IDX_CPW = HIDDEN // 64 // WAVES
            for tb in range(start_at(base["qkv_a"] + N_QKV_FP8), N_QKV - N_QKV_FP8, G):
                tb = fx.Int32(tb)
                in_ik = tb < N_IK
                r_wb = _rsrc(in_ik.select(tv["w_idx"], tv["w_gate"]))
                lrow0 = in_ik.select(tb, tb - N_IK) * QKV_TILE
                nrows = in_ik.select(
                    fx.Int32(INDEX_DIM + INDEX_HEADS), fx.Int32(INDEX_DIM)
                )

                def ub(c):
                    kc = wave * IDX_CPW + c
                    return unit_rm(
                        r_wb, lrow0, kc, HIDDEN, nrows, n_sel() * XW + kc * 32
                    )

                pre = [ub(c) for c in range(4)]
                wait_ready("box_a_in")
                stage_plain(mb("a_in"), HIDDEN)
                gpu.barrier()
                acc = run_units(ub, IDX_CPW, 4, pre)

                def emit_idx(rl, n, v):
                    row = tb * QKV_TILE + rl
                    if row < INDEX_DIM:
                        put(mb("ik"), n * INDEX_DIM + row, bf16_round(v))
                    elif row < INDEX_DIM + INDEX_HEADS:
                        put(mb("iw"), n * INDEX_HEADS + row - INDEX_DIM, v)
                    else:
                        put(
                            mb("ig"),
                            n * INDEX_DIM + row - INDEX_DIM - INDEX_HEADS,
                            bf16_round(v),
                        )

                reduce_rows(1, acc, emit_idx)
                gpu.barrier()

        # ======================================= q_a RMSNorm, one task a row
        def dsa_qnorm(tv, t32):
            r_gqa = _rsrc(tv["g_qa"])
            NQ4 = Q_LORA // 4
            for t in range(start("qnorm"), S, G):
                s = fx.Int32(t)
                live = tid < NQ4
                e0 = fx.min(tid, NQ4 - 1) * 4
                gam = ld_bf16s(r_gqa, e0, 4)
                v2 = get2_many(
                    [(mb("qa"), s * Q_LORA + e0), (mb("qa"), s * Q_LORA + e0 + 2)]
                )
                xv = [v2[0][0], v2[0][1], v2[1][0], v2[1][1]]
                ss = xv[0] * xv[0] + xv[1] * xv[1] + xv[2] * xv[2] + xv[3] * xv[3]
                total = block_sum(live.select(ss, fx.Float32(0.0)))
                rstd = _rsq(total * (1.0 / Q_LORA) + EPS)
                if live:
                    put_bf(
                        mb("q_an"),
                        s * Q_LORA + e0,
                        [xv[j] * rstd * gam[j] for j in range(4)],
                    )
                gpu.barrier()

        # ===== KV RMSNorm -> MLA cache; index k LayerNorm; kpool update, a row
        # A row whose position completes a pool (pos % 4 == 3) compresses the
        # pool into the index cache: per dim softmax(gate + ape) over its four
        # tokens' (k, gate) (three from the tail ring, the row's own), the
        # weighted k, bf16, Hadamard, bf16, ue8m0 FP8. Then the row's (k, gate)
        # goes to ring slot pos % TAIL (SGLang's ring of 4 slots a request, plus a
        # verify step's draft tokens). ``kv_ready`` releases both caches. In a
        # verify step one task runs a request's rows in order: a row's pool reads
        # the ring entries of the chain's earlier rows.
        def dsa_cache(tv, t32):
            KVB = t32["kv_block_tokens"]
            KVE = tv["kv_block_elems"]
            KVR = t32["kv_row_elems"]
            IPB = t32["idx_block_pools"]
            IBB = t32["idx_block_bytes"]
            TBE = t32["tail_block_elems"]
            TAIL = TBE // INDEX_DIM
            r_gkv = _rsrc(tv["g_kv"])
            r_lnw, r_lnb = _rsrc(tv["ln_w"]), _rsrc(tv["ln_b"])
            r_ape = _rsrc(tv["ape"])

            # a verify chain's earlier rows wrote the ring in this task: read it past
            # the CU's cache
            ring_cm = {"cache_modifier": CM_DEV} if VERIFY else {}

            def ring_ld(r, i):
                return fx.Float32(
                    fx.BFloat16(
                        bo.buffer_load(r, i, vec_width=1, dtype=T.bf16, **ring_cm)
                    )
                )

            def cache_row(s):
                pos = row_word(s, ROW_POS)
                slot = row_word(s, ROW_SLOT)
                pslot = row_word(s, ROW_POOL_SLOT)
                tslot = row_word(s, ROW_TAIL_SLOT)
                g = ld_bf16(r_gkv, tid)
                d = fx.min(tid, INDEX_DIM - 1)
                lk = tid < INDEX_DIM
                lnw, lnb = ld_f32(r_lnw, d), ld_f32(r_lnb, d)
                kv, kraw, graw = getf_many(
                    [
                        (mb("kva"), s * KV_LORA + tid),
                        (mb("ik"), s * INDEX_DIM + d),
                        (mb("ig"), s * INDEX_DIM + d),
                    ]
                )
                rstd = _rsq(block_sum(kv * kv) * (1.0 / KV_LORA) + EPS)
                # a padding row (position < 0, or MLA slot 0: SGLang's reserved
                # dummy, which its graph runner gives pad rows) writes nothing
                live = (pos >= 0) & (slot > 0)
                if live:
                    row_addr = tv["kv_cache"] + (
                        fx.Int64(slot // KVB) * KVE + fx.Int64((slot % KVB) * KVR)
                    ) * fx.Int64(2)
                    bo.buffer_store(
                        fx.Float32(kv * rstd * g).to(fx.BFloat16),
                        _rsrc(row_addr),
                        tid,
                        cache_modifier=CM_DEV,
                    )
                mean = block_sum(lk.select(kraw, fx.Float32(0.0))) * (1.0 / INDEX_DIM)
                cen = lk.select(kraw - mean, fx.Float32(0.0))
                var = block_sum(cen * cen) * (1.0 / INDEX_DIM)
                kn = bf16_round(cen * _rsq(var + INDEX_EPS) * lnw + lnb)
                phase = fx.max(pos, 0) % KPOOL
                # caches can be views of pools past 4 GB: 64-bit row bases
                tail_row = (
                    fx.Int64(fx.max(tslot, 0) // KPOOL) * fx.Int64(TBE) * fx.Int64(2)
                )
                r_tail = _rsrc(tv["tail_k"] + tail_row)
                r_gate = _rsrc(tv["tail_g"] + tail_row)
                complete = live & (pslot >= 0) & (phase == KPOOL - 1)
                if complete:
                    if lk:
                        sj, kj = [], []
                        for j in range_constexpr(KPOOL - 1):
                            ring = (pos - (KPOOL - 1) + j) % TAIL * INDEX_DIM + d
                            sj.append(ring_ld(r_gate, ring))
                            kj.append(ring_ld(r_tail, ring))
                        sj.append(graw)
                        kj.append(kn)
                        sj = [
                            sj[j] + ld_f32(r_ape, j * INDEX_DIM + d)
                            for j in range(KPOOL)
                        ]
                        mx = sj[0]
                        for j in range_constexpr(1, KPOOL):
                            mx = fx.max(mx, sj[j])
                        num = fx.Float32(0.0)
                        den = fx.Float32(0.0)
                        for j in range_constexpr(KPOOL):
                            p = _exp(sj[j] - mx)
                            den = den + p
                            num = num + kj[j] * p
                        lds_st(xs, d, bf16_round(div(num, den)))
                    gpu.barrier()
                    if wave == 0:
                        x0, x1 = fwht128(lds_ld(xs, lane), lds_ld(xs, lane + 64))
                        q0, q1, kscale = ue8m0_quant(bf16_round(x0), bf16_round(x1))
                        lds_st(xs, INDEX_DIM + lane, q0)
                        lds_st(xs, INDEX_DIM + lane + 64, q1)
                        if lane == 0:
                            lds_st(misc, 0, kscale)
                    gpu.barrier()
                    r_icache = _rsrc(
                        tv["idx_cache"] + fx.Int64(pslot // IPB) * fx.Int64(IBB)
                    )
                    poff = pslot % IPB
                    if tid < INDEX_DIM // 4:
                        d4 = tid * 4
                        word = fp8_pack4(
                            *[lds_ld(xs, INDEX_DIM + d4 + j) for j in range(4)]
                        )
                        # 16-pool x 16-byte tiles (a block holds 16 n pools)
                        byte = (
                            (poff // 16) * (16 * INDEX_DIM)
                            + (d4 // 16) * 256
                            + (poff % 16) * 16
                            + d4 % 16
                        )
                        bo.buffer_store(
                            word, r_icache, byte // 4, cache_modifier=CM_DEV
                        )
                    if tid == 0:
                        bo.buffer_store(
                            lds_ld(misc, 0),
                            r_icache,
                            IPB * INDEX_DIM // 4 + poff,
                            cache_modifier=CM_DEV,
                        )
                if live & (tslot >= 0) & lk:
                    e = fx.max(pos, 0) % TAIL * INDEX_DIM + d
                    bo.buffer_store(
                        kn.to(fx.BFloat16), r_tail, e, cache_modifier=CM_DEV
                    )
                    bo.buffer_store(
                        graw.to(fx.BFloat16), r_gate, e, cache_modifier=CM_DEV
                    )
                # the cache writes are device-scope stores: their completion, not a
                # write-back of the L2, releases them
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                if tid == 0:
                    put(mb("kv_ready"), s, fx.Int32(1))

            gpu.barrier()
            for t in range(start("cache"), S, G):
                s = fx.Int32(t)
                if const_expr(VERIFY):
                    if (s % VL == 0) & (s < NREQ * VL):
                        for j in range_constexpr(VL):
                            cache_row(s + j)
                    if const_expr(NREQ * VL < S):
                        if s >= NREQ * VL:
                            cache_row(s)
                else:
                    cache_row(s)

        # ======================= q_b (FP8 or bf16): two 16-row tiles a task
        def dsa_q_b(tv, t32):
            r_wqb, r_sqb = _rsrc(tv["w_qb"]), _rsrc(tv["s_qb"])
            QB_CPW = Q_LORA // (128 if QB8 else 64) // (WAVES // 2)
            for t in range(start("q_b"), QB_ROWS // QB_TILE, G):
                t = fx.Int32(t)
                row0 = t * QB_TILE + (wave // 4) * 16

                def u_qb(c):
                    return proj_unit(
                        QB8,
                        r_wqb,
                        r_sqb,
                        row0,
                        (wave % 4) * QB_CPW + c,
                        Q_LORA,
                        QB_ROWS,
                        n_sel() * (Q_LORA // 2),
                        QB_SH,
                    )

                QB_AHEAD = min(QB_CPW, 4)
                pre = [u_qb(c) for c in range(QB_AHEAD)]
                stage_rows(mb("q_an"), Q_LORA)
                gpu.barrier()
                acc = run_units(u_qb, QB_CPW, QB_AHEAD, pre)
                reduce_rows(2, acc, emit_out(QB_TILE))
                gpu.barrier()
                if tid < S * QB_TILE // 4:
                    s = tid // (QB_TILE // 4)
                    r = (tid % (QB_TILE // 4)) * 4
                    put_bf(
                        mb("q_nope"),
                        s * QB_ROWS + t * QB_TILE + r,
                        [lds_ld(outs, s * QB_TILE + r + j) for j in range(4)],
                    )
                gpu.barrier()

        # ===== index q (bf16): wq_b in 16-row slices, one a task, every CTA; each
        # wave takes IQ_UPW of the 64-K units and the slice sums them in unit order.
        # Then, one head a task: Hadamard, ue8m0 FP8 and the head weight
        # w * q_scale * 128^-0.5 * 32^-0.5. Only scored rows read it, so a launch
        # without one skips both (the rows are the same on every rank).
        IQ_SLICES = INDEX_DIM // 16
        IQ_UPW = Q_LORA // 64 // WAVES
        IQ_PART = S * Q_LORA // 2
        assert not dsa or IQ_PART + WAVES * IQ_UPW * 64 * 4 <= X_WORDS

        def iq_scored():
            any_scored = row_keys(row_word(0, ROW_POS))[1]
            for s in range_constexpr(1, S):
                any_scored = any_scored | row_keys(row_word(s, ROW_POS))[1]
            return any_scored

        def dsa_index_q(tv, t32):
            r_wiq = _rsrc(tv["w_iq"])
            part = xs + IQ_PART  # each unit's MFMA tile, past the staged rows
            n_tasks = iq_scored().select(fx.Int32(INDEX_HEADS * IQ_SLICES), fx.Int32(0))
            for t in range(start("index_q"), n_tasks, G):
                t = fx.Int32(t)
                row0 = t * 16

                def u_iq(c):
                    kc = wave * IQ_UPW + c
                    return unit_rm(
                        r_wiq,
                        row0,
                        kc,
                        Q_LORA,
                        INDEX_HEADS * INDEX_DIM,
                        n_sel() * (Q_LORA // 2) + kc * 32,
                    )

                pre = [u_iq(c) for c in range(IQ_UPW)]
                stage_rows(mb("q_an"), Q_LORA)
                gpu.barrier()
                zero = [fx.Float32(0.0) for _ in range(4)]
                for c in range_constexpr(IQ_UPW):
                    tile = mma_units(zero, [pre[c]])
                    fx.ptr_store(
                        fx.Vector.from_elements(tile, fx.Float32),
                        part + ((wave * IQ_UPW + c) * 64 + lane) * 4,
                    )
                gpu.barrier()
                # C[r][n] sits in lane n + 16 (r // 4), element r % 4
                if tid < S * 16:
                    r = tid % 16
                    n = tid // 16
                    tot = fx.Float32(0.0)
                    for k in range_constexpr(WAVES * IQ_UPW):
                        tot = tot + lds_ld(
                            part, (k * 64 + n + 16 * (r // 4)) * 4 + r % 4
                        )
                    lds_st(outs, n * 16 + r, bf16_round(tot))
                gpu.barrier()
                if tid < S * 4:
                    n = tid // 4
                    r = (tid % 4) * 4
                    put_bf(
                        mb("iq"),
                        n * INDEX_HEADS * INDEX_DIM + row0 + r,
                        [lds_ld(outs, n * 16 + r + j) for j in range(4)],
                    )
                gpu.barrier()

        def dsa_iq_quant(tv, t32):
            IQW_SCALE = INDEX_DIM**-0.5 * INDEX_HEADS**-0.5
            n_heads = iq_scored().select(fx.Int32(INDEX_HEADS), fx.Int32(0))
            for t in range(start("iq_quant"), n_heads, G):
                head = fx.Int32(t)
                # an instance no index_q task has: the last slice releases the wait
                if tid < S * INDEX_DIM // 4:
                    n = tid // (INDEX_DIM // 4)
                    j = (tid % (INDEX_DIM // 4)) * 4
                    v = poll(
                        [
                            (
                                mb("iq"),
                                (n * INDEX_HEADS * INDEX_DIM + head * INDEX_DIM + j)
                                // 2,
                                2,
                            )
                        ]
                    )[0]
                    for i in range_constexpr(2):
                        lo, hi = bf2_f32(v[i])
                        lds_st(xs, n * INDEX_DIM + j + 2 * i, lo)
                        lds_st(xs, n * INDEX_DIM + j + 2 * i + 1, hi)
                gpu.barrier()
                # row wave + WAVES i a wave
                for wi in range_constexpr((S + WAVES - 1) // WAVES):
                    sw = wave + wi * WAVES
                    if sw < S:
                        x0, x1 = fwht128(
                            lds_ld(xs, sw * INDEX_DIM + lane),
                            lds_ld(xs, sw * INDEX_DIM + lane + 64),
                        )
                        q0, q1, qscale = ue8m0_quant(bf16_round(x0), bf16_round(x1))
                        lds_st(xs, (S + sw) * INDEX_DIM + lane, q0)
                        lds_st(xs, (S + sw) * INDEX_DIM + lane + 64, q1)
                        if lane == 0:
                            w_h = getf_many([(mb("iw"), sw * INDEX_HEADS + head)])[0]
                            put(
                                mb("iqw"),
                                sw * INDEX_HEADS + head,
                                w_h * qscale * IQW_SCALE,
                            )
                gpu.barrier()
                if tid < S * INDEX_DIM // 4:
                    s = tid // (INDEX_DIM // 4)
                    j = tid % (INDEX_DIM // 4)
                    q8 = (S + s) * INDEX_DIM + 4 * j
                    put(
                        mb("iq8"),
                        (s * INDEX_HEADS + head) * (INDEX_DIM // 4) + j,
                        fp8_pack4(*[lds_ld(xs, q8 + i) for i in range(4)]),
                    )
                gpu.barrier()

        # ========== absorbed query q_lat = W_UK^T q_nope: 128 latent rows a task
        def dsa_uk(tv, t32):
            r_wuk = _rsrc(tv["w_uk"])
            UK_CPW = NOPE_DIM // 64
            UK_PER_HEAD = KV_LORA // UK_TILE
            for t in range(start("uk"), MH * UK_PER_HEAD, G):
                t = fx.Int32(t)
                head = t // UK_PER_HEAD
                row0 = t * UK_TILE + wave * 16

                def u_uk(c):
                    return unit_rm(
                        r_wuk,
                        row0,
                        c,
                        NOPE_DIM,
                        MH * KV_LORA,
                        n_sel() * (NOPE_DIM // 2) + c * 32,
                    )

                pre = [u_uk(c) for c in range(UK_CPW)]
                for pi in range_constexpr((S * NOPE_DIM // 4 + THREADS - 1) // THREADS):
                    q = tid + pi * THREADS
                    if q < S * NOPE_DIM // 4:
                        n = q // (NOPE_DIM // 4)
                        k4 = (q % (NOPE_DIM // 4)) * 4
                        v = poll(
                            [
                                (
                                    mb("q_nope"),
                                    (n * QB_ROWS + head * NOPE_DIM + k4) // 2,
                                    2,
                                )
                            ]
                        )[0]
                        for j in range_constexpr(2):
                            lds_st(
                                xs,
                                n * (NOPE_DIM // 2) + k4 // 2 + j,
                                v[j].bitcast(fx.Float32),
                            )
                gpu.barrier()
                acc = run_units(u_uk, UK_CPW, UK_CPW, pre)
                if lane % 16 < S:
                    put_bf(
                        mb("q_lat"),
                        ((lane % 16) * MH + head) * KV_LORA
                        + (t % UK_PER_HEAD) * UK_TILE
                        + wave * 16
                        + 4 * (lane // 16),
                        [acc[e] for e in range(4)],
                    )
                gpu.barrier()

        # ===== index scores of a row's complete pools: 256 pools a task
        # score(p) = kscale(p) * sum_h w'_h relu(q8_h . k8_p), FP8 x FP8 MFMAs. Rows
        # of at most TOPK_POOLS complete pools attend to every token and are not scored.
        MAXP = max(max_pools, 1)
        QW8 = INDEX_DIM // 4  # words of one head's FP8 query

        def dsa_iscore(tv, t32):
            IPB = t32["idx_block_pools"]
            IBB = t32["idx_block_bytes"]
            # entries of the index block table an index block spans: its entry is
            # the first one's value over this (SGLang's 64-token page table)
            IBG = t32["idx_bt_group"]
            one = fx.Int32(127)  # E8M0 1.0
            HALVES = POOLS_PER_TASK // (16 * WAVES)
            g4 = lane // 16
            e_w = lane % 16

            def half_block(r_ibt, n_pools, tile, half):
                """Issue the load of the block-table entry of the wave's 16 pools of a
                tile half (clamped to the row's): they share one block."""
                p0 = tile * POOLS_PER_TASK + (half * WAVES + wave) * 16
                return p0, ld_i32(r_ibt, fx.min(p0, n_pools - 1) // IPB * IBG)

            def load_half(n_pools, block):
                """Issue the loads of the wave's 16 pools of a tile half: lane % 16's
                FP8 key, bytes 32 g4 .., and the scale of the pool lane scores."""
                p0, entry = block

                def off_of(p):
                    return fx.min(p, n_pools - 1) % IPB

                # a uniform 64-bit block base
                blk = _uniform(entry) // IBG
                r_blk = _rsrc(tv["idx_cache"] + fx.Int64(blk) * fx.Int64(IBB))
                po = off_of(p0 + lane % 16)
                b0 = (po // 16) * (16 * INDEX_DIM) + (2 * g4) * 256 + (po % 16) * 16
                kw = [
                    fx.Vector(bo.buffer_load(r_blk, bb // 4, vec_width=4, dtype=T.i32))
                    for bb in (b0, b0 + 256)
                ]
                pw = p0 + 4 * g4 + e_w
                ksc = ld_f32(r_blk, IPB * INDEX_DIM // 4 + off_of(pw))
                return pw, kw, ksc

            def score_half(s, n_pools, loaded, q_off, w_off):
                """Row ``s``'s scores of a loaded tile half: its FP8 query at LDS word
                ``q_off``, its head weights at ``w_off``."""
                pw, kw, ksc = loaded
                a = fx.Vector.from_elements(
                    [kw[h][e] for h in range(2) for e in range(4)], fx.Int32
                )
                parts = [fx.Float32(0.0) for _ in range(4)]
                for hh in range_constexpr(INDEX_HEADS // 16):
                    hq = hh * 16 + lane % 16
                    qb = [
                        fx.Vector(
                            fx.ptr_load(
                                xs + (q_off + hq * QW8 + 8 * g4 + 4 * h),
                                result_type=v4f,
                            )
                        ).bitcast(fx.Int32)
                        for h in range(2)
                    ]
                    b = fx.Vector.from_elements(
                        [qb[h][e] for h in range(2) for e in range(4)], fx.Int32
                    )
                    c = fx.Vector(
                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.vec(4, T.f32),
                            [
                                a,
                                b,
                                fx.Vector.filled(4, 0.0, fx.Float32),
                                0,
                                0,
                                0,
                                one,
                                0,
                                one,
                            ],
                        )
                    )
                    # C[pool][head] in lane head + 16 (pool // 4), elem pool % 4
                    w_h = lds_ld(xs, w_off + hq)
                    for e in range_constexpr(4):
                        parts[e] = parts[e] + fx.max(c[e], fx.Float32(0.0)) * w_h
                for off in (1, 2, 4, 8):
                    parts = [_xred(v, off, lambda x, y: x + y) for v in parts]
                val = parts[0]
                for e in range_constexpr(1, 4):
                    val = (e_w == e).select(parts[e], val)
                # every lane stores, the others past num_records (dropped): FlyDSL's
                # lowering merges a divergent store here with the first tile's, and
                # the tile loop's MFMAs then run with the storing lanes off and
                # overwrite their index (https://github.com/ROCm/FlyDSL/issues/1358)
                keep = (e_w < 4) & (pw < n_pools)
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [(val * ksc).bitcast(fx.Int32), tag], fx.Int32
                    ),
                    _rsrc(mb("iscores"), S * MAXP * 8),
                    keep.select(s * MAXP + pw, fx.Int32(S * MAXP)) * 2,
                    cache_modifier=CM_DEV,
                )

            # a task a (unit, tile): a unit is a verify request's rows, which share its
            # page table (SGLang repeats the request's row) and so each tile's keys,
            # or one other row. A tile's keys load once for the unit's rows; each row
            # scores them with its own query and weights, masked to its own pools
            # (pools past a row's are a later row's, loaded in place).
            if const_expr(VERIFY):
                UNITS = [list(range(r * VL, (r + 1) * VL)) for r in range(NREQ)]
                UNITS += [[s] for s in range(NREQ * VL, S)]
            else:
                UNITS = [[s] for s in range(S)]
            NQW = INDEX_HEADS * QW8 // THREADS
            for u in range_constexpr(len(UNITS)):
                urows = UNITS[u]
                UL = len(urows)
                row_pools = []
                for s in urows:
                    bound, scored, _ = row_keys(row_word(s, ROW_POS))
                    row_pools.append(scored.select(bound, fx.Int32(0)))
                n_pools = row_pools[0]
                for v in row_pools[1:]:
                    n_pools = fx.max(n_pools, v)
                n_tiles = (n_pools + POOLS_PER_TASK - 1) // POOLS_PER_TASK
                first = start_at(base["iscore"] + u * (G // len(UNITS)))
                if first < n_tiles:
                    bt = row_word(urows[0], ROW_BT)
                    r_ibt = _rsrc(
                        page_table + fx.Int64(bt) * fx.Int64(pt_stride) * fx.Int64(4)
                    )
                    # the first tile's block-table entries load while one lane polls
                    # the cache's flags (the pools this launch completed), then its
                    # keys load while the queries are awaited
                    blocks = [
                        half_block(r_ibt, n_pools, first, h) for h in range(HALVES)
                    ]
                    if tid == 0:
                        for r in chain_rows(urows[-1]):
                            get(mb("kv_ready"), r)
                    memory_fence(syncscope="agent", ordering="acquire")
                    gpu.barrier()
                    pre = [load_half(n_pools, b) for b in blocks]
                    # the rows' FP8 queries -> LDS words i * Q_ROW + h * 32 .., their
                    # weights -> words W_OFF + i * INDEX_HEADS + h, in one poll
                    Q_ROW = INDEX_HEADS * QW8
                    W_OFF = UL * Q_ROW
                    assert W_OFF + UL * INDEX_HEADS <= MISC_OFF
                    q0 = urows[0] * Q_ROW
                    specs = [
                        (mb("iq8"), q0 + tid + i * THREADS, 1) for i in range(UL * NQW)
                    ]
                    specs.append(
                        (
                            mb("iqw"),
                            urows[0] * INDEX_HEADS
                            + fx.min(tid, fx.Int32(UL * INDEX_HEADS - 1)),
                            1,
                        )
                    )
                    qv = poll(specs, batch=len(specs))
                    for i in range_constexpr(UL * NQW):
                        lds_st(xs, tid + i * THREADS, qv[i][0].bitcast(fx.Float32))
                    if tid < UL * INDEX_HEADS:
                        lds_st(xs, W_OFF + tid, qv[-1][0].bitcast(fx.Float32))
                    gpu.barrier()

                    def score_tile(loaded):
                        for i in range_constexpr(UL):
                            score_half(
                                urows[i],
                                row_pools[i],
                                loaded,
                                i * Q_ROW,
                                W_OFF + i * INDEX_HEADS,
                            )

                    for h in range_constexpr(HALVES):
                        score_tile(pre[h])
                    for tile in range(first + G, n_tiles, G):
                        tile = fx.Int32(tile)
                        for h in range_constexpr(HALVES):
                            score_tile(
                                load_half(n_pools, half_block(r_ibt, n_pools, tile, h))
                            )
                    gpu.barrier()

        # ====== top-512 pools of a row (one CTA a row), then the attended tokens:
        # the pools' 2048 tokens, then the row's tail tokens past its last complete
        # pool.
        def dsa_iselect(tv, t32):
            hist = fx.recast_iter(fx.Int32, xs)
            r_scores = _rsrc(mb("iscores"))

            def radix_key(bits):
                return (bits >= 0).select(bits ^ fx.Int32(-(2**31)), ~bits)

            def select_digit(shift, prefix, remain):
                digit = 255 - fx.min(tid, 255)
                cnt = (tid < 256).select(lds_ld(keys, digit), fx.Int32(0))
                inclusive = fx.coop.warp_inclusive_scan(
                    cnt, fx.ReductionOp.ADD, width=64
                )
                wave_total = read_lane_i32(inclusive, 63)
                if (wave < 4) & (lane == 63):
                    lds_st(keys, 256 + wave, wave_total)
                gpu.barrier()
                before_wave = fx.Int32(0)
                for w in range_constexpr(4):
                    before_wave = before_wave + (wave > w).select(
                        lds_ld(keys, 256 + w), fx.Int32(0)
                    )
                above = before_wave + inclusive - cnt
                hit = (tid < 256) & (above < remain) & (above + cnt >= remain)
                gpu.barrier()
                if hit:
                    lds_st(keys, 256, fx.Int32(prefix | (fx.Uint32(digit) << shift)))
                    lds_st(keys, 257, remain - above)
                gpu.barrier()
                return fx.Uint32(lds_ld(keys, 256)), lds_ld(keys, 257)

            for t in range(start("iselect"), S, G):
                s = fx.Int32(t)
                pos = row_word(s, ROW_POS)
                bound, scored, _ = row_keys(pos)
                if scored:
                    count = fx.Int32(TOPK_POOLS)
                    n_chunks = (bound + SEL_CHUNK - 1) // SEL_CHUNK

                    def chunk_items(chunk):
                        return [
                            chunk * SEL_CHUNK + tid * SEL_ITEMS + j
                            for j in range_constexpr(SEL_ITEMS)
                        ]

                    def poll_keys(items):
                        vals = poll(
                            [
                                (mb("iscores"), s * MAXP + fx.min(i, bound - 1), 1)
                                for i in items
                            ]
                        )
                        return [fx.Uint32(radix_key(v[0])) for v in vals]

                    def load_keys(items):
                        """Keys of consecutive item pairs, two (value, tag) pairs a load."""
                        out = []
                        for j in range_constexpr(0, len(items), 2):
                            w = fx.Vector(
                                bo.buffer_load(
                                    r_scores,
                                    (s * MAXP + fx.min(items[j], MAXP - 2)) * 2,
                                    vec_width=4,
                                    dtype=T.i32,
                                    cache_modifier=CM_DEV,
                                )
                            )
                            out += [
                                fx.Uint32(radix_key(w[0])),
                                fx.Uint32(radix_key(w[2])),
                            ]
                        return out

                    def clear_hist():
                        for b in range_constexpr(SEL_BINS // THREADS):
                            lds_st(hist, tid + b * THREADS, fx.Int32(0))
                        gpu.barrier()

                    def select_wide(shift, prefix, remain):
                        """The 12-bit digit at ``shift`` holding the remain-th largest
                        key: (prefix with it, keys still needed, keys in that bin)."""
                        per = SEL_BINS // THREADS
                        cs = [
                            lds_ld(hist, SEL_BINS - 1 - (tid * per + j))
                            for j in range(per)
                        ]
                        local = cs[0]
                        for c in cs[1:]:
                            local = local + c
                        inclusive = fx.coop.warp_inclusive_scan(
                            local, fx.ReductionOp.ADD, width=64
                        )
                        wave_total = read_lane_i32(inclusive, 63)
                        if lane == 63:
                            lds_st(keys, 256 + wave, wave_total)
                        gpu.barrier()
                        running = inclusive - local
                        for w in range_constexpr(WAVES):
                            running = running + (wave > w).select(
                                lds_ld(keys, 256 + w), fx.Int32(0)
                            )
                        for j in range_constexpr(per):
                            digit = SEL_BINS - 1 - (tid * per + j)
                            hit = (running < remain) & (running + cs[j] >= remain)
                            if hit:
                                lds_st(
                                    keys,
                                    280,
                                    fx.Int32(prefix | (fx.Uint32(digit) << shift)),
                                )
                                lds_st(keys, 281, remain - running)
                                lds_st(keys, 282, cs[j])
                            running = running + cs[j]
                        gpu.barrier()
                        return (
                            fx.Uint32(lds_ld(keys, 280)),
                            lds_ld(keys, 281),
                            lds_ld(keys, 282),
                        )

                    def scan_pair(first_flags, second_flags, parity):
                        """Exclusive block offsets of two flag lists in one scan
                        (running bases double-buffered by ``parity``)."""
                        local = fx.Int32(0)
                        local_offsets = []
                        for a, b in zip(first_flags, second_flags):
                            local_offsets.append(local)
                            local = (
                                local
                                + a.select(fx.Int32(1), fx.Int32(0))
                                + b.select(fx.Int32(1 << 16), fx.Int32(0))
                            )
                        inclusive = fx.coop.warp_inclusive_scan(
                            local, fx.ReductionOp.ADD, width=64
                        )
                        slot = 256 + parity * WAVES
                        if lane == 63:
                            lds_st(keys, slot + wave, read_lane_i32(inclusive, 63))
                        gpu.barrier()
                        before = fx.Int32(0)
                        total = fx.Int32(0)
                        for w in range_constexpr(WAVES):
                            wave_count = lds_ld(keys, slot + w)
                            before = before + (wave > w).select(wave_count, fx.Int32(0))
                            total = total + wave_count
                        thread_base = before + inclusive - local
                        base_a = lds_ld(keys, 272 + parity * 2)
                        base_b = lds_ld(keys, 273 + parity * 2)
                        offs = [thread_base + o for o in local_offsets]
                        n_a = total & fx.Int32(0xFFFF)
                        n_b = total >> 16
                        if tid == 0:
                            lds_st(keys, 272 + (1 - parity) * 2, base_a + n_a)
                            lds_st(keys, 273 + (1 - parity) * 2, base_b + n_b)
                        return (
                            [base_a + (o & fx.Int32(0xFFFF)) for o in offs],
                            [base_b + (o >> 16) for o in offs],
                        )

                    def set_bases(a, b):
                        if tid == 0:
                            lds_st(keys, 272, a)
                            lds_st(keys, 273, b)
                        gpu.barrier()

                    def gather(
                        chunk, valid, item_keys, values, threshold, out_gt, remain
                    ):
                        """Keys above ``threshold`` append to red from the running
                        base; the first ``remain`` ties fill red[out_gt:]."""
                        gt = [v & (k > threshold) for v, k in zip(valid, item_keys)]
                        eq = [v & (k == threshold) for v, k in zip(valid, item_keys)]
                        gt_offsets, eq_offsets = scan_pair(gt, eq, chunk & 1)
                        for j in range_constexpr(len(values)):
                            if gt[j]:
                                lds_st(
                                    red, gt_offsets[j], values[j].bitcast(fx.Float32)
                                )
                            if eq[j] & (eq_offsets[j] < remain):
                                lds_st(
                                    red,
                                    out_gt + eq_offsets[j],
                                    values[j].bitcast(fx.Float32),
                                )

                    def publish_all():
                        """red[0 .. 512) holds the pools; write the row's tokens, tagged."""
                        gpu.barrier()
                        for b in range_constexpr(TOPK_TOKENS // THREADS):
                            j = tid + b * THREADS
                            pool = lds_ld(red, j // KPOOL).bitcast(fx.Int32)
                            put(
                                mb("tok_sel"),
                                s * SEL_STRIDE + j,
                                pool * KPOOL + j % KPOOL,
                            )
                        if tid < (pos + 1) - bound * KPOOL:
                            put(
                                mb("tok_sel"),
                                s * SEL_STRIDE + TOPK_TOKENS + tid,
                                bound * KPOOL + tid,
                            )

                    def candidates(level, prefix, remain, n_cand):
                        """Keys above the ``level`` bin go to red, the bin's to LDS."""
                        set_bases(fx.Int32(0), fx.Int32(0))

                        def cand_chunk(chunk, items, item_keys):
                            high = [
                                (i < bound) & ((k >> level) > (prefix >> level))
                                for i, k in zip(items, item_keys)
                            ]
                            same = [
                                (i < bound) & ((k >> level) == (prefix >> level))
                                for i, k in zip(items, item_keys)
                            ]
                            win_offsets, cand_offsets = scan_pair(high, same, chunk & 1)
                            for j in range_constexpr(SEL_ITEMS):
                                if high[j]:
                                    lds_st(
                                        red,
                                        win_offsets[j],
                                        items[j].bitcast(fx.Float32),
                                    )
                                c = SEL_BINS + cand_offsets[j] * 2
                                if same[j]:
                                    lds_st(xs, c, item_keys[j].bitcast(fx.Float32))
                                    lds_st(xs, c + 1, items[j].bitcast(fx.Float32))

                        cand_chunk(fx.Int32(0), items0, keys0)
                        for chunk in range(fx.Int32(1), n_chunks, fx.Int32(1)):
                            chunk = fx.Int32(chunk)
                            items = chunk_items(chunk)
                            cand_chunk(chunk, items, load_keys(items))
                        gpu.barrier()
                        n_win = count - remain

                        def cand_pass(shift, mask, prefix, remain):
                            if tid < 256:
                                lds_st(keys, tid, fx.Int32(0))
                            gpu.barrier()
                            for c in range(tid, n_cand, fx.Int32(THREADS)):
                                key = fx.Uint32(
                                    lds_ld(xs, SEL_BINS + fx.Int32(c) * 2).bitcast(
                                        fx.Int32
                                    )
                                )
                                if (key & fx.Uint32(mask)) == prefix:
                                    digit = fx.Int32((key >> shift) & fx.Uint32(255))
                                    lds_atomic_add(
                                        keys + digit,
                                        fx.Int32(1),
                                        syncscope="workgroup",
                                    )
                            gpu.barrier()
                            return select_digit(shift, prefix, remain)

                        def rank_select():
                            """Each candidate (one a thread) counts the keys above and
                            equal to its own: the threshold key is the one whose count
                            above is below ``remain`` and with its ties reaches it."""
                            on = tid < n_cand
                            c = fx.min(tid, n_cand - 1)
                            key = fx.Uint32(
                                lds_ld(xs, SEL_BINS + c * 2).bitcast(fx.Int32)
                            )
                            item = lds_ld(xs, SEL_BINS + c * 2 + 1).bitcast(fx.Int32)
                            above = fx.Int32(0)
                            ties = fx.Int32(0)
                            for j in range(fx.Int32(0), n_cand, fx.Int32(1)):
                                other = fx.Uint32(
                                    lds_ld(xs, SEL_BINS + fx.Int32(j) * 2).bitcast(
                                        fx.Int32
                                    )
                                )
                                above = above + (other > key).select(
                                    fx.Int32(1), fx.Int32(0)
                                )
                                ties = ties + (other == key).select(
                                    fx.Int32(1), fx.Int32(0)
                                )
                            if on & (above < remain) & (above + ties >= remain):
                                lds_st(keys, 280, fx.Int32(key))
                                lds_st(keys, 281, remain - above)
                            set_bases(n_win, fx.Int32(0))
                            threshold = fx.Uint32(lds_ld(keys, 280))
                            remain_t = lds_ld(keys, 281)
                            gather(
                                fx.Int32(0),
                                [on],
                                [key],
                                [item],
                                threshold,
                                count - remain_t,
                                remain_t,
                            )
                            publish_all()

                        def radix_select(prefix, remain):
                            passes = (
                                ((12, 0xFFF00000), (4, 0xFFFFF000), (0, 0xFFFFFFF0))
                                if level == 20
                                else ((0, 0xFFFFFF00),)
                            )
                            for shift, mask in passes:
                                prefix, remain = cand_pass(shift, mask, prefix, remain)
                            set_bases(n_win, fx.Int32(0))
                            out_gt = count - remain
                            n_cand_chunks = (n_cand + SEL_CHUNK - 1) // SEL_CHUNK
                            for chunk in range(fx.Int32(0), n_cand_chunks, fx.Int32(1)):
                                chunk = fx.Int32(chunk)
                                idx = chunk_items(chunk)
                                safe = [fx.min(c, n_cand - 1) for c in idx]
                                cand = [
                                    lds_ld(xs, SEL_BINS + c * 2 + k).bitcast(fx.Int32)
                                    for k in range(2)
                                    for c in safe
                                ]
                                gather(
                                    chunk,
                                    [c < n_cand for c in idx],
                                    [fx.Uint32(k) for k in cand[: len(safe)]],
                                    cand[len(safe) :],
                                    prefix,
                                    out_gt,
                                    remain,
                                )
                            publish_all()

                        # one candidate a thread ranks directly; a larger bin takes
                        # radix passes
                        if n_cand <= THREADS:
                            rank_select()
                        if n_cand > THREADS:
                            radix_select(prefix, remain)

                    def full_radix():
                        """Byte-wise radix over every key, when the threshold bin
                        overflows LDS."""

                        def radix_pass(shift, prefix_mask, prefix, remain):
                            if tid < 256:
                                lds_st(keys, tid, fx.Int32(0))
                            gpu.barrier()
                            for chunk in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                                items = chunk_items(fx.Int32(chunk))
                                item_keys = load_keys(items)
                                for j in range_constexpr(SEL_ITEMS):
                                    key = item_keys[j]
                                    if (items[j] < bound) & (
                                        (key & fx.Uint32(prefix_mask)) == prefix
                                    ):
                                        digit = fx.Int32(
                                            (key >> shift) & fx.Uint32(255)
                                        )
                                        lds_atomic_add(
                                            keys + digit,
                                            fx.Int32(1),
                                            syncscope="workgroup",
                                        )
                            gpu.barrier()
                            return select_digit(shift, prefix, remain)

                        prefix, remain = radix_pass(24, 0, fx.Uint32(0), count)
                        prefix, remain = radix_pass(16, 0xFF000000, prefix, remain)
                        prefix, remain = radix_pass(8, 0xFFFF0000, prefix, remain)
                        prefix, remain = radix_pass(0, 0xFFFFFF00, prefix, remain)
                        set_bases(fx.Int32(0), fx.Int32(0))
                        out_gt = count - remain
                        for chunk in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                            chunk = fx.Int32(chunk)
                            items = chunk_items(chunk)
                            gather(
                                chunk,
                                [i < bound for i in items],
                                load_keys(items),
                                items,
                                prefix,
                                out_gt,
                                remain,
                            )
                        publish_all()

                    def hist_chunk(items, item_keys):
                        for j in range_constexpr(SEL_ITEMS):
                            if items[j] < bound:
                                lds_atomic_add(
                                    hist + fx.Int32(item_keys[j] >> 20),
                                    fx.Int32(1),
                                    syncscope="workgroup",
                                )

                    clear_hist()
                    # the first chunk's keys stay in registers for the passes after
                    items0 = chunk_items(fx.Int32(0))
                    keys0 = poll_keys(items0)
                    hist_chunk(items0, keys0)
                    for chunk in range(fx.Int32(1), n_chunks, fx.Int32(1)):
                        items = chunk_items(fx.Int32(chunk))
                        hist_chunk(items, poll_keys(items))
                    gpu.barrier()
                    p12, r12, c12 = select_wide(20, fx.Uint32(0), count)
                    if c12 <= SEL_CAP:
                        candidates(20, p12, r12, c12)
                    if c12 > SEL_CAP:
                        clear_hist()

                        def hist2_chunk(items, item_keys):
                            for j in range_constexpr(SEL_ITEMS):
                                k = item_keys[j]
                                if (items[j] < bound) & ((k >> 20) == (p12 >> 20)):
                                    lds_atomic_add(
                                        hist + fx.Int32((k >> 8) & fx.Uint32(0xFFF)),
                                        fx.Int32(1),
                                        syncscope="workgroup",
                                    )

                        hist2_chunk(items0, keys0)
                        for chunk in range(fx.Int32(1), n_chunks, fx.Int32(1)):
                            items = chunk_items(fx.Int32(chunk))
                            hist2_chunk(items, load_keys(items))
                        gpu.barrier()
                        p24, r24, c24 = select_wide(8, p12, r12)
                        if c24 <= SEL_CAP:
                            candidates(8, p24, r24, c24)
                        if c24 > SEL_CAP:
                            full_radix()

        # ================= sparse MLA split: 64 selected keys x HT heads a task
        # Scored rows attend to the tokens the index selected plus their tail, other
        # rows to every token. Splits past the row's keys do nothing (split 0
        # always publishes, so empty rows merge to zero). The tail splits come first:
        # their tokens follow the row's last complete pool, so they do not wait for
        # the selection and run before the CTA's regular split waits for it.
        def dsa_split(tv, t32):
            KVB = t32["kv_block_tokens"]
            KVE = tv["kv_block_elems"]
            KVR = t32["kv_row_elems"]
            KPW = SPLIT_KEYS // WAVES
            HPW = HT // WAVES
            N_TAIL = S * HG
            ktile = xs + KT_OFF
            pl = xs  # q is dead once the scores are in ``red``
            for tt in range(start("split"), S * HG * N_SPLIT, G):
                tt = fx.Int32(tt)
                is_tail = tt < N_TAIL
                rt = tt - N_TAIL
                s = is_tail.select(tt // HG, rt // (HG * BASE_SPLITS))
                hg = is_tail.select(tt % HG, (rt // BASE_SPLITS) % HG)
                t = is_tail.select(fx.Int32(BASE_SPLITS), rt % BASE_SPLITS)
                pos = row_word(s, ROW_POS)
                bound, long_row, nk = row_keys(pos)
                picked_keys = long_row & (t < BASE_SPLITS)
                n_used = fx.max((nk + SPLIT_KEYS - 1) // SPLIT_KEYS, fx.Int32(1))
                if t < n_used:
                    bt = row_word(s, ROW_BT)
                    r_bt = _rsrc(
                        page_table + fx.Int64(bt) * fx.Int64(pt_stride) * fx.Int64(4)
                    )

                    def resolve(tok):
                        """Wave 0: lane's key's latent block and offset -> LDS keys."""
                        lds_st(keys, lane, ld_i32(r_bt, tok // KVB))
                        lds_st(keys, SPLIT_KEYS + lane, tok % KVB)

                    def kv_row(blk, off):
                        return fx.Vector(
                            bo.buffer_load(
                                _rsrc(
                                    tv["kv_cache"]
                                    + (fx.Int64(blk) * KVE + fx.Int64(off * KVR))
                                    * fx.Int64(2)
                                ),
                                lane * 4,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )

                    kidx = t * SPLIT_KEYS + lane
                    # A task that reads the selection waits for it: wave 0 polls its 64
                    # tagged tokens and the cache's flag together, then acquires; the
                    # barrier below orders the other waves' loads after it. Any other task
                    # knows its tokens and gathers them at once; of those only the row's
                    # own token (its last key) is written by this launch's cache task, and
                    # it is read again once kv_ready is in.
                    if picked_keys:
                        if wave == 0:
                            picked = poll(
                                [(mb("tok_sel"), s * SEL_STRIDE + kidx, 1)]
                                + chain_ready(s)
                            )[0][0]
                            memory_fence(syncscope="agent", ordering="acquire")
                            resolve((kidx < nk).select(picked, fx.Int32(0)))
                    else:
                        if wave == 0:
                            if const_expr(VERIFY):
                                # the chain's earlier rows' keys are among the
                                # task's: wait for all of them first
                                poll(chain_ready(s))
                                memory_fence(syncscope="agent", ordering="acquire")
                            # a scored row's tail follows its last complete pool; keys
                            # past the row's address its first token (masked)
                            own = long_row.select(
                                kidx - TOPK_TOKENS + bound * KPOOL, kidx
                            )
                            resolve((kidx < nk).select(own, fx.Int32(0)))
                    gpu.barrier()
                    # the wave's latent rows: every load issued before any store
                    locs = [
                        (
                            _uniform(lds_ld(keys, wave * KPW + jj)),
                            _uniform(lds_ld(keys, SPLIT_KEYS + wave * KPW + jj)),
                        )
                        for jj in range(KPW)
                    ]
                    rows_kv = [kv_row(blk, off) for blk, off in locs]
                    for jj in range_constexpr(KPW):
                        fx.ptr_store(
                            rows_kv[jj].bitcast(fx.Float32),
                            ktile + ((wave * KPW + jj) * QS + lane * 4),
                        )
                    # q_lat of the task's HT heads -> Q[h] (words h * QS ..)
                    NQ = HT * KV_LORA // 4 // THREADS
                    qv = poll(
                        [
                            (
                                mb("q_lat"),
                                ((s * MH + hg * HT) * KV_LORA + (tid + i * THREADS) * 4)
                                // 2,
                                2,
                            )
                            for i in range(NQ)
                        ]
                    )
                    for i in range_constexpr(NQ):
                        q4 = tid + i * THREADS
                        qw = (q4 // (KV_LORA // 4)) * QS + (q4 % (KV_LORA // 4)) * 2
                        lds_st(xs, qw, qv[i][0].bitcast(fx.Float32))
                        lds_st(xs, qw + 1, qv[i][1].bitcast(fx.Float32))
                    gpu.barrier()
                    # (a verify row waited for its chain's keys before loading them)
                    own_j = (
                        fx.Int32(-1)
                        if VERIFY
                        else picked_keys.select(fx.Int32(-1), nk - 1 - t * SPLIT_KEYS)
                    )
                    if (own_j >= 0) & (own_j < SPLIT_KEYS):
                        if wave == 0:
                            if lane == 0:
                                get(mb("kv_ready"), s)
                            memory_fence(syncscope="agent", ordering="acquire")
                            own_row = kv_row(
                                _uniform(lds_ld(keys, own_j)),
                                _uniform(lds_ld(keys, SPLIT_KEYS + own_j)),
                            )
                            fx.ptr_store(
                                own_row.bitcast(fx.Float32),
                                ktile + (own_j * QS + lane * 4),
                            )
                        gpu.barrier()
                    # scores = K Q^T: wave (16-key row group wave % 4, K half wave / 4)
                    hn = fx.min(lane % 16, HT - 1)
                    rgk = wave % 4
                    c = fx.Vector.filled(4, 0.0, fx.Float32)
                    for st in range_constexpr(KV_LORA // 32 // 2):
                        kst = (wave // 4) * (KV_LORA // 32 // 2) + st
                        key = rgk * 16 + lane % 16
                        a = fx.ptr_load(
                            ktile + (key * QS + kst * 16 + (lane // 16) * 4),
                            result_type=v4f,
                        ).bitcast(fx.BFloat16)
                        b = fx.ptr_load(
                            xs + (hn * QS + kst * 16 + (lane // 16) * 4),
                            result_type=v4f,
                        ).bitcast(fx.BFloat16)
                        c = fx.Vector(
                            rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c])
                        )
                    fx.ptr_store(c, red + (wave * 64 + lane) * 4)
                    gpu.barrier()
                    # split-local softmax: heads wave + WAVES j, lane = key
                    valid = (t * SPLIT_KEYS + lane) < nk
                    r16 = lane % 16
                    krg = lane // 16
                    m_l = []
                    for j in range_constexpr(HPW):
                        cl = wave + WAVES * j + 16 * (r16 // 4)
                        raw_s = lds_ld(red, (krg * 64 + cl) * 4 + r16 % 4) + lds_ld(
                            red, ((krg + 4) * 64 + cl) * 4 + r16 % 4
                        )
                        sc_v = valid.select(raw_s * MLA_SCALE, fx.Float32(NEG))
                        m = wave_max(sc_v)
                        p = valid.select(_exp(sc_v - m), fx.Float32(0.0))
                        m_l.append((m, wave_sum(p)))
                        p_n = _xshfl(p, 1)
                        if lane % 2 == 0:
                            lds_st(
                                pl,
                                (wave + WAVES * j) * (SPLIT_KEYS // 2) + lane // 2,
                                bf16_pair(p, p_n),
                            )
                    gpu.barrier()
                    # O = P V: heads M, keys K, latent dim pairs N (even / odd MFMAs)
                    for gq in range_constexpr(KV_LORA // 32 // WAVES):
                        dw = (wave * (KV_LORA // 32 // WAVES) + gq) * 16 + lane % 16
                        c0 = fx.Vector.filled(4, 0.0, fx.Float32)
                        c1 = fx.Vector.filled(4, 0.0, fx.Float32)
                        for js in range_constexpr(SPLIT_KEYS // 32):
                            a = fx.ptr_load(
                                pl
                                + (hn * (SPLIT_KEYS // 2) + js * 16 + (lane // 16) * 4),
                                result_type=v4f,
                            ).bitcast(fx.BFloat16)
                            ws = [
                                fx.ptr_load(
                                    ktile + ((js * 32 + (lane // 16) * 8 + i) * QS + dw)
                                ).bitcast(fx.Int32)
                                for i in range(8)
                            ]
                            w_lo = [
                                (ws[2 * i] & 0xFFFF) | (ws[2 * i + 1] << 16)
                                for i in range(4)
                            ]
                            w_hi = [
                                fx.Int32(fx.Uint32(ws[2 * i]) >> 16)
                                | (ws[2 * i + 1] & -65536)
                                for i in range(4)
                            ]
                            b0 = fx.Vector.from_elements(w_lo, fx.Int32).bitcast(
                                fx.BFloat16
                            )
                            b1 = fx.Vector.from_elements(w_hi, fx.Int32).bitcast(
                                fx.BFloat16
                            )
                            c0 = fx.Vector(
                                rocdl.mfma_f32_16x16x32_bf16(
                                    T.vec(4, T.f32), [a, b0, c0]
                                )
                            )
                            c1 = fx.Vector(
                                rocdl.mfma_f32_16x16x32_bf16(
                                    T.vec(4, T.f32), [a, b1, c1]
                                )
                            )
                        if lane < HT * 4:
                            for e in range_constexpr(4):
                                hh = hg * HT + (lane // 16) * 4 + e
                                put_bf(
                                    mb("sp_acc"),
                                    ((s * N_SPLIT + t) * MH + hh) * KV_LORA + dw * 2,
                                    [c0[e], c1[e]],
                                )
                    if lane == 0:
                        for j in range_constexpr(HPW):
                            ml = (s * N_SPLIT + t) * MH + hg * HT + wave + WAVES * j
                            put(mb("sp_m"), ml, m_l[j][0])
                            put(mb("sp_l"), ml, m_l[j][1])
                    gpu.barrier()

        # ===== split merge + W_UV: 64 v rows of one (row, head) a task
        def dsa_uv(tv, t32):
            r_wkvb = _rsrc(tv["w_kvb"])
            UV_R = UV_TILE // 16
            UV_WPR = WAVES // UV_R
            UV_CPW = KV_LORA // 64 // UV_WPR
            SPH = BASE_SPLITS // 2
            UV_TPT = uv_tiles(S)
            UV_NT = V_DIM // UV_TILE // UV_TPT
            for tt in range(start("uv"), S * MH * UV_NT, G):
                tt = fx.Int32(tt)
                s = tt // (MH * UV_NT)
                head = (tt // UV_NT) % MH
                tile0 = (tt % UV_NT) * UV_TPT
                pos = row_word(s, ROW_POS)
                _, _, nk = row_keys(pos)
                n_used = fx.max((nk + SPLIT_KEYS - 1) // SPLIT_KEYS, fx.Int32(1))

                def u_uv(tile, c):
                    row0 = head * (NOPE_DIM + V_DIM) + NOPE_DIM + tile * UV_TILE
                    row0 = row0 + (wave // UV_WPR) * 16
                    kc = (wave % UV_WPR) * UV_CPW + c
                    return unit_rm(
                        r_wkvb, row0, kc, KV_LORA, MH * (NOPE_DIM + V_DIM), kc * 32
                    )

                pre = [
                    [u_uv(tile0 + j, c) for c in range(UV_CPW)] for j in range(UV_TPT)
                ]
                if wave == 0:  # split weights exp(m - M) / L -> misc[split]
                    spi = fx.min(lane, n_used - 1)
                    sp_ml = (s * N_SPLIT + spi) * MH + head
                    mlv = poll([(mb("sp_m"), sp_ml, 1), (mb("sp_l"), sp_ml, 1)])
                    ok_sp = lane < n_used
                    m_sp = ok_sp.select(mlv[0][0].bitcast(fx.Float32), fx.Float32(NEG))
                    l_sp = ok_sp.select(mlv[1][0].bitcast(fx.Float32), fx.Float32(0.0))
                    m_all = wave_max(m_sp)
                    w_sp = _exp(m_sp - m_all)
                    den = wave_sum(l_sp * w_sp)
                    if lane < N_SPLIT:
                        lds_st(
                            misc,
                            lane,
                            ok_sp.select(
                                w_sp * (den > 0.0).select(_rcp(den), fx.Float32(0.0)),
                                fx.Float32(0.0),
                            ),
                        )
                gpu.barrier()
                dp = tid % (KV_LORA // 2)
                hf = tid // (KV_LORA // 2)
                got = poll(
                    [
                        (
                            mb("sp_acc"),
                            (
                                (s * N_SPLIT + fx.min(hf * SPH + q, n_used - 1)) * MH
                                + head
                            )
                            * (KV_LORA // 2)
                            + dp,
                            1,
                        )
                        for q in range(SPH)
                    ]
                )
                o0 = fx.Float32(0.0)
                o1 = fx.Float32(0.0)
                for q in range_constexpr(SPH):
                    wj = lds_ld(misc, hf * SPH + q)
                    a0, a1 = bf2_f32(got[q][0])
                    o0 = o0 + a0 * wj
                    o1 = o1 + a1 * wj
                # the split past the first BASE_SPLITS (the tail's keys) joins the
                # second half
                if (n_used > BASE_SPLITS) & (hf == 1):
                    last = poll(
                        [
                            (
                                mb("sp_acc"),
                                ((s * N_SPLIT + BASE_SPLITS) * MH + head)
                                * (KV_LORA // 2)
                                + dp,
                                1,
                            )
                        ]
                    )[0][0]
                    wj = lds_ld(misc, BASE_SPLITS)
                    a0, a1 = bf2_f32(last)
                    o0 = o0 + a0 * wj
                    o1 = o1 + a1 * wj
                lds_st(red, (hf * (KV_LORA // 2) + dp) * 2, o0)
                lds_st(red, (hf * (KV_LORA // 2) + dp) * 2 + 1, o1)
                gpu.barrier()
                if tid < KV_LORA // 2:
                    o0 = lds_ld(red, tid * 2) + lds_ld(red, (KV_LORA // 2 + tid) * 2)
                    o1 = lds_ld(red, tid * 2 + 1) + lds_ld(
                        red, (KV_LORA // 2 + tid) * 2 + 1
                    )
                    lds_st(xs, tid, bf16_pair(o0, o1))
                gpu.barrier()
                for j in range_constexpr(UV_TPT):
                    acc = run_units(None, UV_CPW, UV_CPW, pre[j])
                    reduce_rows(UV_R, acc, emit_out(UV_TILE), 1)
                    gpu.barrier()
                    if tid < UV_TILE // 4:
                        r = tid * 4
                        put_bf(
                            mb("o_in"),
                            (s * MH + head) * V_DIM + (tile0 + j) * UV_TILE + r,
                            [lds_ld(outs, r + i) for i in range(4)],
                        )
                    gpu.barrier()

        # ======================= o_proj (FP8 or bf16) partial -> every peer
        def dsa_oproj(tv, t32):
            r_wo, r_so = _rsrc(tv["w_o"]), _rsrc(tv["s_o"])
            OK = MH * V_DIM
            O8_CPW = OK // (128 if O8 else 64) // WAVES
            run_oproj(
                lambda row0, kc: proj_unit(
                    O8, r_wo, r_so, row0, kc, OK, HIDDEN, n_sel() * (OK // 2), O_SH
                ),
                OK,
                O8_CPW,
                min(4, O8_CPW),
            )

        def run_dsa():
            # the layer's argument table (``DSA_TAB`` order): pointers, then int32 geometry;
            # AKA passes its entries as kernel arguments (upstream: a device table read
            # one entry a lane)
            tab_args = [
                tab_w_qkv,
                tab_s_qkv,
                tab_w_idx,
                tab_w_gate,
                tab_g_qa,
                tab_g_kv,
                tab_ln_w,
                tab_ln_b,
                tab_ape,
                tab_w_qb,
                tab_s_qb,
                tab_w_iq,
                tab_w_uk,
                tab_w_kvb,
                tab_w_o,
                tab_s_o,
                tab_kv_cache,
                tab_idx_cache,
                tab_tail_k,
                tab_tail_g,
                tab_idx_bt_group,
                tab_kv_block_elems,
                tab_kv_block_tokens,
                tab_kv_row_elems,
                tab_idx_block_bytes,
                tab_idx_block_pools,
                tab_tail_block_elems,
            ]
            assert len(tab_args) == len(DSA_TAB)
            tv = {n: fx.Int64(v) for n, v in zip(DSA_TAB, tab_args)}
            t32 = {n: fx.Int32(v & fx.Int64(0xFFFFFFFF)) for n, v in tv.items()}
            run_seam("aseam", "attn", early_seam)
            run_gate("agate", "attn", early_gate)
            # the step's row table -> LDS, read from the cache stage on (which
            # begins with a barrier)
            if tid < S * ROW_WORDS:
                lds_st(rows_lds, tid, row_words)
            dsa_qkv_fp8(tv, t32)
            dsa_qkv_bf16(tv, t32)
            dsa_qnorm(tv, t32)
            dsa_cache(tv, t32)
            dsa_q_b(tv, t32)
            dsa_index_q(tv, t32)
            dsa_iq_quant(tv, t32)
            dsa_uk(tv, t32)
            dsa_iscore(tv, t32)
            dsa_iselect(tv, t32)
            dsa_split(tv, t32)
            dsa_uv(tv, t32)
            dsa_oproj(tv, t32)

        # ===================== the model's tail: mHC post, stream mean, final norm, LM head
        # hpost: one task a 32-column tile, every row: the four streams as the terminal
        # layer's hc_post rounds them, their mean (hc_contract) to mailbox ``hc``, and the
        # tile's sum of squares of it to ``hss``. lmhead: every CTA normalizes the rows
        # (rstd from the tiles' sums in a fixed order; bf16 as SGLang's RMSNorm stores
        # them), then runs 16-row tiles of this rank's vocabulary and pushes the bf16
        # logits to every rank. gather: each CTA turns its slice of the gathered logits
        # into f32.
        VR = SH.vocab
        N_VT = VR // HEAD_TILE
        HD_CPW = HIDDEN // 64 // WAVES
        HD_AHEAD = HD_CPW if S <= 4 else HD_CPW // 2
        HD_ROUNDS = (N_VT + G - 1) // G
        G_PAIRS = S * VOCAB // 2
        G_PER = (G_PAIRS + G - 1) // G
        assert VR % HEAD_TILE == 0 and N_ROW_TILES <= 2 * 64

        def run_head():
            r_x, r_res = _rsrc(attn_part), _rsrc(res_in)
            r_post, r_comb = _rsrc(post_in), _rsrc(comb_in)
            for t in range(start("hpost"), N_ROW_TILES, G):
                t = fx.Int32(t)
                col0 = t * ROW_TILE
                if tid < S * (ROW_TILE // 2):
                    s = tid // (ROW_TILE // 2)
                    c = (tid % (ROW_TILE // 2)) * 2
                    x0, x1 = ld_pair(r_x, s * HIDDEN + col0 + c)
                    rs = [
                        ld_pair(r_res, (s * HC_MULT + i) * HIDDEN + col0 + c)
                        for i in range(HC_MULT)
                    ]
                    h = []
                    for e in range_constexpr(2):
                        xe = x0 if e == 0 else x1
                        tot = fx.Float32(0.0)
                        for j in range_constexpr(HC_MULT):
                            acc = fx.Float32(0.0)
                            for i in range_constexpr(HC_MULT):
                                acc = (
                                    acc
                                    + ld_f32(r_comb, (s * HC_MULT + i) * HC_MULT + j)
                                    * rs[i][e]
                                )
                            tot = tot + bf16_round(
                                acc + ld_f32(r_post, s * HC_MULT + j) * xe
                            )
                        h.append(bf16_round(tot * (1.0 / HC_MULT)))
                    put(mb("hc"), (s * HIDDEN + col0 + c) // 2, bf16_pair(h[0], h[1]))
                    ss = h[0] * h[0] + h[1] * h[1]
                    for off in (1, 2, 4, 8):
                        ss = _xred(ss, off, lambda a, b: a + b)
                    if c == 0:
                        put(mb("hss"), s * N_ROW_TILES + t, ss)
                gpu.barrier()
            r_wh, r_g = _rsrc(w_r), _rsrc(g_norm)
            logits_base = fx.Int64(SY["logits"]) + fx.Int64(peer_slot) * fx.Int64(
                SY["_logits_stride"]
            )
            t0 = start("lmhead")

            def u_h(row0, c):
                kc = wave * HD_CPW + c
                return unit_rm(
                    r_wh, row0, kc, HIDDEN, VR, n_sel() * (HIDDEN // 2) + kc * 32
                )

            def ahead(tile):
                # past the last tile: the last row only (dropped)
                row0 = (tile < N_VT).select(tile * HEAD_TILE, fx.Int32(VR))
                return [u_h(row0, c) for c in range(HD_AHEAD)]

            def tile_task(k, tile, pre):
                acc = run_units(
                    lambda c: u_h(tile * HEAD_TILE, c), HD_CPW, HD_AHEAD, pre
                )
                nxt = ahead(tile + G) if k + 1 < HD_ROUNDS else None
                reduce_rows(1, acc, emit_out(HEAD_TILE))
                gpu.barrier()
                if wave < W:
                    for li in range_constexpr((S * HEAD_TILE // 2 + 63) // 64):
                        pq = lane + li * 64
                        if pq < S * HEAD_TILE // 2:
                            si = pq // (HEAD_TILE // 2)
                            ri = (pq % (HEAD_TILE // 2)) * 2
                            put_bf(
                                peer_dst + logits_base,
                                si * VOCAB + rank * VR + tile * HEAD_TILE + ri,
                                [
                                    bf16_round(lds_ld(outs, si * HEAD_TILE + ri)),
                                    bf16_round(lds_ld(outs, si * HEAD_TILE + ri + 1)),
                                ],
                                CM_SYS,
                            )
                gpu.barrier()
                if const_expr(k + 1 < HD_ROUNDS):
                    if tile + G < N_VT:
                        tile_task(k + 1, tile + G, nxt)

            pre = ahead(t0)
            gam = ld_bf16s(r_g, tid * CPT, CPT)
            # the rows' rstd: wave w sums rows w, w + WAVES, ..'s tile sums (lanes l,
            # l + 64), then the wave in a fixed order
            for wi in range_constexpr((S + WAVES - 1) // WAVES):
                sw = wave + wi * WAVES
                if sw < S:
                    parts = getf_many(
                        [
                            (
                                mb("hss"),
                                sw * N_ROW_TILES
                                + fx.min(lane + 64 * i, N_ROW_TILES - 1),
                            )
                            for i in range(2)
                        ]
                    )
                    tot = parts[0] + (lane + 64 < N_ROW_TILES).select(
                        parts[1], fx.Float32(0.0)
                    )
                    tot = wave_sum(tot)
                    if lane == 0:
                        lds_st(misc, sw, _rsq(tot * (1.0 / HIDDEN) + EPS))
            stage_rows(mb("hc"), HIDDEN)
            gpu.barrier()
            # normalize in place: thread tid holds columns tid * CPT .. of every row
            for s in range_constexpr(S):
                rstd = lds_ld(misc, s)
                for q in range_constexpr(CPT // 2):
                    wd = s * (HIDDEN // 2) + tid * (CPT // 2) + q
                    v0, v1 = bf2_f32(lds_ld(xs, wd).bitcast(fx.Int32))
                    lds_st(
                        xs,
                        wd,
                        bf16_pair(
                            (v0 * rstd) * gam[2 * q], (v1 * rstd) * gam[2 * q + 1]
                        ),
                    )
            gpu.barrier()
            tile_task(0, t0, pre)
            own = sym + logits_base
            r_out = _rsrc(x_out)
            n_g = (G_PER + THREADS - 1) // THREADS
            qs = [
                fx.min(bid * G_PER + fx.min(tid + i * THREADS, G_PER - 1), G_PAIRS - 1)
                for i in range(n_g)
            ]
            got = poll([(own, q, 1) for q in qs], "one-as")
            for i in range_constexpr(n_g):
                if tid + i * THREADS < G_PER:
                    v0, v1 = bf2_f32(got[i][0])
                    bo.buffer_store(
                        fx.Vector.from_elements([v0, v1], fx.Float32), r_out, qs[i] * 2
                    )

        # ------------------------------------------------------------ the epoch
        # The first stage's loads that wait for nothing go out with the prologue's, then
        # the prologue's are read.
        early_seam = early_gate = None
        if const_expr(EARLY):
            early_seam = seam_early()
            early_gate = gate_early()
        if const_expr(PRO):
            tag, peer_slot = read_epoch(step_word)
            peer_dst = read_peer(pv)
            launch_guard = new_guard()

        if const_expr(head):
            run_head()
            return
        if const_expr(kda):
            r_wo_kda = _rsrc(w_o)
            O_CPW = P // 64 // WAVES
            if const_expr(tokens):
                write_step_tables()
            run_seam("aseam", "attn", early_seam)
            run_gate("agate", "attn", early_gate)
            run_inproj()
            run_fgb()

            def kda_oproj(done=None):
                run_oproj(
                    lambda row0, kc: unit_rm(
                        r_wo_kda, row0, kc, P, HIDDEN, n_sel() * (P // 2) + kc * 32
                    ),
                    P,
                    O_CPW,
                    min(4, O_CPW),
                    done,
                )

            # a CTA with a kda task (at most one) runs its o_proj task inside it
            assert S * H <= G
            if const_expr(VERIFY):
                run_kda_verify()
            else:
                run_kda(kda_oproj)
            kda_oproj(start("kda") < S * H if OPROJ_IN_KDA else None)
            run_seam("seam", "ffn_own")
        elif const_expr(dsa):
            run_dsa()
            run_seam("seam", "ffn_own")
        else:
            run_seam("seam", "ffn")
        run_gate("gate", "ffn")
        # ======== dense MLP (layers 0-2): FP8 gate | up + clamped SiLU, FP8 down + TP reduce
        # bf16 activations; gate and up rounded to bf16 as SGLang's GEMM stores them, mid
        # rounded to bf16 as its activation stores it
        DI = SH.dense_inter
        DXW = HIDDEN // 2
        DUG_CPW = HIDDEN // FP8_BLOCK // (WAVES // 2)
        DUG_AHEAD = DUG_CPW if S <= 4 else DUG_CPW // 2
        DDN_KB = DI // FP8_BLOCK
        DDN_CPW = -(-DDN_KB // WAVES)

        def store_x(s, row, v0, v1):
            bo.buffer_store(
                fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16),
                _rsrc(x_out),
                s * HIDDEN + row,
            )

        def run_dense():
            r_gu, r_gus = _rsrc(w_ug), _rsrc(s_ug)
            r_dn, r_dns = _rsrc(w_dn), _rsrc(s_dn)
            # one task 16 gate rows (waves 0-3) and the same 16 up rows (waves 4-7)
            for t in range(start("dug"), DI // DUG_TILE, G):
                t = fx.Int32(t)
                row0 = (wave // (WAVES // 2)) * DI + t * DUG_TILE

                def u_gu(c):
                    kb = (wave % (WAVES // 2)) * DUG_CPW + c
                    return unit_fp8_rm(
                        r_gu,
                        r_gus,
                        row0,
                        kb,
                        HIDDEN,
                        2 * DI,
                        n_sel() * DXW + kb * 64,
                        HIDDEN // FP8_BLOCK,
                        DENSE_SH,
                    )

                pre = [u_gu(c) for c in range(DUG_AHEAD)]
                wait_ready("box_a")
                stage_rows(mb("a"), HIDDEN)
                gpu.barrier()
                acc = run_units(u_gu, DUG_CPW, DUG_AHEAD, pre)
                reduce_rows(2, acc, emit_out(2 * DUG_TILE))
                gpu.barrier()
                if tid < S * DUG_TILE // 2:
                    n = tid // (DUG_TILE // 2)
                    r = (tid % (DUG_TILE // 2)) * 2
                    mids = []
                    for j in range_constexpr(2):
                        g = fx.min(
                            bf16_round(lds_ld(outs, n * 2 * DUG_TILE + r + j)),
                            SWIGLU_LIMIT,
                        )
                        v = fx.min(
                            fx.max(
                                bf16_round(
                                    lds_ld(outs, n * 2 * DUG_TILE + DUG_TILE + r + j)
                                ),
                                -SWIGLU_LIMIT,
                            ),
                            SWIGLU_LIMIT,
                        )
                        mids.append(g * _rcp(1.0 + _exp(-g)) * v)
                    put(
                        mb("dmid"),
                        (n * DI + t * DUG_TILE + r) // 2,
                        bf16_pair(mids[0], mids[1]),
                    )
                gpu.barrier()
            # one task 16 output rows of the whole K, then the TP reduce of the tile
            for t in range(start("down"), HIDDEN // DN_TILE, G):
                t = fx.Int32(t)
                gpu.barrier()
                row0 = t * DN_TILE

                def u_dn(c):
                    kb = wave * DDN_CPW + c
                    if const_expr(DDN_KB % WAVES):
                        kb = fx.min(kb, DDN_KB - 1)
                    unit = unit_fp8_rm(
                        r_dn,
                        r_dns,
                        row0,
                        kb,
                        DI,
                        HIDDEN,
                        n_sel() * (DI // 2) + kb * 64,
                        DDN_KB,
                        DENSE_SH,
                    )
                    if const_expr(DDN_KB % WAVES):
                        fmt, wv, sc, bw = unit
                        live = wave * DDN_CPW + c < DDN_KB
                        unit = (fmt, wv, live.select(sc, fx.Float32(0.0)), bw)
                    return unit

                pre = [u_dn(c) for c in range(DDN_CPW)]
                stage_rows(mb("dmid"), DI)
                gpu.barrier()
                acc = run_units(u_dn, DDN_CPW, DDN_CPW, pre)
                reduce_rows(1, acc, emit_out(DN_TILE))
                gpu.barrier()
                peer_reduce("ffn", t, store_x, DN_TILE)
                gpu.barrier()

        if const_expr(dense):
            run_dense()
            return
        # ====== 2. router scores + the row's MXFP8 activation blocks (one sample a CTA)
        r_wr = _rsrc(w_r)
        R_NKC = HIDDEN // 64
        R_CPW = R_NKC // WAVES // 2
        ROUTER_OWN_ROW = S * N_ROUTER <= G
        for tt in range(start("router"), S * N_ROUTER, G):
            tt = fx.Int32(tt)
            t = tt % N_ROUTER
            router_sample = tt // N_ROUTER
            # K-fold: MFMA rows / B columns 0..7 take this wave's first K half, rows /
            # columns 8..15 the second; logit = C[r][n] + C[8 + r][8 + n]
            r_sub = t * ROUTER_TILE % 16
            r_ln = (lane & -16) | (r_sub + lane % ROUTER_TILE)
            r_fold = (lane % 16) // ROUTER_TILE

            def u_r(c):
                kc = wave * (R_NKC // WAVES) + r_fold * R_CPW + c
                return unit_bf16(r_wr, t * ROUTER_TILE // 16, kc, R_NKC, kc * 32, r_ln)

            pre = [u_r(c) for c in range(R_CPW)]
            # the row, bf16, into LDS words 0 .. HIDDEN / 2: computed from the seam's
            # streams as its gate computes it when no CTA runs two router tasks, else
            # the gate's
            if const_expr(ROUTER_OWN_ROW):
                li, rstd, gam, _ = gate_row(
                    "router", "seam", router_sample, False, xs + PM_WORDS
                )
                for q in range_constexpr(CPT // 2):
                    lds_st(
                        xs,
                        tid * (CPT // 2) + q,
                        bf16_pair(
                            li[2 * q] * rstd * gam[2 * q],
                            li[2 * q + 1] * rstd * gam[2 * q + 1],
                        ),
                    )
            else:
                wait_ready("box_a", router_sample)
                nq = HIDDEN // 4
                vals = poll(
                    [
                        (
                            mb("a"),
                            (router_sample * HIDDEN + (tid + i * THREADS) * 4) // 2,
                            2,
                        )
                        for i in range(nq // THREADS)
                    ]
                )
                for i in range_constexpr(nq // THREADS):
                    for j in range_constexpr(2):
                        lds_st(
                            xs,
                            (tid + i * THREADS) * 2 + j,
                            vals[i][j].bitcast(fx.Float32),
                        )
            gpu.barrier()
            acc = run_units(u_r, R_CPW, R_CPW, pre)
            fx.ptr_store(
                fx.Vector.from_elements(acc, fx.Float32),
                red + (wave * 64 + lane) * 4,
            )
            gpu.barrier()
            if tid < ROUTER_TILE:
                r = tid % ROUTER_TILE
                logit = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    for f in range_constexpr(2):
                        m = f * ROUTER_TILE + r
                        logit = logit + lds_ld(
                            red,
                            (w * 64 + f * ROUTER_TILE + 16 * (m // 4)) * 4 + m % 4,
                        )
                # SGLang rounds the router GEMM's logits to bf16
                logit = bf16_round(logit)
                put(
                    mb("scores"),
                    router_sample * N_EXPERTS + t * ROUTER_TILE + r,
                    _rcp(1.0 + _exp(-logit)),
                )
            gpu.barrier()

        def dn_route(bs):
            """Expert routing: one whole wave per sample, in wave-sized batches.
            Slot 0 is the shared expert (weight 1), slot k + 1 pick k."""
            for sample_batch in range_constexpr(sample_wave_batches(S)):
                route_sample = wave + sample_batch * WAVES
                if route_sample < S:
                    e, w = route_top8(route_sample, bs)
                    if lane < MOE_SLOTS:
                        q = route_sample * MOE_SLOTS + (lane + 1) % MOE_SLOTS
                        lds_st(keys, q, (lane == TOP_K).select(fx.Int32(SHARED), e))
                        lds_st(dnw, q, (lane == TOP_K).select(fx.Float32(1.0), w))

        # ===================== 3. expert up/gate + clamped SiLU
        UG_W_BYTES = 2 * expert_inter * HIDDEN // 2
        UG_S_BYTES = 2 * expert_inter * (HIDDEN // 32)
        UG8 = 8
        UG_KU = (HIDDEN // 128) // WAVES
        assert UG_KU == 4 and UG_RG == 4
        XW = HIDDEN // 4

        def run_grouped():
            """The expert-major MoE. Every CTA builds the step's route table; a unit is
            an expert's next (at most) 16 picks, its rows the MFMA columns. An up/gate
            task is a unit's 32 intermediate values, a down task a unit's HC hidden
            rows (each pick's output, unweighted, to ``dpart``), a sum task 16 hidden
            rows of every row (its slots in order, route-weighted) and their TP
            reduce. A row's values never depend on the other rows of its units. A CTA
            runs a contiguous range of a stage's tasks, so it stages a unit's rows
            once."""
            rti = fx.recast_iter(fx.Int32, xs)
            dn_route(load_bias())
            gpu.barrier()
            # each expert's row mask: wave w the experts w, w + 8, ..; lane s row s
            wv = _uniform(wave)
            row_keys = [
                lds_ld(keys, fx.min(lane, S - 1) * MOE_SLOTS + k)
                for k in range(MOE_SLOTS)
            ]
            for i in range_constexpr((NE1 + WAVES - 1) // WAVES):
                e = wv + i * WAVES
                if e < NE1:
                    hit = row_keys[0] == e
                    for k in range_constexpr(1, MOE_SLOTS):
                        hit = hit | (row_keys[k] == e)
                    bits = ballot(hit & (lane < S))
                    if lane == 0:
                        lds_st(rti, RT_MASK + 2 * e, fx.Int32(bits))
                        lds_st(rti, RT_MASK + 2 * e + 1, fx.Int32(bits >> 32))
                        lds_st(rti, RT_CNT + e, popcount(bits))
            gpu.barrier()
            # an exclusive scan of (picks | units << 16) in expert order -> each
            # expert's first pick and its units
            cnt = lds_ld(rti, RT_CNT + fx.min(tid, NE1 - 1))
            v = (tid < NE1).select(
                cnt | (((cnt + UNIT_ROWS - 1) // UNIT_ROWS) << 16), fx.Int32(0)
            )
            inc = v
            for d in range_constexpr(6):
                o = bpermute_i32(fx.max(lane - (1 << d), fx.Int32(0)) * 4, inc)
                inc = inc + (lane >= (1 << d)).select(o, fx.Int32(0))
            if lane == 63:
                lds_st(rti, RT_WSUM + wave, inc)
            gpu.barrier()
            exc = inc - v
            for w in range_constexpr(WAVES - 1):
                exc = exc + (wave > w).select(lds_ld(rti, RT_WSUM + w), fx.Int32(0))
            if tid < NE1:
                poff = exc & 0xFFFF
                uoff = exc >> 16
                lds_st(rti, RT_POFF + tid, poff)
                for j in range_constexpr((S + UNIT_ROWS - 1) // UNIT_ROWS):
                    if cnt > j * UNIT_ROWS:
                        n = fx.min(cnt - j * UNIT_ROWS, UNIT_ROWS)
                        lds_st(
                            rti,
                            RT_UNIT + uoff + j,
                            tid | ((n - 1) << 9) | ((poff + j * UNIT_ROWS) << 13),
                        )
            if tid == NE1 - 1:
                lds_st(rti, RT_N, (exc + v) >> 16)
            gpu.barrier()
            # pick (row s, slot) at its expert's first pick + its rows below s
            if tid < S * MOE_SLOTS:
                e = lds_ld(keys, tid)
                s = tid // MOE_SLOTS
                m = (fx.Int64(lds_ld(rti, RT_MASK + 2 * e + 1)) << 32) | (
                    fx.Int64(lds_ld(rti, RT_MASK + 2 * e)) & fx.Int64(0xFFFFFFFF)
                )
                below = popcount(m & ((fx.Int64(1) << fx.Int64(s)) - 1))
                lds_st(rti, RT_PICK + lds_ld(rti, RT_POFF + e) + below, tid)
            gpu.barrier()
            n_units = _uniform(lds_ld(rti, RT_N))

            def unit(u):
                """Unit u's (expert, rows, first pick)."""
                w = _uniform(lds_ld(rti, RT_UNIT + u))
                return w & 511, ((w >> 9) & 15) + 1, w >> 13

            def pick(first, n, j):
                return lds_ld(rti, RT_PICK + first + fx.min(j, n - 1))

            def cta_tasks(stage, total):
                b = start(stage)
                return b * total // G, (b + 1) * total // G

            # ---- up/gate: a wave the gate and up row groups of 4 of the 32 K units
            N_UG_C = expert_inter // UG_BLK
            ug_red = xs + UNIT_ROWS * XW if UG_RED_IN_X else red
            lo, hi = cta_tasks("ug", n_units * N_UG_C)
            for ti in range(lo, hi, 1):
                ti = fx.Int32(ti)
                u = ti // N_UG_C
                c = ti % N_UG_C
                e, n, first = unit(u)
                is_sh = e == SHARED
                rw = _rsrc(
                    is_sh.select(w_sh_ug, w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES))
                )
                rs = _rsrc(
                    is_sh.select(s_sh_ug, s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES))
                )
                col = n_sel(n)
                rgs = [c * 2 + h for h in range(2)] + [
                    expert_inter // 16 + c * 2 + h for h in range(2)
                ]
                scale_words = {}

                def g_unit(q):
                    """Unit q of this wave: gate or up, K pair m, row group parity h,
                    K parity d."""
                    side, rest = divmod(q, UG_RG * UG_KU // 2)
                    m, rest = divmod(rest, 4)
                    h, d = divmod(rest, 2)
                    rgi, k = side * 2 + h, m * 2 + d
                    kc = wave * UG_KU + k
                    if const_expr((side, m) not in scale_words):
                        scale_words[(side, m)] = mxfp4_scale_word(
                            rs, rgs[side * 2], kc - d, HIDDEN
                        )
                    return rgi, unit_native_mxfp4(
                        rw,
                        rs,
                        rgs[rgi],
                        kc,
                        HIDDEN,
                        col * XW + kc * 32,
                        8 + col * XQ_GROUPS + kc * 4 + lane // 16,
                        None,
                        lane,
                        (scale_words[(side, m)], 2 * d + h),
                    )

                units = [g_unit(q) for q in range(UG_RG * UG_KU)]
                if (ti == lo) | (c == 0):
                    gpu.barrier()
                    stage_xq(
                        [
                            _uniform(pick(first, n, j)) // MOE_SLOTS
                            for j in range(UNIT_ROWS)
                        ]
                    )
                gpu.barrier()
                accs = [[fx.Float32(0.0) for _ in range(4)] for _ in range(UG_RG)]
                for rgi, un in units:
                    accs[rgi] = mma_units(accs[rgi], [un])
                for rgi in range_constexpr(UG_RG):
                    fx.ptr_store(
                        fx.Vector.from_elements(accs[rgi], fx.Float32),
                        ug_red + ((rgi * WAVES + wave) * 64 + lane) * 4,
                    )
                gpu.barrier()
                # 16 threads a pick: the gate and up sums of 2 of its 32 values, then mid
                if tid < n * 16:
                    i = tid // 16
                    v2 = (tid % 16) * 2
                    p = lds_ld(rti, RT_PICK + first + i)
                    mids = []
                    for j in range_constexpr(2):
                        r = (v2 + j) % 16
                        sums = []
                        for h in range_constexpr(2):
                            rgi = 2 * h + v2 // 16
                            tot = fx.Float32(0.0)
                            for w in range_constexpr(WAVES):
                                tot = tot + lds_ld(
                                    ug_red,
                                    ((rgi * WAVES + w) * 64 + i + 16 * (r // 4)) * 4
                                    + r % 4,
                                )
                            sums.append(tot)
                        g = fx.min(sums[0], SWIGLU_LIMIT)
                        vu = fx.min(fx.max(sums[1], -SWIGLU_LIMIT), SWIGLU_LIMIT)
                        mids.append(g * _rcp(1.0 + _exp(-g)) * vu)
                    q0, q1, qs = quant_mxfp8(mids[0], mids[1])
                    w2 = (
                        fx.Int32(
                            rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)
                        )
                        & 0xFFFF
                    )
                    w4 = w2 | (_xshfl(w2, 1) << 16)
                    w4n = _xshfl(w4, 2)
                    blk = p * N_UG_C + c
                    if v2 % 8 == 0:
                        put2(
                            mb("midq"),
                            blk * (UG_BLK // 4) + v2 // 4,
                            w4.bitcast(fx.Float32),
                            w4n.bitcast(fx.Float32),
                        )
                    if v2 == 0:
                        put(mb("midqs"), blk, qs)
                    if (c == 0) & (v2 == 0):
                        put(mb("sel"), p, lds_ld(keys, p))
                        put(mb("prob"), p, lds_ld(dnw, p))

            # ---- down: a wave RGW row groups of every K unit of its picks' mid
            DN_NK = expert_inter // 128
            RGW = max(1, 16 // DN_NK)
            HC = RGW * WAVES * 16
            N_HC = HIDDEN // HC
            MW = expert_inter // 4
            MB = expert_inter // 32
            GDN_W_BYTES = HIDDEN * expert_inter // 2
            GDN_S_BYTES = HIDDEN * MB
            NWQ = UNIT_ROWS * MW // 2 // THREADS
            assert NWQ * THREADS * 2 == UNIT_ROWS * MW and UNIT_ROWS * MB <= THREADS
            lo, hi = cta_tasks("down", n_units * N_HC)
            for ti in range(lo, hi, 1):
                ti = fx.Int32(ti)
                u = ti // N_HC
                hc = ti % N_HC
                e, n, first = unit(u)
                is_sh = e == SHARED
                wb = _rsrc(
                    is_sh.select(w_sh_dn, w_dn + fx.Int64(e) * fx.Int64(GDN_W_BYTES))
                )
                sb = _rsrc(
                    is_sh.select(s_sh_dn, s_dn + fx.Int64(e) * fx.Int64(GDN_S_BYTES))
                )
                col = n_sel(n)
                rg0 = hc * (HC // 16) + wave * RGW
                dn_words = {}

                def d_unit(r, kc):
                    scale = None
                    if const_expr(RGW % 2 == 0 and DN_NK % 2 == 0):
                        if const_expr((r // 2, kc // 2) not in dn_words):
                            dn_words[(r // 2, kc // 2)] = mxfp4_scale_word(
                                sb, rg0 + r - r % 2, kc - kc % 2, expert_inter
                            )
                        scale = (dn_words[(r // 2, kc // 2)], 2 * (kc % 2) + r % 2)
                    return unit_native_mxfp4(
                        wb,
                        sb,
                        rg0 + r,
                        kc,
                        expert_inter,
                        col * MW + kc * 32,
                        8 + col * MB + kc * 4 + lane // 16,
                        None,
                        lane,
                        scale,
                    )

                units = [[d_unit(r, kc) for kc in range(DN_NK)] for r in range(RGW)]
                if (ti == lo) | (hc == 0):
                    gpu.barrier()
                    specs = []
                    for q in range_constexpr(NWQ):
                        f = tid + q * THREADS
                        p = pick(first, n, f // (MW // 2))
                        specs.append((mb("midq"), p * MW + (f % (MW // 2)) * 2, 2))
                    fs = fx.min(tid, UNIT_ROWS * MB - 1)
                    ps = pick(first, n, fs // MB)
                    specs.append((mb("midqs"), ps * MB + fs % MB, 1))
                    got = poll(specs)
                    for q in range_constexpr(NWQ):
                        f = tid + q * THREADS
                        w0 = (f // (MW // 2)) * MW + (f % (MW // 2)) * 2
                        for jj in range_constexpr(2):
                            lds_st(xs, w0 + jj, got[q][jj].bitcast(fx.Float32))
                    if tid < UNIT_ROWS * MB:
                        lds_st(misc, 8 + tid, got[NWQ][0].bitcast(fx.Float32))
                gpu.barrier()
                for r in range_constexpr(RGW):
                    acc = mma_units([fx.Float32(0.0) for _ in range(4)], units[r])
                    if lane % 16 < n:
                        p = lds_ld(rti, RT_PICK + first + lane % 16)
                        o = p * HIDDEN + (rg0 + r) * 16 + 4 * (lane // 16)
                        put2(mb("dpart"), o, acc[0], acc[1])
                        put2(mb("dpart"), o + 2, acc[2], acc[3])

            # ---- sum: 16 hidden rows of every row, slots in order, route-weighted
            ST = GROUPED_SUM_TILE

            def store_x(s, row, v0, v1):
                bo.buffer_store(
                    fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16),
                    _rsrc(x_out),
                    s * HIDDEN + row,
                )

            for t in range(start("dsum"), HIDDEN // ST, G):
                t = fx.Int32(t)
                gpu.barrier()
                if tid < S * ST // 2:
                    s = tid // (ST // 2)
                    r = (tid % (ST // 2)) * 2
                    got = poll(
                        [
                            (
                                mb("dpart"),
                                (s * MOE_SLOTS + k) * HIDDEN + t * ST + r,
                                2,
                            )
                            for k in range(MOE_SLOTS)
                        ]
                    )
                    y0, y1 = fx.Float32(0.0), fx.Float32(0.0)
                    for k in range_constexpr(MOE_SLOTS):
                        wk = lds_ld(dnw, s * MOE_SLOTS + k)
                        y0 = y0 + got[k][0].bitcast(fx.Float32) * wk
                        y1 = y1 + got[k][1].bitcast(fx.Float32) * wk
                    lds_st(outs, s * ST + r, y0)
                    lds_st(outs, s * ST + r + 1, y1)
                gpu.barrier()
                peer_reduce("ffn", t, store_x, ST)
                gpu.barrier()

        if const_expr(GROUPED):
            run_grouped()
            return
        # the shared expert needs only the activation blocks: it runs before the router
        # scores, its weights loading before the gate's flag on CTAs that run no router
        # task, else after the blocks (a poll behind them would wait for them too). One
        # task 8 intermediate rows (gate and up) of every row; the other CTAs' loads are
        # empty
        N_SH = expert_inter // UG8
        assert N_SH <= G
        SH_EARLY = S * N_ROUTER + N_SH <= G
        u = start("ug")
        has_sh = u < N_SH
        r_sh = bo.create_buffer_resource_from_addr(
            w_sh_ug, num_records_bytes=has_sh.select(fx.Int32(UG_W_BYTES), fx.Int32(0))
        )
        r_sh_s = bo.create_buffer_resource_from_addr(
            s_sh_ug, num_records_bytes=has_sh.select(fx.Int32(UG_S_BYTES), fx.Int32(0))
        )
        w_rg = ((lane % 16) // 8) * (expert_inter // 16) + u // 2
        w_ln = (lane & -16) | ((u % 2) * 8 + lane % 8)
        bias_c = load_bias()

        def sh_units():
            return [
                unit_native_mxfp4(
                    r_sh,
                    r_sh_s,
                    w_rg,
                    wave * UG_KU + k,
                    HIDDEN,
                    n_sel() * XW + (wave * UG_KU + k) * 32,
                    8 + n_sel() * XQ_GROUPS + (wave * UG_KU + k) * 4 + lane // 16,
                    ln=w_ln,
                )
                for k in range(UG_KU)
            ]

        if const_expr(SH_EARLY):
            sh_pre = sh_units()

        # the activation blocks are the gate's, ready with its row: a CTA idle since the
        # attention stages stages them before the router scores. When the router tasks
        # compute their rows (they end with the scores), one that ran a gate or router
        # task (and no shared task) stages them behind its routed up/gate task's first
        # weight batch instead
        def stage_all_xq():
            wait_ready("box_a")
            stage_xq(list(range(S)))

        if const_expr(ROUTER_OWN_ROW):
            late_xq = (
                ((start("router") < S * N_ROUTER) | (start("gate") < S)) & (u >= N_SH)
            ).select(fx.Int32(1), fx.Int32(0))
            if late_xq == 0:
                stage_all_xq()
        else:
            stage_all_xq()
        if const_expr(not SH_EARLY):
            sh_pre = sh_units()
        if has_sh:
            gpu.barrier()
            reduce_rows(
                1, mma_units([fx.Float32(0.0) for _ in range(4)], sh_pre), emit_out(16)
            )
            gpu.barrier()
            if tid < S * UG8 // 2:
                n = tid // (UG8 // 2)
                r = (tid % (UG8 // 2)) * 2
                mids = []
                for j in range_constexpr(2):
                    g = fx.min(lds_ld(outs, n * 16 + r + j), SWIGLU_LIMIT)
                    v = fx.min(
                        fx.max(lds_ld(outs, n * 16 + UG8 + r + j), -SWIGLU_LIMIT),
                        SWIGLU_LIMIT,
                    )
                    mids.append(g * _rcp(1.0 + _exp(-g)) * v)
                put2(mb("mid"), n * expert_inter + u * UG8 + r, mids[0], mids[1])
            if (u == 0) & (tid < S):
                put(mb("sel"), tid * MOE_SLOTS, fx.Int32(SHARED))
                put(mb("prob"), tid * MOE_SLOTS, fx.Float32(1.0))
        dn_route(bias_c)
        gpu.barrier()

        # routed: task t is slot 1 + t / (2 N_UG_C), 32-value chunk (t / 2) % N_UG_C, rows
        # of half t % 2. A wave takes 4 K units of the 4 row groups of every row; a row's
        # MFMAs read every row's activation (B column n = row n of the half) and add only
        # column n, so the task reduces its tiles once
        N_UG_C = expert_inter // UG_BLK
        UG_NQ = UG_HALF * UG_RG * UG_KU
        UG_BATCH = 8 if S <= 4 else 4
        ug_red = xs + S * XW if UG_RED_IN_X else red
        for t in range(start("ug"), N_UG, G):
            t = fx.Int32(t)
            slot = 1 + t // (2 * N_UG_C)
            c = (t // 2) % N_UG_C
            s0 = (t % 2) * UG_HALF
            col = fx.min(s0 + fx.min(lane % 16, UG_HALF - 1), S - 1)
            rgs = [c * 2 + h for h in range(2)] + [
                expert_inter // 16 + c * 2 + h for h in range(2)
            ]
            experts = [
                _uniform(lds_ld(keys, fx.min(s0 + i, S - 1) * MOE_SLOTS + slot))
                for i in range(UG_HALF)
            ]

            scale_words = {}

            def ug_unit(q):
                """(row group, unit) q of this wave: row i, gate or up, K pair m, row
                group parity h, K parity d."""
                i, rest = divmod(q, UG_RG * UG_KU)
                side, rest = divmod(rest, UG_RG * UG_KU // 2)
                m, rest = divmod(rest, 4)
                h, d = divmod(rest, 2)
                rgi, k = side * 2 + h, m * 2 + d
                rw = _rsrc(w_ug + fx.Int64(experts[i]) * fx.Int64(UG_W_BYTES))
                rs = _rsrc(s_ug + fx.Int64(experts[i]) * fx.Int64(UG_S_BYTES))
                kc = wave * UG_KU + k
                if const_expr((i, side, m) not in scale_words):
                    scale_words[(i, side, m)] = mxfp4_scale_word(
                        rs, rgs[side * 2], kc - d, HIDDEN
                    )
                return rgi, unit_native_mxfp4(
                    rw,
                    rs,
                    rgs[rgi],
                    kc,
                    HIDDEN,
                    col * XW + kc * 32,
                    8 + col * XQ_GROUPS + kc * 4 + lane // 16,
                    None if UG_HALF == 1 else (lane % 16 == i, None),
                    lane,
                    (scale_words[(i, side, m)], 2 * d + h),
                )

            accs = [[fx.Float32(0.0) for _ in range(4)] for _ in range(UG_RG)]
            starts = list(range(0, UG_NQ, UG_BATCH))
            cur = [ug_unit(q) for q in range(min(UG_BATCH, UG_NQ))]
            if const_expr(ROUTER_OWN_ROW):
                if (late_xq == 1) & (t == u):
                    stage_all_xq()
                    gpu.barrier()
            for bi in range_constexpr(len(starts)):
                nxt = None
                if const_expr(bi + 1 < len(starts)):
                    n0 = starts[bi + 1]
                    nxt = [ug_unit(q) for q in range(n0, min(n0 + UG_BATCH, UG_NQ))]
                for z in range_constexpr(len(cur)):
                    accs[cur[z][0]] = mma_units(accs[cur[z][0]], [cur[z][1]])
                cur = nxt
            for rgi in range_constexpr(UG_RG):
                fx.ptr_store(
                    fx.Vector.from_elements(accs[rgi], fx.Float32),
                    ug_red + ((rgi * WAVES + wave) * 64 + lane) * 4,
                )
            gpu.barrier()
            # 16 threads a row: the gate and up sums of 2 of its 32 values, then mid
            if tid < UG_HALF * 16:
                i = tid // 16
                v2 = (tid % 16) * 2
                s = s0 + i
                if s < S:
                    mids = []
                    for j in range_constexpr(2):
                        r = (v2 + j) % 16
                        sums = []
                        for h in range_constexpr(2):
                            rgi = 2 * h + v2 // 16
                            tot = fx.Float32(0.0)
                            for w in range_constexpr(WAVES):
                                tot = tot + lds_ld(
                                    ug_red,
                                    ((rgi * WAVES + w) * 64 + i + 16 * (r // 4)) * 4
                                    + r % 4,
                                )
                            sums.append(tot)
                        g = fx.min(sums[0], SWIGLU_LIMIT)
                        v = fx.min(fx.max(sums[1], -SWIGLU_LIMIT), SWIGLU_LIMIT)
                        mids.append(g * _rcp(1.0 + _exp(-g)) * v)
                    # the block's MXFP8 words and exponent, as down's B operand holds them
                    q0, q1, qs = quant_mxfp8(mids[0], mids[1])
                    w2 = (
                        fx.Int32(
                            rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)
                        )
                        & 0xFFFF
                    )
                    w4 = w2 | (_xshfl(w2, 1) << 16)
                    w4n = _xshfl(w4, 2)
                    blk = (s * MOE_SLOTS + slot) * (expert_inter // UG_BLK) + c
                    if v2 % 8 == 0:
                        put2(
                            mb("midq"),
                            blk * (UG_BLK // 4) + v2 // 4,
                            w4.bitcast(fx.Float32),
                            w4n.bitcast(fx.Float32),
                        )
                    if v2 == 0:
                        put(mb("midqs"), blk, qs)
                    if (c == 0) & (v2 == 0):
                        put(
                            mb("sel"),
                            s * MOE_SLOTS + slot,
                            lds_ld(keys, s * MOE_SLOTS + slot),
                        )
                        put(
                            mb("prob"),
                            s * MOE_SLOTS + slot,
                            lds_ld(dnw, s * MOE_SLOTS + slot),
                        )

        # ======== 4. mid MXFP8 quant + expert down + route weighting + MoE TP reduce
        DN_R = (DN_TILE + 15) // 16
        DN_WPR = WAVES // DN_R
        DN_BATCH = down_prefetch_batch(S)
        DN_W_BYTES = HIDDEN * expert_inter // 2
        DN_S_BYTES = HIDDEN * (expert_inter // 32)
        DN_NKC = expert_inter // 64
        for t in range(start("down"), N_DN_TILES, G):
            t = fx.Int32(t)
            gpu.barrier()
            gu = wave // DN_WPR
            dn_rg = t * DN_TILE // 16
            dn_off = t * DN_TILE % 16
            # rows outside the tile load their lane ^ 8 twin and are dropped
            dn_lr = gu * 16 + lane % 16 - dn_off
            dn_ln = ((dn_lr >= 0) & (dn_lr < DN_TILE)).select(lane, lane ^ 8)

            DN_NU = S * MOE_SLOTS * DN_NKC // 2
            DN_UPW = (DN_NU + DN_WPR - 1) // DN_WPR
            # poll specs a thread: word pairs of the routed slots' MXFP8 mid, their
            # exponents, value pairs of the shared expert's mid
            DN_RQ = (MOE_SLOTS - 1) * expert_inter // 8
            DN_BQ = (MOE_SLOTS - 1) * expert_inter // 32
            DN_SQ = expert_inter // 2
            DN_NRQ = (S * DN_RQ + THREADS - 1) // THREADS
            DN_NBQ = (S * DN_BQ + THREADS - 1) // THREADS
            DN_NSQ = (S * DN_SQ + THREADS - 1) // THREADS

            def dn_word(q):
                p = fx.min(tid + q * THREADS, S * DN_RQ - 1)
                return (p // DN_RQ * MOE_SLOTS + 1) * (
                    expert_inter // 4
                ) + p % DN_RQ * 2

            def dn_blk(q):
                p = fx.min(tid + q * THREADS, S * DN_BQ - 1)
                return (p // DN_BQ * MOE_SLOTS + 1) * (expert_inter // 32) + p % DN_BQ

            def dn_sh(q):
                return fx.min(tid + q * THREADS, S * DN_SQ - 1)

            def dn_sh_flat(q):
                p = dn_sh(q)
                return p // DN_SQ * MOE_SLOTS * expert_inter + p % DN_SQ * 2

            # with an even count of units a wave, units 2 j and 2 j + 1 are the two
            # K units of one scale word
            dn_scales = {}

            def u_dn(cc):  # cc: 128-k chunk of this wave
                qu = (wave % DN_WPR) * DN_UPW + cc
                live = qu < DN_NU
                unit = fx.min(qu, DN_NU - 1)
                q = unit * 2  # 64-k chunk index over (s, slot, kc)
                s_q = q // (MOE_SLOTS * DN_NKC)
                slot_q = (q // DN_NKC) % MOE_SLOTS
                e = _uniform(lds_ld(keys, s_q * MOE_SLOTS + slot_q))
                is_sh = e == SHARED
                wb = bo.create_buffer_resource_from_addr(
                    is_sh.select(w_sh_dn, w_dn + fx.Int64(e) * fx.Int64(DN_W_BYTES)),
                    num_records_bytes=(
                        None
                        if DN_NU % DN_WPR == 0
                        else live.select(fx.Int32(DN_W_BYTES), fx.Int32(0))
                    ),
                )
                sb = bo.create_buffer_resource_from_addr(
                    is_sh.select(s_sh_dn, s_dn + fx.Int64(e) * fx.Int64(DN_S_BYTES)),
                    num_records_bytes=(
                        None
                        if DN_NU % DN_WPR == 0
                        else live.select(fx.Int32(DN_S_BYTES), fx.Int32(0))
                    ),
                )

                def route_coef():
                    return (
                        lane % 16 == s_q,
                        _uniform_f32(lds_ld(dnw, s_q * MOE_SLOTS + slot_q)),
                    )

                scale = None
                if const_expr(DN_UPW % 2 == 0):
                    if const_expr(cc % 2 == 0):
                        dn_scales[cc // 2] = mxfp4_scale_word(
                            sb,
                            dn_rg + gu,
                            unit % (expert_inter // 128),
                            expert_inter,
                            dn_ln,
                        ) >> (8 * ((dn_rg + gu) % 2))
                    scale = (dn_scales[cc // 2], 2 * (cc % 2))
                return unit_native_mxfp4(
                    wb,
                    sb,
                    dn_rg + gu,
                    unit % (expert_inter // 128),
                    expert_inter,
                    unit * 32,
                    unit * 4 + lane // 16,
                    route_coef,
                    dn_ln,
                    scale,
                )

            if const_expr(S <= 8):
                pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
            # every row's mid into LDS as MXFP8: the routed slots' words and exponents as
            # up/gate published them, the shared expert's f32 values quantized here
            specs = [(mb("midq"), dn_word(q), 2) for q in range(DN_NRQ)]
            specs += [(mb("midqs"), dn_blk(q), 1) for q in range(DN_NBQ)]
            specs += [(mb("mid"), dn_sh(q) * 2, 2) for q in range(DN_NSQ)]
            got = poll(specs)
            for q in range_constexpr(DN_NRQ):
                if tid + q * THREADS < S * DN_RQ:
                    for j in range_constexpr(2):
                        lds_st(xs, dn_word(q) + j, got[q][j].bitcast(fx.Float32))
            for q in range_constexpr(DN_NBQ):
                if tid + q * THREADS < S * DN_BQ:
                    lds_st(misc, dn_blk(q), got[DN_NRQ + q][0].bitcast(fx.Float32))
            for q in range_constexpr(DN_NSQ):
                if tid + q * THREADS < S * DN_SQ:
                    v = got[DN_NRQ + DN_NBQ + q]
                    q0, q1, qs = quant_mxfp8(
                        v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32)
                    )
                    k = dn_sh_flat(q)
                    st_mxfp8(k, q0, q1)
                    if lane % 16 == 0:
                        lds_st(misc, k // 32, qs.bitcast(fx.Float32))
            gpu.barrier()
            if const_expr(S > 8):
                pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
            acc = run_units(u_dn, DN_UPW, DN_BATCH, pre)

            def emit_dn(rl, n, v):
                if (rl >= dn_off) & (rl < dn_off + DN_TILE):
                    lds_st(outs, n * DN_TILE + rl - dn_off, v)

            reduce_rows(DN_R, acc, emit_dn)
            gpu.barrier()

            def store_x(s, row, v0, v1):
                bo.buffer_store(
                    fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16),
                    _rsrc(x_out),
                    s * HIDDEN + row,
                )

            peer_reduce("ffn", t, store_x, DN_TILE)
            gpu.barrier()

    @flyc.jit
    def launch(
        attn_part: Int64,
        res_in: Int64,
        post_in: Int64,
        comb_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        g_norm: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        x_out: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        w_sh_ug: Int64,
        s_sh_ug: Int64,
        w_sh_dn: Int64,
        s_sh_dn: Int64,
        hc_a_fn: Int64,
        hc_a_scale: Int64,
        hc_a_base: Int64,
        g_in: Int64,
        w_in: Int64,
        w_fb: Int64,
        w_gb: Int64,
        conv_w: Int64,
        a_log: Int64,
        dt_bias: Int64,
        w_on: Int64,
        w_o: Int64,
        conv_state: Int64,
        rec_state: Int64,
        state_idx: Int64,
        conv_stride: Int64,
        rec_stride: Int64,
        inter_rec: Int64,
        inter_conv: Int64,
        inter_rec_stride: Int64,
        inter_conv_stride: Int64,
        dsa_tab: Int64,
        tab_w_qkv: Int64,
        tab_s_qkv: Int64,
        tab_w_idx: Int64,
        tab_w_gate: Int64,
        tab_g_qa: Int64,
        tab_g_kv: Int64,
        tab_ln_w: Int64,
        tab_ln_b: Int64,
        tab_ape: Int64,
        tab_w_qb: Int64,
        tab_s_qb: Int64,
        tab_w_iq: Int64,
        tab_w_uk: Int64,
        tab_w_kvb: Int64,
        tab_w_o: Int64,
        tab_s_o: Int64,
        tab_kv_cache: Int64,
        tab_idx_cache: Int64,
        tab_tail_k: Int64,
        tab_tail_g: Int64,
        tab_idx_bt_group: Int64,
        tab_kv_block_elems: Int64,
        tab_kv_block_tokens: Int64,
        tab_kv_row_elems: Int64,
        tab_idx_block_bytes: Int64,
        tab_idx_block_pools: Int64,
        tab_tail_block_elems: Int64,
        dsa_rows: Int64,
        page_table: Int64,
        pt_stride: Int32,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        flash_ffn_kernel(
            attn_part,
            res_in,
            post_in,
            comb_in,
            hc_fn,
            hc_scale,
            hc_base,
            g_norm,
            res_out,
            post_out,
            comb_out,
            x_out,
            w_r,
            bias,
            w_ug,
            s_ug,
            w_dn,
            s_dn,
            w_sh_ug,
            s_sh_ug,
            w_sh_dn,
            s_sh_dn,
            hc_a_fn,
            hc_a_scale,
            hc_a_base,
            g_in,
            w_in,
            w_fb,
            w_gb,
            conv_w,
            a_log,
            dt_bias,
            w_on,
            w_o,
            conv_state,
            rec_state,
            state_idx,
            conv_stride,
            rec_stride,
            inter_rec,
            inter_conv,
            inter_rec_stride,
            inter_conv_stride,
            dsa_tab,
            tab_w_qkv,
            tab_s_qkv,
            tab_w_idx,
            tab_w_gate,
            tab_g_qa,
            tab_g_kv,
            tab_ln_w,
            tab_ln_b,
            tab_ape,
            tab_w_qb,
            tab_s_qb,
            tab_w_iq,
            tab_w_uk,
            tab_w_kvb,
            tab_w_o,
            tab_s_o,
            tab_kv_cache,
            tab_idx_cache,
            tab_tail_k,
            tab_tail_g,
            tab_idx_bt_group,
            tab_kv_block_elems,
            tab_kv_block_tokens,
            tab_kv_row_elems,
            tab_idx_block_bytes,
            tab_idx_block_pools,
            tab_tail_block_elems,
            dsa_rows,
            page_table,
            pt_stride,
            scratch,
            sym,
            peers,
            step,
            rank,
            layer,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    return launch, SC, SY
