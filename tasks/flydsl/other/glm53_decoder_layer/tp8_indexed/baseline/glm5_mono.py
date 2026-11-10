# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Adapted from ROCm/aiter#6173 at 56d6390def (Apache-2.0):
# aiter/ops/flydsl/kernels/glm5_mono/glm/kernel.py
# Modified for AgentKernelArena (sgl-amd-mono 52486703 glm53/gfx950/kernel.py: imports made package-relative; the source-hash build key replaced by the build arguments; reload_args returns the reloaded pointers instead of rebinding them with nonlocal; the indexer's seven pointers are kernel arguments instead of a device pointer table; the unused _ld_pair removed; fx.atomic_add (FlyDSL >= 0.3.4) replaced by the llvm atomicrmw of compat.py).

"""GLM-5 indexed decode MonoKernel: one persistent launch per TP rank.

One launch of ``grid = 256 CTAs x 512 threads`` (one CTA per MI355X CU) runs the
whole decoder-layer body for this rank's TP shard.  The indexed path adds the
selection refresh to the same resident grid instead of consuming externally
prepared sparse indices::

    input RMSNorm -> q_a / kv_a projection -> q_a RMSNorm -> q_b (+RoPE)
      -> KV RMSNorm / k_pe RoPE -> KV/PE cache publish
      -> index K/Q/W projection -> index K norm/RoPE/Hadamard/cache
      -> index score -> exact radix top-2048
      -> absorbed q (W_UK) -> sparse MLA split softmax -> merge -> W_UV -> W_o
      -> attention TP peer reduce + residual                      (sym_attn)
      -> post-attention RMSNorm -> router sigmoid + activation FP8 quant
      -> top-8 -> 1 shared + 8 routed expert up/gate/SiLU
      -> mid FP8 quant -> expert down + route weighting
      -> MoE TP peer reduce + residual -> x_out                   (sym_ffn)

Scheduling: every stage is a list of tasks; task ``t`` of a stage runs on CTA
``(stage_base + t) % 256`` and every CTA walks the stages in order.  There is
no grid-wide barrier: dependencies only point to earlier stages and all CTAs
are co-resident, so every spin wait makes progress.

Mailboxes are *tagged pairs*: every 32-bit value a task hands to another CTA
(or GPU) is stored next to this launch's epoch tag, ``(value, tag)``, with
device- (``sc1``) or system-coherent (``sc0 sc1``) 8 / 16-byte stores.  A
consumer polls the payload itself until the tags match, so a hand-off costs
one memory round trip: no store drain, no separate flag, no second load.

GEMVs run on the matrix cores: weights are host-packed (``pack_fp8`` /
``pack_bf16``) or, with ``fp8_shuffled``, read in place in AITER's (16, 16)
layout, so one wave loads 16 rows x 64 k as one contiguous 1 KB, FP8 is
widened exactly to bf16 and fed to ``mfma_f32_16x16x32_bf16`` with the samples
as the N dimension.  Each 64-k chunk's partial is scaled by its f32 block
scale (times any activation scale / route weight) into the accumulator, so the
math is exact block-scaled FP8 on bf16 activations.  Weight loads that do not
depend on upstream results are issued before the task waits for its inputs.

Cross-GPU: each rank pushes its partial rows as tagged pairs into every peer's
symmetric buffer and polls its own; every rank sums the TP partials in rank
order, so all ranks produce bit-identical hidden states (and routing).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from .dpp_utils import update_dpp_i32
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from . import guard
from .compat import lds_atomic_add
from .epoch import spin_until
from .ops import (
    ballot,
    bpermute_i32,
)
from .ops import clock as mem_realtime
from .ops import (
    div_rn,
    f8_word,
    fp8_pack4,
)
from .ops import fp8_roundtrip as _fp8_roundtrip
from .ops import fp8_to_bf16x8 as _fp8_to_bf16x8
from .ops import fresh as _fresh
from .ops import hw_exp as _exp
from .ops import hw_rcp as _rcp
from .ops import hw_rsq as _rsq
from .ops import (
    lanes_below,
)
from .ops import mxfp4_to_bf16x8 as _mxfp4_to_bf16x8
from .ops import (
    permlane_swap,
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

from .config import (
    EPS,
    FP8_MAX,
    HIDDEN,
    INTER,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    ROUTE_SCALE,
    SCALE_BM,
    SHARED_EXPERT,
    SOFTMAX_SCALE,
    TOP_K,
    V_DIM,
    AttentionWeight,
    KvCacheLayout,
    as_kv_cache_layout,
)
from .layout import (
    BLOCKS,
    DCP_SUMMARY_PAIRS,
    EH_ARGS,
    FRONT_K_BIG,
    FRONT_SMALL_CTAS,
    HEAD_TILE,
    INDEX_CAND_STAGE_CAP,
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_KEYS_PER_TASK,
    INDEX_Q_ROWS,
    INDEX_SEL_DIGITS,
    INDEX_SEL_SLICES,
    INDEX_TILE,
    N_QKV_A,
    N_ROUTER,
    N_ROW_TILES,
    Q_B_TILE,
    QKV_A_TILE,
    ROUTER_TILE,
    ROW_TILE,
    UG_TILE,
    UK_TILE,
    UV_TILE,
    WAVES,
    XQ_BLOCKS,
    XQ_GROUPS,
    XQ_WAVES,
    dcp_local_uv_tile,
    dcp_summary_index,
    dcp_uv_owner,
    dn_tile,
    down_prefetch_batch,
    down_x_words,
    eh_base,
    eh_parts,
    fp8_kv_upper_pair_lane,
    fp8_pe_upper_pair_lane,
    layout,
    sample_wave_batches,
    select_slices,
    sparse_keys_per_task,
    split_acc_head,
    split_score_column,
    stage_bases,
    stage_tasks,
    ug_split,
    ug_task_rounds,
)
from .tiles import (
    CM_DEV,
    CM_NT,
    CM_SYS,
    LAYER_SLOTS,
    NEG,
    POLL_MAX,
    THREADS,
    TL_COLS,
    atom_mxfp4_scale_index,
)

class _RowSwapped:
    """A shuffled FP8 weight load's 16 bytes, its halves swapped between lane rows
    2i and 2i + 1 where an element is read: words (lo0, lo1, hi0, hi1)."""

    def __init__(self, raw):
        self.raw = raw

    def __getitem__(self, e):
        lo, hi = permlane_swap(16, self.raw[e % 2], self.raw[e % 2 + 2])
        return lo if e < 2 else hi


def build_glm5_monokernel(
    S: int = 1,
    heads: int = 8,
    npes: int = 8,
    topk: int = 2048,
    launches_per_step: int = 1,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    expert_mxfp4: bool = False,
    atom_experts: bool = False,
    attention_weight: AttentionWeight | str = AttentionWeight.FP8_BLOCK128,
    kv_cache_layout: KvCacheLayout | str = KvCacheLayout.SPLIT,
    kv_cache_dtype: str = "bf16",
    inter: int = INTER,
    output_heads: int | None = None,
    dcp_size: int = 1,
    scale: float = SOFTMAX_SCALE,
    uv_scale_rows: int = 128,
    native_fp4_mfma: bool = False,
    timeline: bool = False,
    index_paged: bool = False,
    index_block_size: int = 64,
    index_block_bytes: int = 0,
    index_shuffled: bool = False,
    index_k_bf16: bool = False,
    dense_experts: int = 0,
    fp8_shuffled: bool = False,
    fault_site: int = 0,
    fault_after: int = 0,
    stop: bool = True,
    tokens: bool = False,
    vocab_rows: int = 0,
    tail: bool = False,
    eh_entry: bool = False,
):
    """Return the ``@flyc.jit`` launcher for one rank's whole layer.

    ``timeline=True`` records ``s_memrealtime`` (100 MHz) at the start and end of
    every task, and once its inputs have arrived, into the ``timeline`` buffer:
    int64 ``[sum(task counts), TL_COLS]`` (start, hint seen, inputs staged, compute
    done, end, then free debug marks) in ``stage_tasks`` order.

    ``index_paged=True`` keeps the index keys in SGLang's paged FP8 cache: per
    block, ``index_block_size x 128`` E4M3 values (``index_shuffled``: 16-token x
    16-byte tiles) followed by one power-of-two f32 scale per token, with blocks
    ``index_block_bytes`` apart.  ``req_ids`` picks each row's ``block_table`` row
    (rows ``bt_stride`` entries apart, a launch argument; the rows of one verify
    request may share it), keys cover positions ``0 .. positions[s]``, a key that
    a row of this launch writes is scored from that row's ``index_k8`` pairs (its
    cached FP8 values and scale), and
    the top-k is published as global cache slots to ``out_indices`` at the row's
    ``sparse_kv_indptr`` offset.
    ``index_k_bf16`` reads the index K projection as packed bf16.

    ``dense_experts > 0`` runs a dense MLP instead of the MoE: its intermediate
    dimension is stored as that many expert-shaped slices, every row routes to all
    of them with weight 1, and the router GEMV is skipped. With FP8 experts the
    slots past the slices load nothing.

    ``tokens=True`` (a step's first layer) starts from the step's tokens: ``h_in``
    holds their ids (int32 [S], -1 for a pad row) and ``w_r`` this rank's
    ``vocab_rows`` rows of the vocab-parallel embedding (bf16 [vocab_rows, HIDDEN]).
    The rank that holds a token's row pushes it to every rank (a pad row's zeros come
    from rank 0), which equals the embedding's all-reduce: one rank holds each row.

    ``eh_entry=True`` (a NextN layer) starts from ``eh_proj(enorm(e) | hnorm(h))``:
    ``h_in`` holds the e rows and ``cur_pos`` the h rows (bf16 [S, HIDDEN]; a pad
    row's are read as zeros), and ``timeline_buf`` a pointer table whose entries
    EH_ARGS are this rank's HIDDEN / npes rows of eh_proj (bf16 [rows, 2 * HIDDEN],
    row-major), the enorm and hnorm weights and, with ``tokens``, this rank's
    ``vocab_rows`` embedding rows (entries 0-6 the indexer's, 7 the timeline). Each
    rank computes its rows of x (f32 sums, rounded to bf16) and pushes them to every
    rank, where the layer reads x. With ``tokens`` the e rows come from the token ids
    in ``h_in``, as layer 0's do.

    ``tail=True`` builds the tail instead of a layer: the final RMSNorm of the rows
    ``h_in`` (bf16 [S, HIDDEN]) with weight ``g_in``, written to ``w_o`` (bf16
    [S, HIDDEN]); this rank's ``vocab_rows`` LM-head rows ``w_r`` (bf16, row-major) on
    the normed rows, rounded to bf16 as SGLang's LM head leaves them; and their
    all-gather over the TP ranks into ``x_out`` (f32 [S, vocab_rows * npes]).
    """
    keyed = (
        S, heads, npes, topk, launches_per_step, with_indexer, index_max_seq, expert_mxfp4,
        atom_experts, str(attention_weight), str(kv_cache_layout), kv_cache_dtype, inter, output_heads,
        dcp_size, scale, uv_scale_rows, native_fp4_mfma, timeline, index_paged, index_block_size,
        index_block_bytes, index_shuffled, index_k_bf16, dense_experts, fp8_shuffled, fault_site,
        fault_after, stop, tokens, vocab_rows, tail, eh_entry,
    )
    assert uv_scale_rows in (64, 128)
    assert heads % WAVES == 0, "split attention maps one local head to each wave"
    assert kv_cache_dtype in ("bf16", "fp8")
    attention_weight = AttentionWeight(attention_weight)
    cache_layout = as_kv_cache_layout(kv_cache_layout)
    attention_bf16 = attention_weight is AttentionWeight.BF16
    attention_ptpc = attention_weight is AttentionWeight.FP8_PTPC
    use_atom_kv_cache = cache_layout is KvCacheLayout.ATOM
    cache_fp8 = kv_cache_dtype == "fp8"
    assert not cache_fp8 or use_atom_kv_cache
    assert not atom_experts or expert_mxfp4
    assert not native_fp4_mfma or atom_experts
    attention_k_chunks_per_unit = 1 if attention_bf16 else 2
    SPLIT_KEYS = sparse_keys_per_task(S, heads)
    assert topk % SPLIT_KEYS == 0 and 1 <= S <= 12
    assert 1 <= launches_per_step <= LAYER_SLOTS
    assert not with_indexer or (
        topk == 2048 and index_max_seq % INDEX_KEYS_PER_TASK == 0
    )
    assert not (attention_bf16 and with_indexer), (
        "BF16 attention uses the external indexer"
    )
    assert not with_indexer or index_paged, (
        "the fused indexer reads SGLang's paged keys"
    )
    assert not index_paged or (
        with_indexer
        and use_atom_kv_cache
        and index_block_bytes >= index_block_size * (INDEX_DIM + 4)
        and index_block_bytes % 4 == 0
        and (not index_shuffled or index_block_size % 16 == 0)
    )
    assert not tokens or (
        vocab_rows > 0 and (dense_experts or eh_entry) and not attention_ptpc
    )
    assert not tail or (vocab_rows % HEAD_TILE == 0 and not tokens)
    assert not eh_entry or (
        not tail and not dense_experts and HIDDEN % (npes * HEAD_TILE) == 0
    )
    # x arrives as tagged pairs in xin (the embedding rows or the NextN entry's)
    x_pushed = tokens or eh_entry
    # the kinds whose stages load their pointer arguments again from the kernarg segment
    RELOAD_ARGS = bool(eh_entry or with_indexer or dense_experts)
    EH_TILES = HIDDEN // npes // HEAD_TILE if eh_entry else 0
    # the kernel's Int64 arguments in signature order: argument i is the 8 bytes at
    # 8 * i of the kernarg segment
    KERNARG_I64 = (
        "h_in x_out cur_pos positions slot_mapping sparse_kv_indptr kv_cache pe_cache"
        " indices rope_cos rope_sin g_in g_q g_kv g_post w_qkv_a s_qkv_a w_q_b s_q_b"
        " w_uk s_uk w_uv s_uv w_o s_o w_r bias w_ug s_ug w_dn s_dn scratch sym peers"
        " timeline_buf step block_table req_ids out_indices ix_wk ix_sk ix_ww ix_wq ix_sq ix_gk ix_bk"
    ).split()
    assert 0 <= dense_experts <= MOE_SLOTS
    assert not dense_experts or (S > 1 and (atom_experts or not expert_mxfp4))
    # the slots past an FP8 dense MLP's slices: weight 0, no weight loads
    dense_fp8 = dense_experts > 0 and not expert_mxfp4
    H = heads
    L = H if output_heads is None else output_heads
    W = npes
    D = dcp_size
    expert_inter = inter
    SHARED = 0 if dense_experts else SHARED_EXPERT
    assert D == 1 or (D == W and H == L * D)
    I_PER_SLOT = expert_inter // UG_TILE
    G = BLOCKS
    SC, SY = layout(
        S,
        H,
        W,
        topk,
        with_indexer,
        index_max_seq,
        inter=expert_inter,
        output_heads=L,
        dcp_size=D,
        native_fp4_mfma=native_fp4_mfma,
        tokens=tokens,
        head_vocab=vocab_rows * W if tail else 0,
        eh_tiles=EH_TILES,
    )
    N_SPLIT = topk // SPLIT_KEYS
    QB_ROWS = H * (NOPE_DIM + PE_DIM)
    N_QB = QB_ROWS // Q_B_TILE
    assert not with_indexer or (INDEX_Q_ROWS // INDEX_TILE == G and N_QB in (G // 2, G))
    QB_PER_HEAD = (NOPE_DIM + PE_DIM) // Q_B_TILE
    N_UK = H * KV_LORA // UK_TILE
    UK_PER_HEAD = KV_LORA // UK_TILE
    N_UV = H * V_DIM // UV_TILE
    O_K = L * V_DIM
    N_UG = S * MOE_SLOTS * I_PER_SLOT
    QK_DIM = KV_LORA + PE_DIM
    # split LDS: bf16 q of all heads, then the KV latent / k_pe tiles (bf16 pairs); row
    # strides are padded by 4 words so the MFMA operand rows spread over the banks
    QS = QK_DIM // 2 + 4
    KS = KV_LORA // 2 + 4
    PS = PE_DIM // 2 + 4
    KT_OFF = H * QS
    PT_OFF = KT_OFF + SPLIT_KEYS * KS
    # The input projection stages up to eight samples' normalized activations in LDS
    # (96 KiB at S=8, inside the arena the split tiles already need) and runs its
    # MFMAs once for all of them; the RMSNorm takes NORM_ROWS samples at a time to
    # bound its registers.  Later stages either consume a smaller tensor or use the
    # FP8 representation and therefore fit all samples at once.
    SAMPLE_TILE = min(S, 8)
    NORM_ROWS = 4
    DN_TILE = dn_tile(S, expert_mxfp4)
    N_DN_TILES = HIDDEN // DN_TILE
    RED_WORDS = WAVES * 64 * 4
    LDS_KEYS = max(
        (288 if index_paged else 264) if with_indexer else 0,
        SPLIT_KEYS,
        S * MOE_SLOTS,
    )

    # one phase-overlaid LDS arena, not the sum of every stage's needs.
    # The largest X users are the four-sample input projection, all-sample MoE FP8
    # activations, and sparse attention.  Metadata, reductions, and outputs live
    # after that common X region because they are simultaneously live in GEMVs.
    SPLIT_X_WORDS = PT_OFF + SPLIT_KEYS * PS
    X_WORDS = max(
        SAMPLE_TILE * HIDDEN // 2,
        S * HIDDEN // 4,
        SPLIT_X_WORDS,
        0 if index_paged else index_max_seq,
        down_x_words(S, expert_inter, expert_mxfp4, native_fp4_mfma),
        # the tail's normed rows, or one half of the NextN entry's, every row at once
        S * HIDDEN // 2 if tail or eh_entry else 0,
    )
    # Expert jobs (S > 1, X region words): down's FP8 mid, then its products a
    # (row group, row, unit); up/gate writes its expert groups past both and its
    # activations: [0] the group count, [1 + g] group g's expert, then each
    # (group, row)'s slot of it (-1: the row does not use it).
    MOE_GROUPS = S * TOP_K + 1
    # LDS words of a row's FP8 activation, a (row, slot)'s FP8 mid, a (row group,
    # row)'s down products; above 4 rows, where MFMA columns read different rows,
    # 4 words longer than their data, so those rows sit in different banks
    XQ_STRIDE = HIDDEN // 4 + (4 if S > 4 else 0)
    MID_STRIDE = expert_inter // 4 + 4
    DN_KU = expert_inter // 128  # down's 128-k units a (row, slot)
    DN_PROD_STRIDE = MOE_SLOTS * DN_KU * 16 + 4
    DN_PROD_OFF = S * MOE_SLOTS * MID_STRIDE
    DN_PROD_WORDS = 2 * S * DN_PROD_STRIDE
    JOB_OFF = max(S * XQ_STRIDE, DN_PROD_OFF + DN_PROD_WORDS)
    JOB_SLOT_OFF = JOB_OFF + 1 + MOE_GROUPS
    assert S == 1 or JOB_SLOT_OFF + MOE_GROUPS * S <= X_WORDS
    # Paged top-k: a 4096-bin histogram, then (key, position) candidates of
    # the threshold bin, both in the X region of the selecting CTA.
    SEL_ITEMS = 8
    SEL_CHUNK = THREADS * SEL_ITEMS
    SEL_BINS = 4096
    SEL_CAP = (X_WORDS - SEL_BINS) // 2
    assert not index_paged or SEL_CAP >= 2048
    # a row selects on up to SEL_SLICES CTAs, whole rows a round of the grid
    SEL_SLICES = INDEX_SEL_SLICES
    assert not index_paged or (SEL_CAP <= INDEX_CAND_STAGE_CAP and G % SEL_SLICES == 0)
    # At 4 rows with one up/gate task round (TP8) a CTA's up/gate jobs are one row's
    # and its first job goes out after that row's first picks (not in the NextN
    # indexer launch: there its first job's weights next to the rest of the rounds
    # add VGPR spills). Its routing scratch sits at the X region's end, past the
    # rows' FP8 activations: a routing wave's picks and a word a lane, then the
    # wave's round state (10 words a lane).
    EARLY_ISSUE = (
        S == 4
        and expert_inter == G
        and S * N_ROUTER < G
        and not dense_experts
        and not (eh_entry and with_indexer)
    )
    ROUTE_WORDS = TOP_K + 64
    STASH_WORDS = 10 * 64
    ROUTE_OFF = X_WORDS - (WAVES * ROUTE_WORDS + S * STASH_WORDS if EARLY_ISSUE else 0)
    assert ROUTE_OFF >= S * HIDDEN // 4
    MISC_OFF = X_WORDS
    MISC_WORDS = max(
        8 + S * (XQ_GROUPS if native_fp4_mfma else XQ_BLOCKS),
        S * MOE_SLOTS * (expert_inter // (32 if native_fp4_mfma else 128)),
        N_SPLIT + 2 + (4 if attention_ptpc else 0),
    )
    KEYS_OFF = MISC_OFF + MISC_WORDS
    DNW_OFF = KEYS_OFF + LDS_KEYS
    RED_OFF = DNW_OFF + S * MOE_SLOTS
    OUT_OFF = RED_OFF + RED_WORDS
    # UK writes its 128-row result straight from the reduction tile to q_lat;
    # all remaining stages need at most these compact output tiles.
    OUT_WORDS = max(S * ROW_TILE, S * 2 * UG_TILE)
    WORK_WORDS = OUT_OFF + OUT_WORDS
    # Above 4 rows, with the paged indexer, the scoring layout (written once a CTA's
    # front tile is done, read at the scoring stage) sits ahead of the arena: the row
    # groups scored together and their requests' first rows, then each row's index
    # bound and page-table row. Up to 4 rows the scoring stage computes it from its own
    # loads (the build's registers come out worse with it ahead)
    PLANNED = index_paged and S > 4
    assert not PLANNED or use_atom_kv_cache
    PLAN_WORDS = (5 + 2 * S + 3) // 4 * 4 if PLANNED else 0
    assert WORK_WORDS + PLAN_WORDS <= 32768, (
        "keep static LDS below the MI355X per-workgroup budget"
    )

    first, acc = {}, 0
    for name, n in stage_tasks(
        S,
        H,
        topk,
        with_indexer,
        index_max_seq,
        expert_mxfp4,
        inter=expert_inter,
        tokens=tokens,
        head_rows=vocab_rows if tail else 0,
        eh_tiles=EH_TILES,
    ):
        first[name] = acc
        acc += n
    base = stage_bases(
        S, H, topk, with_indexer, index_max_seq, expert_mxfp4, inter=expert_inter
    )
    # the embedding push runs ahead of qkv_a on the CTAs from 0; the tail's stages
    # start at CTA 0
    base["embed"] = base["lmhead"] = base["gather"] = 0
    base["eh"] = eh_base(EH_TILES, with_indexer)

    @fx.struct
    class Smem:
        work: fx.Array[fx.Float32, PLAN_WORDS + WORK_WORDS, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def glm5_monokernel(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        positions: Int64,
        slot_mapping: Int64,
        sparse_kv_indptr: Int64,
        kv_cache: Int64,
        pe_cache: Int64,
        indices: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_in: Int64,
        g_q: Int64,
        g_kv: Int64,
        g_post: Int64,
        w_qkv_a: Int64,
        s_qkv_a: Int64,
        w_q_b: Int64,
        s_q_b: Int64,
        w_uk: Int64,
        s_uk: Int64,
        w_uv: Int64,
        s_uv: Int64,
        w_o: Int64,
        s_o: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        timeline_buf: Int64,
        step: Int64,
        block_table: Int64,
        req_ids: Int64,
        out_indices: Int64,
        ix_wk: Int64,
        ix_sk: Int64,
        ix_ww: Int64,
        ix_wq: Int64,
        ix_sq: Int64,
        ix_gk: Int64,
        ix_bk: Int64,
        rank: Int32,
        layer: Int32,
        bt_stride: Int32,
    ):
        _ = keyed
        # every wait of this launch runs under the guard (deadline, error words);
        # the scope is entered by hand so the traced body keeps its indentation
        guarded = guard.scope(
            guard.Guard(
                step,
                own=sym,
                peers=peers,
                rank=rank,
                npes=W,
                flags_offset=SY["flags"],
                fault_site=fault_site or None,
                fault_after=fault_after,
                stop=stop,
            )
        )
        guarded.__enter__()
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Smem).peek()
        xs = lds.work.ptr + PLAN_WORDS if const_expr(PLAN_WORDS) else lds.work.ptr
        plan = fx.recast_iter(fx.Int32, lds.work.ptr)
        misc = xs + MISC_OFF
        keys = fx.recast_iter(fx.Int32, xs + KEYS_OFF)
        jobs = fx.recast_iter(fx.Int32, xs + JOB_OFF)
        dnw = xs + DNW_OFF
        red = xs + RED_OFF
        route_scr = fx.recast_iter(fx.Int32, xs + ROUTE_OFF)
        outs = xs + OUT_OFF
        pl = xs  # score Q is dead before split probabilities are written
        attn_keys = keys
        ktile = xs + KT_OFF  # f32-typed views holding raw bf16 pairs
        petile = xs + PT_OFF
        v4f = fx.Vector.make_type(4, fx.Float32)

        r_h = _rsrc(h_in)
        # this launch's epoch: every mailbox tag must equal it.  ``step`` is a
        # device counter bumped once per decode step (graph friendly); ``layer``
        # makes it unique per layer within the step.
        step_value = _uniform(bo.buffer_load(_rsrc(step), 0, vec_width=1, dtype=T.i32))
        tag = step_value * launches_per_step + layer + 1
        peer_slot = (step_value * launches_per_step + layer) & 1
        pos0 = _uniform(bo.buffer_load(_rsrc(cur_pos), 0, vec_width=1, dtype=T.i32))
        r_peers = _rsrc(peers)
        # One wave sends to one peer, so retain only that wave's destination.
        pv = fx.Vector(
            bo.buffer_load(r_peers, fx.min(wave, W - 1) * 2, vec_width=2, dtype=T.i32)
        )
        peer_dst = (fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(
            fx.Uint32(_uniform(pv[0]))
        )

        # ------------------------------------------------------------ helpers
        def ld_f32(r, i):
            return fx.Float32(bo.buffer_load(r, i, vec_width=1, dtype=T.f32))

        def sld_f32(addr):
            """The f32 at a wave-uniform byte address, as a scalar load issued here
            (volatile: not sunk to its use)."""
            ptr = llvm.IntToPtrOp(
                ir.Type.parse("!llvm.ptr<4>"), as_ir_value(fx.Int64(addr))
            ).result
            return fx.Float32(
                llvm.LoadOp(T.f32, ptr, alignment=4, volatile_=True).result
            )

        def sld_i32(addr):
            """The i32 at a wave-uniform byte address, as a scalar load issued here."""
            return sld_f32(addr).bitcast(fx.Int32)

        def ld_bf16(r, i):
            return fx.Float32(
                fx.BFloat16(bo.buffer_load(r, i, vec_width=1, dtype=T.bf16))
            )

        def row_index_bounds(s):
            begin = fx.Int32(
                bo.buffer_load(_rsrc(sparse_kv_indptr), s, vec_width=1, dtype=T.i32)
            )
            end = fx.Int32(
                bo.buffer_load(_rsrc(sparse_kv_indptr), s + 1, vec_width=1, dtype=T.i32)
            )
            return begin, end

        def row_active(s):
            if const_expr(use_atom_kv_cache):
                begin, end = row_index_bounds(s)
                return end > begin
            return True

        def row_position(s):
            if const_expr(use_atom_kv_cache):
                position = fx.Int32(
                    bo.buffer_load(_rsrc(positions), s * 2, vec_width=1, dtype=T.i32)
                )
                present = row_active(s) | (row_slot(s) >= 0)
                return present.select(position, fx.Int32(0))
            return pos0 + s

        def row_slot(s):
            if const_expr(use_atom_kv_cache):
                return fx.Int32(
                    bo.buffer_load(_rsrc(slot_mapping), s * 2, vec_width=1, dtype=T.i32)
                )
            return pos0 + s

        def row_writes_cache(s):
            return row_slot(s) >= 0

        def index_bound(s):
            return row_writes_cache(s).select(row_position(s) + 1, fx.Int32(0))

        def index_request(s):
            return fx.Int32(bo.buffer_load(_rsrc(req_ids), s, vec_width=1, dtype=T.i32))

        def index_block(req, key):
            return fx.Int32(
                bo.buffer_load(
                    _rsrc(block_table),
                    req * bt_stride + key // index_block_size,
                    vec_width=1,
                    dtype=T.i32,
                )
            )

        def index_value_word(block, off, d):
            """Dword holding index-cache values d .. d + 3 (d % 4 == 0) of one token."""
            if const_expr(index_shuffled):
                byte = (off // 16) * (16 * INDEX_DIM) + (off % 16) * 16
                byte = byte + (d // 16) * 256 + d % 16
            else:
                byte = off * INDEX_DIM + d
            return block * (index_block_bytes // 4) + byte // 4

        def index_scale_word(block, off):
            return (
                block * (index_block_bytes // 4)
                + index_block_size * INDEX_DIM // 4
                + off
            )

        def lds_ld(ptr, i):
            return fx.ptr_load(ptr + i)

        def lds_st(ptr, i, v):
            fx.ptr_store(v, ptr + i)

        # The stages of a NextN, indexer or dense-MLP launch (after a NextN launch's
        # entry) load the pointer arguments they read again from the kernarg segment,
        # so no argument stays live in registers across the launch: with the NextN
        # entry's registers on top, the compiler restores such values from SGPR spill
        # lanes that were never written, and those kinds spill far fewer SGPRs. Only
        # the segment's address stays live.
        kernarg_ptr = ir.Type.parse("!llvm.ptr<4>")
        if const_expr(RELOAD_ARGS):
            kernarg = fx.Int64(
                llvm.PtrToIntOp(
                    T.i64,
                    llvm.call_intrinsic(
                        kernarg_ptr, "llvm.amdgcn.kernarg.segment.ptr", [], [], []
                    ),
                ).result
            )

        RELOADED = ('x_out', 'positions', 'slot_mapping', 'sparse_kv_indptr', 'kv_cache', 'pe_cache', 'indices', 'rope_cos', 'rope_sin', 'g_in', 'g_q', 'g_kv', 'g_post', 'w_qkv_a', 's_qkv_a', 'w_q_b', 's_q_b', 'w_uk', 's_uk', 'w_uv', 's_uv', 'w_o', 's_o', 'w_r', 'bias', 'w_ug', 's_ug', 'w_dn', 's_dn', 'scratch', 'sym', 'timeline_buf', 'block_table', 'req_ids', 'out_indices', 'ix_wk', 'ix_sk', 'ix_ww', 'ix_wq', 'ix_sq', 'ix_gk', 'ix_bk')

        def reload_args():
            """The pointer arguments as fresh loads from the kernarg segment (the current values
            when this kind does not reload). The segment's address passes through an empty asm
            first, so LLVM cannot merge these loads with another stage's. The caller rebinds them
            (and ``xin``, when x is pushed, from the reloaded ``sym``)."""
            if const_expr(not RELOAD_ARGS):
                return (x_out, positions, slot_mapping, sparse_kv_indptr, kv_cache, pe_cache, indices, rope_cos, rope_sin, g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o, s_o, w_r, bias, w_ug, s_ug, w_dn, s_dn, scratch, sym, timeline_buf, block_table, req_ids, out_indices, ix_wk, ix_sk, ix_ww, ix_wq, ix_sq, ix_gk, ix_bk,)
            base = fx.Int64(
                llvm.inline_asm(
                    T.i64, [kernarg.ir_value()], "", "=s,0", has_side_effects=True
                )
            )

            def load(name):
                at = base + fx.Int64(8 * KERNARG_I64.index(name))
                ptr = llvm.IntToPtrOp(kernarg_ptr, as_ir_value(at)).result
                return fx.Int64(
                    llvm.LoadOp(T.i64, ptr, alignment=8, invariant=True).result
                )

            return tuple(load(n) for n in RELOADED)

        def fwht128(v0, v1):
            """The orthonormal 128-point Hadamard transform (Sylvester order) of the
            values one wave holds, lane l holding entries 2l and 2l + 1."""
            v0, v1 = v0 + v1, v0 - v1
            for j in range_constexpr(6):
                m = 1 << j
                p0, p1 = _xshfl(v0, m), _xshfl(v1, m)
                upper = (lane & m) != 0
                v0 = upper.select(p0 - v0, v0 + p0)
                v1 = upper.select(p1 - v1, v1 + p1)
            return v0 * (128**-0.5), v1 * (128**-0.5)

        def bf16_pair(a, b):
            """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
            return (
                fx.Vector.from_elements([a, b], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Float32)[0]
            )

        def bf16_round(a):
            return fx.Float32(fx.Float32(a).to(fx.BFloat16))

        def index_arg(i):
            """The indexer's pointer ``i`` (0-6: wk, its scales, weights_proj packed, wq_b, its
            scales, the k LayerNorm weight and bias), a kernel argument; the NextN entry's and the
            timeline's (7 on) from the compact parameter table.

            Fused-indexer launches carry this table in the otherwise independent
            timeline argument, so both kinds share one signature.  The indices
            argument similarly carries index_cache.
            """
            if const_expr(i < 7):
                return (ix_wk, ix_sk, ix_ww, ix_wq, ix_sq, ix_gk, ix_bk)[i]
            pv = fx.Vector(
                bo.buffer_load(_rsrc(timeline_buf), i * 2, vec_width=2, dtype=T.i32)
            )
            return (fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(
                fx.Uint32(_uniform(pv[0]))
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
            """Pairs i, i+1 (i even) in one 16-byte store; f32 values, or int32 bits."""

            def bits(v):
                return v if isinstance(v, fx.Int32) else fx.Float32(v).bitcast(fx.Int32)

            vec = fx.Vector.from_elements([bits(v0), tag, bits(v1), tag], fx.Int32)
            bo.buffer_store(vec, _rsrc(base_addr), i * 2, cache_modifier=cm)

        def put_bf(base_addr, i, vs, cm=CM_DEV):
            """Elements i .. i + len(vs) (2 or 4, i aligned) as packed bf16 pairs: pair
            i / 2 + j := (bf16(vs[2j]) | bf16(vs[2j + 1]) << 16, tag), one 8 / 16-byte store.
            """
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

        def poll(specs, scope="agent", batch=POLL_MAX, between=None):
            """Batched poll of mailbox pairs: ``specs`` = [(base_addr, pair index, npairs in {1, 2})].

            All pairs are loaded together with plain 8 / 16-byte coherent buffer loads
            (sc1 locally, sc0 sc1 for peer memory); while any tag is not this launch's
            the whole batch is re-loaded, so a batch costs one round trip after its
            last producer lands.  A side-effecting (compiler-opaque) asm statement in
            the retry loop keeps the loads from being hoisted.  Returns one list of
            Int32 value bits per spec.  ``between()`` runs after the first loads are
            issued and before they are checked (one batch only)."""
            if const_expr(len(specs) == 0):
                return []
            assert between is None or len(specs) <= batch
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

            issued = []
            if const_expr(between is not None):
                issued.append(load_all())
                between()

            def first_then_all():
                return issued.pop() if issued else load_all()

            v = spin_until(first_then_all, pending)
            outs_, e = [], 0
            for _, _, n in specs:
                outs_.append([v[e + 2 * q] for q in range(n)])
                e += 2 * n
            return outs_

        def hint_wait(n, addr_of, mark=None):
            """Consumers poll their payload directly (tight per-wave spins); a wave-0
            pre-poll of each producer's last pair only added a hop of latency."""
            if const_expr(mark is not None):
                stamp(mark[0], mark[1], 1)
            gpu.barrier()

        def pre_poll(n, addr_of):
            """Wave 0 spins on one small pair per producer (lane j -> producer j < n <= 64)
            before a large payload poll, so waiting CTAs do not flood memory."""
            if wave == 0:
                b, i = addr_of(fx.min(lane, n - 1))
                poll([(b, i, 1)])
            gpu.barrier()

        def get(base_addr, i):
            return poll([(base_addr, i, 1)])[0][0]

        def getf(base_addr, i):
            return get(base_addr, i).bitcast(fx.Float32)

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

        def get2(base_addr, i):
            return get2_many([(base_addr, i)])[0]

        def get_bf2_many(specs):
            """[(base, i)] packed bf16 elements i, i + 1 (i even) -> list of (f32, f32)."""
            return [bf2_f32(v[0]) for v in poll([(b, i // 2, 1) for b, i in specs])]

        # ---- wave reductions
        def wave_sum(v):
            for sh in range_constexpr(6):
                v = _xred(v, 32 >> sh, lambda a, b: a + b)
            return v

        def block_reduce(vs, wave_op, op):
            """Block-wide reductions of several per-thread values with one LDS exchange."""
            ws = [wave_op(v) for v in vs]
            if lane == 0:
                for i in range_constexpr(len(vs)):
                    lds_st(red, i * WAVES + wave, ws[i])
            gpu.barrier()
            tots = []
            for i in range_constexpr(len(vs)):
                t = lds_ld(red, i * WAVES)
                for w in range_constexpr(1, WAVES):
                    t = op(t, lds_ld(red, i * WAVES + w))
                tots.append(t)
            gpu.barrier()
            return tots

        def block_sums(vs):
            return block_reduce(vs, wave_sum, lambda a, b: a + b)

        def block_maxs(vs):
            return block_reduce(vs, wave_max, fx.max)

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
        # Weights read once a launch load nontemporal: at 2 and 8 rows the attention
        # weights and the experts; at 4 rows the up/gate tiles when a job streams each
        # expert once for every row (EARLY_ISSUE). Down's 4-row tiles re-read the
        # shared expert's and repeated experts' lines per row.
        AT_CM = CM_NT if S != 4 else 0

        def ld_fp8_chunk(w_rsrc, rg, kc, NKC, K, ln, shuffled, cm=0):
            """The 16 bytes lane ``ln`` holds of 64-k chunk ``kc`` of row group ``rg``:
            packed (``pack_fp8``), or two 8-byte halves of AITER's (16, 16) layout
            ([rows / 16, K / 32, 2, 16 rows, 16 bytes]) in the same order.

            Shuffled: a lane of an even 16-lane row loads the whole 16-byte segment of
            the first K32 block, its partner 16 lanes up the same segment of the second,
            and one 8-byte swap gives each lane its two halves. Partner lanes must load
            the same row group and chunk, and ``ln`` must keep the lane's row and its
            partner's offset of 16 (``(lane & -16) | f(lane % 16)``). The swap is
            made where the words are first read, so the load stays in flight until then.
            ``cm``: the load's cache policy."""
            if const_expr(shuffled):
                b, r = ln // 16, ln % 16
                word = (
                    (rg * (K // 32) + 2 * kc) * 512
                    + (b // 2) * 256
                    + r * 16
                    + (b % 2) * 512
                ) // 4
                return _RowSwapped(
                    fx.Vector(
                        bo.buffer_load(
                            w_rsrc, word, vec_width=4, dtype=T.i32, cache_modifier=cm
                        )
                    )
                )
            return fx.Vector(
                bo.buffer_load(
                    w_rsrc,
                    ((rg * NKC + kc) * 64 + ln) * 4,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=cm,
                )
            )

        def unit_fp8(
            w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word, coef=None, shuffled=False, cm=0
        ):
            """Issue one 64-k chunk of row group ``rg`` of an FP8 matrix; the
            bf16 activation chunk starts at LDS word ``b_word``."""
            wv = ld_fp8_chunk(w_rsrc, rg, kc, NKC, K, lane, shuffled, cm)
            s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // BK) + kc * 64 // BK)
            if const_expr(callable(coef)):  # factor known only after a later wait
                return ("fp8", [wv], lambda: s * coef(), b_word + (lane // 16) * 4)
            if const_expr(coef is not None):
                s = s * coef
            return ("fp8", [wv], s, b_word + (lane // 16) * 4)

        def unit_fp8x2(
            w_rsrc,
            s_rsrc,
            rg,
            kc,
            NKC,
            K,
            b_word,
            coef=None,
            scale_rows=SCALE_BM,
            shuffled=False,
            cm=0,
        ):
            """Issue both 64-k halves of one 128-k FP8 weight-scale block."""
            wv = [
                ld_fp8_chunk(w_rsrc, rg, kc + h, NKC, K, lane, shuffled, cm)
                for h in range(2)
            ]
            s = ld_f32(s_rsrc, (rg * 16 // scale_rows) * (K // 128) + kc // 2)
            if const_expr(callable(coef)):
                return ("fp8x2", wv, lambda: s * coef(), b_word + (lane // 16) * 4)
            if const_expr(coef is not None):
                s = s * coef
            return ("fp8x2", wv, s, b_word + (lane // 16) * 4)

        def unit_f8f8(
            w_rsrc,
            s_rsrc,
            rg,
            kc,
            NKC,
            K,
            b_word,
            coef,
            ln=None,
            shuffled=False,
            s_addr=None,
            cm=0,
        ):
            """Issue one 128-k chunk (packed 64-k chunks kc, kc + 1; kc even) of row group
            ``rg`` against the FP8 activation of LDS words ``b_word`` + [0, 32) (``f8_word``
            order); ``coef()`` = activation block scale (times route weight).  ``ln``
            = the lane whose weights are loaded (default: own lane). With ``s_addr``
            (the byte address of the expert's scales; ``rg`` and ``kc`` wave-uniform)
            the scale is a scalar load. ``cm``: the weight loads' cache policy."""
            ln = lane if ln is None else ln
            wv = [
                ld_fp8_chunk(w_rsrc, rg, kc + h, NKC, K, ln, shuffled, cm)
                for h in range(2)
            ]
            si = (rg * 16 // SCALE_BM) * (K // 128) + kc // 2
            if const_expr(s_addr is None):
                s = ld_f32(s_rsrc, si)
            else:
                s = sld_f32(s_addr + fx.Int64(si) * 4)
            return ("f8f8", wv, lambda: s * coef(), b_word + (lane // 16) * 4)

        def unit_ptpc(w_rsrc, rg, kc, NKC, b_word, coef=None, ln=None):
            """Issue one packed 128-K PTPC FP8 tile; output-channel scale is applied after reduction."""
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        ((rg * NKC + kc + h) * 64 + ln) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for h in range(2)
            ]
            return ("ptpc", wv, coef, b_word + (lane // 16) * 4)

        def unit_ptpc64(w_rsrc, rg, kc, NKC, b_word, coef=None, ln=None):
            """Issue one packed 64-K PTPC FP8 tile."""
            ln = lane if ln is None else ln
            weight = fx.Vector(
                bo.buffer_load(
                    w_rsrc,
                    ((rg * NKC + kc) * 64 + ln) * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            return ("ptpc64", [weight], coef, b_word + (lane // 16) * 4)

        def unit_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln=None):
            """Issue one native packed 128-K MXFP4 tile and four E8M0 row scales."""
            ln = lane if ln is None else ln
            row = rg * 16 + ln % 16
            if const_expr(atom_experts):
                g4 = ln // 16
                raw = fx.Vector.from_elements(
                    [
                        fx.Int32(
                            bo.buffer_load(
                                w_rsrc,
                                (rg * (K // 64) + kc * 2 + sp // 2) * 128
                                + (sp % 2) * 64
                                + (row % 16) * 4
                                + g4,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
                        for sp in range_constexpr(4)
                    ],
                    fx.Int32,
                )
                scales = [
                    (
                        fx.Int32(
                            bo.buffer_load(
                                s_rsrc,
                                atom_mxfp4_scale_index(row, kc * 4 + sp, K // 32),
                                vec_width=1,
                                dtype=T.i8,
                            )
                        )
                        << fx.Int32(23)
                    ).bitcast(fx.Float32)
                    for sp in range_constexpr(4)
                ]
            else:
                raw = fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        ((rg * (K // 128) + kc) * 64 + ln) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                packed_scale = fx.Int32(
                    bo.buffer_load(
                        s_rsrc, row * (K // 128) + kc, vec_width=1, dtype=T.i32
                    )
                )
                scales = [
                    (
                        (packed_scale.shrui(fx.Int32(sp * 8)) & fx.Int32(0xFF))
                        << fx.Int32(23)
                    ).bitcast(fx.Float32)
                    for sp in range_constexpr(4)
                ]
            return ("mxfp4", (raw, scales), coef, b_word + (lane // 16) * 4)

        def unit_mxfp4_bf16(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln=None):
            fmt, weights, factor, _ = unit_mxfp4(
                w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln
            )
            return ("mxfp4_bf16", weights, factor, b_word + (lane // 16) * 4)

        def unit_native_mxfp4(
            w_rsrc, s_rsrc, rg, kc, K, b_word, b_scale_word, coef=None, ln=None
        ):
            """Issue ATOM's 16-byte/lane W4 tile directly on the scaled MFMA."""
            ln = lane if ln is None else ln
            raw = fx.Vector(
                bo.buffer_load(
                    w_rsrc,
                    (rg * (K // 64) + kc * 2) * 128 + ln * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            scale = fx.Int32(
                bo.buffer_load(
                    s_rsrc,
                    atom_mxfp4_scale_index(
                        rg * 16 + ln % 16, kc * 4 + ln // 16, K // 32
                    ),
                    vec_width=1,
                    dtype=T.i8,
                )
            )
            return ("native_mxfp4", (raw, scale, b_scale_word), coef, b_word)

        def lds_mxfp8(chunk_word):
            """One scaled-MFMA B operand from a row-major 128-value MXFP8 tile."""
            words = [
                fx.ptr_load(
                    xs + chunk_word + 16 * half + 4 * (lane // 16), result_type=v4f
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

        def unit_attention(
            w_rsrc,
            s_rsrc,
            rg,
            kc,
            NKC,
            K,
            BK,
            b_word,
            ln=None,
            scale_rows=SCALE_BM,
            coef=None,
            shuffled=False,
            cm=0,
        ):
            if const_expr(attention_bf16):
                return unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln)
            if const_expr(attention_ptpc):
                if const_expr(BK == 64):
                    return unit_ptpc64(w_rsrc, rg, kc, NKC, b_word, coef=coef, ln=ln)
                return unit_ptpc(w_rsrc, rg, kc, NKC, b_word, coef=coef, ln=ln)
            if const_expr(BK == 64):
                return unit_fp8(
                    w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word, shuffled=shuffled, cm=cm
                )
            return unit_fp8x2(
                w_rsrc,
                s_rsrc,
                rg,
                kc,
                NKC,
                K,
                b_word,
                scale_rows=scale_rows,
                shuffled=shuffled,
                cm=cm,
            )

        def mma_units(acc, units):
            """acc[4] += coef * (W_chunk @ X_chunk) for every issued unit."""
            for fmt, wv, coef, bw in units:
                if const_expr(callable(coef)):
                    coef = coef()
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                if const_expr(fmt in ("mxfp4", "mxfp4_bf16")):
                    raw, scales = wv
                    for sp in range_constexpr(4):
                        a = _mxfp4_to_bf16x8(raw[sp], scales[sp])
                        if const_expr(fmt == "mxfp4"):
                            wh, ws = sp // 2, sp % 2
                            bv = fx.ptr_load(
                                xs + (bw + wh * 16), result_type=v4f
                            ).bitcast(fx.Int32)
                            b = _fp8_to_bf16x8(bv[ws * 2], bv[ws * 2 + 1])
                        else:
                            b = fx.ptr_load(
                                xs + (bw + sp * 16), result_type=v4f
                            ).bitcast(fx.BFloat16)
                        c = fx.Vector(
                            rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c])
                        )
                if const_expr(fmt == "f8f8"):  # one FP8 x FP8 MFMA (E8M0 scales = 1)
                    a = fx.Vector.from_elements(
                        [wv[h][e] for h in range(2) for e in range(4)], fx.Int32
                    )
                    bv = [
                        fx.ptr_load(xs + (bw + h * 16), result_type=v4f).bitcast(
                            fx.Int32
                        )
                        for h in range(2)
                    ]
                    b = fx.Vector.from_elements(
                        [bv[h][e] for h in range(2) for e in range(4)], fx.Int32
                    )
                    one = fx.Int32(127)
                    c = fx.Vector(
                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.vec(4, T.f32), [a, b, c, 0, 0, 0, one, 0, one]
                        )
                    )
                if const_expr(fmt == "native_mxfp4"):
                    raw, weight_scale, activation_scale_word = wv
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
                                0,
                                weight_scale,
                                0,
                                activation_scale,
                            ],
                        )
                    )
                if const_expr(fmt in ("ptpc", "ptpc64")):
                    halves64 = 2 if fmt == "ptpc" else 1
                    for half64 in range_constexpr(halves64):
                        a = wv[half64].bitcast(fx.Int64)
                        b = fx.ptr_load(
                            xs + (bw + half64 * 16), result_type=v4f
                        ).bitcast(fx.Int64)
                        for half32 in range_constexpr(2):
                            c = fx.Vector(
                                rocdl.mfma_f32_16x16x32_fp8_fp8(
                                    T.vec(4, T.f32),
                                    [a[half32], b[half32], c, 0, 0, 0],
                                )
                            )
                nsp = (
                    4
                    if fmt == "fp8x2"
                    else (
                        2
                        if fmt
                        not in (
                            "f8f8",
                            "ptpc",
                            "ptpc64",
                            "mxfp4",
                            "mxfp4_bf16",
                            "native_mxfp4",
                        )
                        else 0
                    )
                )
                for sp in range_constexpr(nsp):
                    if const_expr(fmt in ("fp8", "fp8x2")):
                        wh = sp // 2 if fmt == "fp8x2" else 0
                        ws = sp % 2 if fmt == "fp8x2" else sp
                        a = _fp8_to_bf16x8(wv[wh][ws * 2], wv[wh][ws * 2 + 1])
                    else:
                        a = wv[sp].bitcast(fx.BFloat16)
                    b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(
                        fx.BFloat16
                    )
                    c = fx.Vector(
                        rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c])
                    )
                if const_expr(coef is None):
                    acc = [acc[e] + c[e] for e in range(4)]
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

        def pick(values, index):
            value = values[0]
            for i in range_constexpr(1, len(values)):
                value = (index == i).select(values[i], value)
            return value

        def attention_b_word(element):
            return element // 4 if const_expr(attention_ptpc) else element // 2

        def emit_ptpc(stride, row_base, scale_rsrc, x_scales):
            def f(rl, n, v):
                lds_st(
                    outs,
                    n * stride + rl,
                    bf16_round(
                        v * pick(x_scales, n) * ld_f32(scale_rsrc, row_base + rl)
                    ),
                )

            return f

        def emit_ptpc_weight(stride, row_base, scale_rsrc):
            def f(rl, n, v):
                lds_st(
                    outs,
                    n * stride + rl,
                    bf16_round(v * ld_f32(scale_rsrc, row_base + rl)),
                )

            return f

        def emit_ptpc_scalar(stride, scale):
            def f(rl, n, v):
                lds_st(outs, n * stride + rl, bf16_round(v * scale))

            return f

        def stage_x_rmsnorm_ptpc(ld4s, n, gamma, loaded=None, count=S, first=0):
            """RMSNorm followed by production per-token FP8 quantization, into LDS
            sample rows first ... first + count - 1."""
            per = n // (4 * THREADS)
            ks = [(tid + i * THREADS) * 4 for i in range(per)]
            gs, vals = (
                loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
            )
            sums = []
            for s in range_constexpr(count):
                ss = fx.Float32(0.0)
                for i in range_constexpr(per):
                    for value in vals[s * per + i]:
                        ss = ss + value * value
                sums.append(ss)
            totals = block_sums(sums)
            normed = []
            for s in range_constexpr(count):
                rstd = _rsq(totals[s] * (1.0 / n) + EPS)
                for i in range_constexpr(per):
                    normed.append(
                        [
                            vals[s * per + i][j] * rstd * gs[i][j]
                            for j in range_constexpr(4)
                        ]
                    )
            local_max = []
            for s in range_constexpr(count):
                amax = fx.Float32(0.0)
                for i in range_constexpr(per):
                    for value in normed[s * per + i]:
                        amax = fx.max(amax, fmath.absf(value))
                local_max.append(amax)
            maxima = block_maxs(local_max)
            scales = []
            for s in range_constexpr(count):
                scale = (maxima[s] == 0.0).select(
                    fx.Float32(1.0), maxima[s] * (1.0 / FP8_MAX)
                )
                scales.append(scale)
                reciprocal = 1.0 / scale
                for i in range_constexpr(per):
                    values = [
                        div_rn(value, scale, reciprocal)
                        for value in normed[s * per + i]
                    ]
                    lds_st(
                        xs,
                        ((first + s) * n + ks[i]) // 4,
                        fp8_pack4(values[0], values[1], values[2], values[3]).bitcast(
                            fx.Float32
                        ),
                    )
            return scales

        def stage_x_pairs_ptpc(name, count, n, src_of, group=None):
            """Quantize packed-BF16 mailbox rows with one scale per activation group."""
            group = n if group is None else group
            groups = (n + group - 1) // group
            scales = []
            for s in range_constexpr(count):
                for g in range_constexpr(groups):
                    width = min(group, n - g * group)
                    words = width // 4
                    per = (words + THREADS - 1) // THREADS
                    local_max = fx.Float32(0.0)
                    row_values = []
                    valids = []
                    for i in range_constexpr(per):
                        word = tid + i * THREADS
                        source_word = fx.min(word, words - 1)
                        element = s * n + g * group + source_word * 4
                        got = poll([(mb(name), src_of(element) // 2, 2)])[0]
                        first = bf2_f32(got[0])
                        second = bf2_f32(got[1])
                        values = [first[0], first[1], second[0], second[1]]
                        valid = word < words
                        row_values.append(values)
                        valids.append(valid)
                        for value in values:
                            local_max = fx.max(
                                local_max,
                                valid.select(fmath.absf(value), fx.Float32(0.0)),
                            )
                    maximum = block_maxs([local_max])[0]
                    scale = (maximum == 0.0).select(
                        fx.Float32(1.0), maximum * (1.0 / FP8_MAX)
                    )
                    scales.append(scale)
                    reciprocal = 1.0 / scale
                    for i in range_constexpr(per):
                        word = tid + i * THREADS
                        if valids[i]:
                            values = row_values[i]
                            quantized = [
                                div_rn(value, scale, reciprocal) for value in values
                            ]
                            lds_st(
                                xs,
                                (s * n + g * group + word * 4) // 4,
                                fp8_pack4(
                                    quantized[0],
                                    quantized[1],
                                    quantized[2],
                                    quantized[3],
                                ).bitcast(fx.Float32),
                            )
            return scales

        def stage_x_rmsnorm(ld4s, n, gamma, mark=None, loaded=None, count=S, first=0):
            """LDS bf16 X[first + s][0:n] = bf16(rmsnorm(x_s) * gamma) for every sample s,
            where ld4s([(s, k)]) -> [(x_s[k], .., x_s[k+3])] (one batched load); returns the
            rstds. ``loaded``: the (gamma, x) loads already issued by load_x_rmsnorm."""
            per = n // (4 * THREADS)
            ks = [(tid + i * THREADS) * 4 for i in range(per)]
            gs, vals = (
                loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
            )
            sss = []
            for s in range_constexpr(count):
                ss = fx.Float32(0.0)
                for i in range_constexpr(per):
                    for a in vals[s * per + i]:
                        ss = ss + a * a
                sss.append(ss)
            if const_expr(mark is not None):
                stamp(mark[0], mark[1], 6)
            rstds = [_rsq(tot * (1.0 / n) + EPS) for tot in block_sums(sss)]
            if const_expr(mark is not None):
                stamp(mark[0], mark[1], 7)
            for s in range_constexpr(count):
                for i in range_constexpr(per):
                    a = vals[s * per + i]
                    for j in range_constexpr(2):
                        lds_st(
                            xs,
                            ((first + s) * n + ks[i]) // 2 + j,
                            bf16_pair(
                                a[2 * j] * rstds[s] * gs[i][2 * j],
                                a[2 * j + 1] * rstds[s] * gs[i][2 * j + 1],
                            ),
                        )
            return rstds

        def load_x_rmsnorm(ld4s, n, gamma, count=S):
            """The gamma loads (issued ahead of the wait), then ld4s -> (gammas, x values)."""
            rg_ = _rsrc(gamma)
            ks = [(tid + i * THREADS) * 4 for i in range(n // (4 * THREADS))]
            gs = []
            for k in ks:
                g = (
                    fx.Vector(bo.buffer_load(rg_, k // 2, vec_width=2, dtype=T.i32))
                    .bitcast(fx.BFloat16)
                    .to(fx.Float32)
                )
                gs.append([g[j] for j in range(4)])
            return gs, ld4s([(s, k) for s in range(count) for k in ks])

        def stage_x_pairs(name, n_total, src_of):
            """LDS bf16 X[k] = packed bf16 mailbox ``name`` element src_of(k) for k < n_total
            (src_of contiguous over aligned groups of 4): one 16-byte poll per 4 elements.
            """
            nq = n_total // 4
            full = nq // THREADS
            vals = poll(
                [
                    (mb(name), src_of((tid + i * THREADS) * 4) // 2, 2)
                    for i in range(full)
                ]
            )
            for i in range_constexpr(full):
                for j in range_constexpr(2):
                    lds_st(
                        xs, (tid + i * THREADS) * 2 + j, vals[i][j].bitcast(fx.Float32)
                    )
            if const_expr(nq % THREADS):
                w = tid + full * THREADS
                if w < nq:
                    v = poll([(mb(name), src_of(w * 4) // 2, 2)])[0]
                    for j in range_constexpr(2):
                        lds_st(xs, w * 2 + j, v[j].bitcast(fx.Float32))

        def quant_scaled(a0, a1):
            """Per-wave FP8 quant of a 128-block held as 2 f32 per lane -> (scaled q0, q1, scale)."""
            amax = wave_max(fx.max(fmath.absf(a0), fmath.absf(a1)))
            nz = amax > 0.0
            qs = nz.select(amax * (1.0 / FP8_MAX), fx.Float32(1.0))
            inv = nz.select(
                _rcp(amax) * FP8_MAX, fx.Float32(1.0)
            )  # hardware rcp, no IEEE divide
            q0 = fx.min(fx.max(a0 * inv, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(a1 * inv, -FP8_MAX), FP8_MAX)
            return q0, q1, qs

        def quant_mxfp8(a0, a1):
            """Per-32 E4M3 values and E8M0 scale used by ATOM/AITER W4A8."""
            amax = fx.max(fmath.absf(a0), fmath.absf(a1))
            for offset in (1, 2, 4, 8):
                amax = _xred(amax, offset, fx.max)
            bits = (amax * fx.Float32(1.0 / FP8_MAX)).bitcast(fx.Int32)
            exponent = (bits >> 23) & fx.Int32(0xFF)
            exponent = exponent + ((bits & fx.Int32(0x7FFFFF)) != 0).select(
                fx.Int32(1), fx.Int32(0)
            )
            exponent = fx.max(fx.min(exponent, fx.Int32(254)), fx.Int32(1))
            inverse = _rcp((exponent << 23).bitcast(fx.Float32))
            q0 = fx.min(fx.max(a0 * inverse, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(a1 * inverse, -FP8_MAX), FP8_MAX)
            return q0, q1, exponent

        def quant_block(a0, a1):
            """quant_scaled, values returned as the FP8-rounded f32s."""
            q0, q1, qs = quant_scaled(a0, a1)
            d0, d1 = _fp8_roundtrip(q0, q1)
            return d0, d1, qs

        def stage_xq(samples, between=None):
            """Poll the router's packed FP8 activation + block scales of ``samples``
            (sample list, or one runtime sample) into LDS, slot 0 for one runtime sample.
            ``between()``: run after the poll's loads are issued (every row only)."""
            nxw = HIDDEN // 4 // THREADS
            nscale = XQ_GROUPS if native_fp4_mfma else XQ_BLOCKS
            if const_expr(list(samples) == list(range(S)) and S % 2 == 0):
                # every row: two tagged words (16 bytes) a load, all scales in one
                # spec, so the whole activation is one poll
                n2 = S * HIDDEN // 8 // THREADS
                nsc = (S * nscale + THREADS - 1) // THREADS
                got = poll(
                    [(mb("xq"), (tid + i * THREADS) * 2, 2) for i in range(n2)]
                    + [
                        (mb("xqs"), fx.min(tid + i * THREADS, S * nscale - 1), 1)
                        for i in range(nsc)
                    ],
                    batch=n2 + nsc,
                    between=between,
                )
                for i in range_constexpr(n2):
                    for e in range_constexpr(2):
                        w = (tid + i * THREADS) * 2 + e
                        at = w if const_expr(native_fp4_mfma) else f8_word(w * 4)
                        if const_expr(XQ_STRIDE > HIDDEN // 4):  # padded rows
                            at = at + w // (HIDDEN // 4) * (XQ_STRIDE - HIDDEN // 4)
                        lds_st(xs, at, got[i][e].bitcast(fx.Float32))
                for i in range_constexpr(nsc):
                    if tid + i * THREADS < S * nscale:
                        lds_st(
                            misc,
                            8 + tid + i * THREADS,
                            got[n2 + i][0].bitcast(fx.Float32),
                        )
                return
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
                    wd = (
                        tid + i * THREADS
                        if const_expr(native_fp4_mfma)
                        else f8_word((tid + i * THREADS) * 4)
                    )
                    lds_st(
                        xs,
                        j * XQ_STRIDE + wd,
                        got[j * nxw + i][0].bitcast(fx.Float32),
                    )
                if tid < nscale:
                    lds_st(
                        misc,
                        8 + j * nscale + tid,
                        got[len(samples) * nxw + j][0].bitcast(fx.Float32),
                    )

        def st_f8(k, q0, q1):
            """LDS FP8 activation bytes k, k + 1 (k even, held by this lane; lane ^ 1 holds
            k ^ 2) in ``f8_word`` order.  Call from the whole wave."""
            w = (
                fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
                & 0xFFFF
            )
            nb = _xshfl(w, 1)
            if lane % 2 == 0:
                lds_st(xs, f8_word(k), (w | (nb << 16)).bitcast(fx.Float32))

        def st_mxfp8(k, q0, q1):
            """Store one row-major MXFP8 pair for the scaled MFMA."""
            w = (
                fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
                & 0xFFFF
            )
            nb = _xshfl(w, 1)
            if lane % 2 == 0:
                lds_st(xs, k // 4, (w | (nb << 16)).bitcast(fx.Float32))

        def st_ptpc(k, q0, q1):
            """Store one wave's PTPC FP8 pairs in the AITER GEMM's linear K order."""
            word = (
                fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
                & 0xFFFF
            )
            neighbor = _xshfl(word, 1)
            if lane % 2 == 0:
                lds_st(xs, k // 4, (word | (neighbor << 16)).bitcast(fx.Float32))

        def load_bias():
            """This lane's 4 expert biases (issue before the scores wait)."""
            if const_expr(dense_experts):
                return None
            return [ld_f32(_rsrc(bias), lane + i * 64) for i in range(N_EXPERTS // 64)]

        def wave_max_all(v):
            """Signed maximum of ``v`` over the wave, in every lane."""
            for ctrl in (0xB1, 0x4E, 0x124, 0x128):  # quad xor 1, 2; row_ror 4, 8
                v = fx.max(v, fx.Int32(update_dpp_i32(v, v, ctrl, 0xF, 0xF, True)))
            for off in (16, 32):
                v = _xred(v, off, fx.max)
            return v

        def route_scores(s):
            """This lane's 4 router scores of sample s (candidate i: expert lane + 64 i)."""
            raws = getf_many(
                [
                    (mb("scores"), s * N_EXPERTS + lane + i * 64)
                    for i in range(N_EXPERTS // 64)
                ]
            )
            stamp("ug", bid, 7)
            return raws

        def route_keys(raws, bs):
            """This lane's 4 keys (all FP32 bits of sigmoid + bias, in unsigned order)
            sorted descending, equal keys smaller ID first, and their IDs."""
            ks = []
            for i in range_constexpr(N_EXPERTS // 64):
                kb = (raws[i] + bs[i]).bitcast(fx.Int32)
                ok = (kb >= 0).select(kb ^ fx.Int32(-(2**31)), ~kb)
                ks.append(fx.Uint32(ok))
            ids = [lane + i * 64 for i in range(N_EXPERTS // 64)]
            for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)):
                first = (ks[a] > ks[b]) | ((ks[a] == ks[b]) & (ids[a] < ids[b]))
                ka, kb, ia, ib = ks[a], ks[b], ids[a], ids[b]
                ks[a], ks[b] = first.select(ka, kb), first.select(kb, ka)
                ids[a], ids[b] = first.select(ia, ib), first.select(ib, ia)
            return ks, ids

        def route_weights(e, raws):
            """Route weight of pick ``lane`` (expert ``e``): its raw score / sum of the 8
            raw scores * ROUTE_SCALE, valid in lanes < TOP_K."""
            src = (e % 64) * 4
            got = [bpermute_i32(src, r.bitcast(fx.Int32)) for r in raws]
            raw = got[0]
            for i in range_constexpr(1, N_EXPERTS // 64):
                raw = (e // 64 == i).select(got[i], raw)
            raw = (lane < TOP_K).select(raw.bitcast(fx.Float32), fx.Float32(0.0))
            tot = raw
            for off in (1, 2, 4):
                tot = _xred(tot, off, lambda a, b: a + b)
            return raw * (_rcp(tot) * ROUTE_SCALE)

        def route_top8(s, raws=None, bs=None):
            """Top-8 of sample s (call from one whole wave, after the router scores landed).

            Preserve all FP32 bits of (sigmoid + bias). Each round reduces the score
            first, then the expert ID only among exactly equal winners. Packing the
            ID into the score's low byte can change GLM's top-k on real checkpoints.
            Returns (expert id, route weight = raw score / sum of
            the 8 raw scores * ROUTE_SCALE) of pick ``lane`` in score order, valid in
            lanes < TOP_K."""
            if const_expr(bs is None):
                bs = load_bias()
            if const_expr(raws is None):
                raws = route_scores(s)
            ks, ids = route_keys(raws, bs)
            # each round takes the wave max of the lane heads and shifts the winning
            # lane's list (0 is below every key)
            ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
            ids = ids + [fx.Int32(N_EXPERTS)]
            mv = fx.Int32(0)  # lane k: the expert ID of pick k
            for k in range_constexpr(TOP_K):
                m = wave_umax(ks[0])
                top = ks[0] == m
                tied = ballot(top)
                # one lane holds the maximum: its head; equal keys: the smallest ID
                low = tied & (fx.Int64(0) - tied)
                expert = read_lane_i32(ids[0], popcount(low - 1))
                if popcount(tied) > 1:
                    expert = 255 - wave_umax(top.select(255 - ids[0], fx.Int32(0)))
                hit = top & (ids[0] == expert)
                ks = [hit.select(ks[i + 1], ks[i]) for i in range(4)] + [ks[4]]
                ids = [hit.select(ids[i + 1], ids[i]) for i in range(4)] + [ids[4]]
                mv = write_lane_i32(expert, k, mv)
            return mv, route_weights(mv, raws)

        ROUTE_NEG = -(2**31)

        def route_heads(ks, ids):
            """State of the early-issue rounds: the lane's keys in signed order (the last
            one below every key) and its IDs a byte each, the head in the low byte."""
            hk = [fx.Int32(k) ^ fx.Int32(ROUTE_NEG) for k in ks]
            hp = ids[0] | (ids[1] << 8) | (ids[2] << 16) | (ids[3] << 24)
            return hk, hp

        def route_rounds(scr, k0, k1, hk, hp):
            """Picks k0 .. k1 - 1 of route_top8, a part of the rounds at a time: each round
            takes the wave maximum of the lane heads into every lane, and the lane whose
            head equals it records its ID at ``scr[k]`` and pops it. Exact while every
            round's maximum is in one lane; returns the state and a nonzero word if a
            round's maximum was in several lanes (then route_top8 decides)."""
            tie = fx.Int64(0)
            for k in range_constexpr(k0, k1):
                m = wave_max_all(hk[0])
                top = hk[0] == m
                t = ballot(top)
                tie = tie | (t & (t - 1))
                lds_st(scr, top.select(fx.Int32(k), TOP_K + lane), hp)
                hk = [top.select(hk[i + 1], hk[i]) for i in range(3)] + [
                    top.select(fx.Int32(ROUTE_NEG), hk[3])
                ]
                hp = top.select(hp >> 8, hp)
            return hk, hp, tie

        def route_tied(s, scr, raws, bs):
            """route_top8's picks of sample s into ``scr`` (after equal heads)."""
            e, _ = route_top8(s, raws, bs)
            if lane < TOP_K:
                lds_st(scr, lane, e)

        def peer_reduce(region, t, residual, out_fn, tile=ROW_TILE):
            """Push BF16 partials to every peer, then sum them in rank order.

            One wave owns each destination, allowing the peer stores to progress
            concurrently without retaining all peer pointers in every wave."""
            region_base = fx.Int64(SY[region]) + fx.Int64(peer_slot) * fx.Int64(
                SY["_part_stride"]
            )
            if const_expr(W > 1):
                if wave < W:
                    pair_count = S * tile // 2
                    for batch in range_constexpr((pair_count + 63) // 64):
                        pair = lane + batch * 64
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
            if tid < S * tile // 2:
                s = tid // (tile // 2)
                r = (tid % (tile // 2)) * 2
                row = t * tile + r
                if const_expr(callable(residual)):
                    r0, r1 = residual(s, row)
                v0 = lds_ld(outs, s * tile + r)
                v1 = lds_ld(outs, s * tile + r + 1)
                if const_expr(W == 1):  # no TP peers: the sum is the local value
                    parts = [(v0, v1)]
                    got = []
                    if const_expr(not callable(residual)):
                        got = poll([(residual, (s * HIDDEN + row) // 2, 1)])
                else:
                    own = sym + region_base
                    specs = [
                        (own, ((src * S + s) * HIDDEN + row) // 2, 1)
                        for src in range(W)
                    ]
                    if const_expr(not callable(residual)):  # packed bf16 pair
                        specs.append((residual, (s * HIDDEN + row) // 2, 1))
                    got = poll(specs, "one-as")
                    parts = [bf2_f32(v[0]) for v in got[:W]]
                    got = got[W:]
                if const_expr(not callable(residual)):
                    r0, r1 = bf2_f32(got[0][0])
                t0 = fx.Float32(0.0)
                t1 = fx.Float32(0.0)
                for src in range_constexpr(W):
                    t0 = t0 + parts[src][0]
                    t1 = t1 + parts[src][1]
                out_fn(s, row, r0 + t0, r1 + t1)

        def dcp_merge_latent(s, t):
            if const_expr(D == 1):
                return True
            region_base = fx.Int64(SY["dcp"]) + fx.Int64(peer_slot) * fx.Int64(
                SY["_dcp_part_stride"]
            )
            summary_base = dcp_summary_index(rank, s, t, 0, S, N_UV)
            if wave < D:
                for batch in range_constexpr((DCP_SUMMARY_PAIRS + 63) // 64):
                    item = lane + batch * 64
                    if item < DCP_SUMMARY_PAIRS:
                        value = fx.Float32(0.0)
                        if item < KV_LORA // 2:
                            value = lds_ld(xs, item)
                        elif item == KV_LORA // 2:
                            value = lds_ld(misc, N_SPLIT)
                        else:
                            value = lds_ld(misc, N_SPLIT + 1)
                        put(peer_dst + region_base, summary_base + item, value, CM_SYS)
            gpu.barrier()
            owner = dcp_uv_owner(t, L, rank)
            if owner:
                own = sym + region_base
                if wave == 0:
                    src = fx.min(lane, D - 1)
                    src_base = dcp_summary_index(src, s, t, 0, S, N_UV)
                    values = poll(
                        [
                            (own, src_base + KV_LORA // 2, 1),
                            (own, src_base + KV_LORA // 2 + 1, 1),
                        ],
                        "one-as",
                    )
                    valid = lane < D
                    m = valid.select(values[0][0].bitcast(fx.Float32), fx.Float32(NEG))
                    local_sum = valid.select(
                        values[1][0].bitcast(fx.Float32), fx.Float32(0.0)
                    )
                    m_all = wave_max(m)
                    z = local_sum * _exp(m - m_all)
                    den = wave_sum(z)
                    if valid:
                        lds_st(
                            misc,
                            lane,
                            z * (den > 0.0).select(_rcp(den), fx.Float32(0.0)),
                        )
                gpu.barrier()
                if tid < KV_LORA // 2:
                    parts = poll(
                        [
                            (
                                own,
                                dcp_summary_index(src, s, t, tid, S, N_UV),
                                1,
                            )
                            for src in range(D)
                        ],
                        "one-as",
                    )
                    o0, o1 = fx.Float32(0.0), fx.Float32(0.0)
                    for src in range_constexpr(D):
                        a0, a1 = bf2_f32(parts[src][0])
                        weight = lds_ld(misc, src)
                        o0, o1 = o0 + a0 * weight, o1 + a1 * weight
                    lds_st(xs, tid, bf16_pair(o0, o1))
                gpu.barrier()
            return owner

        def start(name):
            return (bid + (G - base[name])) & (G - 1)

        # the timeline's address, loaded once: a load per mark would wait for every
        # load in flight before it
        if const_expr(timeline):
            tl_base = (
                index_arg(7) if const_expr(with_indexer or eh_entry) else timeline_buf
            )

        def stamp(name, t, which, lead=0):
            if const_expr(timeline):
                if tid == lead:
                    now = mem_realtime()
                    tl_addr = tl_base
                    fx.generic_store(
                        fx.inttoptr(
                            fx.PointerType.get(
                                fx.Int64.ir_type, fx.AddressSpace.Global, 8
                            ),
                            tl_addr + fx.Int64((first[name] + t) * TL_COLS + which) * 8,
                        ),
                        now,
                    )

        def warm_code(nbytes):
            """Up to ``nbytes`` of this kernel's code from here into this XCD's L2 (a word
            of each 64-byte line), ahead of code that runs once a launch. The buffer ends
            at the code's end: its size is the assembler's distance from the ``s_getpc``
            to ``.Lfunc_end0``, capped at ``nbytes``, and loads past it read nothing."""
            pc_left = llvm.inline_asm(
                llvm.StructType.get_literal([T.i64, T.i32]),
                [],
                "s_getpc_b64 $0\n.Lwarm_code${:uid}:\n"
                "s_mov_b32 $1, .Lfunc_end0-.Lwarm_code${:uid}",
                "=s,=s",
                has_side_effects=True,
            )
            pc = fx.Int64(llvm.extractvalue(T.i64, pc_left, [0]))
            left = fx.Int32(llvm.extractvalue(T.i32, pc_left, [1]))
            r_code = _rsrc(pc, fx.min(left, fx.Int32(nbytes)))
            words = fx.Int32(0)
            for i in range_constexpr(nbytes // 64 // THREADS):
                words = words | fx.Int32(
                    bo.buffer_load(
                        r_code,
                        (fx.Int32(tid) + i * THREADS) * 16,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
            _fresh(words)

        def n_sel(count=S):
            """This lane's local MFMA B column; inactive columns duplicate the last one."""
            return fx.min(lane % 16, count - 1)

        # ======================================= the tail (a ``tail`` build)
        def run_head():
            """Every CTA normalizes all rows into LDS (CTA b also writes columns
            HIDDEN / G * b .. of them to ``w_o``), runs 16-row tiles of this rank's
            LM-head rows and pushes their bf16 logits to every rank (two slots by
            launch parity), then turns its slice of the gathered logits into f32.
            A wave keeps HD_AHEAD weight units in flight across its tiles: each unit
            it multiplies is replaced by the next one of its tile sequence."""
            VR = vocab_rows
            VOCAB = VR * W
            N_VT = VR // HEAD_TILE
            HD_CPW = HIDDEN // 64 // WAVES
            HD_AHEAD = HD_CPW if S <= 4 else HD_CPW // 2
            HD_ROUNDS = (N_VT + G - 1) // G
            G_PAIRS = S * VOCAB // 2
            G_PER = (G_PAIRS + G - 1) // G
            NC = HIDDEN // G
            r_x, r_wh = _rsrc(h_in), _rsrc(w_r)
            logits_base = fx.Int64(SY["logits"]) + fx.Int64(peer_slot) * fx.Int64(
                SY["_logits_stride"]
            )

            def u_h(row0, c):
                # rows past the end load the last row; the caller drops them
                kc = wave * HD_CPW + c
                r = fx.min(row0 + lane % 16, VR - 1)
                wv = [
                    fx.Vector(
                        bo.buffer_load(
                            r_wh,
                            (r * HIDDEN + kc * 64 + sp * 32 + (lane // 16) * 8) // 2,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    for sp in range(2)
                ]
                b_word = n_sel() * (HIDDEN // 2) + kc * 32 + (lane // 16) * 4
                return ("bf16", wv, None, b_word)

            def u_next(tile, c):
                """Unit c of the tile after ``tile`` (past the last tile: the last row)."""
                nt = tile + G
                return u_h((nt < N_VT).select(nt * HEAD_TILE, fx.Int32(VR)), c)

            def tile_task(k, tile, window):
                stamp("lmhead", tile, 0)
                acc = [fx.Float32(0.0) for _ in range(4)]
                for c in range_constexpr(HD_CPW):
                    u = window.pop(0)
                    n = c + HD_AHEAD
                    if const_expr(n < HD_CPW):
                        window.append(u_h(tile * HEAD_TILE, n))
                    elif const_expr(k + 1 < HD_ROUNDS):
                        window.append(u_next(tile, n - HD_CPW))
                    acc = mma_units(acc, [u])
                reduce_rows(1, acc, emit_out(HEAD_TILE))
                gpu.barrier()
                if wave < W:
                    if lane < S * HEAD_TILE // 2:
                        si = lane // (HEAD_TILE // 2)
                        ri = (lane % (HEAD_TILE // 2)) * 2
                        put_bf(
                            peer_dst + logits_base,
                            si * VOCAB + rank * VR + tile * HEAD_TILE + ri,
                            [
                                lds_ld(outs, si * HEAD_TILE + ri),
                                lds_ld(outs, si * HEAD_TILE + ri + 1),
                            ],
                            CM_SYS,
                        )
                gpu.barrier()
                stamp("lmhead", tile, 4)
                if const_expr(k + 1 < HD_ROUNDS):
                    if tile + G < N_VT:
                        tile_task(k + 1, tile + G, window)

            def ld_x(sks):
                res = []
                for s, k in sks:
                    w = fx.Vector(
                        bo.buffer_load(
                            r_x, (s * HIDDEN + k) // 2, vec_width=2, dtype=T.i32
                        )
                    )
                    v = w.bitcast(fx.BFloat16).to(fx.Float32)
                    res.append([v[j] for j in range(4)])
                return res

            # the rows' loads go out first: the norm waits for them, not for the weights
            t0 = start("lmhead")
            x_ld = load_x_rmsnorm(ld_x, HIDDEN, g_in, S)
            window = [
                u_h((t0 < N_VT).select(t0 * HEAD_TILE, fx.Int32(VR)), c)
                for c in range(HD_AHEAD)
            ]
            stage_x_rmsnorm(ld_x, HIDDEN, g_in, loaded=x_ld, count=S)
            gpu.barrier()
            if tid < S * NC // 4:
                s = tid // (NC // 4)
                word = (s * HIDDEN + bid * NC) // 2 + (tid % (NC // 4)) * 2
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [lds_ld(xs, word + j).bitcast(fx.Int32) for j in range(2)],
                        fx.Int32,
                    ),
                    _rsrc(w_o),
                    word,
                )
            tile_task(0, t0, window)
            stamp("gather", bid, 0)
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
            stamp("gather", bid, 4)

        if const_expr(tail):
            run_head()
            guarded.__exit__(None, None, None)
            return

        if const_expr(x_pushed):
            xin = (
                sym
                + fx.Int64(SY["xin"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
            xin_dst = (
                peer_dst
                + fx.Int64(SY["xin"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
        if const_expr(tokens and eh_entry):  # a NextN layer's e rows
            ein = (
                sym
                + fx.Int64(SY["ein"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
            ein_dst = (
                peer_dst
                + fx.Int64(SY["ein"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )

        # ============================== 0. the step's embedding rows (layer 0)
        # The token's owner rank pushes its row's 32-column tile to every rank, one
        # wave a destination, as tagged packed bf16 pairs; the layer reads them back
        # from its own buffer where it would read x (a NextN layer: where its entry
        # reads e).
        if const_expr(tokens):
            e_ids = _rsrc(h_in)
            e_emb = _rsrc(index_arg(EH_ARGS[3]) if const_expr(eh_entry) else w_r)
            e_dst = ein_dst if const_expr(eh_entry) else xin_dst
            EMB_PAIRS = S * (ROW_TILE // 2)
            for e_t in range(start("embed"), N_ROW_TILES, G):
                e_t = fx.Int32(e_t)
                stamp("embed", e_t, 0)
                if wave < W:
                    for e_b in range_constexpr((EMB_PAIRS + 63) // 64):
                        e_pair = lane + e_b * 64
                        if e_pair < EMB_PAIRS:
                            e_s = e_pair // (ROW_TILE // 2)
                            e_col = e_t * ROW_TILE + (e_pair % (ROW_TILE // 2)) * 2
                            e_tok = fx.Int32(
                                bo.buffer_load(e_ids, e_s, vec_width=1, dtype=T.i32)
                            )
                            e_live = e_tok >= 0
                            e_own = e_live.select(e_tok // vocab_rows, fx.Int32(0))
                            if e_own == rank:
                                e_row = e_live.select(
                                    e_tok - rank * vocab_rows, fx.Int32(0)
                                )
                                e_word = fx.Int32(
                                    bo.buffer_load(
                                        e_emb,
                                        (e_row * HIDDEN + e_col) // 2,
                                        vec_width=1,
                                        dtype=T.i32,
                                    )
                                )
                                bo.buffer_store(
                                    fx.Vector.from_elements(
                                        [e_live.select(e_word, fx.Int32(0)), tag],
                                        fx.Int32,
                                    ),
                                    _rsrc(e_dst),
                                    e_s * HIDDEN + e_col,
                                    cache_modifier=CM_SYS,
                                )
                stamp("embed", e_t, 4)

        # ============================================ 0. the NextN entry (eh_entry)
        # Task p * EH_TILES + k: x columns 16 k .. of this rank's rows of eh_proj over
        # K part p of [enorm(e) | hnorm(h)] (EH_PARTS equal parts, the first half of
        # them e's). Each part stages its whole half's normalized rows; every part but
        # the first sends its f32 partial to task k's CTA, which adds them in part
        # order, rounds to bf16 and pushes the tile to every rank, one wave a
        # destination, as tagged packed bf16 pairs.
        if const_expr(eh_entry):
            EH_K = 2 * HIDDEN
            EH_PARTS = eh_parts(EH_TILES)
            EH_CPW = EH_K // 64 // EH_PARTS // WAVES
            EH_AHEAD = EH_CPW if S <= 4 else EH_CPW // 2
            r_weh = _rsrc(index_arg(EH_ARGS[0]))
            for et in range(start("eh"), EH_PARTS * EH_TILES, G):
                et = fx.Int32(et)
                stamp("eh", et, 0)
                e_part = et // EH_TILES
                e_half = e_part // (EH_PARTS // 2)
                e_tile = et % EH_TILES
                # the part's first 64-column chunk within its half
                e_kc0 = (e_part % (EH_PARTS // 2)) * (EH_CPW * WAVES)

                def u_eh(c):
                    kc = e_kc0 + wave * EH_CPW + c
                    r = e_tile * HEAD_TILE + lane % 16
                    wv = [
                        fx.Vector(
                            bo.buffer_load(
                                r_weh,
                                (
                                    r * EH_K
                                    + e_half * HIDDEN
                                    + kc * 64
                                    + sp * 32
                                    + (lane // 16) * 8
                                )
                                // 2,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                        for sp in range(2)
                    ]
                    b_word = n_sel() * (HIDDEN // 2) + kc * 32 + (lane // 16) * 4
                    return ("bf16", wv, None, b_word)

                pre = [u_eh(c) for c in range(EH_AHEAD)]
                if const_expr(tokens):  # e from the pushed embedding rows
                    r_src = _rsrc(cur_pos)
                else:
                    r_src = _rsrc((e_half == 0).select(h_in, cur_pos))
                # a pad row reads row 0 and takes zeros
                e_live = [row_writes_cache(s) for s in range(S)]

                def ld_ein(sks):
                    got = poll(
                        [(ein, (s * HIDDEN + k) // 2, 2) for s, k in sks], "one-as"
                    )
                    return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in got]

                def ld_eh(sks):
                    res = []
                    for s, k in sks:
                        live = e_live[s]
                        w = fx.Vector(
                            bo.buffer_load(
                                r_src,
                                (live.select(fx.Int32(s), fx.Int32(0)) * HIDDEN + k)
                                // 2,
                                vec_width=2,
                                dtype=T.i32,
                            )
                        )
                        v = w.bitcast(fx.BFloat16).to(fx.Float32)
                        res.append(
                            [live.select(v[j], fx.Float32(0.0)) for j in range(4)]
                        )
                    return res

                # NORM_ROWS rows at a time
                for e_first in range_constexpr(0, S, NORM_ROWS):
                    e_count = min(NORM_ROWS, S - e_first)

                    def rows_of(ld, e_first=e_first):
                        return lambda sks: ld([(e_first + s, k) for s, k in sks])

                    if const_expr(tokens):
                        if e_half == 0:
                            stage_x_rmsnorm(
                                rows_of(ld_ein),
                                HIDDEN,
                                index_arg(EH_ARGS[1]),
                                count=e_count,
                                first=e_first,
                            )
                        else:
                            stage_x_rmsnorm(
                                rows_of(ld_eh),
                                HIDDEN,
                                index_arg(EH_ARGS[2]),
                                count=e_count,
                                first=e_first,
                            )
                    else:
                        e_gamma = (e_half == 0).select(
                            index_arg(EH_ARGS[1]), index_arg(EH_ARGS[2])
                        )
                        stage_x_rmsnorm(
                            rows_of(ld_eh),
                            HIDDEN,
                            e_gamma,
                            count=e_count,
                            first=e_first,
                        )
                gpu.barrier()
                stamp("eh", et, 2)
                acc = run_units(u_eh, EH_CPW, EH_AHEAD, pre)
                reduce_rows(1, acc, emit_out(HEAD_TILE))
                stamp("eh", et, 3)
                gpu.barrier()
                if e_part > 0:
                    if tid < S * HEAD_TILE:
                        put(
                            mb("ehp"),
                            (et - EH_TILES) * S * HEAD_TILE + tid,
                            lds_ld(outs, tid),
                        )
                else:
                    if tid < S * HEAD_TILE:
                        parts = poll(
                            [
                                (
                                    mb("ehp"),
                                    (p * EH_TILES + e_tile) * S * HEAD_TILE + tid,
                                    1,
                                )
                                for p in range(EH_PARTS - 1)
                            ]
                        )
                        v = lds_ld(outs, tid)
                        for hp in parts:
                            v = v + hp[0].bitcast(fx.Float32)
                        lds_st(outs, tid, v)
                    gpu.barrier()
                    if wave < W:
                        if lane < S * HEAD_TILE // 2:
                            si = lane // (HEAD_TILE // 2)
                            ri = (lane % (HEAD_TILE // 2)) * 2
                            put_bf(
                                xin_dst,
                                si * HIDDEN
                                + rank * (EH_TILES * HEAD_TILE)
                                + e_tile * HEAD_TILE
                                + ri,
                                [
                                    lds_ld(outs, si * HEAD_TILE + ri),
                                    lds_ld(outs, si * HEAD_TILE + ri + 1),
                                ],
                                CM_SYS,
                            )
                    gpu.barrier()
                stamp("eh", et, 4)

        # ================================================= 1. q_a / kv_a GEMV
        # 1 row group x 96 chunks: 8 waves split K, 12 chunks each (all prefetched).
        # With the indexer, the index K and weights projections are further 16-row tiles
        # of this stage, each alone on one of the grid's last CTAs: the first N_INDEX_K
        # W_k's tiles, the last N_INDEX_W the weights projection's.
        assert attention_weight is AttentionWeight.FP8_BLOCK128, (
            "the K-split front reads FP8 block-128 q_a / kv_a weights"
        )
        (x_out, positions, slot_mapping, sparse_kv_indptr, kv_cache, pe_cache, indices, rope_cos, rope_sin, g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o, s_o, w_r, bias, w_ug, s_ug, w_dn, s_dn, scratch, sym, timeline_buf, block_table, req_ids, out_indices, ix_wk, ix_sk, ix_ww, ix_wq, ix_sq, ix_gk, ix_bk) = reload_args()  # fmt: skip
        if const_expr(RELOAD_ARGS and x_pushed):
            xin = (
                sym
                + fx.Int64(SY["xin"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
        r_wqa, r_sqa = _rsrc(w_qkv_a), _rsrc(s_qkv_a)
        QA_NKC = HIDDEN // 64
        QA_UNITS = QA_NKC // (attention_k_chunks_per_unit * WAVES)
        IW_CPW = QA_NKC // WAVES
        N_INDEX_K = INDEX_DIM // QKV_A_TILE if const_expr(with_indexer) else 0
        N_INDEX_W = INDEX_HEADS // QKV_A_TILE if const_expr(with_indexer) else 0
        if const_expr(with_indexer):
            r_wik, r_sik = _rsrc(index_arg(0)), _rsrc(index_arg(1))
            r_wiw = _rsrc(index_arg(2))

        def ld_h(sample_base):
            """Loader of input rows sample_base + s, elements k .. k + 3, for (s, k) pairs."""

            def f(sks):
                if const_expr(x_pushed):  # the rows pushed to every rank
                    got = poll(
                        [
                            (xin, ((sample_base + s) * HIDDEN + k) // 2, 2)
                            for s, k in sks
                        ],
                        "one-as",
                    )
                    return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in got]
                res = []
                for s, k in sks:
                    w = fx.Vector(
                        bo.buffer_load(
                            r_h,
                            ((sample_base + s) * HIDDEN + k) // 2,
                            vec_width=2,
                            dtype=T.i32,
                        )
                    )
                    v = w.bitcast(fx.BFloat16).to(fx.Float32)
                    res.append([v[j] for j in range(4)])
                return res

            return f

        def front_tile(t, make_unit, n_units, publish, ptpc=False, ahead=None):
            """One 16-row tile of a projection of the RMS-normalized input rows: the
            first NORM_ROWS samples' input loads, the weight units (issued once and kept
            in registers for every group of SAMPLE_TILE samples), then per group the
            RMSNorm into LDS (NORM_ROWS samples at a time), the MFMAs, the row reduction
            and ``publish(sample_base, group_count, acc, scales)``. A later group with
            fewer samples reuses the first group's units: their extra MFMA columns read
            stale LDS and are not emitted. ``ahead``: only the first ``ahead`` units go
            out before the staging, the rest after it (one group only); the sum keeps
            the units' order."""
            ahead = n_units if ahead is None else ahead
            assert ahead == n_units or S <= SAMPLE_TILE

            def stage(sample_base, count, loaded, first):
                ld = ld_h(sample_base)
                if const_expr(ptpc):
                    return stage_x_rmsnorm_ptpc(
                        ld, HIDDEN, g_in, loaded=loaded, count=count, first=first
                    )
                stage_x_rmsnorm(
                    ld, HIDDEN, g_in, loaded=loaded, count=count, first=first
                )
                return []

            pre = None
            for sample_base in range_constexpr(0, S, SAMPLE_TILE):
                group_count = min(SAMPLE_TILE, S - sample_base)
                scales = []
                # a NextN layer polls the pushed rows two at a time: their tags then fit
                # in registers next to the weights in flight
                NR = 2 if const_expr(eh_entry and S <= 4) else NORM_ROWS
                for first in range_constexpr(0, group_count, NR):
                    count = min(NR, group_count - first)
                    if const_expr(x_pushed):  # the weights go out before the rows' wait
                        if const_expr(pre is None):
                            pre = [make_unit(c, group_count) for c in range(ahead)]
                        scales += stage(sample_base + first, count, None, first)
                    else:
                        h_ld = load_x_rmsnorm(
                            ld_h(sample_base + first), HIDDEN, g_in, count
                        )
                        if const_expr(pre is None):
                            pre = [make_unit(c, group_count) for c in range(ahead)]
                        scales += stage(sample_base + first, count, h_ld, first)
                gpu.barrier()
                stamp("qkv_a", t, 2)
                if const_expr(ahead == n_units):
                    acc = mma_units([fx.Float32(0.0) for _ in range(4)], pre)
                else:
                    acc = run_units(
                        lambda c, g=group_count: make_unit(c, g), n_units, ahead, pre
                    )
                publish(sample_base, group_count, acc, scales)

        def front_part(t, tiles, k0, kn, boxes):
            """The K range [k0, k0 + kn) of q_a / kv_a tiles ``tiles`` against bf16(x *
            gamma) of the input rows (not normalized: the consumers scale the summed
            parts by the row's rstd): every tile's weight units issued after the first
            NORM_ROWS samples' input loads, the scaled inputs into LDS, then per tile the
            MFMAs, the row reduction and its rows' partial sums into ``boxes`` (the q_a
            and kv_a mailboxes of this K range). ``tiles``: (tile, published) pairs."""
            per = kn // (4 * THREADS)
            units = kn // 128 // WAVES

            def load(sample_base, count):
                ks = [k0 + (tid + i * THREADS) * 4 for i in range(per)]
                gs = []
                for k in ks:
                    g = fx.Vector(
                        bo.buffer_load(_rsrc(g_in), k // 2, vec_width=2, dtype=T.i32)
                    )
                    g = g.bitcast(fx.BFloat16).to(fx.Float32)
                    gs.append([g[j] for j in range(4)])
                vals = ld_h(sample_base)([(s, k) for s in range(count) for k in ks])
                return ks, gs, vals

            def unit(tile, c, group_count):
                kc = k0 // 64 + (wave * units + c) * 2
                return unit_fp8x2(
                    r_wqa,
                    r_sqa,
                    tile,
                    kc,
                    QA_NKC,
                    HIDDEN,
                    (n_sel(group_count) * kn + kc * 64 - k0) // 2,
                    shuffled=fp8_shuffled,
                    cm=AT_CM,
                )

            pre = None
            for sample_base in range_constexpr(0, S, SAMPLE_TILE):
                group_count = min(SAMPLE_TILE, S - sample_base)
                for first in range_constexpr(0, group_count, NORM_ROWS):
                    count = min(NORM_ROWS, group_count - first)
                    if const_expr(
                        x_pushed and pre is None
                    ):  # weights before the rows' wait
                        pre = [
                            [unit(tile, c, group_count) for c in range(units)]
                            for tile, _ in tiles
                        ]
                    ks, gs, vals = load(sample_base + first, count)
                    if const_expr(pre is None):
                        pre = [
                            [unit(tile, c, group_count) for c in range(units)]
                            for tile, _ in tiles
                        ]
                    for s in range_constexpr(count):
                        for i in range_constexpr(per):
                            a = vals[s * per + i]
                            for j in range_constexpr(2):
                                lds_st(
                                    xs,
                                    ((first + s) * kn + ks[i] - k0) // 2 + j,
                                    bf16_pair(
                                        a[2 * j] * gs[i][2 * j],
                                        a[2 * j + 1] * gs[i][2 * j + 1],
                                    ),
                                )
                gpu.barrier()
                stamp("qkv_a", t, 2)
                for (tile, published), tile_units in zip(tiles, pre):
                    acc = mma_units([fx.Float32(0.0) for _ in range(4)], tile_units)
                    reduce_rows(1, acc, emit_out(QKV_A_TILE), group_count)
                    stamp("qkv_a", t, 3)
                    gpu.barrier()
                    live = tid < group_count * QKV_A_TILE
                    if const_expr(published is not True):
                        live = live & published
                    if live:
                        s = sample_base + tid // QKV_A_TILE
                        row = tile * QKV_A_TILE + tid % QKV_A_TILE
                        v = lds_ld(outs, tid)
                        if row < Q_LORA:
                            put(mb(boxes[0]), s * Q_LORA + row, v)
                        else:
                            put(mb(boxes[1]), s * (KV_LORA + PE_DIM) + row - Q_LORA, v)

        # with the indexer, the index tiles on the last CTAs, alone, and the K rests two
        # a CTA on the CTAs before them
        N_REST_CTAS = FRONT_SMALL_CTAS - N_INDEX_K - N_INDEX_W
        assert 2 * N_REST_CTAS >= N_QKV_A

        def rests(t, j, n_rest_ctas):
            second = j + n_rest_ctas
            front_part(
                t,
                [(j, True), (fx.min(second, N_QKV_A - 1), second < N_QKV_A)],
                FRONT_K_BIG,
                HIDDEN - FRONT_K_BIG,
                ("q_a2", "kv_a2"),
            )

        for t in range(start("qkv_a"), G, G):
            t = fx.Int32(t)
            stamp("qkv_a", t, 0)
            j = t - N_QKV_A
            if t < N_QKV_A:
                front_part(t, [(t, True)], 0, FRONT_K_BIG, ("q_a", "kv_a"))
            else:
                if const_expr(with_indexer):
                    if j < N_REST_CTAS:
                        rests(t, j, N_REST_CTAS)
                else:
                    rests(t, j, FRONT_SMALL_CTAS)
            if const_expr(with_indexer):
                tk = j - N_REST_CTAS
                tw = tk - N_INDEX_K

                def put_index(name, rows, tile):
                    def f(sample_base, group_count, acc, _):
                        reduce_rows(1, acc, emit_out(INDEX_TILE), group_count)
                        stamp("qkv_a", t, 3)
                        gpu.barrier()
                        if tid < group_count * INDEX_TILE:
                            s = sample_base + tid // INDEX_TILE
                            row = tile * INDEX_TILE + tid % INDEX_TILE
                            put(mb(name), s * rows + row, lds_ld(outs, tid))

                    return f

                def u_index_k(c, group_count):
                    if const_expr(index_k_bf16):
                        kc = wave * IW_CPW + c
                        return unit_bf16(
                            r_wik,
                            tk,
                            kc,
                            QA_NKC,
                            (n_sel(group_count) * HIDDEN + kc * 64) // 2,
                        )
                    kc = (wave * QA_UNITS + c) * 2
                    return unit_fp8x2(
                        r_wik,
                        r_sik,
                        tk,
                        kc,
                        QA_NKC,
                        HIDDEN,
                        (n_sel(group_count) * HIDDEN + kc * 64) // 2,
                        shuffled=fp8_shuffled,
                    )

                def u_index_w(c, group_count):
                    kc = wave * IW_CPW + c
                    return unit_bf16(
                        r_wiw,
                        tw,
                        kc,
                        QA_NKC,
                        (n_sel(group_count) * HIDDEN + kc * 64) // 2,
                    )

                # the index tiles: above 4 rows the K tiles issue half their weights
                # and the weights tiles a quarter before the rows' staging, the rest
                # after, so they and the 8 rows' inputs fit in registers
                if (tk >= 0) & (tk < N_INDEX_K):
                    n_k = IW_CPW if const_expr(index_k_bf16) else QA_UNITS
                    front_tile(
                        t,
                        u_index_k,
                        n_k,
                        put_index("index_k", INDEX_DIM, tk),
                        ahead=n_k // 2 if const_expr(S > 4) else n_k,
                    )
                if (tw >= 0) & (tw < N_INDEX_W):
                    front_tile(
                        t,
                        u_index_w,
                        IW_CPW,
                        put_index("index_w", INDEX_HEADS, tw),
                        ahead=IW_CPW // 4 if const_expr(S > 4) else IW_CPW,
                    )
            stamp("qkv_a", t, 4)

        def x_rstd(s):
            """The rstd of input row s (sum of squares in stage_x_rmsnorm's order)."""
            per = HIDDEN // (4 * THREADS)
            vals = ld_h(s)([(0, (tid + i * THREADS) * 4) for i in range(per)])
            ss = fx.Float32(0.0)
            for v in vals:
                for j in range_constexpr(4):
                    ss = ss + v[j] * v[j]
            return _rsq(block_sums([ss])[0] * (1.0 / HIDDEN) + EPS)

        def ld_qa(sks, rstd):
            """q_a elements k .. k + 3 of rows s: its K parts' sum times the row's rstd."""
            v = get2_many(
                [
                    (mb(box), s * Q_LORA + k + j)
                    for s, k in sks
                    for box in ("q_a", "q_a2")
                    for j in (0, 2)
                ]
            )
            out = []
            for i in range_constexpr(len(sks)):
                a = list(v[4 * i]) + list(v[4 * i + 1])
                b = list(v[4 * i + 2]) + list(v[4 * i + 3])
                out.append([(a[e] + b[e]) * rstd for e in range(4)])
            return out

        # ===================== 2. one q_a RMSNorm CTA per sample, shared downstream
        (x_out, positions, slot_mapping, sparse_kv_indptr, kv_cache, pe_cache, indices, rope_cos, rope_sin, g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o, s_o, w_r, bias, w_ug, s_ug, w_dn, s_dn, scratch, sym, timeline_buf, block_table, req_ids, out_indices, ix_wk, ix_sk, ix_ww, ix_wq, ix_sq, ix_gk, ix_bk) = reload_args()  # fmt: skip
        if const_expr(RELOAD_ARGS and x_pushed):
            xin = (
                sym
                + fx.Int64(SY["xin"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
        # a request's consecutive rows (up to SCORE_ROWS) score its keys together
        SCORE_ROWS = 1 if S == 1 else 2 if S == 2 else 4

        def score_groups(bounds, page_of, slot_of):
            """Bit s of the first value: row s leads a group of rows scored together,
            up to SCORE_ROWS consecutive rows of one request; 3 bits a row of the
            second: the rows of the group it leads, and of the third: its request's
            first row in the launch. A row is of the previous row's request when its
            page table maps that row's position to the slot that row writes (a page
            being written is its request's alone); ``page_of(r, key)``: the block of
            row r's page table holding key ``key``. From 4 rows a row joins only at the
            next position, so a run spans at most two of each row's slices."""
            same = [fx.Int32(0)]
            for r in range_constexpr(1, S):
                prev = fx.max(bounds[r - 1] - 1, fx.Int32(0))
                at = page_of(r, prev) * index_block_size + (prev % index_block_size)
                same.append(
                    ((bounds[r - 1] > 0) & (at == slot_of(r - 1))).select(
                        fx.Int32(1), fx.Int32(0)
                    )
                )
            leads, sizes, firsts = fx.Int32(0), fx.Int64(0), fx.Int64(0)
            lead, run, first = fx.Int32(0), fx.Int32(0), fx.Int32(0)
            for r in range_constexpr(S):
                live = bounds[r] > 0
                first = (live & (same[r] != 0)).select(first, fx.Int32(r))
                joins = live & (same[r] != 0) & (run > 0) & (run < SCORE_ROWS)
                if const_expr(S >= 4 and r > 0):
                    joins = joins & (bounds[r] == bounds[r - 1] + 1)
                starts = live & (joins.select(fx.Int32(1), fx.Int32(0)) == 0)
                leads = leads | starts.select(fx.Int32(1 << r), fx.Int32(0))
                firsts = firsts | starts.select(fx.Int64(first) << (3 * r), fx.Int64(0))
                lead = starts.select(fx.Int32(r), lead)
                run = live.select(joins.select(run + 1, fx.Int32(1)), fx.Int32(0))
                shift = fx.Int64(lead) * 3
                sizes = live.select(
                    (sizes & ~(fx.Int64(7) << shift)) | (fx.Int64(run) << shift),
                    sizes,
                )
            return _uniform(leads), sizes, firsts

        def plan_layout():
            """The scoring layout into the plan words, from scalar loads of each row's
            cache slot, position and page-table row."""
            p_slot = [sld_i32(slot_mapping + fx.Int64(8 * r)) for r in range(S)]
            p_pos = [sld_i32(positions + fx.Int64(8 * r)) for r in range(S)]
            p_req = [sld_i32(req_ids + fx.Int64(4 * r)) for r in range(S)]
            # index_bound's value
            p_bound = [
                (p_slot[r] >= 0).select(p_pos[r] + 1, fx.Int32(0)) for r in range(S)
            ]
            p_page = [None] + [
                sld_i32(
                    block_table
                    + fx.Int64(
                        p_req[r] * bt_stride
                        + fx.max(p_bound[r - 1] - 1, fx.Int32(0)) // index_block_size
                    )
                    * 4
                )
                for r in range(1, S)
            ]
            p_leads, p_sizes, p_firsts = score_groups(
                p_bound, lambda r, key: p_page[r], lambda r: p_slot[r]
            )
            if tid == 0:
                lds_st(plan, 0, p_leads)
                lds_st(plan, 1, fx.Int32(p_sizes))
                lds_st(plan, 2, fx.Int32(p_sizes >> 32))
                lds_st(plan, 3, fx.Int32(p_firsts))
                lds_st(plan, 4, fx.Int32(p_firsts >> 32))
                for r in range_constexpr(S):
                    lds_st(plan, 5 + r, p_bound[r])
                    lds_st(plan, 5 + S + r, p_req[r])
            gpu.barrier()

        if const_expr(PLANNED):
            # on the CTAs that score (past the q_norm ones), once their front tile is
            # done
            if bid >= S:
                plan_layout()
        for s_norm in range(start("q_norm"), S, G):
            s_norm = fx.Int32(s_norm)
            stamp("q_norm", s_norm, 0)
            rstd_x = x_rstd(s_norm)
            hint_wait(
                Q_LORA // QKV_A_TILE,
                lambda k: (
                    mb("q_a"),
                    s_norm * Q_LORA + k * QKV_A_TILE + QKV_A_TILE - 1,
                ),
                mark=("q_norm", s_norm),
            )

            def ld_qa_one(sks):
                return ld_qa([(s_norm, k) for _, k in sks], rstd_x)

            stage_x_rmsnorm(ld_qa_one, Q_LORA, g_q, count=1)
            stamp("q_norm", s_norm, 2)
            gpu.barrier()
            k = tid * 4
            w0 = lds_ld(xs, k // 2)
            w1 = lds_ld(xs, k // 2 + 1)
            a0, a1 = bf2_f32(w0.bitcast(fx.Int32))
            a2, a3 = bf2_f32(w1.bitcast(fx.Int32))
            put_bf(mb("q_an"), s_norm * Q_LORA + k, [a0, a1, a2, a3])
            stamp("q_norm", s_norm, 4)

        (x_out, positions, slot_mapping, sparse_kv_indptr, kv_cache, pe_cache, indices, rope_cos, rope_sin, g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o, s_o, w_r, bias, w_ug, s_ug, w_dn, s_dn, scratch, sym, timeline_buf, block_table, req_ids, out_indices, ix_wk, ix_sk, ix_ww, ix_wq, ix_sq, ix_gk, ix_bk) = reload_args()  # fmt: skip
        if const_expr(RELOAD_ARGS and x_pushed):
            xin = (
                sym
                + fx.Int64(SY["xin"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
        if const_expr(with_indexer):
            # The 128 CTAs without q_b work produce the other 128 index-query tiles
            # concurrently, the q_norm CTAs theirs right after q_norm (ahead of the
            # row's index key and KV rows). They reload q_a, but remove one full GEMV
            # from the q_b CTAs' serialized critical path.
            N_INDEX_Q_EXTRA = INDEX_Q_ROWS // INDEX_TILE - N_QB
            IQ_NKC = Q_LORA // 64
            IQ_UNITS = IQ_NKC // (2 * WAVES)
            r_wiq, r_siq = _rsrc(index_arg(3)), _rsrc(index_arg(4))

            def u_index_q(iq_t, units, c):
                """K unit c of index-query tile iq_t, each wave taking ``units`` units
                of two 64-k chunks of the normalized q_lora rows."""
                kc = (wave * units + c) * 2
                return unit_fp8x2(
                    r_wiq,
                    r_siq,
                    iq_t,
                    kc,
                    IQ_NKC,
                    Q_LORA,
                    (n_sel() * Q_LORA + kc * 64) // 2,
                    shuffled=fp8_shuffled,
                )

            def index_q_extra(tt):
                iq_t = N_QB + tt
                stamp("index_q", iq_t, 0)
                pre = [u_index_q(iq_t, IQ_UNITS, c) for c in range(IQ_UNITS)]
                hint_wait(
                    S,
                    lambda s: (mb("q_an"), (s * Q_LORA + Q_LORA - 2) // 2),
                    mark=("index_q", iq_t),
                )
                stage_x_pairs("q_an", S * Q_LORA, lambda k: k)
                stamp("index_q", iq_t, 2)
                gpu.barrier()
                iq_acc = mma_units([fx.Float32(0.0) for _ in range(4)], pre)
                reduce_rows(1, iq_acc, emit_out(INDEX_TILE))
                stamp("index_q", iq_t, 3)
                gpu.barrier()
                if tid < S * INDEX_TILE // 4:
                    s = tid // (INDEX_TILE // 4)
                    r = (tid % (INDEX_TILE // 4)) * 4
                    put_bf(
                        mb("index_q"),
                        s * INDEX_Q_ROWS + iq_t * INDEX_TILE + r,
                        [lds_ld(outs, s * INDEX_TILE + r + j) for j in range(4)],
                    )
                stamp("index_q", iq_t, 4)

            for tt in range(start("index_q"), N_INDEX_Q_EXTRA, G):
                index_q_extra(fx.Int32(tt))

            # ========= each row's index query, rotated and Hadamard-transformed once:
            # 8 heads a task, a wave a head, lane l holding values 2l and 2l + 1
            def index_qh_tasks():
                for t in range(start("index_qh"), S * INDEX_HEADS // 8, G):
                    t = fx.Int32(t)
                    s = t // (INDEX_HEADS // 8)
                    h = (t % (INDEX_HEADS // 8)) * 8 + fx.Int32(wave)
                    stamp("index_qh", t, 0)
                    kq = fx.Int32(lane) * 2
                    pos_q = fx.max(index_bound(s) - 1, fx.Int32(0))
                    at = pos_q * (PE_DIM // 2) + fx.min(kq, PE_DIM - 2) // 2
                    c = ld_bf16(_rsrc(rope_cos), at)
                    sn = ld_bf16(_rsrc(rope_sin), at)
                    q_pair = (
                        s * (INDEX_Q_ROWS // 2) + h * (INDEX_DIM // 2) + fx.Int32(lane)
                    )
                    q0, q1 = bf2_f32(poll([(mb("index_q"), q_pair, 1)])[0][0])
                    stamp("index_qh", t, 2)
                    if kq < PE_DIM:
                        q0, q1 = q0 * c - q1 * sn, q0 * sn + q1 * c
                    q0, q1 = fwht128(q0, q1)
                    put(mb("index_qh"), q_pair, bf16_pair(q0, q1))
                    stamp("index_qh", t, 4)

            # At TP4 q_b runs on every CTA and its tasks, later in a CTA's order,
            # produce every index-query tile, so these tasks run after q_b there
            assert N_INDEX_Q_EXTRA >= 0
            if const_expr(N_INDEX_Q_EXTRA > 0):
                index_qh_tasks()

        if const_expr(with_indexer):
            # ========= 3a. each row's index key: LayerNorm, RoPE, Hadamard -> index cache
            # one task a row, on the row's q_norm CTA after its index-query tile
            for s in range(start("index_key"), S, G):
                s = fx.Int32(s)
                stamp("index_key", s, 0)
                # the LayerNorm weight and bias and the RoPE factors ahead of the wait
                r_gik, r_bik = _rsrc(index_arg(5)), _rsrc(index_arg(6))
                r_index_cache = _rsrc(indices)
                i0 = fx.min(tid, INDEX_DIM // 2 - 1) * 2
                ik_g = [ld_f32(r_gik, i0 + j) for j in range(2)]
                ik_b = [ld_f32(r_bik, i0 + j) for j in range(2)]
                tpe = tid % (PE_DIM // 2)
                c = ld_bf16(_rsrc(rope_cos), row_position(s) * (PE_DIM // 2) + tpe)
                sn = ld_bf16(_rsrc(rope_sin), row_position(s) * (PE_DIM // 2) + tpe)
                slot = row_slot(s)
                active = row_writes_cache(s)
                stamp("index_key", s, 1)
                # Index keys use LayerNorm (not RMSNorm), interleaved RoPE on the
                # first 64 dimensions, then the 128-point Hadamard that SGLang's
                # gfx950 indexer applies to the keys it caches and to its queries.
                # the row's key in one poll
                live = tid < INDEX_DIM
                ik_one = fx.min(tid, INDEX_DIM - 1)
                ik_two = fx.min(tid, INDEX_DIM // 2 - 1) * 2
                got = poll(
                    [
                        (mb("index_k"), s * INDEX_DIM + ik_one, 1),
                        (mb("index_k"), s * INDEX_DIM + ik_two, 2),
                    ]
                )
                stamp("index_key", s, 2)
                ik = got[0][0].bitcast(fx.Float32)
                k0, k1 = got[1][0].bitcast(fx.Float32), got[1][1].bitcast(fx.Float32)
                mean = block_sums([live.select(ik, fx.Float32(0.0))])[0] * (
                    1.0 / INDEX_DIM
                )
                centered = live.select(ik - mean, fx.Float32(0.0))
                rstd = _rsq(
                    block_sums([centered * centered])[0] * (1.0 / INDEX_DIM) + 1.0e-6
                )
                if tid < INDEX_DIM // 2:
                    v0 = (k0 - mean) * rstd * ik_g[0] + ik_b[0]
                    v1 = (k1 - mean) * rstd * ik_g[1] + ik_b[1]
                    if tid < PE_DIM // 2:
                        v0, v1 = v0 * c - v1 * sn, v0 * sn + v1 * c
                    v0, v1 = fwht128(v0, v1)
                    # the index cache's ue8m0 quant: scale = 2^ceil(log2(max(amax, 1e-4) / 448))
                    amax = wave_max(fx.max(fmath.absf(v0), fmath.absf(v1)))
                    bits = div_rn(
                        fx.max(amax, fx.Float32(1.0e-4)),
                        fx.Float32(FP8_MAX),
                        fx.Float32(1.0 / FP8_MAX),
                    ).bitcast(fx.Int32)
                    exponent = (bits >> 23) & fx.Int32(0xFF)
                    exponent = exponent + ((bits & fx.Int32(0x7FFFFF)) != 0).select(
                        fx.Int32(1), fx.Int32(0)
                    )
                    k_scale = (exponent << 23).bitcast(fx.Float32)
                    k_inv = ((fx.Int32(254) - exponent) << 23).bitcast(fx.Float32)
                    q0, q1 = v0 * k_inv, v1 * k_inv
                    word = fp8_pack4(q0, q1, _xshfl(q0, 1), _xshfl(q1, 1))
                    block = slot // index_block_size
                    off = slot % index_block_size
                    if active & (tid % 2 == 0):
                        bo.buffer_store(
                            word, r_index_cache, index_value_word(block, off, i0)
                        )
                    if tid % 2 == 0:
                        put(mb("index_k8"), s * (INDEX_DIM // 4 + 1) + tid // 2, word)
                    if tid == 0:
                        put(
                            mb("index_k8"),
                            s * (INDEX_DIM // 4 + 1) + INDEX_DIM // 4,
                            k_scale,
                        )
                    if active & (tid == 0):
                        bo.buffer_store(
                            k_scale, r_index_cache, index_scale_word(block, off)
                        )
                gpu.barrier()
                stamp("index_key", s, 4)

        # ================ 3. KV RMSNorm + k_pe RoPE -> cache (+ this launch's rows)
        # one task a row
        (x_out, positions, slot_mapping, sparse_kv_indptr, kv_cache, pe_cache, indices, rope_cos, rope_sin, g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o, s_o, w_r, bias, w_ug, s_ug, w_dn, s_dn, scratch, sym, timeline_buf, block_table, req_ids, out_indices, ix_wk, ix_sk, ix_ww, ix_wq, ix_sq, ix_gk, ix_bk) = reload_args()  # fmt: skip
        if const_expr(RELOAD_ARGS and x_pushed):
            xin = (
                sym
                + fx.Int64(SY["xin"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
        for s in range(start("cache"), S, G):
            s = fx.Int32(s)
            stamp("cache", s, 0)
            r_kv = _rsrc(kv_cache)
            r_pe = _rsrc(pe_cache)
            # gamma and the RoPE factors are issued ahead of the wait
            g = ld_bf16(_rsrc(g_kv), tid)
            tpe = tid % (PE_DIM // 2)
            c = ld_bf16(_rsrc(rope_cos), row_position(s) * (PE_DIM // 2) + tpe)
            sn = ld_bf16(_rsrc(rope_sin), row_position(s) * (PE_DIM // 2) + tpe)
            rstd_x = x_rstd(s)
            hint_wait(
                (KV_LORA + PE_DIM) // QKV_A_TILE,
                lambda k: (
                    mb("kv_a"),
                    s * (KV_LORA + PE_DIM) + k * QKV_A_TILE + QKV_A_TILE - 1,
                ),
                mark=("cache", s),
            )
            # the row's kv latent and k_pe pair, both K parts, in one poll
            got = poll(
                [
                    (mb(box), s * (KV_LORA + PE_DIM) + tid, 1)
                    for box in ("kv_a", "kv_a2")
                ]
                + [
                    (
                        mb(box),
                        s * (KV_LORA + PE_DIM) + KV_LORA + (tid % (PE_DIM // 2)) * 2,
                        2,
                    )
                    for box in ("kv_a", "kv_a2")
                ]
            )
            v = (got[0][0].bitcast(fx.Float32) + got[1][0].bitcast(fx.Float32)) * rstd_x
            x0 = (
                got[2][0].bitcast(fx.Float32) + got[3][0].bitcast(fx.Float32)
            ) * rstd_x
            x1 = (
                got[2][1].bitcast(fx.Float32) + got[3][1].bitcast(fx.Float32)
            ) * rstd_x
            stamp("cache", s, 2)
            ssq = block_sums([v * v])[0]
            pos = row_position(s)
            slot = row_slot(s)
            active = row_writes_cache(s)
            value = v * _rsq(ssq * (1.0 / KV_LORA) + EPS) * g
            if const_expr(cache_fp8):
                value = bf16_round(value)
                kvn, _ = _fp8_roundtrip(value, value)
            else:
                kvn = bf16_round(value)
            if const_expr(use_atom_kv_cache):
                if active:
                    if const_expr(cache_fp8):
                        pair = fx.Int32(
                            rocdl.cvt_pk_fp8_f32(
                                T.i32, kvn, _xshfl(kvn, 1), fx.Int32(0), False
                            )
                        ) & fx.Int32(0xFFFF)
                        upper_pair = bpermute_i32(
                            fp8_kv_upper_pair_lane(lane) * 4, pair
                        )
                        if lane % 4 == 0:
                            bo.buffer_store(
                                pair | (upper_pair << 16),
                                r_kv,
                                slot * (QK_DIM // 4) + tid // 4,
                            )
                    else:
                        bo.buffer_store(kvn.to(fx.BFloat16), r_kv, slot * QK_DIM + tid)
            else:
                bo.buffer_store(kvn.to(fx.BFloat16), r_kv, pos * KV_LORA + tid)
            put(mb("kvnew"), s * KV_LORA + tid, kvn)
            if tid < PE_DIM // 2:
                p0, p1 = x0 * c - x1 * sn, x0 * sn + x1 * c
                if const_expr(cache_fp8):
                    p0, p1 = bf16_round(p0), bf16_round(p1)
                    p0, p1 = _fp8_roundtrip(p0, p1)
                else:
                    p0, p1 = bf16_round(p0), bf16_round(p1)
                if const_expr(use_atom_kv_cache):
                    if active:
                        if const_expr(cache_fp8):
                            pair = fx.Int32(
                                rocdl.cvt_pk_fp8_f32(T.i32, p0, p1, fx.Int32(0), False)
                            ) & fx.Int32(0xFFFF)
                            upper_pair = bpermute_i32(
                                fp8_pe_upper_pair_lane(lane) * 4, pair
                            )
                            if lane % 2 == 0:
                                bo.buffer_store(
                                    pair | (upper_pair << 16),
                                    r_pe,
                                    slot * (QK_DIM // 4) + KV_LORA // 4 + tid // 2,
                                )
                        else:
                            pe_offset = slot * QK_DIM + KV_LORA + tid * 2
                            bo.buffer_store(p0.to(fx.BFloat16), r_pe, pe_offset)
                            bo.buffer_store(p1.to(fx.BFloat16), r_pe, pe_offset + 1)
                else:
                    pe_offset = pos * PE_DIM + tid * 2
                    bo.buffer_store(p0.to(fx.BFloat16), r_pe, pe_offset)
                    bo.buffer_store(p1.to(fx.BFloat16), r_pe, pe_offset + 1)
                put2(mb("penew"), s * PE_DIM + tid * 2, p0, p1)
            stamp("cache", s, 4)

        # ================ split's KV cache rows (section 5). Above 4 rows, in launches
        # without the indexer (the selection is the previous launch's), most split
        # tiles share a CTA with a q_norm, q_b or uk task: every tile's rows go out
        # here, after the CTA's q_norm and cache tasks and ahead of its q_b or uk task
        KPW = SPLIT_KEYS // WAVES
        N_SPLIT_TILES = S * (H // WAVES) * N_SPLIT
        SPLIT_EARLY = S > 4 and not with_indexer and N_SPLIT_TILES <= G

        def split_bounds(s, indptr=None):
            """(first index, keys, sparse) of sample s's selection."""
            if const_expr(use_atom_kv_cache):
                rs = _rsrc(sparse_kv_indptr) if indptr is None else indptr
                begin = fx.Int32(bo.buffer_load(rs, s, vec_width=1, dtype=T.i32))
                end = fx.Int32(bo.buffer_load(rs, s + 1, vec_width=1, dtype=T.i32))
                return begin, end - begin, end > begin
            kv_len = pos0 + s + 1
            sparse = kv_len > topk
            return s * topk, sparse.select(fx.Int32(topk), kv_len), sparse

        def split_kv_loads(krows, kv_rs, pe_rs):
            """Issue the KV latent and k_pe cache rows of this wave's keys."""
            kv_row_words = (
                QK_DIM // 2 if const_expr(use_atom_kv_cache) else KV_LORA // 2
            )
            pe_lanes = PE_DIM // 8 if const_expr(cache_fp8) else PE_DIM // 2
            pe_lane = fx.min(lane, pe_lanes - 1)
            raws = []
            for jj in range_constexpr(KPW):
                if const_expr(cache_fp8):
                    row = krows[jj] * (QK_DIM // 4)
                    kv = fx.Vector(
                        bo.buffer_load(kv_rs, row + lane * 2, vec_width=2, dtype=T.i32)
                    )
                    pe = fx.Vector(
                        bo.buffer_load(
                            pe_rs,
                            row + KV_LORA // 4 + pe_lane * 2,
                            vec_width=2,
                            dtype=T.i32,
                        )
                    )
                else:
                    kv = fx.Vector(
                        bo.buffer_load(
                            kv_rs,
                            krows[jj] * kv_row_words + lane * 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    pe_row = (
                        krows[jj] * (QK_DIM // 2) + KV_LORA // 2 + pe_lane
                        if const_expr(use_atom_kv_cache)
                        else krows[jj] * (PE_DIM // 2) + pe_lane
                    )
                    pe = ld_f32(pe_rs, pe_row)
                raws.append((kv, pe))
            return raws

        (x_out, positions, slot_mapping, sparse_kv_indptr, kv_cache, pe_cache, indices, rope_cos, rope_sin, g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o, s_o, w_r, bias, w_ug, s_ug, w_dn, s_dn, scratch, sym, timeline_buf, block_table, req_ids, out_indices, ix_wk, ix_sk, ix_ww, ix_wq, ix_sq, ix_gk, ix_bk) = reload_args()  # fmt: skip
        if const_expr(RELOAD_ARGS and x_pushed):
            xin = (
                sym
                + fx.Int64(SY["xin"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
        if const_expr(SPLIT_EARLY):
            e_tt = fx.Int32(start("split"))
            e_has = e_tt < N_SPLIT_TILES
            e_tt = fx.min(e_tt, N_SPLIT_TILES - 1)

            def e_rsrc(addr):
                """A CTA without a split tile loads nothing (no round trip to wait)."""
                return bo.create_buffer_resource_from_addr(
                    addr,
                    num_records_bytes=e_has.select(fx.Int64(0xFFFFFFFF), fx.Int64(0)),
                )

            # lane j < KPW: key j of this wave
            e_base, e_keys, e_sparse = split_bounds(
                e_tt // ((H // WAVES) * N_SPLIT), e_rsrc(sparse_kv_indptr)
            )
            e_pos = (e_tt % N_SPLIT) * SPLIT_KEYS + wave * KPW + fx.min(lane, KPW - 1)
            e_cl = (e_pos < e_keys).select(e_pos, 0)
            e_key = fx.Int32(
                bo.buffer_load(e_rsrc(indices), e_base + e_cl, vec_width=1, dtype=T.i32)
            )
            if const_expr(use_atom_kv_cache):
                e_key = e_sparse.select(e_key, fx.Int32(0))
            else:
                e_key = e_sparse.select(e_key, e_cl)
            split_early = (
                e_key,
                split_kv_loads(
                    [read_lane_i32(e_key, jj) for jj in range(KPW)],
                    e_rsrc(kv_cache),
                    e_rsrc(pe_cache),
                ),
            )

        # =============================================== 4. normalized q_a -> q_b (+RoPE)
        (x_out, positions, slot_mapping, sparse_kv_indptr, kv_cache, pe_cache, indices, rope_cos, rope_sin, g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o, s_o, w_r, bias, w_ug, s_ug, w_dn, s_dn, scratch, sym, timeline_buf, block_table, req_ids, out_indices, ix_wk, ix_sk, ix_ww, ix_wq, ix_sq, ix_gk, ix_bk) = reload_args()  # fmt: skip
        if const_expr(RELOAD_ARGS and x_pushed):
            xin = (
                sym
                + fx.Int64(SY["xin"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
        r_wqb, r_sqb = _rsrc(w_q_b), _rsrc(s_q_b)
        QB_NKC = Q_LORA // 64
        QB_UNITS = QB_NKC // (attention_k_chunks_per_unit * WAVES)
        if const_expr(with_indexer):
            r_wiq, r_siq = _rsrc(index_arg(3)), _rsrc(index_arg(4))

        for t in range(start("q_b"), N_QB, G):
            t = fx.Int32(t)
            stamp("q_b", t, 0)

            def u_qb(c):
                kc = (wave * QB_UNITS + c) * attention_k_chunks_per_unit
                return unit_attention(
                    r_wqb,
                    r_sqb,
                    t,
                    kc,
                    QB_NKC,
                    Q_LORA,
                    128,
                    attention_b_word(n_sel() * Q_LORA + kc * 64),
                    shuffled=fp8_shuffled,
                    cm=AT_CM,
                )

            pre = [u_qb(c) for c in range(QB_UNITS)]
            if const_expr(with_indexer):
                # this CTA's index-query tile reuses the normalized q_lora tile: its
                # weights are issued before the wait too
                iq_t = t
                iq_pre = [u_index_q(iq_t, QB_UNITS, c) for c in range(QB_UNITS)]
            hint_wait(
                S,
                lambda s: (mb("q_an"), (s * Q_LORA + Q_LORA - 2) // 2),
                mark=("q_b", t),
            )
            if const_expr(attention_ptpc):
                qb_scales = stage_x_pairs_ptpc("q_an", S, Q_LORA, lambda k: k)
            else:
                stage_x_pairs("q_an", S * Q_LORA, lambda k: k)
            stamp("q_b", t, 2)
            gpu.barrier()
            if const_expr(with_indexer):
                # This CTA's index-query row tile first, on the normalized q_lora tile
                # (the index chain waits on it, split on the selection after it).
                # Complementary CTAs compute the remaining half below.
                stamp("index_q", iq_t, 0)
                iq_acc = mma_units([fx.Float32(0.0) for _ in range(4)], iq_pre)
                reduce_rows(1, iq_acc, emit_out(INDEX_TILE))
                stamp("index_q", iq_t, 3)
                gpu.barrier()
                if tid < S * INDEX_TILE // 4:
                    s = tid // (INDEX_TILE // 4)
                    r = (tid % (INDEX_TILE // 4)) * 4
                    put_bf(
                        mb("index_q"),
                        s * INDEX_Q_ROWS + iq_t * INDEX_TILE + r,
                        [lds_ld(outs, s * INDEX_TILE + r + j) for j in range(4)],
                    )
                stamp("index_q", iq_t, 4)
            acc = run_units(u_qb, QB_UNITS, QB_UNITS, pre)
            if const_expr(attention_ptpc):
                reduce_rows(
                    1,
                    acc,
                    emit_ptpc(Q_B_TILE, t * Q_B_TILE, r_sqb, qb_scales),
                )
            else:
                reduce_rows(1, acc, emit_out(Q_B_TILE))
            stamp("q_b", t, 3)
            gpu.barrier()
            head = t // QB_PER_HEAD
            hoff = (t % QB_PER_HEAD) * Q_B_TILE
            if hoff < NOPE_DIM:
                if tid < S * Q_B_TILE // 4:
                    s = tid // (Q_B_TILE // 4)
                    r = (tid % (Q_B_TILE // 4)) * 4
                    put_bf(
                        mb("q_nope"),
                        (s * H + head) * NOPE_DIM + hoff + r,
                        [lds_ld(outs, s * Q_B_TILE + r + j) for j in range(4)],
                    )
            else:
                if tid < S * Q_B_TILE // 2:
                    s = tid // (Q_B_TILE // 2)
                    pr = tid % (Q_B_TILE // 2)
                    i = hoff - NOPE_DIM + pr * 2
                    x0 = lds_ld(outs, s * Q_B_TILE + pr * 2)
                    x1 = lds_ld(outs, s * Q_B_TILE + pr * 2 + 1)
                    c = ld_bf16(
                        _rsrc(rope_cos), row_position(s) * (PE_DIM // 2) + i // 2
                    )
                    sn = ld_bf16(
                        _rsrc(rope_sin), row_position(s) * (PE_DIM // 2) + i // 2
                    )
                    put_bf(
                        mb("q_pe"),
                        (s * H + head) * PE_DIM + i,
                        [x0 * c - x1 * sn, x0 * sn + x1 * c],
                    )
            stamp("q_b", t, 4)

        if const_expr(with_indexer and N_INDEX_Q_EXTRA == 0):
            index_qh_tasks()

        # ==================================== 4. absorbed query: q_lat = W_UK^T q_nope
        # 8 row groups (128 latent rows of one head) x 3 chunks: one row group per wave
        (x_out, positions, slot_mapping, sparse_kv_indptr, kv_cache, pe_cache, indices, rope_cos, rope_sin, g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o, s_o, w_r, bias, w_ug, s_ug, w_dn, s_dn, scratch, sym, timeline_buf, block_table, req_ids, out_indices, ix_wk, ix_sk, ix_ww, ix_wq, ix_sq, ix_gk, ix_bk) = reload_args()  # fmt: skip
        if const_expr(RELOAD_ARGS and x_pushed):
            xin = (
                sym
                + fx.Int64(SY["xin"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
        r_wuk, r_suk = _rsrc(w_uk), _rsrc(s_uk)
        UK_NKC = NOPE_DIM // 64
        for t in range(start("uk"), N_UK, G):
            t = fx.Int32(t)
            stamp("uk", t, 0)
            head = t // UK_PER_HEAD

            def u_uk(c):
                return unit_attention(
                    r_wuk,
                    r_suk,
                    t * WAVES + wave,
                    c,
                    UK_NKC,
                    NOPE_DIM,
                    64,
                    attention_b_word(n_sel() * NOPE_DIM + c * 64),
                    coef=(
                        (
                            lambda: pick(
                                [
                                    uk_scales[
                                        sample * ((NOPE_DIM + 127) // 128) + c // 2
                                    ]
                                    for sample in range(S)
                                ],
                                n_sel(),
                            )
                        )
                        if const_expr(attention_ptpc)
                        else None
                    ),
                    cm=AT_CM,
                )

            pre = [u_uk(c) for c in range(UK_NKC)]
            hint_wait(
                NOPE_DIM // Q_B_TILE,
                lambda k: (
                    mb("q_nope"),
                    ((S - 1) * H + head) * NOPE_DIM + k * Q_B_TILE + Q_B_TILE - 1,
                ),
                mark=("uk", t),
            )
            if const_expr(attention_ptpc):
                uk_scales = stage_x_pairs_ptpc(
                    "q_nope",
                    S,
                    NOPE_DIM,
                    lambda k: ((k // NOPE_DIM) * H + head) * NOPE_DIM + k % NOPE_DIM,
                    group=128,
                )
            else:
                stage_x_pairs(
                    "q_nope",
                    S * NOPE_DIM,
                    lambda k: ((k // NOPE_DIM) * H + head) * NOPE_DIM + k % NOPE_DIM,
                )
            stamp("uk", t, 2)
            gpu.barrier()
            acc = run_units(u_uk, UK_NKC, UK_NKC, pre)
            fx.ptr_store(
                fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4
            )
            stamp("uk", t, 3)
            gpu.barrier()
            if tid < S * UK_TILE // 4:
                k = tid * 4
                s = k // UK_TILE
                r0 = k % UK_TILE
                vals = []
                if const_expr(attention_ptpc):
                    ptpc_scale = ld_f32(r_suk, 0)
                for j in range_constexpr(4):
                    r = r0 + j
                    ww = r // 16
                    value = lds_ld(
                        red, (ww * 64 + s + 16 * ((r % 16) // 4)) * 4 + r % 4
                    )
                    if const_expr(attention_ptpc):
                        value = bf16_round(value * ptpc_scale)
                    vals.append(value)
                put_bf(
                    mb("q_lat"),
                    (s * H + head) * KV_LORA + (t % UK_PER_HEAD) * UK_TILE + r0,
                    vals,
                )
            stamp("uk", t, 4)

        # ====================== 4b. index scores and the exact top-2048 (paged keys)
        (x_out, positions, slot_mapping, sparse_kv_indptr, kv_cache, pe_cache, indices, rope_cos, rope_sin, g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o, s_o, w_r, bias, w_ug, s_ug, w_dn, s_dn, scratch, sym, timeline_buf, block_table, req_ids, out_indices, ix_wk, ix_sk, ix_ww, ix_wq, ix_sq, ix_gk, ix_bk) = reload_args()  # fmt: skip
        if const_expr(RELOAD_ARGS and x_pushed):
            xin = (
                sym
                + fx.Int64(SY["xin"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
        if const_expr(index_paged):
            r_index_cache = _rsrc(indices)
            r_out_indices = _rsrc(out_indices)
            hist = fx.recast_iter(fx.Int32, xs)

            def load_index_q8(head, k):
                words = fx.Vector.from_elements(
                    [lds_ld(xs, (head * INDEX_DIM + k) // 2 + j) for j in range(4)],
                    fx.Float32,
                )
                return words.bitcast(fx.BFloat16)

            def radix_key(bits):
                return (bits >= 0).select(bits ^ fx.Int32(-(2**31)), ~bits)

            # Scoring. A request's consecutive rows (up to SCORE_ROWS) score its keys
            # together: the request's pages are split into runs of whole pages, one a
            # CTA (counted from a CTA its first row offsets). Below 4 rows a CTA copies
            # its run into LDS a batch at a time, the next batch's loads in flight while
            # it scores one, and each row's waves score the batch, the row's rotated
            # query in registers. A key this launch writes is scored from its row's
            # index_k8 pairs (its FP8 values and scale as cached; the cache store need
            # not be visible yet).
            # From 4 rows a wave scores two rows, converting each key once for them
            # (their queries staged in MFMA operand order), long rows' runs split
            # evenly; a set of rows takes SET_WAVES waves, which split the run's 16-key
            # groups, each wave's groups loaded straight into its registers and its
            # scores stored from them
            PAIRED = S >= 4
            WAVE_ROWS = 2 if PAIRED else 1
            ROW_WAVES = WAVES // SCORE_ROWS
            SET_WAVES = WAVES * WAVE_ROWS // SCORE_ROWS
            PAGE_WORDS = index_block_bytes // 4
            BATCH_PAGES = 8
            BATCH_WORDS = BATCH_PAGES * PAGE_WORDS
            BATCH_LOADS = (BATCH_WORDS + 4 * THREADS - 1) // (4 * THREADS)
            # a CTA scores at least SCORE_PAGES pages of a request, or all of them
            SCORE_PAGES = 2
            # runs go to CTAs S .. G - 1: the q_norm CTAs (with the rows' index-key and
            # cache tasks) score nothing
            SCORE_CTAS = G - S
            MAX_RUN_PAGES = max(
                SCORE_PAGES,
                (index_max_seq // index_block_size + SCORE_CTAS - 1) // SCORE_CTAS,
            )
            BT_OFF = BATCH_WORDS  # the run's block ids, after the batch
            # from 4 rows, after them: the run's first-digit counts, 64 bins a (row,
            # slice segment); a run spans at most two of a row's slices
            SH_OFF = BT_OFF + MAX_RUN_PAGES
            SH_WORDS = SCORE_ROWS * 2 * 64
            assert not PAIRED or (SH_OFF + SH_WORDS <= X_WORDS and SH_WORDS == THREADS)
            GROUP_PAGE = index_block_size // 16  # 16-key groups a page
            GROUP_SHIFT = GROUP_PAGE.bit_length() - 1
            assert GROUP_PAGE == 1 << GROUP_SHIFT

            def group_words(g):
                """Group g's key: the lane's value word 0 and scale word in the batch. g is
                wave-uniform, so its page and first key are scalar math and the lane's key
                is added to their words."""
                p = g >> GROUP_SHIFT
                off0 = (g & (GROUP_PAGE - 1)) * 16
                return (
                    index_value_word(p, off0, 0) + index_value_word(0, lane % 16, 0),
                    index_scale_word(p, off0) + lane % 16,
                )

            def swap(off, x, y):
                """v_permlane{16,32}_swap of two f32 registers."""
                a, b = permlane_swap(off, x.bitcast(fx.Int32), y.bitcast(fx.Int32))
                return a.bitcast(fx.Float32), b.bitcast(fx.Float32)

            def head_sums(halves):
                """Each row's score from its two head halves' lane partials (``halves``: one or
                two rows of [h0, h1]): a half sums its four lane rows as (r0 + r1) + (r2 + r3),
                then the row's score is h0 + h1, as two butterflies a half would. Lanes 0-31
                hold the first row's score, lanes 32-63 the second's."""
                ts = []
                for h0, h1 in halves:
                    a, b = swap(16, h0, h1)
                    ts.append(
                        a + b
                    )  # lane rows: h0 (0 + 1), h1 (0 + 1), h0 (2 + 3), h1 (2 + 3)
                c, d = swap(32, ts[0], ts[-1])
                u = c + d  # lane rows: h0, h1 of the first row, then of the second
                e, f = swap(16, u, u)
                return e + f

            K8 = INDEX_DIM // 4 + 1  # a new key's pairs: its value words, its scale
            assert INDEX_Q_ROWS == THREADS * 8
            assert index_block_size == INDEX_KEYS_PER_TASK
            # paired scoring: a wave's next DEPTH groups' keys are in flight while it
            # scores one; a lane's words of a group for 32-dim chunk k32 are KEY_STEP[k32]
            # words past those for chunk 0
            DEPTH = 2
            KEY_STEP = [
                index_value_word(0, 0, k32 * 32) - index_value_word(0, 0, 0)
                for k32 in range(INDEX_DIM // 32)
            ]
            r_scores = _rsrc(mb("index_scores"), S * index_max_seq * 8)
            assert SCORE_ROWS * INDEX_Q_ROWS // 2 <= BATCH_WORDS
            assert SCORE_ROWS * INDEX_HEADS <= LDS_KEYS and S * K8 <= THREADS
            kw = fx.recast_iter(fx.Int32, xs)
            # a request's runs from this CTA on (past the q_norm and uk CTAs' work)
            SCORE_BASE = 101

            def plan_bound(i):
                """Row i's index bound."""
                if const_expr(PLANNED):
                    return _uniform(lds_ld(plan, 5 + i))
                return index_bound(i)

            def plan_request(i):
                """Row i's page-table row."""
                if const_expr(PLANNED):
                    return _uniform(lds_ld(plan, 5 + S + i))
                return _uniform(index_request(i))

            def fetch_batch(page0, n_pages):
                """This thread's words of the run's pages page0 .. page0 + n_pages - 1
                (at most a batch; block ids in LDS), as 16-byte loads. Every block id is
                read before the first load, and words past the run (or the batch) load the
                cache's first 16 bytes, which are never read back."""
                ws = [(tid + i * THREADS) * 4 for i in range(BATCH_LOADS)]
                ps = [fx.Int32(w // PAGE_WORDS) for w in ws]
                blocks = [
                    lds_ld(kw, BT_OFF + page0 + fx.max(fx.min(p, n_pages - 1), 0))
                    for p in ps
                ]
                out = []
                for i in range_constexpr(BATCH_LOADS):
                    w, p = ws[i], ps[i]
                    at = ((p < n_pages) & (fx.Int32(w) < BATCH_WORDS)).select(
                        blocks[i] * PAGE_WORDS + w % PAGE_WORDS, fx.Int32(0)
                    )
                    out.append(
                        fx.Vector(
                            bo.buffer_load(
                                r_index_cache,
                                at,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                    )
                return out

            def group_keys(i, n_g, g_lo, n_run_groups):
                """The words of the wave's i-th group of the run (16 keys from group g_lo +
                SET_WAVES i) a lane holds as MFMA B operands, for each 32-dim chunk two, then
                the lane's key's scale; past the wave's n_g groups, the cache's first word."""
                ok = i < n_g
                g = fx.min(g_lo + i * SET_WAVES, fx.Int32(n_run_groups - 1))
                blk = lds_ld(kw, BT_OFF + (g >> GROUP_SHIFT))
                off0 = (g & (GROUP_PAGE - 1)) * 16
                vw = (
                    index_value_word(blk, off0, 0)
                    + index_value_word(0, lane % 16, 0)
                    + index_value_word(0, 0, (lane // 16) * 8)
                )
                words = []
                for k32 in range_constexpr(INDEX_DIM // 32):
                    v = fx.Vector(
                        bo.buffer_load(
                            r_index_cache,
                            ok.select(vw + KEY_STEP[k32], fx.Int32(0)),
                            vec_width=2,
                            dtype=T.i32,
                        )
                    )
                    words += [v[0], v[1]]
                words.append(
                    fx.Int32(
                        bo.buffer_load(
                            r_index_cache,
                            ok.select(
                                index_scale_word(blk, off0) + lane % 16, fx.Int32(0)
                            ),
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                )
                return words

            def with_new_keys(ws, k0, new_bounds):
                """A group's words (keys k0 .. k0 + 15) with the keys this launch writes (key
                new_bounds[r] - 1 of each row r of the request) taken from their rows'
                index_k8 pairs: the FP8 values and scale as cached, before the cache store
                is visible."""
                key = k0 + lane % 16
                r_new = fx.Int32(-1)
                near = fx.Int32(0)
                for r in range_constexpr(S):
                    pos = new_bounds[r] - 1
                    r_new = (key == pos).select(fx.Int32(r), r_new)
                    near = near | ((pos >= k0) & (pos < k0 + 16)).select(
                        fx.Int32(1), fx.Int32(0)
                    )
                w0, w1, w2, w3, w4, w5, w6, w7, w8 = ws
                if _uniform(near) != 0:
                    items = [
                        8 * k32 + 2 * (lane // 16) + j
                        for k32 in range(INDEX_DIM // 32)
                        for j in range(2)
                    ] + [fx.Int32(K8 - 1)]
                    got = poll(
                        [
                            (
                                mb("index_k8"),
                                fx.max(r_new, fx.Int32(0)) * K8 + it,
                                1,
                            )
                            for it in items
                        ],
                        batch=len(items),
                    )
                    mine = r_new >= 0
                    w0 = mine.select(got[0][0], w0)
                    w1 = mine.select(got[1][0], w1)
                    w2 = mine.select(got[2][0], w2)
                    w3 = mine.select(got[3][0], w3)
                    w4 = mine.select(got[4][0], w4)
                    w5 = mine.select(got[5][0], w5)
                    w6 = mine.select(got[6][0], w6)
                    w7 = mine.select(got[7][0], w7)
                    w8 = mine.select(got[8][0], w8)
                return [w0, w1, w2, w3, w4, w5, w6, w7, w8]

            def group_accs(ws, g, key_lo, n_rows, qf, new_bounds):
                """Group g's (16 keys from key_lo + 16 g; words ``ws``) head-half
                accumulators for the wave's ``n_rows`` rows, and the keys' scales."""
                ws = with_new_keys(ws, key_lo + g * 16, new_bounds)
                accs = [
                    [
                        fx.Vector.filled(4, 0.0, fx.Float32),
                        fx.Vector.filled(4, 0.0, fx.Float32),
                    ]
                    for _ in range(n_rows)
                ]
                for k32 in range_constexpr(INDEX_DIM // 32):
                    kv = _fp8_to_bf16x8(ws[2 * k32], ws[2 * k32 + 1])
                    for ri in range_constexpr(n_rows):
                        for hg in range_constexpr(2):
                            accs[ri][hg] = fx.Vector(
                                rocdl.mfma_f32_16x16x32_bf16(
                                    T.vec(4, T.f32), [qf[ri][hg][k32], kv, accs[ri][hg]]
                                )
                            )
                return accs, ws[8].bitcast(fx.Float32)

            def store_scores(accs, k_scale, g, key_lo, s0, rows_w, wf, seg):
                """Group g's scores for the wave's rows ``rows_w`` [(row in the group, bound,
                live)], stored as pairs where a key is below its row's bound, and counted
                in the run's first-digit bins: ``seg`` = (the lane's row's bins, the
                first key of its second slice segment). Each head half is k_scale * the
                sum over its heads of relu(acc) * weight, summed from 0 in head order;
                the first term is one fma with 0, which rounds as 0 + product does."""
                halves = []
                for ri in range_constexpr(len(rows_w)):
                    parts = []
                    for hg in range_constexpr(2):
                        acc, w = accs[ri][hg], wf[ri][hg]
                        partial = fmath.fma(
                            fx.max(acc[0], fx.Float32(0.0)), w[0], fx.Float32(0.0)
                        )
                        for e in range_constexpr(1, 4):
                            partial = partial + fx.max(acc[e], fx.Float32(0.0)) * w[e]
                        parts.append(partial * k_scale)
                    halves.append(parts)
                # lanes 0-15 hold the first row's scores, lanes 32-47 the second's
                score = head_sums(halves)
                second = lane >= 32
                r_w = second.select(rows_w[-1][0], rows_w[0][0])
                bound_r = second.select(
                    rows_w[-1][2].select(rows_w[-1][1], fx.Int32(0)),
                    rows_w[0][2].select(rows_w[0][1], fx.Int32(0)),
                )
                key = key_lo + g * 16 + lane % 16
                ok = (lane % 32 < 16) & (key < bound_r)
                bo.buffer_store(
                    fx.Vector.from_elements([score.bitcast(fx.Int32), tag], fx.Int32),
                    r_scores,
                    ((s0 + r_w) * index_max_seq + key) * 2,
                    mask=ok,
                    cache_modifier=CM_DEV,
                )
                bins, seg1 = seg
                digit = fx.Int32(fx.Uint32(radix_key(score.bitcast(fx.Int32))) >> 26)
                at = bins + (key >= seg1).select(fx.Int32(64), fx.Int32(0)) + digit
                if ok:
                    lds_atomic_add(kw + at, fx.Int32(1), syncscope="workgroup")

            def score_keys_row(g, key0, s_w, bound_w, qf, wf, kofs):
                """The wave's group g (16 keys) of the batch, from key key0, for row s_w."""
                k = g * 16 + lane % 16
                base_w, scale_w = group_words(g)
                k_scale = lds_ld(kw, scale_w).bitcast(fx.Float32)
                acc0 = fx.Vector.filled(4, 0.0, fx.Float32)
                acc1 = acc0
                for k32 in range_constexpr(INDEX_DIM // 32):
                    kv = _fp8_to_bf16x8(
                        lds_ld(kw, base_w + kofs[k32]),
                        lds_ld(kw, base_w + kofs[k32] + 1),
                    )
                    acc0 = fx.Vector(
                        rocdl.mfma_f32_16x16x32_bf16(
                            T.vec(4, T.f32), [qf[0][k32], kv, acc0]
                        )
                    )
                    acc1 = fx.Vector(
                        rocdl.mfma_f32_16x16x32_bf16(
                            T.vec(4, T.f32), [qf[1][k32], kv, acc1]
                        )
                    )
                parts = []
                for hg in range_constexpr(2):
                    acc = acc1 if hg else acc0
                    partial = fx.Float32(0.0)
                    for e in range_constexpr(4):
                        partial = partial + fx.max(acc[e], fx.Float32(0.0)) * wf[hg][e]
                    parts.append(partial * k_scale)
                score = head_sums([parts])
                key_pos = key0 + k
                if (lane < 16) & (key_pos < bound_w):
                    put(mb("index_scores"), s_w * index_max_seq + key_pos, score)

            def score_group(s0, n, req, c0):
                """This CTA's run of the keys of rows s0 .. s0 + n - 1 of a request whose
                first row in the launch is c0."""
                rows = [fx.min(s0 + r, fx.Int32(S - 1)) for r in range(SCORE_ROWS)]
                bounds = [
                    (fx.Int32(r) < n).select(plan_bound(rows[r]), fx.Int32(0))
                    for r in range(SCORE_ROWS)
                ]
                bound_g = bounds[0]
                for r in range_constexpr(1, SCORE_ROWS):
                    bound_g = fx.max(bound_g, bounds[r])
                bound_g = _uniform(bound_g)
                n_pages = (bound_g + index_block_size - 1) // index_block_size
                runs = fx.min(
                    fx.Int32(SCORE_CTAS), (n_pages + SCORE_PAGES - 1) // SCORE_PAGES
                )
                per = (n_pages + runs - 1) // runs
                first_run = (SCORE_BASE - S + s0 * (G // S)) % SCORE_CTAS
                run = (fx.Int32(bid) - S + (SCORE_CTAS - first_run)) % SCORE_CTAS
                p_lo = run * per
                p_hi = fx.min(p_lo + per, n_pages)
                # from 4 rows, when every scoring CTA has a run, runs differ by at
                # most a page
                if const_expr(PAIRED):
                    even = runs == SCORE_CTAS
                    p_lo = even.select(run * n_pages // runs, p_lo)
                    p_hi = even.select((run + 1) * n_pages // runs, p_hi)
                if (fx.Int32(bid) >= S) & (run < runs) & (p_lo < p_hi):
                    stamp("index_score", bid, 0)
                    # the run's block ids: loaded with the query's poll, stored after it
                    bt_word = index_block(
                        req,
                        (p_lo + fx.min(fx.Int32(tid), p_hi - p_lo - 1))
                        * index_block_size,
                    )
                    # the rows' queries, rotated and transformed by index_qh, in one poll
                    specs = []
                    for r in range_constexpr(SCORE_ROWS):
                        for i in range_constexpr(2):
                            specs.append(
                                (
                                    mb("index_qh"),
                                    rows[r] * (INDEX_Q_ROWS // 2)
                                    + fx.Int32(tid) * 4
                                    + 2 * i,
                                    2,
                                )
                            )
                    specs.append(
                        (
                            mb("index_w"),
                            s0 * INDEX_HEADS
                            + fx.min(fx.Int32(tid), n * INDEX_HEADS - 1),
                            1,
                        )
                    )
                    got = poll(specs, batch=len(specs))
                    if tid < p_hi - p_lo:
                        lds_st(kw, BT_OFF + tid, bt_word)
                    gpu.barrier()
                    if const_expr(not PAIRED):
                        first = fetch_batch(fx.Int32(0), p_hi - p_lo)
                    for r in range_constexpr(SCORE_ROWS):
                        # a thread's 4 words: head tid // 16, dims 8 * (tid % 16) ..
                        q_at = r * (INDEX_Q_ROWS // 2) + fx.Int32(tid) * 4
                        if const_expr(PAIRED):
                            # in MFMA operand order (head half, 32-dim chunk, lane), so a
                            # wave reads its fragments without bank conflicts
                            head = fx.Int32(tid) // 16
                            q_at = (
                                r * (INDEX_Q_ROWS // 2)
                                + (
                                    (
                                        (head // 16) * (INDEX_DIM // 32)
                                        + (fx.Int32(tid) % 16) // 4
                                    )
                                    * 64
                                    + head % 16
                                    + 16 * (fx.Int32(tid) % 4)
                                )
                                * 4
                            )
                        for i in range_constexpr(4):
                            lds_st(
                                xs,
                                q_at + i,
                                got[r * 2 + i // 2][i % 2].bitcast(fx.Float32),
                            )
                    if tid < SCORE_ROWS * INDEX_HEADS:
                        lds_st(keys, tid, got[SCORE_ROWS * 2][0])
                    if const_expr(PAIRED):
                        lds_st(kw, SH_OFF + tid, fx.Int32(0))
                    gpu.barrier()
                    stamp("index_score", bid, 2)
                    if const_expr(PAIRED):
                        # this wave's rows: their queries (A operands of both head halves)
                        # and head weights, held for the whole run
                        wave_u = _uniform(wave)
                        r_first = (wave_u // SET_WAVES) * WAVE_ROWS
                        rows_w, qf, wf = [], [], []
                        bound_max = fx.Int32(0)
                        for ri in range_constexpr(WAVE_ROWS):
                            r_w = r_first + ri
                            s_w = s0 + fx.min(r_w, n - 1)
                            bound_w = _uniform(plan_bound(s_w))
                            live_w = r_w < n
                            bound_max = fx.max(
                                bound_max, live_w.select(bound_w, fx.Int32(0))
                            )
                            rows_w.append((r_w, bound_w, live_w))
                            qf.append(
                                [
                                    [
                                        fx.Vector.from_elements(
                                            [
                                                lds_ld(
                                                    xs,
                                                    r_w * (INDEX_Q_ROWS // 2)
                                                    + (
                                                        (hg * (INDEX_DIM // 32) + k32)
                                                        * 64
                                                        + lane
                                                    )
                                                    * 4
                                                    + j,
                                                )
                                                for j in range(4)
                                            ],
                                            fx.Float32,
                                        ).bitcast(fx.BFloat16)
                                        for k32 in range(INDEX_DIM // 32)
                                    ]
                                    for hg in range(2)
                                ]
                            )
                            wf.append(
                                [
                                    [
                                        lds_ld(
                                            keys,
                                            r_w * INDEX_HEADS
                                            + hg * 16
                                            + (lane // 16) * 4
                                            + e,
                                        ).bitcast(fx.Float32)
                                        for e in range(4)
                                    ]
                                    for hg in range(2)
                                ]
                            )
                    else:
                        # this wave's row: its query (A operands of both head halves) and
                        # head weights, held for the whole run; a lone live row takes every
                        # wave
                        wave_u = _uniform(wave)
                        lone = _uniform(n) == 1
                        r_w = lone.select(fx.Int32(0), wave_u // ROW_WAVES)
                        g_first = lone.select(wave_u, wave_u % ROW_WAVES)
                        g_step = lone.select(fx.Int32(WAVES), fx.Int32(ROW_WAVES))
                        s_w = s0 + fx.min(r_w, n - 1)
                        bound_w = _uniform(plan_bound(s_w))
                        qf = [
                            [
                                load_index_q8(
                                    r_w * INDEX_HEADS + hg * 16 + lane % 16,
                                    k32 * 32 + (lane // 16) * 8,
                                )
                                for k32 in range(INDEX_DIM // 32)
                            ]
                            for hg in range(2)
                        ]
                        wf = [
                            [
                                lds_ld(
                                    keys,
                                    r_w * INDEX_HEADS + hg * 16 + (lane // 16) * 4 + e,
                                ).bitcast(fx.Float32)
                                for e in range(4)
                            ]
                            for hg in range(2)
                        ]
                    # the keys this launch writes among the rows': rows c0 .. s0 + n - 1
                    n_new = s0 + n - c0
                    new_bounds = [
                        ((fx.Int32(r) >= c0) & (fx.Int32(r) < s0 + n)).select(
                            index_bound(r), fx.Int32(0)
                        )
                        for r in range(S)
                    ]
                    if const_expr(PAIRED):
                        # each wave set's waves split the run's 16-key groups; a wave's
                        # groups load straight into registers, DEPTH ahead
                        key_lo = p_lo * index_block_size
                        n_run_groups = (p_hi - p_lo) * GROUP_PAGE
                        g_lo = wave_u % SET_WAVES
                        g_hi = fx.min(
                            n_run_groups,
                            (fx.max(bound_max - key_lo, fx.Int32(0)) + 15) // 16,
                        )
                        n_g = (
                            fx.max(g_hi - g_lo, fx.Int32(0)) + SET_WAVES - 1
                        ) // SET_WAVES
                        n_g = (r_first < n).select(n_g, fx.Int32(0))
                        # the lane's row's first-digit bins, and the first key of the
                        # row's slice after the one holding the run's first key
                        seg1 = []
                        for _row, bound_w, _live in rows_w:
                            _, per_w = select_slices(bound_w, fx.min, fx.max)
                            seg1.append((key_lo // per_w + 1) * per_w)
                        second = lane >= 32
                        seg = (
                            SH_OFF + second.select(rows_w[-1][0], rows_w[0][0]) * 128,
                            second.select(seg1[-1], seg1[0]),
                        )
                        bufs = []
                        for i in range_constexpr(DEPTH):
                            bufs += group_keys(fx.Int32(i), n_g, g_lo, n_run_groups)
                        nw = len(bufs) // DEPTH

                        def scan_group(ws, gi, refill):
                            """The wave's gi-th group; its words' registers take the group
                            DEPTH on once its MFMAs have read them (the scale moved out first)."""
                            g = g_lo + gi * SET_WAVES
                            accs, k_scale = group_accs(
                                ws, g, key_lo, WAVE_ROWS, qf, new_bounds
                            )
                            if const_expr(refill is not None):
                                k_scale = _fresh(k_scale.bitcast(fx.Int32)).bitcast(
                                    fx.Float32
                                )
                                refill += group_keys(
                                    gi + DEPTH, n_g, g_lo, n_run_groups
                                )
                            store_scores(accs, k_scale, g, key_lo, s0, rows_w, wf, seg)

                        for j, held in range(0, n_g // DEPTH, 1, init=bufs):
                            refill = []
                            for i in range_constexpr(DEPTH):
                                scan_group(
                                    [fx.Int32(held[i * nw + e]) for e in range(nw)],
                                    fx.Int32(j) * DEPTH + i,
                                    refill,
                                )
                            last = yield refill
                        for i in range_constexpr(DEPTH - 1):
                            gi = (n_g // DEPTH) * DEPTH + i
                            if gi < n_g:
                                scan_group(
                                    [fx.Int32(last[i * nw + e]) for e in range(nw)],
                                    gi,
                                    None,
                                )
                        # each live row's counts, a pair of bins a thread: pairs 0-31 its
                        # first slice segment's, 32-63 its second's (a later run's LDS
                        # stores also wait here for every wave's last block-id read)
                        gpu.barrier()
                        if fx.Int32(tid) < n * 64:
                            put(
                                mb("score_hist"),
                                ((s0 + fx.Int32(tid) // 64) * G + run) * 64
                                + fx.Int32(tid) % 64,
                                lds_ld(kw, SH_OFF + 2 * fx.Int32(tid))
                                | (lds_ld(kw, SH_OFF + 2 * fx.Int32(tid) + 1) << 16),
                            )
                    else:
                        kofs = [
                            index_value_word(
                                fx.Int32(0), 0, k32 * 32 + (lane // 16) * 8
                            )
                            for k32 in range(INDEX_DIM // 32)
                        ]
                        n_run = p_hi - p_lo
                        n_batches = (n_run + BATCH_PAGES - 1) // BATCH_PAGES
                        for bi, words in range(0, n_batches, 1, init=first):
                            bi = fx.Int32(bi)
                            page0 = bi * BATCH_PAGES
                            n_pages_b = fx.min(fx.Int32(BATCH_PAGES), n_run - page0)
                            key0 = (p_lo + page0) * index_block_size
                            key_end = key0 + n_pages_b * index_block_size
                            gpu.barrier()
                            for i in range_constexpr(BATCH_LOADS):
                                w = (tid + i * THREADS) * 4
                                if fx.Int32(w) < BATCH_WORDS:
                                    for e in range_constexpr(4):
                                        lds_st(kw, w + e, fx.Vector(words[i])[e])
                            nxt = fetch_batch(
                                page0 + BATCH_PAGES, n_run - page0 - BATCH_PAGES
                            )
                            gpu.barrier()
                            # the keys this launch writes, from their rows' pairs
                            hit = fx.Int32(0)
                            for r in range_constexpr(S):
                                hit = hit | (
                                    (new_bounds[r] > key0) & (new_bounds[r] <= key_end)
                                ).select(fx.Int32(1), fx.Int32(0))
                            if _uniform(hit) != 0:
                                r_p = fx.min(c0 + tid // K8, fx.Int32(S - 1))
                                pos_p = new_bounds[0] - 1
                                for r in range_constexpr(1, S):
                                    pos_p = (r_p == r).select(new_bounds[r] - 1, pos_p)
                                if (
                                    (fx.Int32(tid) < n_new * K8)
                                    & (pos_p >= key0)
                                    & (pos_p < key_end)
                                ):
                                    item = tid % K8
                                    v = get(mb("index_k8"), r_p * K8 + item)
                                    kk = pos_p - key0
                                    p = kk // index_block_size
                                    off = kk % index_block_size
                                    at = (item < K8 - 1).select(
                                        index_value_word(p, off, item * 4),
                                        index_scale_word(p, off),
                                    )
                                    lds_st(kw, at, v)
                                gpu.barrier()
                            if r_w < n:
                                for g in range(
                                    g_first,
                                    n_pages_b * (index_block_size // 16),
                                    g_step,
                                ):
                                    g = fx.Int32(g)
                                    if key0 + g * 16 < bound_w:
                                        score_keys_row(
                                            g, key0, s_w, bound_w, qf, wf, kofs
                                        )
                            _ = yield nxt
                        gpu.barrier()
                    stamp("index_score", bid, 4)

            if const_expr(PLANNED):
                # the scoring layout; the q_norm CTAs plan it now, for their selection
                # slices (they score nothing)
                if bid < S:
                    plan_layout()
                row_leads = _uniform(lds_ld(plan, 0))
                leads = (fx.Int32(bid) >= S).select(row_leads, fx.Int32(0))
                sizes = fx.Int64(fx.Uint32(_uniform(lds_ld(plan, 1)))) | (
                    fx.Int64(_uniform(lds_ld(plan, 2))) << 32
                )
                firsts = fx.Int64(fx.Uint32(_uniform(lds_ld(plan, 3)))) | (
                    fx.Int64(_uniform(lds_ld(plan, 4))) << 32
                )
            else:
                leads, sizes, firsts = score_groups(
                    [index_bound(r) for r in range(S)],
                    lambda r, key: index_block(index_request(r), key),
                    row_slot,
                )
                row_leads = leads
            for s in range(0, S, 1):
                s = fx.Int32(s)
                if ((leads >> s) & 1) != 0:
                    n = fx.Int32((sizes >> (fx.Int64(s) * 3)) & fx.Int64(7))
                    c0 = fx.Int32((firsts >> (fx.Int64(s) * 3)) & fx.Int64(7))
                    score_group(s, n, plan_request(s), _uniform(c0))

            def scan_pair(first, second, parity):
                """Exclusive block offsets of two flag lists in one scan.

                Wave totals and running bases are double-buffered by ``parity``
                (the chunk index), so consecutive chunks need one barrier each.
                Returns (first offsets, second offsets, first total, second total),
                offsets already include the running bases.
                """
                local = fx.Int32(0)
                local_offsets = []
                for a, b in zip(first, second):
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
                offs = [thread_base + off for off in local_offsets]
                n_a = total & fx.Int32(0xFFFF)
                n_b = total >> 16
                if tid == 0:
                    lds_st(keys, 272 + (1 - parity) * 2, base_a + n_a)
                    lds_st(keys, 273 + (1 - parity) * 2, base_b + n_b)
                return (
                    [base_a + (o & fx.Int32(0xFFFF)) for o in offs],
                    [base_b + (o >> 16) for o in offs],
                    base_a + n_a,
                    base_b + n_b,
                )

            def set_bases(a, b):
                if tid == 0:
                    lds_st(keys, 272, a)
                    lds_st(keys, 273, b)
                gpu.barrier()

            # Selection: each row's exact top-k over its slices, SEL_SLICES CTAs at
            # most, each a run of the row's keys in position order. A radix select by
            # 6-bit digits from the top: per digit every slice publishes the histogram
            # of its keys under the prefix found so far, and every slice reads all of
            # them, finds the digit holding the remain-th key and counts the keys above
            # it in the slices before its own. Once the prefix has 12 or 24 bits and its
            # keys fit the candidate stage, or has all 32, the prefix is the threshold's:
            # each slice writes its keys above it (their places known), and its keys
            # with the prefix go to the row's first slice, which resolves them (or, at
            # 32 bits, ties, take their places directly). The lists are the one-CTA
            # radix select's: keys above the threshold's 12-bit (24-, 32-bit) prefix in
            # position order, then the prefix's keys above the threshold, then its ties.
            # the keys a slice holds at most, as radix keys in LDS, in chunks of SEL_CHUNK
            SLICE_KEYS = (
                ((index_max_seq + SEL_SLICES - 1) // SEL_SLICES + SEL_CHUNK - 1)
                // SEL_CHUNK
                * SEL_CHUNK
            )
            DIGITS = ((26, 6), (20, 6), (14, 6), (8, 6), (2, 6), (0, 2))
            assert len(DIGITS) == INDEX_SEL_DIGITS
            HIST_PAIRS = 32  # a slice's 64 counts, two 16-bit counts a pair
            # LDS words: the slice's keys, its block ids, its bins (and the digit
            # found), the row's bins
            SBT = SLICE_KEYS
            SBINS = SBT + SLICE_KEYS // index_block_size + 2
            STOT = SBINS + 72
            SALIST = STOT + 64  # the slice's keys above the threshold's prefix
            assert SALIST + topk <= X_WORDS and SEL_SLICES * HIST_PAIRS <= 4 * THREADS
            assert SLICE_KEYS < 1 << 16
            skeys = fx.recast_iter(fx.Int32, xs)
            sbins = fx.recast_iter(fx.Int32, xs + SBINS)
            stot = fx.recast_iter(fx.Int32, xs + STOT)

            red_i = fx.recast_iter(fx.Int32, red)

            def block_isums(vs):
                """Block-wide sums of several per-thread Int32 values."""
                ws = [
                    read_lane_i32(
                        fx.coop.warp_inclusive_scan(v, fx.ReductionOp.ADD, width=64), 63
                    )
                    for v in vs
                ]
                if lane == 0:
                    for i in range_constexpr(len(vs)):
                        lds_st(red_i, i * WAVES + wave, ws[i])
                gpu.barrier()
                tots = []
                for i in range_constexpr(len(vs)):
                    t = lds_ld(red_i, i * WAVES)
                    for w in range_constexpr(1, WAVES):
                        t = t + lds_ld(red_i, i * WAVES + w)
                    tots.append(t)
                gpu.barrier()
                return tots

            def chunk_slots(chunk):
                """This thread's places in a chunk of the slice (position order)."""
                first_ = chunk * SEL_CHUNK + fx.Int32(tid) * SEL_ITEMS
                return [first_ + j for j in range(SEL_ITEMS)]

            # the selection's stores take resources made once, outside its branches
            r_indices = _rsrc(mb("indices"))

            def publish_slot(s, out_begin, p, slot):
                """Place p of row s's list := cache slot ``slot``."""
                bo.buffer_store(
                    fx.Vector.from_elements([slot, tag], fx.Int32),
                    r_indices,
                    (s * topk + p) * 2,
                    cache_modifier=CM_DEV,
                )
                bo.buffer_store(
                    slot, r_out_indices, out_begin + p, cache_modifier=CM_DEV
                )

            def resolve(s, req, out_begin, count, level, prefix, remain, n_cand):
                """The row's candidates (the keys under ``prefix``'s top 32 - ``level``
                bits with their cache slots, in position order, staged in LDS): the
                remain-th largest, then
                places count - remain_at_entry .. count - 1 of the list: the
                candidates above it, then its first ties."""
                n_win = count - remain

                def cand_pass(shift, bits, prefix, remain):
                    """The next ``bits`` (≤ 10) of the threshold below ``prefix``: a
                    histogram of the candidates under it (2^bits bins in LDS, the slice's
                    dead keys), then the bin holding the remain-th."""
                    nb = 1 << bits
                    mask = (0xFFFFFFFF << (shift + bits)) & 0xFFFFFFFF
                    per = (nb + THREADS - 1) // THREADS
                    for b in range_constexpr(per):
                        if fx.Int32(tid) + b * THREADS < nb:
                            lds_st(skeys, fx.Int32(tid) + b * THREADS, fx.Int32(0))
                    gpu.barrier()
                    for c in range(fx.Int32(tid), n_cand, fx.Int32(THREADS)):
                        key = fx.Uint32(
                            lds_ld(xs, SEL_BINS + fx.Int32(c) * 2).bitcast(fx.Int32)
                        )
                        if (key & fx.Uint32(mask)) == prefix:
                            digit = fx.Int32((key >> shift) & fx.Uint32(nb - 1))
                            lds_atomic_add(
                                skeys + digit, fx.Int32(1), syncscope="workgroup"
                            )
                    gpu.barrier()
                    # the bins from the top: per bins a thread, then waves, then the block
                    cs = [
                        (fx.Int32(tid) * per + q < nb).select(
                            lds_ld(
                                skeys,
                                fx.max(nb - 1 - (fx.Int32(tid) * per + q), fx.Int32(0)),
                            ),
                            fx.Int32(0),
                        )
                        for q in range(per)
                    ]
                    local = cs[0]
                    for q in range_constexpr(1, per):
                        local = local + cs[q]
                    inc = fx.coop.warp_inclusive_scan(
                        local, fx.ReductionOp.ADD, width=64
                    )
                    if lane == 63:
                        lds_st(keys, 256 + wave, inc)
                    gpu.barrier()
                    running = inc - local
                    for w in range_constexpr(WAVES):
                        running = running + (wave > w).select(
                            lds_ld(keys, 256 + w), fx.Int32(0)
                        )
                    for q in range_constexpr(per):
                        digit = nb - 1 - (fx.Int32(tid) * per + q)
                        if (running < remain) & (running + cs[q] >= remain):
                            lds_st(
                                keys,
                                280,
                                fx.Int32(prefix | (fx.Uint32(digit) << shift)),
                            )
                            lds_st(keys, 281, remain - running)
                        running = running + cs[q]
                    gpu.barrier()
                    return fx.Uint32(lds_ld(keys, 280)), lds_ld(keys, 281)

                if const_expr(level == 20):
                    prefix, remain = cand_pass(10, 10, prefix, remain)
                    prefix, remain = cand_pass(0, 10, prefix, remain)
                else:
                    prefix, remain = cand_pass(0, 8, prefix, remain)
                set_bases(n_win, fx.Int32(0))
                out_gt = count - remain
                n_cand_chunks = (n_cand + SEL_CHUNK - 1) // SEL_CHUNK
                for chunk in range(fx.Int32(0), n_cand_chunks, fx.Int32(1)):
                    chunk = fx.Int32(chunk)
                    idx = chunk_slots(chunk)
                    safe = [fx.min(c, n_cand - 1) for c in idx]
                    valid = [c < n_cand for c in idx]
                    item_keys = [
                        fx.Uint32(lds_ld(xs, SEL_BINS + c * 2).bitcast(fx.Int32))
                        for c in safe
                    ]
                    values = [
                        lds_ld(xs, SEL_BINS + c * 2 + 1).bitcast(fx.Int32) for c in safe
                    ]
                    gt = [v & (k > prefix) for v, k in zip(valid, item_keys)]
                    eq = [v & (k == prefix) for v, k in zip(valid, item_keys)]
                    gt_offsets, eq_offsets, _, _ = scan_pair(gt, eq, chunk & 1)
                    for j in range_constexpr(SEL_ITEMS):
                        if gt[j]:
                            lds_st(red, gt_offsets[j], values[j].bitcast(fx.Float32))
                        if eq[j] & (eq_offsets[j] < remain):
                            lds_st(
                                red,
                                out_gt + eq_offsets[j],
                                values[j].bitcast(fx.Float32),
                            )
                gpu.barrier()
                for batch in range_constexpr((topk + THREADS - 1) // THREADS):
                    j = tid + batch * THREADS
                    if (j >= n_win) & (j < count):
                        publish_slot(s, out_begin, j, lds_ld(red, j).bitcast(fx.Int32))

            def sel_digit(
                s, c, n_sl, level, shift, bits, lo, hi, n_chunks, prefix, remain
            ):
                """One radix digit of row s on its slice c (of n_sl; keys lo .. hi - 1 in
                LDS): this slice's histogram of its keys under ``prefix``, every
                slice's, the digit holding the remain-th key. Returns (digit, keys still
                needed in it, keys in it, keys above it, those in the slices before c,
                keys in it in the slices before c). From 4 rows the first digit's
                histogram is already published, and its totals zeroed."""
                top = shift + bits
                hbase = ((s * len(DIGITS) + level) * SEL_SLICES) * HIST_PAIRS
                if const_expr(not (PAIRED and level == 0)):
                    if tid < 64:
                        lds_st(sbins, tid, fx.Int32(0))
                        lds_st(stot, tid, fx.Int32(0))
                    gpu.barrier()
                    for chunk in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                        slots = chunk_slots(fx.Int32(chunk))
                        for j in range_constexpr(SEL_ITEMS):
                            key = fx.Uint32(lds_ld(skeys, slots[j]))
                            under = lo + slots[j] < hi
                            if const_expr(top < 32):
                                under = under & ((key >> top) == (prefix >> top))
                            if under:
                                lds_atomic_add(
                                    sbins
                                    + fx.Int32(
                                        (key >> shift) & fx.Uint32((1 << bits) - 1)
                                    ),
                                    fx.Int32(1),
                                    syncscope="workgroup",
                                )
                    gpu.barrier()
                    if tid < HIST_PAIRS:
                        put(
                            mb("sel_hist"),
                            hbase + c * HIST_PAIRS + tid,
                            lds_ld(sbins, 2 * tid) | (lds_ld(sbins, 2 * tid + 1) << 16),
                        )
                n_pairs = n_sl * HIST_PAIRS
                qs = [fx.Int32(tid) + q * THREADS for q in range(4)]
                got = poll(
                    [(mb("sel_hist"), hbase + fx.min(q, n_pairs - 1), 1) for q in qs],
                    batch=4,
                )
                words = [
                    (qs[q] < n_pairs).select(got[q][0], fx.Int32(0)) for q in range(4)
                ]
                for q in range_constexpr(4):
                    bin2 = (qs[q] % HIST_PAIRS) * 2
                    lds_atomic_add(
                        stot + bin2, words[q] & fx.Int32(0xFFFF), syncscope="workgroup"
                    )
                    lds_atomic_add(
                        stot + bin2 + 1,
                        (words[q] >> 16) & fx.Int32(0xFFFF),
                        syncscope="workgroup",
                    )
                gpu.barrier()
                # the digit holding the remain-th key: wave 0 scans from the top
                if wave == 0:
                    cnt = lds_ld(stot, 63 - fx.Int32(lane))
                    inc = fx.coop.warp_inclusive_scan(cnt, fx.ReductionOp.ADD, width=64)
                    above = inc - cnt
                    if (above < remain) & (inc >= remain):
                        lds_st(sbins, 64, 63 - fx.Int32(lane))
                        lds_st(sbins, 65, remain - above)
                        lds_st(sbins, 66, cnt)
                        lds_st(sbins, 67, above)
                gpu.barrier()
                digit = lds_ld(sbins, 64)
                rem = lds_ld(sbins, 65)
                in_bin = lds_ld(sbins, 66)
                a_all = lds_ld(sbins, 67)
                # the slices before c: their keys above the digit and in it
                mine_above = fx.Int32(0)
                mine_in = fx.Int32(0)
                for q in range_constexpr(4):
                    before = (qs[q] < n_pairs) & (qs[q] // HIST_PAIRS < c)
                    for h in range_constexpr(2):
                        bn = (qs[q] % HIST_PAIRS) * 2 + h
                        n_b = before.select(
                            (words[q] >> (16 * h)) & fx.Int32(0xFFFF), fx.Int32(0)
                        )
                        mine_above = mine_above + (bn > digit).select(n_b, fx.Int32(0))
                        mine_in = mine_in + (bn == digit).select(n_b, fx.Int32(0))
                sums = block_isums([mine_above, mine_in])
                return digit, rem, in_bin, a_all, sums[0], sums[1]

            def first_digit(s, c, bound, lo, hi, per):
                """From 4 rows: slice c's first-digit histogram (keys lo .. hi - 1, per
                a slice, of row s's bound) as the sum of its scoring runs' segments,
                published before its scores are read. The row's group of rows and so
                its runs are the scoring's: its rows are at consecutive positions, the
                group's last one bounds the run layout. Also zeroes the totals."""
                m = row_leads & ((fx.Int32(2) << s) - 1)
                lead = fx.Int32(0)
                for r in range_constexpr(S):
                    lead = (((m >> r) & 1) != 0).select(fx.Int32(r), lead)
                n_rows = fx.Int32((sizes >> (fx.Int64(lead) * 3)) & fx.Int64(7))
                bound_g = bound + lead + n_rows - 1 - s
                n_pg = (bound_g + index_block_size - 1) // index_block_size
                runs = fx.min(
                    fx.Int32(SCORE_CTAS), (n_pg + SCORE_PAGES - 1) // SCORE_PAGES
                )
                per_g = (n_pg + runs - 1) // runs
                even = runs == SCORE_CTAS

                p0 = lo // index_block_size
                r0 = even.select(((p0 + 1) * runs - 1) // n_pg, p0 // per_g)
                # thread: run r0 + tid / 32 (r0's when that run starts past the slice),
                # pair tid % 32 of the run's segment in this slice: its second when the
                # run starts in the slice before
                i = fx.Int32(tid) // HIST_PAIRS
                p_lo = even.select((r0 + i) * n_pg // runs, (r0 + i) * per_g)
                mine = p_lo <= (hi - 1) // index_block_size
                run = mine.select(r0 + i, r0)
                seg = (mine & (p_lo * index_block_size < lo)).select(
                    fx.Int32(1), fx.Int32(0)
                )
                w = poll(
                    [
                        (
                            mb("score_hist"),
                            ((s * G + run) * 2 + seg) * HIST_PAIRS
                            + fx.Int32(tid) % HIST_PAIRS,
                            1,
                        )
                    ],
                    batch=1,
                )[0][0]
                lds_st(red_i, tid, mine.select(w, fx.Int32(0)))
                if tid < 64:
                    lds_st(stot, tid, fx.Int32(0))
                gpu.barrier()
                # a bin's sum is at most the slice's keys (< 2^16): pairs add as words
                if tid < HIST_PAIRS:
                    total = lds_ld(red_i, tid)
                    for q in range_constexpr(1, THREADS // HIST_PAIRS):
                        total = total + lds_ld(red_i, q * HIST_PAIRS + tid)
                    put(
                        mb("sel_hist"),
                        (s * len(DIGITS) * SEL_SLICES + c) * HIST_PAIRS + tid,
                        total,
                    )

            def sel_place(st, p, i):
                """Place p of the slice's row := key i (its block id from LDS)."""
                s, req, out_begin, lo, hi, first_page = st[:6]
                blk = lds_ld(hist, SBT + i // index_block_size - first_page)
                publish_slot(
                    s, out_begin, p, blk * index_block_size + i % index_block_size
                )

            def sel_stage(st, chunk, in_base, e):
                """Entry e of a chunk's compacted keys under the prefix: a tie placed
                (at 32 bits) or a candidate staged for the row's first slice."""
                s, req, out_begin, lo, hi, first_page, low, lp, low_mask = st[:9]
                prefix_i, decided, remain, above_all, in_before = st[9:]
                word = lds_ld(skeys, chunk * SEL_CHUNK + e)
                pos = lo + chunk * SEL_CHUNK + (word & fx.Int32(0xFFF))
                rk = in_before + in_base + e
                if (decided == 32) & (rk < remain):
                    sel_place(st, above_all + rk, pos)
                if decided != 32:
                    blk = lds_ld(hist, SBT + pos // index_block_size - first_page)
                    put2(
                        mb("sel_cand"),
                        (s * SEL_CAP + rk) * 2,
                        prefix_i | fx.Int32(fx.Uint32(word) >> 12),
                        blk * index_block_size + pos % index_block_size,
                    )

            def sel_place_chunk(st, chunk):
                """A chunk's keys under the prefix, compacted in place in its own LDS
                words (a word each: the key's bits below the prefix, its place in the
                chunk), then staged or placed a thread an entry; its keys above the
                prefix listed after the slice's keys."""
                s, req, out_begin, lo, hi, first_page, low, lp, low_mask = st[:9]
                slots = chunk_slots(chunk)
                ks = [fx.Uint32(lds_ld(skeys, q)) for q in slots]
                valid = [lo + q < hi for q in slots]
                above_f = [v & ((k >> low) > lp) for v, k in zip(valid, ks)]
                in_f = [v & ((k >> low) == lp) for v, k in zip(valid, ks)]
                a_offs, i_offs, _, in_end = scan_pair(above_f, in_f, chunk & 1)
                in_base = lds_ld(keys, 273 + (chunk & 1) * 2)
                gpu.barrier()
                for j in range_constexpr(SEL_ITEMS):
                    if above_f[j]:
                        lds_st(skeys, SALIST + a_offs[j], lo + slots[j])
                    if in_f[j]:
                        lds_st(
                            skeys,
                            chunk * SEL_CHUNK + i_offs[j] - in_base,
                            (fx.Int32(ks[j] & low_mask) << 12)
                            | (slots[j] - chunk * SEL_CHUNK),
                        )
                gpu.barrier()
                for e in range(fx.Int32(tid), in_end - in_base, fx.Int32(THREADS)):
                    sel_stage(st, chunk, in_base, fx.Int32(e))

            # Rows' slices interleave (slice c of row s is task c * S + s) when every
            # row has at most G / S slices, so short rows' slices share one compact
            # range of CTAs; else each row's take a run of SEL_SLICES tasks, whole rows
            # a round of the grid, so a row never waits on a slice queued behind another
            # row's.
            max_slices = fx.Int32(0)
            for r in range_constexpr(S):
                max_slices = fx.max(
                    max_slices, select_slices(index_bound(r), fx.min, fx.max)[0]
                )
            interleave = _uniform(max_slices) * S <= G
            for t in range(start("index_slice"), S * SEL_SLICES, G):
                t = fx.Int32(t)
                s = interleave.select(t % S, t // SEL_SLICES)
                c = interleave.select(t // S, t % SEL_SLICES)
                bound = _uniform(index_bound(s))
                n_sl, per = select_slices(bound, fx.min, fx.max)
                # the timeline's task: the row's slice, whatever the CTA
                t = s * SEL_SLICES + c
                if (bound > 0) & (c < n_sl):
                    stamp("index_slice", t, 0)
                    # the selection's and split's code, while the scores are pending
                    # (not in TP8 dense launches above 4 rows: there it costs more than
                    # it saves)
                    if const_expr(S <= 4 or not dense_experts or W != 8):
                        warm_code(65536)
                    req = _uniform(index_request(s))
                    count = fx.min(fx.Int32(topk), bound)
                    out_begin, _ = row_index_bounds(s)
                    lo = c * per
                    hi = fx.min(lo + per, bound)
                    n_chunks = (per + SEL_CHUNK - 1) // SEL_CHUNK
                    first_page = lo // index_block_size
                    n_pages = (
                        hi + index_block_size - 1
                    ) // index_block_size - first_page
                    for pg in range(fx.Int32(tid), n_pages, fx.Int32(THREADS)):
                        pg = fx.Int32(pg)
                        lds_st(
                            hist,
                            SBT + pg,
                            index_block(req, (first_page + pg) * index_block_size),
                        )

                    def stage_scores(chunk, between=None):
                        """A chunk of the slice's scores, as radix keys in LDS."""
                        slots = chunk_slots(chunk)
                        vals = poll(
                            [
                                (
                                    mb("index_scores"),
                                    s * index_max_seq
                                    + fx.max(fx.min(lo + q, hi - 1), lo),
                                    1,
                                )
                                for q in slots
                            ],
                            batch=SEL_ITEMS,
                            between=between,
                        )
                        for j in range_constexpr(SEL_ITEMS):
                            lds_st(skeys, slots[j], radix_key(vals[j][0]))

                    chunk0 = 0
                    if const_expr(PAIRED):
                        # the first chunk's scores load while the first digit is summed
                        stage_scores(
                            fx.Int32(0), lambda: first_digit(s, c, bound, lo, hi, per)
                        )
                        chunk0 = 1
                    for chunk in range(fx.Int32(chunk0), n_chunks, fx.Int32(1)):
                        stage_scores(fx.Int32(chunk))
                    gpu.barrier()
                    stamp("index_slice", t, 1)
                    prefix, remain = fx.Uint32(0), count
                    above_before, above_all = fx.Int32(0), fx.Int32(0)
                    in_before, in_all = fx.Int32(0), fx.Int32(0)
                    done, decided = fx.Int32(0), fx.Int32(0)
                    for level, (shift, bits) in enumerate(DIGITS):
                        if done == 0:
                            digit, remain, in_all, a_all, a_before, in_before = (
                                sel_digit(
                                    s,
                                    c,
                                    n_sl,
                                    level,
                                    shift,
                                    bits,
                                    lo,
                                    hi,
                                    n_chunks,
                                    prefix,
                                    remain,
                                )
                            )
                            above_all = above_all + a_all
                            above_before = above_before + a_before
                            prefix = prefix | (fx.Uint32(digit) << shift)
                            if const_expr(shift == 0):
                                done = fx.Int32(1)
                                decided = fx.Int32(32)
                            elif const_expr(32 - shift in (12, 24)):
                                fits = in_all <= SEL_CAP
                                done = fits.select(fx.Int32(1), fx.Int32(0))
                                decided = fits.select(fx.Int32(32 - shift), fx.Int32(0))
                    stamp("index_slice", t, 2)
                    # this slice's keys above the threshold's prefix take their places;
                    # its keys under it are ties placed, or candidates staged
                    low = fx.Uint32(32 - decided)
                    lp = prefix >> low
                    # the key bits below the prefix (none at 32 bits)
                    low_mask = (decided < 32).select(
                        fx.Uint32(0xFFFFFFFF)
                        >> fx.Uint32(fx.min(decided, fx.Int32(31))),
                        fx.Uint32(0),
                    )

                    st = (
                        s,
                        req,
                        out_begin,
                        lo,
                        hi,
                        first_page,
                        low,
                        lp,
                        low_mask,
                        fx.Int32(prefix.bitcast(fx.Int32)),
                        decided,
                        remain,
                        above_all,
                        in_before,
                    )

                    set_bases(fx.Int32(0), fx.Int32(0))
                    for pchunk in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                        sel_place_chunk(st, fx.Int32(pchunk))
                    gpu.barrier()
                    n_above = lds_ld(keys, 272 + (n_chunks & 1) * 2)
                    for pe in range(fx.Int32(tid), n_above, fx.Int32(THREADS)):
                        sel_place(
                            st,
                            above_before + fx.Int32(pe),
                            lds_ld(skeys, SALIST + fx.Int32(pe)),
                        )
                    stamp("index_slice", t, 3)
                    # the row's first slice resolves the candidates
                    if (c == 0) & (decided < 32):
                        gpu.barrier()
                        for c0 in range(fx.Int32(0), in_all, fx.Int32(4 * THREADS)):
                            cs = [
                                fx.Int32(c0) + fx.Int32(tid) + q * THREADS
                                for q in range(4)
                            ]
                            kv = poll(
                                [
                                    (
                                        mb("sel_cand"),
                                        (s * SEL_CAP + fx.min(ci, in_all - 1)) * 2,
                                        2,
                                    )
                                    for ci in cs
                                ],
                                batch=4,
                            )
                            for q in range_constexpr(4):
                                if cs[q] < in_all:
                                    lds_st(
                                        xs,
                                        SEL_BINS + cs[q] * 2,
                                        kv[q][0].bitcast(fx.Float32),
                                    )
                                    lds_st(
                                        xs,
                                        SEL_BINS + cs[q] * 2 + 1,
                                        kv[q][1].bitcast(fx.Float32),
                                    )
                        gpu.barrier()
                        stamp("index_slice", t, 5)
                        if decided == 12:
                            resolve(
                                s, req, out_begin, count, 20, prefix, remain, in_all
                            )
                        else:
                            resolve(s, req, out_begin, count, 8, prefix, remain, in_all)
                    gpu.barrier()
                    stamp("index_slice", t, 4)

        # ================================== 5. sparse MLA split: 32 keys x 8 heads
        (x_out, positions, slot_mapping, sparse_kv_indptr, kv_cache, pe_cache, indices, rope_cos, rope_sin, g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o, s_o, w_r, bias, w_ug, s_ug, w_dn, s_dn, scratch, sym, timeline_buf, block_table, req_ids, out_indices, ix_wk, ix_sk, ix_ww, ix_wq, ix_sq, ix_gk, ix_bk) = reload_args()  # fmt: skip
        if const_expr(RELOAD_ARGS and x_pushed):
            xin = (
                sym
                + fx.Int64(SY["xin"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
        r_kv = _rsrc(kv_cache)
        r_pe = _rsrc(pe_cache)
        r_idx = _rsrc(indices)

        def split_keys(t, s):
            """(nkeys, sparse) of sample s; wave 0 writes this split's cache rows to LDS
            (without the indexer)."""
            index_base, nkeys, sparse = split_bounds(s)
            if wave == 0:
                if lane < SPLIT_KEYS:
                    k_pos = t * SPLIT_KEYS + lane
                    k_cl = (k_pos < nkeys).select(k_pos, 0)
                    if const_expr(use_atom_kv_cache):
                        idx = fx.Int32(0)
                        if sparse:
                            idx = fx.Int32(
                                bo.buffer_load(
                                    r_idx, index_base + k_cl, vec_width=1, dtype=T.i32
                                )
                            )
                        lds_st(attn_keys, lane, idx)
                    else:
                        idx = fx.Int32(
                            bo.buffer_load(
                                r_idx, index_base + k_cl, vec_width=1, dtype=T.i32
                            )
                        )
                        lds_st(attn_keys, lane, sparse.select(idx, k_cl))
            return nkeys, sparse

        # a row this launch appends, staged by split as bf16 pairs: latent then k_pe;
        # with the indexer, before the selection's wait, in the reduction and output
        # regions (q is staged by then), else in the (still unused) q region
        NEW_ROW_WORDS = (KV_LORA + PE_DIM) // 2
        assert S * NEW_ROW_WORDS <= KT_OFF, "new rows are staged in split's q region"
        NEW_ROWS_OFF = (RED_OFF + 3) // 4 * 4
        assert not with_indexer or (
            NEW_ROWS_OFF + S * NEW_ROW_WORDS <= OUT_OFF + OUT_WORDS
        ), "new rows are staged in split's reduction region"

        def stage_new_rows(base):
            """Every row this launch appends, from the cache tasks' kvnew / penew pairs,
            as bf16 pairs at LDS ``base``."""
            pe_word = fx.min(tid - KV_LORA // 2, PE_DIM // 2 - 1)
            is_kv = tid < KV_LORA // 2
            got = poll(
                [
                    (
                        is_kv.select(fx.Int64(SC["kvnew"]), fx.Int64(SC["penew"]))
                        + scratch,
                        is_kv.select(r * KV_LORA + tid * 2, r * PE_DIM + pe_word * 2),
                        2,
                    )
                    for r in range(S)
                ]
            )
            if tid < NEW_ROW_WORDS:
                for r in range_constexpr(S):
                    a0 = got[r][0].bitcast(fx.Float32)
                    a1 = got[r][1].bitcast(fx.Float32)
                    lds_st(base, r * NEW_ROW_WORDS + tid, bf16_pair(a0, a1))

        def copy_new_rows(base, news):
            """This wave's keys that are rows this launch appends: their staged rows
            (at LDS ``base``) to the keys' tile rows."""
            for jj in range_constexpr(KPW):
                j = wave * KPW + jj
                sn = news[jj]
                if sn >= 0:
                    row = base + (sn * NEW_ROW_WORDS + lane * 4)
                    fx.ptr_store(
                        fx.ptr_load(row, result_type=v4f),
                        ktile + (j * KS + lane * 4),
                    )
                    if lane < PE_DIM // 2:
                        lds_st(
                            petile,
                            j * PS + lane,
                            lds_ld(base, sn * NEW_ROW_WORDS + KV_LORA // 2 + lane),
                        )

        def new_rows():
            """(writes cache, slot) of every row of this launch, for gather_kv."""
            if const_expr(use_atom_kv_cache):
                return [(row_writes_cache(s), row_slot(s)) for s in range(S)]
            return None

        def gather_kv(rows, early=None, staged=None):
            """Each wave copies its 8 keys' KV latent (1 KB) + k_pe (128 B) rows into
            the LDS tiles: the cache rows of all 8 keys are loaded at once (or were,
            ``early``: (the keys a lane, their rows)), then, if any key is a row this
            launch appends, the CTA polls those rows from the cache tasks' kvnew /
            penew pairs (their loads issued behind the cache rows'; or they were,
            ``staged``: their LDS base), then the rest is stored."""
            if const_expr(early is None):
                krows = [lds_ld(attn_keys, wave * KPW + jj) for jj in range(KPW)]
                raws = split_kv_loads(krows, r_kv, r_pe)
            else:
                krows = [read_lane_i32(early[0], jj) for jj in range(KPW)]
                raws = early[1]
            news = []
            if const_expr(use_atom_kv_cache):
                for jj in range_constexpr(KPW):
                    sn = fx.Int32(-1)
                    for new_s in range_constexpr(S):
                        active, slot = rows[new_s]
                        sn = (active & (krows[jj] == slot)).select(fx.Int32(new_s), sn)
                    news.append(sn)
            else:
                news = [
                    (krows[jj] >= pos0).select(krows[jj] - pos0, fx.Int32(-1))
                    for jj in range(KPW)
                ]
            if const_expr(staged is not None):
                copy_new_rows(staged, news)
            else:
                # every row this launch appends, if a key of this task is one: polled
                # once by the whole CTA into the (still unused) q region, then copied
                # to the keys' tile rows
                any_new = news[0] >= 0
                for jj in range_constexpr(1, KPW):
                    any_new = any_new | (news[jj] >= 0)
                if lane == 0:
                    lds_st(misc, wave, any_new.select(fx.Float32(1.0), fx.Float32(0.0)))
                gpu.barrier()
                block_new = lds_ld(misc, 0)
                for w in range_constexpr(1, WAVES):
                    block_new = block_new + lds_ld(misc, w)
                if block_new > 0.0:
                    stage_new_rows(xs)
                    gpu.barrier()
                    copy_new_rows(xs, news)
            for jj in range_constexpr(KPW):
                j = wave * KPW + jj
                kv, pe = raws[jj]
                if news[jj] < 0:
                    if const_expr(cache_fp8):
                        fx.ptr_store(
                            _fp8_to_bf16x8(kv[0], kv[1]).bitcast(fx.Float32),
                            ktile + (j * KS + lane * 4),
                        )
                        if lane < PE_DIM // 8:
                            fx.ptr_store(
                                _fp8_to_bf16x8(pe[0], pe[1]).bitcast(fx.Float32),
                                petile + (j * PS + lane * 4),
                            )
                    else:
                        fx.ptr_store(
                            kv.bitcast(fx.Float32), ktile + (j * KS + lane * 4)
                        )
                        if lane < PE_DIM // 2:
                            lds_st(petile, j * PS + lane, pe)

        N_HEAD_GROUPS = H // WAVES

        def wait_q(tt, s):
            N_PE_T = PE_DIM // Q_B_TILE
            hint_wait(
                N_UK + H * N_PE_T + 1,
                lambda k: (
                    (k < N_UK).select(
                        fx.Int64(SC["q_lat"]),
                        (k < N_UK + H * N_PE_T).select(
                            fx.Int64(SC["q_pe"]), fx.Int64(SC["penew"])
                        ),
                    )
                    + scratch,
                    (k < N_UK).select(
                        (s * H + k // UK_PER_HEAD) * KV_LORA
                        + (k % UK_PER_HEAD) * UK_TILE
                        + UK_TILE
                        - 1,
                        (k < N_UK + H * N_PE_T).select(
                            (s * H + (k - N_UK) // N_PE_T) * PE_DIM
                            + ((k - N_UK) % N_PE_T) * Q_B_TILE
                            + Q_B_TILE
                            - 1,
                            s * PE_DIM + PE_DIM - 1,
                        ),
                    ),
                ),
                mark=("split", tt),
            )

        def stage_q(s):
            """q of all heads -> bf16 Q[h][576] (words h * 288 + d / 2): latent 512 then pe 64."""
            NQ = H * KV_LORA // 4 // THREADS
            tpe = fx.min(tid, H * PE_DIM // 4 - 1)
            qv = poll(
                [
                    (mb("q_lat"), (s * H * KV_LORA + (tid + i * THREADS) * 4) // 2, 2)
                    for i in range(NQ)
                ]
                + [(mb("q_pe"), (s * H * PE_DIM + tpe * 4) // 2, 2)]
            )
            for i in range_constexpr(NQ):
                w4 = tid + i * THREADS
                qw = (w4 // (KV_LORA // 4)) * QS + (w4 % (KV_LORA // 4)) * 2
                lds_st(xs, qw, qv[i][0].bitcast(fx.Float32))
                lds_st(xs, qw + 1, qv[i][1].bitcast(fx.Float32))
            if tid < H * PE_DIM // 4:
                hh = tid // (PE_DIM // 4)
                qw = hh * QS + KV_LORA // 2 + (tid % (PE_DIM // 4)) * 2
                lds_st(xs, qw, qv[NQ][0].bitcast(fx.Float32))
                lds_st(xs, qw + 1, qv[NQ][1].bitcast(fx.Float32))

        def split_wave_keys(t, s):
            """(nkeys, sparse, this wave's keys: lane j < KPW key j) of sample s's
            selection; each wave polls its own keys."""
            index_base, index_end = row_index_bounds(s)
            nkeys = index_end - index_base
            sparse = index_end > index_base
            k_pos = t * SPLIT_KEYS + wave * KPW + fx.min(lane, KPW - 1)
            idx = (k_pos < nkeys).select(k_pos, 0)
            if sparse:
                idx = get(mb("indices"), s * topk + idx)
            return nkeys, sparse, idx

        for tt in range(start("split"), S * N_HEAD_GROUPS * N_SPLIT, G):
            tt = fx.Int32(tt)
            stamp("split", tt, 0)
            s = tt // (N_HEAD_GROUPS * N_SPLIT)
            head_group = (tt // N_SPLIT) % N_HEAD_GROUPS
            t = tt % N_SPLIT
            h = head_group * WAVES + wave
            rows = new_rows()
            if const_expr(with_indexer):
                # q and this launch's rows are staged before the selection's wait;
                # once a wave's keys land it issues their cache rows
                wait_q(tt, s)
                stage_q(s)
                stage_new_rows(xs + NEW_ROWS_OFF)
                stamp("split", tt, 2)
                gpu.barrier()
                nkeys, sparse, wave_keys = split_wave_keys(t, s)
                gather_kv(
                    rows,
                    (
                        wave_keys,
                        split_kv_loads(
                            [read_lane_i32(wave_keys, jj) for jj in range(KPW)],
                            r_kv,
                            r_pe,
                        ),
                    ),
                    staged=xs + NEW_ROWS_OFF,
                )
            else:
                if const_expr(SPLIT_EARLY):  # the tile's rows are already loaded
                    _, nkeys, sparse = split_bounds(s)
                else:
                    nkeys, sparse = split_keys(t, s)
                gpu.barrier()
                # before waiting for q: the cache tasks end before uk
                gather_kv(rows, split_early if const_expr(SPLIT_EARLY) else None)
                wait_q(tt, s)
                stage_q(s)
                stamp("split", tt, 2)
            gpu.barrier()
            stamp("split", tt, 5)
            # scores = K Q^T on MFMA.  The 64-key tile maps the eight waves to
            # (four row groups, two K halves); the batch-8 32-key tile uses two
            # row groups and four waves per K half.  All waves subsequently own
            # one attention head for softmax and P@V.
            hn = head_group * WAVES + fx.min(lane % 16, WAVES - 1)
            rgk = wave % (SPLIT_KEYS // 16)
            c = fx.Vector.filled(4, 0.0, fx.Float32)
            for st in range_constexpr(QK_DIM // 32 // 2):
                kst = ((wave // (SPLIT_KEYS // 16)) % 2) * (QK_DIM // 32 // 2) + st
                key = rgk * 16 + lane % 16
                kw = (kst < KV_LORA // 32).select(
                    KT_OFF + key * KS + kst * 16,
                    PT_OFF + key * PS + (kst - KV_LORA // 32) * 16,
                )
                a = fx.ptr_load(xs + (kw + (lane // 16) * 4), result_type=v4f).bitcast(
                    fx.BFloat16
                )
                b = fx.ptr_load(
                    xs + (hn * QS + kst * 16 + (lane // 16) * 4), result_type=v4f
                ).bitcast(fx.BFloat16)
                c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
            if const_expr(SPLIT_KEYS == 64) or wave < 4:
                fx.ptr_store(c, red + (wave * 64 + lane) * 4)
            gpu.barrier()
            stamp("split", tt, 6)
            # split-local softmax: wave h, lane = key j (score = sum of the two K halves)
            kidx = t * SPLIT_KEYS + lane
            valid = (lane < SPLIT_KEYS) & (kidx < nkeys)
            r16 = lane % 16
            cl = split_score_column(wave, lane)
            key_rg = fx.min(lane // 16, SPLIT_KEYS // 16 - 1)
            half_stride = SPLIT_KEYS // 16
            raw = lds_ld(red, (key_rg * 64 + cl) * 4 + r16 % 4) + lds_ld(
                red, ((key_rg + half_stride) * 64 + cl) * 4 + r16 % 4
            )
            sc_v = valid.select(raw * scale, fx.Float32(NEG))
            m = wave_max(sc_v)
            p = valid.select(_exp(sc_v - m), fx.Float32(0.0))
            lsum = wave_sum(p)
            p_n = _xshfl(p, 1)
            if (lane < SPLIT_KEYS) & (lane % 2 == 0):
                lds_st(pl, h * (SPLIT_KEYS // 2) + lane // 2, bf16_pair(p, p_n))
            gpu.barrier()
            stamp("split", tt, 3)
            # O = P V on MFMA: heads M, keys K, latent dims N.  Each V word holds
            # a dim pair (even dim low), so one read feeds two MFMAs (even / odd dims):
            # each wave owns 2 groups of 32 dims.  V is read key-strided from the tile.
            for g in range_constexpr(KV_LORA // 32 // WAVES):
                dw = (
                    wave * (KV_LORA // 32 // WAVES) + g
                ) * 16 + lane % 16  # dim pair word
                c0 = fx.Vector.filled(4, 0.0, fx.Float32)
                c1 = fx.Vector.filled(4, 0.0, fx.Float32)
                for js in range_constexpr(SPLIT_KEYS // 32):
                    a = fx.ptr_load(
                        pl + (hn * (SPLIT_KEYS // 2) + js * 16 + (lane // 16) * 4),
                        result_type=v4f,
                    ).bitcast(fx.BFloat16)
                    ws = [
                        fx.ptr_load(
                            ktile + ((js * 32 + (lane // 16) * 8 + i) * KS + dw)
                        ).bitcast(fx.Int32)
                        for i in range(8)
                    ]
                    w_lo = [
                        (ws[2 * i] & 0xFFFF) | (ws[2 * i + 1] << 16) for i in range(4)
                    ]
                    w_hi = [
                        fx.Int32(fx.Uint32(ws[2 * i]) >> 16) | (ws[2 * i + 1] & -65536)
                        for i in range(4)
                    ]
                    b0 = fx.Vector.from_elements(w_lo, fx.Int32).bitcast(fx.BFloat16)
                    b1 = fx.Vector.from_elements(w_hi, fx.Int32).bitcast(fx.BFloat16)
                    c0 = fx.Vector(
                        rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b0, c0])
                    )
                    c1 = fx.Vector(
                        rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b1, c1])
                    )
                if lane < 32:
                    for e in range_constexpr(4):
                        hh = split_acc_head(head_group, lane // 16, e)
                        put_bf(
                            mb("sp_acc"),
                            ((s * N_SPLIT + t) * H + hh) * KV_LORA + dw * 2,
                            [c0[e], c1[e]],
                        )
            if lane == 0:  # written last: the merge's readiness hint
                put(mb("sp_m"), (s * N_SPLIT + t) * H + h, m)
                put(mb("sp_l"), (s * N_SPLIT + t) * H + h, lsum)
            stamp("split", tt, 4)

        # ========================== 6. split merge + W_UV: o = W_UV (softmax . KV)
        # 4 row groups x 8 chunks: 2 waves per row group, 4 chunks each
        (x_out, positions, slot_mapping, sparse_kv_indptr, kv_cache, pe_cache, indices, rope_cos, rope_sin, g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o, s_o, w_r, bias, w_ug, s_ug, w_dn, s_dn, scratch, sym, timeline_buf, block_table, req_ids, out_indices, ix_wk, ix_sk, ix_ww, ix_wq, ix_sq, ix_gk, ix_bk) = reload_args()  # fmt: skip
        if const_expr(RELOAD_ARGS and x_pushed):
            xin = (
                sym
                + fx.Int64(SY["xin"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
        r_wuv, r_suv = _rsrc(w_uv), _rsrc(s_uv)
        UV_NKC = KV_LORA // 64
        UV_R = UV_TILE // 16
        UV_WPR = WAVES // UV_R
        UV_UNITS = UV_NKC // (attention_k_chunks_per_unit * UV_WPR)
        for tt in range(start("uv"), S * N_UV, G):
            tt = fx.Int32(tt)
            stamp("uv", tt, 0)
            s = tt // N_UV  # sample
            t = tt % N_UV  # global 64-row tile
            local_t = dcp_local_uv_tile(t, L)
            head = t // (V_DIM // UV_TILE)

            def u_uv(c):
                kc = ((wave % UV_WPR) * UV_UNITS + c) * attention_k_chunks_per_unit
                return unit_attention(
                    r_wuv,
                    r_suv,
                    local_t * UV_R + wave // UV_WPR,
                    kc,
                    UV_NKC,
                    KV_LORA,
                    128,
                    attention_b_word(kc * 64),
                    scale_rows=uv_scale_rows,
                    coef=(
                        (lambda: lds_ld(misc, N_SPLIT + 2 + kc // 2))
                        if const_expr(attention_ptpc)
                        else None
                    ),
                )

            pre = [u_uv(c) for c in range(UV_UNITS)]
            hint_wait(
                N_SPLIT,
                lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head),
                mark=("uv", tt),
            )
            pre_poll(N_SPLIT, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head))
            stamp("uv", tt, 5)
            # Four 16-split quarters keep each poll and its live payload bounded.
            # Every thread owns one (quarter, half-of-latent) pair and processes
            # its two latent-pair positions sequentially; red holds quarter sums.
            SPH = N_SPLIT // 4
            dp_lo = tid % (KV_LORA // 4)
            hf = tid // (KV_LORA // 4)
            spi = fx.min(lane, N_SPLIT - 1)
            ml = (s * N_SPLIT + spi) * H + head
            ml_got = poll([(mb("sp_m"), ml, 1), (mb("sp_l"), ml, 1)])
            if wave == 0:  # per-split weights exp(m - M) / L for this head -> misc[sp]
                ok_sp = lane < N_SPLIT
                m_sp = ok_sp.select(ml_got[0][0].bitcast(fx.Float32), fx.Float32(NEG))
                l_sp = ok_sp.select(ml_got[1][0].bitcast(fx.Float32), fx.Float32(0.0))
                m_all = wave_max(m_sp)
                w_sp = _exp(m_sp - m_all)
                den = wave_sum(l_sp * w_sp)
                if ok_sp:
                    lds_st(
                        misc,
                        lane,
                        w_sp * (den > 0.0).select(_rcp(den), fx.Float32(0.0)),
                    )
                if lane == 0:
                    lds_st(misc, N_SPLIT, m_all)
                    lds_st(misc, N_SPLIT + 1, den)
            stamp("uv", tt, 2)
            gpu.barrier()
            for dh in range_constexpr(2):
                dp = dp_lo + dh * (KV_LORA // 4)
                got = poll(
                    [
                        (
                            mb("sp_acc"),
                            ((s * N_SPLIT + hf * SPH + j) * H + head) * (KV_LORA // 2)
                            + dp,
                            1,
                        )
                        for j in range(SPH)
                    ]
                )
                o0 = fx.Float32(0.0)
                o1 = fx.Float32(0.0)
                for j in range_constexpr(SPH):
                    wj = lds_ld(misc, hf * SPH + j)
                    a0, a1 = bf2_f32(got[j][0])
                    o0 = o0 + a0 * wj
                    o1 = o1 + a1 * wj
                lds_st(red, (hf * (KV_LORA // 2) + dp) * 2, o0)
                lds_st(red, (hf * (KV_LORA // 2) + dp) * 2 + 1, o1)
            gpu.barrier()
            uv_valid = tid < KV_LORA // 2
            uv0, uv1 = fx.Float32(0.0), fx.Float32(0.0)
            if uv_valid:
                o0 = fx.Float32(0.0)
                o1 = fx.Float32(0.0)
                for q in range_constexpr(4):
                    o0 = o0 + lds_ld(red, (q * (KV_LORA // 2) + tid) * 2)
                    o1 = o1 + lds_ld(red, (q * (KV_LORA // 2) + tid) * 2 + 1)
                uv0, uv1 = bf16_round(o0), bf16_round(o1)
                if const_expr(not attention_ptpc):
                    lds_st(xs, tid, bf16_pair(uv0, uv1))
            if const_expr(attention_ptpc):
                uv_max = wave_max(
                    uv_valid.select(
                        fx.max(fmath.absf(uv0), fmath.absf(uv1)), fx.Float32(0.0)
                    )
                )
                if (lane == 0) & (wave < KV_LORA // 128):
                    lds_st(
                        misc,
                        N_SPLIT + 2 + wave,
                        (uv_max == 0.0).select(
                            fx.Float32(1.0), uv_max * (1.0 / FP8_MAX)
                        ),
                    )
            gpu.barrier()
            if const_expr(attention_ptpc):
                if uv_valid:
                    uv_scale = lds_ld(misc, N_SPLIT + 2 + tid // 64)
                    uv_rcp = 1.0 / uv_scale
                    st_ptpc(
                        tid * 2,
                        div_rn(uv0, uv_scale, uv_rcp),
                        div_rn(uv1, uv_scale, uv_rcp),
                    )
                gpu.barrier()
            owner = dcp_merge_latent(s, t)
            if owner:
                acc = run_units(u_uv, UV_UNITS, UV_UNITS, pre)
                if const_expr(attention_ptpc):
                    reduce_rows(
                        UV_R,
                        acc,
                        emit_ptpc_scalar(UV_TILE, ld_f32(r_suv, 0)),
                    )
                else:
                    reduce_rows(UV_R, acc, emit_out(UV_TILE))
                stamp("uv", tt, 3)
                gpu.barrier()
                if tid < UV_TILE // 4:
                    r = tid * 4
                    put_bf(
                        mb("o"),
                        s * O_K + local_t * UV_TILE + r,
                        [lds_ld(outs, r + j) for j in range(4)],
                    )
            stamp("uv", tt, 4)

        # ====================== 7. W_o + attention TP peer reduce + residual -> a
        # 2 row groups x 32 chunks: 4 waves per row group, 8 chunks each
        (x_out, positions, slot_mapping, sparse_kv_indptr, kv_cache, pe_cache, indices, rope_cos, rope_sin, g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o, s_o, w_r, bias, w_ug, s_ug, w_dn, s_dn, scratch, sym, timeline_buf, block_table, req_ids, out_indices, ix_wk, ix_sk, ix_ww, ix_wq, ix_sq, ix_gk, ix_bk) = reload_args()  # fmt: skip
        if const_expr(RELOAD_ARGS and x_pushed):
            xin = (
                sym
                + fx.Int64(SY["xin"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
        r_wo, r_so = _rsrc(w_o), _rsrc(s_o)
        O_NKC = O_K // 64
        O_R = ROW_TILE // 16
        O_WPR = WAVES // O_R
        O_UNITS = O_NKC // (attention_k_chunks_per_unit * O_WPR)
        for t in range(start("o"), N_ROW_TILES, G):
            t = fx.Int32(t)
            stamp("o", t, 0)

            def u_o(c):
                kc = ((wave % O_WPR) * O_UNITS + c) * attention_k_chunks_per_unit
                return unit_attention(
                    r_wo,
                    r_so,
                    t * O_R + wave // O_WPR,
                    kc,
                    O_NKC,
                    O_K,
                    128,
                    attention_b_word(n_sel() * O_K + kc * 64),
                    shuffled=fp8_shuffled,
                    cm=AT_CM,
                )

            pre = [u_o(c) for c in range(O_UNITS)]
            hint_wait(
                S * N_UV,
                lambda k: (
                    mb("o"),
                    (k // N_UV) * O_K + (k % N_UV) * UV_TILE + UV_TILE - 1,
                ),
                mark=("o", t),
            )
            if const_expr(attention_ptpc):
                o_scales = stage_x_pairs_ptpc("o", S, O_K, lambda k: k)
            else:
                stage_x_pairs("o", S * O_K, lambda k: k)
            stamp("o", t, 2)
            gpu.barrier()
            acc = run_units(u_o, O_UNITS, O_UNITS, pre)
            if const_expr(attention_ptpc):
                reduce_rows(
                    O_R,
                    acc,
                    emit_ptpc(ROW_TILE, t * ROW_TILE, r_so, o_scales),
                )
            else:
                reduce_rows(O_R, acc, emit_out(ROW_TILE))
            stamp("o", t, 3)
            gpu.barrier()

            def resid_h(s, row):
                if const_expr(x_pushed):
                    return bf2_f32(
                        poll([(xin, (s * HIDDEN + row) // 2, 1)], "one-as")[0][0]
                    )
                w = fx.Vector.from_elements(
                    [
                        fx.Int32(
                            bo.buffer_load(
                                r_h, (s * HIDDEN + row) // 2, vec_width=1, dtype=T.i32
                            )
                        )
                    ],
                    fx.Int32,
                )
                v = w.bitcast(fx.BFloat16).to(fx.Float32)
                return v[0], v[1]

            peer_reduce(
                "attn",
                t,
                resid_h,
                lambda s, row, v0, v1: put_bf(mb("a"), s * HIDDEN + row, [v0, v1]),
            )
            stamp("o", t, 4)

        # ====== 8. post-attn RMSNorm -> router scores + this task's FP8 activation blocks
        # One sample per CTA: 1 row group x 96 chunks (bf16), 8 waves split K.
        # S > 1 gets S times as many independent router CTAs instead of serializing
        # every sample's normalization and output columns inside one CTA.
        # at 8 rows the MoE stages' wave index, and so their per-wave address terms,
        # are scalars: the 8-row build then fits in VGPRs
        if const_expr(S > 4):
            wave = _uniform(tid // 64)
        (x_out, positions, slot_mapping, sparse_kv_indptr, kv_cache, pe_cache, indices, rope_cos, rope_sin, g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o, s_o, w_r, bias, w_ug, s_ug, w_dn, s_dn, scratch, sym, timeline_buf, block_table, req_ids, out_indices, ix_wk, ix_sk, ix_ww, ix_wq, ix_sq, ix_gk, ix_bk) = reload_args()  # fmt: skip
        if const_expr(RELOAD_ARGS and x_pushed):
            xin = (
                sym
                + fx.Int64(SY["xin"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
        r_wr = _rsrc(w_r)
        R_NKC = HIDDEN // 64
        for tt in range(start("router"), S * N_ROUTER, G):
            tt = fx.Int32(tt)
            t = tt if const_expr(S == 1) else tt % N_ROUTER
            router_sample = fx.Int32(0) if const_expr(S == 1) else tt // N_ROUTER
            stamp("router", tt, 0)

            # K-fold: MFMA rows / B columns 0..7 take this wave's first K half, rows /
            # columns 8..15 the second, so every loaded weight row is distinct and the
            # whole K slice is prefetched; logit = C[r][n] + C[8 + r][8 + n]
            r_sub = t * ROUTER_TILE % 16  # this task's rows of the 16-row group
            r_ln = (lane & -16) | (r_sub + lane % ROUTER_TILE)
            R_CPW = R_NKC // WAVES // 2
            r_fold = (lane % 16) // ROUTER_TILE
            r_ns = fx.Int32(0)

            def u_r(c):
                kc = wave * (R_NKC // WAVES) + r_fold * R_CPW + c
                return unit_bf16(
                    r_wr,
                    t * ROUTER_TILE // 16,
                    kc,
                    R_NKC,
                    (r_ns * HIDDEN + kc * 64) // 2,
                    r_ln,
                )

            if const_expr(not dense_experts):
                pre = [u_r(c) for c in range(R_CPW)]
            hint_wait(
                N_ROW_TILES,
                lambda k: (
                    mb("a"),
                    router_sample * HIDDEN + k * ROW_TILE + ROW_TILE - 1,
                ),
                mark=("router", tt),
            )
            # this task's FP8 activation block inputs ride along with the staging loads:
            # wave w quantizes block w * N_ROUTER + t of this CTA's sample.
            r_gp = _rsrc(g_post)
            x_blk = wave * N_ROUTER + t
            x_s = router_sample
            x_ok = (wave < XQ_WAVES) & (x_blk < XQ_BLOCKS)
            xk = fx.min(x_blk, XQ_BLOCKS - 1) * 128 + lane * 2
            xg = (ld_bf16(r_gp, xk), ld_bf16(r_gp, xk + 1))
            xa = []

            def ld_a(sks):
                specs = [
                    (mb("a"), (router_sample * HIDDEN + k) // 2, 2) for s, k in sks
                ]
                specs.append((mb("a"), (x_s * HIDDEN + xk) // 2, 1))
                v = poll(specs, batch=len(specs))
                stamp("router", tt, 5, lead=THREADS - 64)
                xa.append(bf2_f32(v[-1][0]))
                return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in v[:-1]]

            # the row times gamma into LDS, not normalized: its sum of squares goes
            # through the same barrier, and the row's rstd scales the activation
            # blocks and the logits
            gs, vals = load_x_rmsnorm(ld_a, HIDDEN, g_post, count=1)
            ss = fx.Float32(0.0)
            for v in vals:
                for a in v:
                    ss = ss + a * a
            ss = wave_sum(ss)
            if lane == 0:
                lds_st(xs, HIDDEN // 2 + wave, ss)
            for i in range_constexpr(len(vals)):
                a, g = vals[i], gs[i]
                for j in range_constexpr(2):
                    lds_st(
                        xs,
                        ((tid + i * THREADS) * 4) // 2 + j,
                        bf16_pair(a[2 * j] * g[2 * j], a[2 * j + 1] * g[2 * j + 1]),
                    )
            gpu.barrier()
            tot = lds_ld(xs, HIDDEN // 2)
            for w in range_constexpr(1, WAVES):
                tot = tot + lds_ld(xs, HIDDEN // 2 + w)
            x_rstd = _rsq(tot * (1.0 / HIDDEN) + EPS)
            stamp("router", tt, 2)
            # this task's FP8 activation blocks go out ahead of the gate GEMV
            if x_ok:
                a0, a1 = xa[0]
                if const_expr(native_fp4_mfma):
                    q0, q1, qs = quant_mxfp8(a0 * x_rstd * xg[0], a1 * x_rstd * xg[1])
                else:
                    q0, q1, qs = quant_scaled(a0 * x_rstd * xg[0], a1 * x_rstd * xg[1])
                w8 = (
                    fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
                    & 0xFFFF
                )
                word = w8 | (_xshfl(w8, 1) << 16)  # even lanes: FP8 bytes k .. k + 3
                word_n = _xshfl(word, 2)  # bytes k + 4 .. k + 7
                if lane % 4 == 0:  # two tagged words in one 16-byte store
                    put2(mb("xq"), (x_s * HIDDEN + xk) // 4, word, word_n)
                d0, d1 = _fp8_roundtrip(q0, q1)
                qscale = (
                    (qs << 23).bitcast(fx.Float32)
                    if const_expr(native_fp4_mfma)
                    else qs
                )
                bo.buffer_store(
                    fx.Vector.from_elements([d0 * qscale, d1 * qscale], fx.Float32),
                    _rsrc(mb("xqd")),
                    x_s * HIDDEN + xk,
                )
                if const_expr(native_fp4_mfma):
                    if lane % 16 == 0:
                        put(
                            mb("xqs"),
                            x_s * XQ_GROUPS + x_blk * 4 + lane // 16,
                            qs,
                        )
                else:
                    if lane == 0:
                        put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
            if const_expr(not dense_experts):
                acc = run_units(u_r, R_CPW, R_CPW, pre)
                fx.ptr_store(
                    fx.Vector.from_elements(acc, fx.Float32),
                    red + (wave * 64 + lane) * 4,
                )
                gpu.barrier()
                stamp("router", tt, 3)
                if tid < ROUTER_TILE:
                    r = tid % ROUTER_TILE
                    n = fx.Int32(0)
                    logit = fx.Float32(0.0)
                    for w in range_constexpr(WAVES):
                        for f in range_constexpr(2):
                            m = f * ROUTER_TILE + r
                            logit = logit + lds_ld(
                                red,
                                (w * 64 + f * ROUTER_TILE + n + 16 * (m // 4)) * 4
                                + m % 4,
                            )
                    put(
                        mb("scores"),
                        router_sample * N_EXPERTS + t * ROUTER_TILE + r,
                        _rcp(1.0 + _exp(-(logit * x_rstd))),
                    )
            stamp("router", tt, 4)

        def dn_route(bs):
            """Expert-down routing: one whole wave per sample, in wave-sized batches."""
            for sample_batch in range_constexpr(sample_wave_batches(S)):
                route_sample = wave + sample_batch * WAVES
                if const_expr(dense_experts):
                    # slot j < dense_experts: slice j, weight 1; spare slots weigh 0
                    if (route_sample < S) & (lane < MOE_SLOTS):
                        q = route_sample * MOE_SLOTS + lane
                        used = lane < dense_experts
                        lds_st(keys, q, used.select(lane, fx.Int32(SHARED)))
                        lds_st(dnw, q, used.select(fx.Float32(1.0), fx.Float32(0.0)))
                elif route_sample < S:
                    e, w = route_top8(route_sample, bs=bs)
                    if (
                        lane < MOE_SLOTS
                    ):  # slot 0: the shared expert, then pick lane (slot lane + 1)
                        q = route_sample * MOE_SLOTS + (lane + 1) % MOE_SLOTS
                        lds_st(keys, q, (lane == TOP_K).select(fx.Int32(SHARED), e))
                        lds_st(dnw, q, (lane == TOP_K).select(fx.Float32(1.0), w))

        # ================================ 9. expert up/gate + SiLU
        (x_out, positions, slot_mapping, sparse_kv_indptr, kv_cache, pe_cache, indices, rope_cos, rope_sin, g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o, s_o, w_r, bias, w_ug, s_ug, w_dn, s_dn, scratch, sym, timeline_buf, block_table, req_ids, out_indices, ix_wk, ix_sk, ix_ww, ix_wq, ix_sq, ix_gk, ix_bk) = reload_args()  # fmt: skip
        if const_expr(RELOAD_ARGS and x_pushed):
            xin = (
                sym
                + fx.Int64(SY["xin"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
        # 2 row groups (16 gate + 16 up rows) x 96 chunks: 4 waves per group, 24 chunks each
        UG_NKC = HIDDEN // 64
        UG_CM = CM_NT if S != 4 or EARLY_ISSUE else 0
        UG_CPW = UG_NKC // (WAVES // 2)
        UG_W_BYTES = 2 * expert_inter * HIDDEN // (2 if expert_mxfp4 else 1)
        UG_S_BYTES = (
            2 * expert_inter * (HIDDEN // 32)
            if expert_mxfp4
            else 2 * expert_inter // SCALE_BM * (HIDDEN // 128) * 4
        )

        def ug_units(e_sel, c, live=None):
            """Unit maker of up/gate tile c of expert e_sel; ``live`` False -> empty
            buffers (loads return 0 without memory traffic)."""
            if const_expr(live is None):
                r_wug = _rsrc(w_ug + fx.Int64(e_sel) * fx.Int64(UG_W_BYTES))
                r_sug = _rsrc(s_ug + fx.Int64(e_sel) * fx.Int64(UG_S_BYTES))
            else:
                r_wug = bo.create_buffer_resource_from_addr(
                    w_ug + fx.Int64(e_sel) * fx.Int64(UG_W_BYTES),
                    num_records_bytes=live.select(fx.Int32(UG_W_BYTES), fx.Int32(0)),
                )
                r_sug = bo.create_buffer_resource_from_addr(
                    s_ug + fx.Int64(e_sel) * fx.Int64(UG_S_BYTES),
                    num_records_bytes=live.select(fx.Int32(UG_S_BYTES), fx.Int32(0)),
                )
            gate_up = wave // (WAVES // 2)  # waves 0-3: gate rows, 4-7: up rows

            def u_ug(cc):  # cc: 128-k chunk of this wave
                kc = (wave % (WAVES // 2)) * UG_CPW + cc * 2
                return unit_f8f8(
                    r_wug,
                    r_sug,
                    gate_up * (expert_inter // 16) + c,
                    kc,
                    UG_NKC,
                    HIDDEN,
                    kc * 16,
                    lambda: _uniform_f32(lds_ld(misc, 8 + kc // 2)),
                    shuffled=fp8_shuffled,
                )

            return u_ug

        def ug_finish(u, s_u, slot, c, e_sel, prob, u_ug, pre):
            """MFMA the staged activation (X, scales in misc[8:]) against the tile,
            SiLU(gate) * up -> mid."""
            acc = run_units(u_ug, UG_CPW // 2, UG_CPW // 2, pre)
            reduce_rows(2, acc, emit_out(UG_TILE * 2))
            stamp("ug", u, 3)
            gpu.barrier()
            if tid < UG_TILE // 2:
                r = tid * 2
                g0, g1 = lds_ld(outs, r), lds_ld(outs, r + 1)
                u0, u1 = lds_ld(outs, UG_TILE + r), lds_ld(outs, UG_TILE + r + 1)
                put2(
                    mb("mid"),
                    (s_u * MOE_SLOTS + slot) * expert_inter + c * UG_TILE + r,
                    g0 * _rcp(1.0 + _exp(-g0)) * u0,
                    g1 * _rcp(1.0 + _exp(-g1)) * u1,
                )
            if (c == 0) & (tid == 0):  # routing record (debug / tests)
                put(mb("sel"), s_u * MOE_SLOTS + slot, e_sel)
                put(mb("prob"), s_u * MOE_SLOTS + slot, prob())
            stamp("ug", u, 4)

        def ug_task(u):
            s_u = u // (MOE_SLOTS * I_PER_SLOT)
            return s_u, (u // I_PER_SLOT) % MOE_SLOTS, u % I_PER_SLOT

        def job_emit(c, g, e):
            """SiLU(gate) * up of eight-intermediate tile ``c`` of expert ``e`` (MFMA
            column n: row n) -> mid at each using row's slot of it; ``g`` 0: the shared
            expert, which every row uses at slot 0 with weight 1."""
            if tid < S * UG8 // 2:
                n = tid // (UG8 // 2)
                r = (tid % (UG8 // 2)) * 2
                sl = fx.Int32(0)
                for k in range_constexpr(1, MOE_SLOTS):
                    sl = (lds_ld(keys, n * MOE_SLOTS + k) == e).select(fx.Int32(k), sl)
                if (g == 0) | (sl != 0):
                    g0, g1 = lds_ld(outs, n * 16 + r), lds_ld(outs, n * 16 + r + 1)
                    v0 = lds_ld(outs, n * 16 + UG8 + r)
                    v1 = lds_ld(outs, n * 16 + UG8 + r + 1)
                    put2(
                        mb("mid"),
                        (n * MOE_SLOTS + sl) * expert_inter + c * UG8 + r,
                        g0 * _rcp(1.0 + _exp(-g0)) * v0,
                        g1 * _rcp(1.0 + _exp(-g1)) * v1,
                    )
                    if (c == 0) & (r == 0):  # routing record (debug / tests)
                        put(mb("sel"), n * MOE_SLOTS + sl, e)
                        put(
                            mb("prob"),
                            n * MOE_SLOTS + sl,
                            (g == 0).select(
                                fx.Float32(1.0), lds_ld(dnw, n * MOE_SLOTS + sl)
                            ),
                        )

        # up/gate tiles of eight intermediates: 8 gate + 8 up rows, K split over the waves
        UG8 = 8
        UG8_UNITS = (HIDDEN // 128) // WAVES

        def ug8_tile(u):
            """(c, slot, w_rg, w_ln, s_rg) of up/gate tile u: intermediates c * 8 of
            routed slot u // (inter / 8), slot 8 for the first inter / 8 tiles,
            which also take the shared expert's."""
            c = u % (expert_inter // UG8)
            has_sh = u < expert_inter // UG8
            slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), u // (expert_inter // UG8))
            w_rg = ((lane % 16) // 8) * (expert_inter // 16) + c // 2
            w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)
            s_rg = (lane // 32) * (expert_inter // 16) + c // 2
            return c, slot, w_rg, w_ln, s_rg

        def ug8_units(tile, e, sample, live=None):
            _, _, w_rg, w_ln, s_rg = tile
            if const_expr(live is None):
                rw = _rsrc(w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES))
                rs = _rsrc(s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES))
            else:
                rw = bo.create_buffer_resource_from_addr(
                    w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES),
                    num_records_bytes=live.select(fx.Int32(UG_W_BYTES), fx.Int32(0)),
                )
                rs = bo.create_buffer_resource_from_addr(
                    s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES),
                    num_records_bytes=live.select(fx.Int32(UG_S_BYTES), fx.Int32(0)),
                )
            sn = n_sel() if sample is None else fx.Int32(sample)
            units = []
            for cc in range_constexpr(UG8_UNITS):
                unit = wave * UG8_UNITS + cc
                kc = unit * 2
                if const_expr(expert_mxfp4):
                    if const_expr(native_fp4_mfma):
                        units.append(
                            unit_native_mxfp4(
                                rw,
                                rs,
                                w_rg,
                                unit,
                                HIDDEN,
                                sn * XQ_STRIDE + unit * 32,
                                8 + sn * XQ_GROUPS + unit * 4 + lane // 16,
                                ln=w_ln,
                            )
                        )
                    else:
                        units.append(
                            unit_mxfp4(
                                rw,
                                rs,
                                w_rg,
                                unit,
                                HIDDEN,
                                sn * XQ_STRIDE + unit * 32,
                                lambda unit=unit, sn=sn: lds_ld(
                                    misc, 8 + sn * XQ_BLOCKS + unit
                                ),
                                w_ln,
                            )
                        )
                    continue
                wv = [
                    ld_fp8_chunk(
                        rw, w_rg, kc + j, UG_NKC, HIDDEN, w_ln, fp8_shuffled, UG_CM
                    )
                    for j in range(2)
                ]
                # the scales of this tile's gate rows (lanes < 32) and up rows
                s_at = s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES)
                si = (tile[0] // 2 * 16 // SCALE_BM) * (HIDDEN // 128) + kc // 2
                sg = sld_f32(s_at + fx.Int64(si) * 4)
                su = sld_f32(
                    s_at + fx.Int64(si + expert_inter // SCALE_BM * (HIDDEN // 128)) * 4
                )
                units.append(
                    (
                        "f8f8",
                        wv,
                        lambda sg=sg, su=su, kc=kc, sn=sn: (
                            (lane < 32).select(sg, su)
                            * lds_ld(misc, 8 + sn * XQ_BLOCKS + kc // 2)
                        ),
                        sn * XQ_STRIDE + kc * 16 + (lane // 16) * 4,
                    )
                )
            return units

        def ug8_emit(tile, sample, shared):
            c, slot = tile[0], tile[1]
            if tid < (S if shared else 1) * UG8 // 2:
                n = tid // (UG8 // 2)
                r = (tid % (UG8 // 2)) * 2
                g0, g1 = (
                    lds_ld(outs, n * 16 + r),
                    lds_ld(outs, n * 16 + r + 1),
                )
                v0 = lds_ld(outs, n * 16 + UG8 + r)
                v1 = lds_ld(outs, n * 16 + UG8 + r + 1)
                sn = n if shared else fx.Int32(sample)
                sl = fx.Int32(0) if shared else slot
                put2(
                    mb("mid"),
                    (sn * MOE_SLOTS + sl) * expert_inter + c * UG8 + r,
                    g0 * _rcp(1.0 + _exp(-g0)) * v0,
                    g1 * _rcp(1.0 + _exp(-g1)) * v1,
                )
            if (c == 0) & (tid < S if shared else tid == 0):
                sn = tid if shared else fx.Int32(sample)
                if const_expr(shared):
                    # slot 0 routes every row to the shared expert with weight 1
                    # (dn_route's keys and weights, not written yet)
                    e_rec, w_rec = fx.Int32(SHARED), fx.Float32(1.0)
                else:
                    e_rec = lds_ld(keys, sn * MOE_SLOTS + slot)
                    w_rec = lds_ld(dnw, sn * MOE_SLOTS + slot)
                sl = fx.Int32(0) if shared else slot
                put(mb("sel"), sn * MOE_SLOTS + sl, e_rec)
                put(mb("prob"), sn * MOE_SLOTS + sl, w_rec)

        if const_expr(S == 1):
            # one task per CTA: task u takes intermediates (u % 32) * 8 of routed slot
            # u // 32 (slot 8 for u < 32, which also take the shared expert's); the 8 gate
            # + 8 up rows are one MFMA row group and all waves split K
            UG8_CPW = UG_NKC // WAVES
            for u in range(start("ug"), G, G):
                u = fx.Int32(u)
                stamp("ug", u, 0)
                s_u, c = fx.Int32(0), u % (expert_inter // UG8)
                has_sh = u < expert_inter // UG8
                slot = has_sh.select(
                    fx.Int32(MOE_SLOTS - 1), u // (expert_inter // UG8)
                )
                # the FP8 activation is computed here from the post-attention state (in
                # parallel with the router): RMSNorm, then per-128 quant with one wave per
                # block -> X[0] (fp8 values in bf16), block scales -> misc[8:]
                NB = XQ_BLOCKS // WAVES
                ks_ = [(wave + j * WAVES) * 128 + lane * 2 for j in range(NB)]
                r_gp = _rsrc(g_post)
                gps = [
                    (ld_bf16(r_gp, k), ld_bf16(r_gp, k + 1)) for k in ks_
                ]  # issued ahead of the wait
                bs = load_bias()
                w_rg = ((lane % 16) // 8) * (
                    expert_inter // 16
                ) + c // 2  # MFMA rows 0-7 gate, 8-15 up
                w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)
                s_rg = (lane // 32) * (
                    expert_inter // 16
                ) + c // 2  # this lane's output rows

                def u_ug8(
                    cc, e, live=None
                ):  # expert e's weights (loads return 0 unless live)
                    nw = (
                        None
                        if live is None
                        else live.select(fx.Int32(UG_W_BYTES), fx.Int32(0))
                    )
                    ns = (
                        None
                        if live is None
                        else live.select(fx.Int32(UG_S_BYTES), fx.Int32(0))
                    )
                    r_wug = bo.create_buffer_resource_from_addr(
                        w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES), num_records_bytes=nw
                    )
                    r_sug = bo.create_buffer_resource_from_addr(
                        s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES), num_records_bytes=ns
                    )
                    unit = wave * (UG8_CPW // 2) + cc
                    kc = unit * 2
                    if const_expr(expert_mxfp4):
                        if const_expr(native_fp4_mfma):
                            return unit_native_mxfp4(
                                r_wug,
                                r_sug,
                                w_rg,
                                unit,
                                HIDDEN,
                                unit * 32,
                                8 + unit * 4 + lane // 16,
                                ln=w_ln,
                            )
                        return unit_mxfp4(
                            r_wug,
                            r_sug,
                            w_rg,
                            unit,
                            HIDDEN,
                            unit * 32,
                            lambda: _uniform_f32(lds_ld(misc, 8 + unit)),
                            w_ln,
                        )
                    wv = [
                        ld_fp8_chunk(
                            r_wug, w_rg, kc + h, UG_NKC, HIDDEN, w_ln, fp8_shuffled
                        )
                        for h in range(2)
                    ]
                    sc = ld_f32(
                        r_sug, (s_rg * 16 // SCALE_BM) * (HIDDEN // 128) + kc // 2
                    )
                    return (
                        "f8f8",
                        wv,
                        lambda: sc * _uniform_f32(lds_ld(misc, 8 + kc // 2)),
                        kc * 16 + (lane // 16) * 4,
                    )

                # the shared expert's weights do not depend on routing: prefetch them (the
                # later zero-weight MMAs of the other tasks are cheaper than a branch)
                pre = [
                    u_ug8(cc, fx.Int32(SHARED), has_sh) for cc in range(UG8_CPW // 2)
                ]
                hint_wait(
                    N_ROW_TILES,
                    lambda k: (mb("a"), s_u * HIDDEN + k * ROW_TILE + ROW_TILE - 1),
                    mark=("ug", u),
                )
                # the sum of squares takes the router's element partition and order
                # (stage_x_rmsnorm), so rstd -- and every FP8 rounding -- is bit-identical
                NQ4 = HIDDEN // (4 * THREADS)
                got = poll(
                    [
                        (mb("a"), (s_u * HIDDEN + (tid + i * THREADS) * 4) // 2, 2)
                        for i in range(NQ4)
                    ]
                    + [(mb("a"), (s_u * HIDDEN + k) // 2, 1) for k in ks_]
                )
                av = [bf2_f32(w[0]) for w in got[NQ4:]]
                ss = fx.Float32(0.0)
                for w in got[:NQ4]:
                    for a in list(bf2_f32(w[0])) + list(bf2_f32(w[1])):
                        ss = ss + a * a
                rstd = _rsq(block_sum(ss) * (1.0 / HIDDEN) + EPS)
                for j in range_constexpr(NB):
                    if const_expr(native_fp4_mfma):
                        q0, q1, qs = quant_mxfp8(
                            av[j][0] * rstd * gps[j][0],
                            av[j][1] * rstd * gps[j][1],
                        )
                        st_mxfp8(ks_[j], q0, q1)
                        if lane % 16 == 0:
                            lds_st(
                                misc,
                                8 + (wave + j * WAVES) * 4 + lane // 16,
                                qs.bitcast(fx.Float32),
                            )
                    else:
                        q0, q1, qs = quant_scaled(
                            av[j][0] * rstd * gps[j][0],
                            av[j][1] * rstd * gps[j][1],
                        )
                        st_f8(ks_[j], q0, q1)
                        if lane == 0:
                            lds_st(misc, 8 + wave + j * WAVES, qs)
                if wave == 0:
                    e, w = route_top8(s_u, bs=bs)
                    if lane == slot - 1:
                        lds_st(keys, 0, e)
                        lds_st(misc, 0, w)
                stamp("ug", u, 2)
                gpu.barrier()
                e_sel = _uniform(lds_ld(keys, 0))
                post = [u_ug8(cc, e_sel) for cc in range(UG8_CPW // 2)]
                reduce_rows(
                    1, mma_units([fx.Float32(0.0) for _ in range(4)], pre), emit_out(16)
                )
                gpu.barrier()
                reduce_rows(
                    1,
                    mma_units([fx.Float32(0.0) for _ in range(4)], post),
                    lambda rl, n, v: lds_st(outs, 16 + rl, v),
                )
                stamp("ug", u, 3)
                gpu.barrier()
                if (
                    tid < UG8
                ):  # threads 0-3: the shared expert's rows, 4-7: the routed slot's
                    r = (tid % (UG8 // 2)) * 2
                    o = (tid // (UG8 // 2)) * 16
                    g0, g1 = lds_ld(outs, o + r), lds_ld(outs, o + r + 1)
                    u0, u1 = lds_ld(outs, o + UG8 + r), lds_ld(outs, o + UG8 + r + 1)
                    if has_sh | (tid >= UG8 // 2):
                        put2(
                            mb("mid"),
                            (tid < UG8 // 2).select(fx.Int32(0), slot) * expert_inter
                            + c * UG8
                            + r,
                            g0 * _rcp(1.0 + _exp(-g0)) * u0,
                            g1 * _rcp(1.0 + _exp(-g1)) * u1,
                        )
                if (c == 0) & (tid == 0):  # routing record (debug / tests)
                    put(mb("sel"), slot, e_sel)
                    put(mb("prob"), slot, lds_ld(misc, 0))
                    if has_sh:
                        put(mb("sel"), 0, fx.Int32(SHARED))
                        put(mb("prob"), 0, fx.Float32(1.0))
                stamp("ug", u, 4)
        elif const_expr(dense_fp8):
            # A dense MLP: a slice is every row's, so one MFMA takes all rows as its B
            # columns. Task u < slices * 32 computes intermediates (u % 32) * 8 of slice
            # u // 32; the tasks of slice 0 also write the zeros of the slots past the
            # slices, which down reads with weight 0.
            D_NT = expert_inter // UG8
            dn_route(load_bias())
            gpu.barrier()
            stage_xq(list(range(S)))
            gpu.barrier()
            u0 = start("ug")
            for task_round in range_constexpr((dense_experts * D_NT + G - 1) // G):
                d_u = fx.Int32(u0 + task_round * G)
                if d_u < dense_experts * D_NT:
                    stamp("ug", d_u, 0)
                    d_c, d_sl = d_u % D_NT, d_u // D_NT
                    w_rg = ((lane % 16) // 8) * (expert_inter // 16) + d_c // 2
                    w_ln = (lane & -16) | ((d_c % 2) * 8 + lane % 8)
                    s_rg = (lane // 32) * (expert_inter // 16) + d_c // 2
                    rw = _rsrc(w_ug + fx.Int64(d_sl) * fx.Int64(UG_W_BYTES))
                    rs = _rsrc(s_ug + fx.Int64(d_sl) * fx.Int64(UG_S_BYTES))
                    d_sn = n_sel()
                    d_units = []
                    for cc in range_constexpr(UG8_UNITS):
                        d_kc = (wave * UG8_UNITS + cc) * 2
                        d_wv = [
                            ld_fp8_chunk(
                                rw,
                                w_rg,
                                d_kc + j,
                                UG_NKC,
                                HIDDEN,
                                w_ln,
                                fp8_shuffled,
                                UG_CM,
                            )
                            for j in range(2)
                        ]
                        d_sc = ld_f32(
                            rs, (s_rg * 16 // SCALE_BM) * (HIDDEN // 128) + d_kc // 2
                        )
                        d_units.append(
                            (
                                "f8f8",
                                d_wv,
                                lambda d_sc=d_sc, d_kc=d_kc: (
                                    d_sc
                                    * lds_ld(misc, 8 + d_sn * XQ_BLOCKS + d_kc // 2)
                                ),
                                d_sn * XQ_STRIDE + d_kc * 16 + (lane // 16) * 4,
                            )
                        )
                    reduce_rows(
                        1,
                        mma_units([fx.Float32(0.0) for _ in range(4)], d_units),
                        emit_out(16),
                    )
                    gpu.barrier()
                    if tid < S * UG8 // 2:
                        d_n = tid // (UG8 // 2)
                        d_r = (tid % (UG8 // 2)) * 2
                        d_g0 = lds_ld(outs, d_n * 16 + d_r)
                        d_g1 = lds_ld(outs, d_n * 16 + d_r + 1)
                        d_v0 = lds_ld(outs, d_n * 16 + UG8 + d_r)
                        d_v1 = lds_ld(outs, d_n * 16 + UG8 + d_r + 1)
                        put2(
                            mb("mid"),
                            (d_n * MOE_SLOTS + d_sl) * expert_inter + d_c * UG8 + d_r,
                            d_g0 * _rcp(1.0 + _exp(-d_g0)) * d_v0,
                            d_g1 * _rcp(1.0 + _exp(-d_g1)) * d_v1,
                        )
                        if d_sl == 0:
                            for d_z in range_constexpr(dense_experts, MOE_SLOTS):
                                put2(
                                    mb("mid"),
                                    (d_n * MOE_SLOTS + d_z) * expert_inter
                                    + d_c * UG8
                                    + d_r,
                                    fx.Float32(0.0),
                                    fx.Float32(0.0),
                                )
                    stamp("ug", d_u, 4)
                if const_expr(task_round + 1 < (dense_experts * D_NT + G - 1) // G):
                    gpu.barrier()
        elif const_expr(S > 4 and not dense_experts):
            # Above 4 rows: eight-intermediate tiles of each expert the step uses
            # (the shared one included) as jobs dealt round-robin over the CTAs, each
            # with every row that picked the expert as an MFMA column. Below, the
            # per-expert overheads cost more than the repeated experts they save.
            # Jobs: tile c (intermediates c * 8 .. c * 8 + 7) of each expert the step
            # uses: the shared expert first (group 0), then the routed experts in
            # order of first use (row-major over the top-8s), each once with every
            # row that picked it as an MFMA column. Job r of a CTA is tile u0 % TPG
            # of group r * GPR + u0 // TPG. A column's sums are those of a one-row
            # task, so mid is the same bits.
            TPG = expert_inter // UG8  # tiles (jobs) a group
            GPR = G // TPG  # groups a round
            assert G % TPG == 0
            ROUNDS = (S * TOP_K + 1 + GPR - 1) // GPR
            u0 = start("ug")
            tile0 = ug8_tile(fx.Int32(u0))
            has_sh = fx.Int32(u0) < TPG
            bs = load_bias()
            SLOT0 = JOB_SLOT_OFF - JOB_OFF
            for i in range_constexpr((MOE_GROUPS * S + THREADS - 1) // THREADS):
                if tid + i * THREADS < MOE_GROUPS * S:
                    lds_st(jobs, SLOT0 + tid + i * THREADS, fx.Int32(-1))
            # up to 4 rows the shared tiles' CTAs run no router task: they stage the
            # activations and run their shared tile before the routing, weights
            # issued ahead of the activations
            EARLY_SH = S * N_ROUTER + TPG <= G
            if const_expr(EARLY_SH):
                early_sh = has_sh & (start("router") >= S * N_ROUTER)
                shared_pre = ug8_units(tile0, fx.Int32(SHARED), None, early_sh)

            if const_expr(EARLY_SH):
                if early_sh:
                    stage_xq(list(range(S)))
                    gpu.barrier()
                    reduce_rows(
                        1,
                        mma_units([fx.Float32(0.0) for _ in range(4)], shared_pre),
                        emit_out(16),
                    )
                    gpu.barrier()
                    ug8_emit(tile0, 0, True)

            # Every CTA routes, then issues the activations' poll, its first job's
            # weights, and checks the poll: the activations are out by then, so the
            # poll's round trip runs under the weights' issue and flight.
            dn_route(bs)
            gpu.barrier()
            # every wave finds the distinct routed experts itself: lane p holds pick
            # p (row p // 8, slot p % 8 + 1); it is the group's first use when no
            # lower lane picked the same expert
            NP = S * TOP_K
            p_ok = lane < NP
            p_e = p_ok.select(
                lds_ld(
                    keys, fx.min(lane, NP - 1) // TOP_K * MOE_SLOTS + lane % TOP_K + 1
                ),
                fx.Int32(-1),
            )
            p_first = fx.Int32(64)
            for p in range_constexpr(NP):
                p_first = ((read_lane_i32(p_e, p) == p_e) & (p_first == 64)).select(
                    fx.Int32(p), p_first
                )
            p_is_first = p_ok & (p_first == lane)
            firsts = ballot(p_is_first)
            p_group = lanes_below(firsts, lane) + 1
            n_groups = popcount(firsts) + 1
            # the groups for down: their experts and each row's slot of them
            if wave == 0:
                if p_is_first:
                    lds_st(jobs, 1 + p_group, p_e)
                if p_ok:
                    p_grp = lanes_below(firsts, fx.min(p_first, 63)) + 1
                    lds_st(jobs, SLOT0 + p_grp * S + lane // TOP_K, lane % TOP_K + 1)
                if lane == 0:
                    lds_st(jobs, 0, n_groups)
                    lds_st(jobs, 1, fx.Int32(SHARED))
                if lane < S:
                    lds_st(jobs, SLOT0 + lane, fx.Int32(0))

            def job(r):
                """(live, group, expert, issued units) of this CTA's job r."""
                g = fx.Int32(r * GPR) + fx.Int32(u0) // TPG
                live = g < n_groups
                if const_expr(r == 0 and EARLY_SH):  # unless it ran before the routing
                    live = live & (
                        (fx.Int32(u0) >= TPG) | (start("router") < S * N_ROUTER)
                    )
                # the group's first-use lane: the one set bit of its ballot
                one = ballot(p_is_first & (p_group == g))
                at = fx.min(
                    popcount((one - 1) & (one ^ fx.Int64(-1))), fx.Int32(NP - 1)
                )
                e = (g == 0).select(fx.Int32(SHARED), read_lane_i32(p_e, at))
                return live, g, e, ug8_units(tile0, e, None, live)

            first_job = []
            stage_xq(list(range(S)), between=lambda: first_job.append(job(0)))
            cur = first_job[0]
            gpu.barrier()
            for r in range_constexpr(ROUNDS):
                live, g, e, pre = cur
                if const_expr(r + 1 < ROUNDS):
                    cur = job(r + 1)
                if live:
                    stamp("ug", r * G + u0, 0)
                    reduce_rows(
                        1,
                        mma_units([fx.Float32(0.0) for _ in range(4)], pre),
                        emit_out(16),
                    )
                    gpu.barrier()
                    job_emit(tile0[0], g, e)
                    stamp("ug", r * G + u0, 4)

        elif const_expr(S > 1):
            # One eight-intermediate tile per CTA and task round. Shared-expert weights
            # feed all sample columns of one MFMA, while routed-expert weights are
            # prefetched one sample ahead.  This avoids the segmented partial
            # tiles and mailbox reduction used by the older S=2/4 schedule.
            # The shared expert's tiles need only the activations: their CTAs run them
            # before the routing, weights issued ahead of the activations' wait, so
            # every CTA starts its routed tiles together.
            u0 = start("ug")
            # round 0 holds every shared-expert tile (inter / 8 <= G of them). At 8 rows
            # every CTA runs a router task before it, and the weights issued after the
            # activations' poll measured faster than ahead of it
            assert expert_inter >= G
            tile0 = ug8_tile(fx.Int32(u0))
            has_sh = fx.Int32(u0) < expert_inter // UG8
            bs = load_bias()
            if const_expr(S <= 4):
                shared_pre = ug8_units(tile0, fx.Int32(SHARED), None, has_sh)

            def stage():
                stage_xq(list(range(S)))
                gpu.barrier()

            def shared(units):
                """The shared expert's tile on its CTAs (``units``: its issued weights)."""
                if has_sh:
                    reduce_rows(
                        1,
                        mma_units([fx.Float32(0.0) for _ in range(4)], units),
                        emit_out(16),
                    )
                    gpu.barrier()
                    ug8_emit(tile0, 0, True)

            def stage_and_shared():
                """The activations, then the shared expert's tile on its CTAs."""
                stage()
                if const_expr(S > 4):
                    shared(ug8_units(tile0, fx.Int32(SHARED), None, has_sh))
                else:
                    shared(shared_pre)

            # A CTA that ran a router task reaches this point after the activations
            # are out: it routes first, issues its first routed weights, and stages
            # the activations behind them. The others stage them while the scores
            # are still being computed.
            if const_expr(S * N_ROUTER < G):
                if start("router") >= S * N_ROUTER:
                    stage_and_shared()
            if const_expr(EARLY_ISSUE):
                # A CTA's jobs are one row's tiles c of every SPAN-th slot from
                # 1 + group / S (each (row, slot, c) tile once), so its first job needs
                # the row's first SPAN picks only. Wave w < S routes row (row + w) % S,
                # wave 0 this CTA's row; its first SPAN rounds go before the first
                # job's weights, the rest after.
                SPAN = TOP_K // S
                assert G // (expert_inter // UG8) == S * SPAN
                grp = fx.Int32(u0) // (expert_inter // UG8)
                my_row = grp % S
                slot0 = 1 + grp // S

                def job_tile(j):
                    return (tile0[0], slot0 + j * SPAN, tile0[2], tile0[3], tile0[4])

                r_row = (my_row + wave) % S
                scr = route_scr + wave * ROUTE_WORDS
                # the round state waits in LDS while the first job's weights go out
                stash = route_scr + WAVES * ROUTE_WORDS + wave * STASH_WORDS
                if wave < S:
                    raws1 = route_scores(r_row)
                    ks1, ids1 = route_keys(raws1, bs)
                    hk1, hp1 = route_heads(ks1, ids1)
                    hk1, hp1, tie1 = route_rounds(scr, 0, SPAN, hk1, hp1)
                    if tie1 != 0:
                        route_tied(r_row, scr, raws1, bs)
                    held = (
                        hk1
                        + [hp1, (tie1 != 0).select(fx.Int32(1), fx.Int32(0))]
                        + [r.bitcast(fx.Int32) for r in raws1]
                    )
                    for i in range_constexpr(len(held)):
                        lds_st(stash, i * 64 + lane, held[i])
                gpu.barrier()
                e0 = _uniform(lds_ld(route_scr, slot0 - 1) & 0xFF)
                units = ug8_units(job_tile(0), e0, None)
                if wave < S:
                    held = [lds_ld(stash, i * 64 + lane) for i in range(10)]
                    raws2 = [r.bitcast(fx.Float32) for r in held[6:]]
                    if held[5] == 0:
                        _, _, tie2 = route_rounds(scr, SPAN, TOP_K, held[:4], held[4])
                        if tie2 != 0:
                            route_tied(r_row, scr, raws2, bs)
                    e = lds_ld(scr, fx.min(lane, TOP_K - 1)) & 0xFF
                    w = route_weights(e, raws2)
                    if lane < MOE_SLOTS:  # slot 0: the shared expert, then pick lane
                        q = r_row * MOE_SLOTS + (lane + 1) % MOE_SLOTS
                        lds_st(keys, q, (lane == TOP_K).select(fx.Int32(SHARED), e))
                        lds_st(dnw, q, (lane == TOP_K).select(fx.Float32(1.0), w))
                gpu.barrier()
                if start("router") < S * N_ROUTER:
                    # its shared tile's weights (if any) issued only here
                    stage()
                    shared(ug8_units(tile0, fx.Int32(SHARED), None, has_sh))
                # An expert several rows picked streams once, every picking row an
                # MFMA column: its owner is the row whose job of it comes first (a
                # row's job j takes its picks j * SPAN ..), the lowest such row on a
                # tie. Other rows' later jobs of it load nothing (a first job is issued
                # before the other rows' picks are known), and only the owner writes mid.
                NP = S * TOP_K
                p_e = (lane < NP).select(
                    lds_ld(
                        keys,
                        fx.min(lane, NP - 1) // TOP_K * MOE_SLOTS + lane % TOP_K + 1,
                    ),
                    fx.Int32(-1),
                )

                def owns(e):
                    m = ballot((p_e == e) & (lane % TOP_K >= (S - 1) * SPAN))
                    for j in range_constexpr(S - 2, -1, -1):
                        mj = ballot((p_e == e) & (lane % TOP_K // SPAN == j))
                        m = (mj != 0).select(mj, m)
                    first = popcount((m - 1) & (m ^ fx.Int64(-1)))
                    return first // TOP_K == my_row

                e_cur = e0
                for j in range_constexpr(S):
                    pre, e_j = units, e_cur
                    if const_expr(j + 1 < S):
                        e_cur = _uniform(
                            lds_ld(keys, my_row * MOE_SLOTS + slot0 + (j + 1) * SPAN)
                        )
                        units = ug8_units(job_tile(j + 1), e_cur, None, owns(e_cur))
                    stamp("ug", j * G + u0, 0)
                    reduce_rows(
                        1,
                        mma_units([fx.Float32(0.0) for _ in range(4)], pre),
                        emit_out(16),
                    )
                    gpu.barrier()
                    if owns(e_j):  # a routed expert: group > 0
                        job_emit(tile0[0], fx.Int32(1), e_j)
                    stamp("ug", j * G + u0, 4)
            else:
                dn_route(bs)
                gpu.barrier()
                row0_units = ug8_units(tile0, _uniform(lds_ld(keys, tile0[1])), 0)
                if const_expr(S * N_ROUTER < G):
                    if start("router") < S * N_ROUTER:
                        stage_and_shared()
                else:
                    stage_and_shared()
                for task_round in range_constexpr(ug_task_rounds(expert_inter)):
                    u = fx.Int32(u0 + task_round * G)
                    if u < expert_inter:
                        tile = tile0 if const_expr(task_round == 0) else ug8_tile(u)
                        slot = tile[1]
                        if const_expr(task_round == 0):
                            row_units = row0_units
                        else:
                            row_units = ug8_units(tile, _uniform(lds_ld(keys, slot)), 0)
                        for sample in range_constexpr(S):
                            stamp("ug", sample * G + u, 0)
                            pre = row_units
                            if const_expr(sample + 1 < S):
                                row_units = ug8_units(
                                    tile,
                                    _uniform(
                                        lds_ld(keys, (sample + 1) * MOE_SLOTS + slot)
                                    ),
                                    sample + 1,
                                )
                            reduce_rows(
                                1,
                                mma_units([fx.Float32(0.0) for _ in range(4)], pre),
                                emit_out(16),
                            )
                            gpu.barrier()
                            ug8_emit(tile, sample, False)
                            stamp("ug", sample * G + u, 4)
                        if const_expr(task_round + 1 < ug_task_rounds(expert_inter)):
                            gpu.barrier()

        elif const_expr(ug_split(S, expert_inter) is not None):
            # S = 2, 4 (the router already quantized every sample's activation): job 0 is
            # this CTA's K-segment of a leftover tile (partial sums -> ugp; the segment-0
            # CTA sums them after its own tiles), jobs 1..NF whole tiles.  Tile x: expert
            # slot x // 16 (0 = the shared expert with MFMA column n = sample n, then
            # sample-major routed slots), intermediates (x % 16) * 16.  Job k + 1 is
            # routed and its weights are in flight while job k computes.
            NF, SEG = ug_split(S, expert_inter)
            KB = XQ_BLOCKS // SEG  # 128-k blocks per segment and row group
            XW = HIDDEN // 4  # LDS words of one sample's FP8 activation
            gu_row = (wave // (WAVES // 2)) * (
                expert_inter // 16
            )  # waves 0-3: gate rows, 4-7: up rows
            wq = wave % (WAVES // 2)
            seg = bid % SEG
            x_seg = NF * G + bid // SEG

            def ug_job(k):
                x = fx.Int32(x_seg if k == 0 else bid + (k - 1) * G)
                es = x // I_PER_SLOT
                c = x % I_PER_SLOT
                shared = es == 0
                s_u = fx.max(es - 1, 0) // TOP_K
                slot = shared.select(fx.Int32(0), (es - 1) % TOP_K + 1)
                e_sel = _uniform(
                    lds_ld(keys, s_u * MOE_SLOTS + slot)
                )  # routed once by dn_route
                prob = lds_ld(dnw, s_u * MOE_SLOTS + slot)
                bsel = shared.select(n_sel(), s_u)  # this lane's activation sample
                if const_expr(k == 0):  # waves wq < KB each own one 128-k block
                    kbs = [seg * KB + fx.min(wq, KB - 1)]
                    nrec = (wq < KB).select(fx.Int32(UG_W_BYTES), fx.Int32(0))
                else:
                    kbs = [wq * (UG_CPW // 2) + cc for cc in range(UG_CPW // 2)]
                    nrec = fx.Int32(UG_W_BYTES)
                r_w = bo.create_buffer_resource_from_addr(
                    w_ug + fx.Int64(e_sel) * fx.Int64(UG_W_BYTES),
                    num_records_bytes=nrec,
                )
                r_s = _rsrc(s_ug + fx.Int64(e_sel) * fx.Int64(UG_S_BYTES))

                def mk(kb):
                    return unit_f8f8(
                        r_w,
                        r_s,
                        gu_row + c,
                        kb * 2,
                        UG_NKC,
                        HIDDEN,
                        bsel * XW + kb * 32,
                        lambda: lds_ld(misc, 8 + bsel * XQ_BLOCKS + kb),
                        shuffled=fp8_shuffled,
                    )

                return (shared, s_u, slot, c, e_sel, prob, [mk(kb) for kb in kbs])

            def ug_mid(shared, s_u, slot, c, e_sel, prob):
                """outs (per column: 16 gate then 16 up sums) -> SiLU(gate) * up -> mid of
                sample s_u, or of every sample n (column n) for the shared expert."""
                if tid < S * UG_TILE // 2:
                    n = tid // (UG_TILE // 2)
                    r = (tid % (UG_TILE // 2)) * 2
                    if shared | (n == 0):
                        g0, g1 = (
                            lds_ld(outs, n * 2 * UG_TILE + r),
                            lds_ld(outs, n * 2 * UG_TILE + r + 1),
                        )
                        v0 = lds_ld(outs, n * 2 * UG_TILE + UG_TILE + r)
                        v1 = lds_ld(outs, n * 2 * UG_TILE + UG_TILE + r + 1)
                        put2(
                            mb("mid"),
                            (shared.select(n, s_u) * MOE_SLOTS + slot) * expert_inter
                            + c * UG_TILE
                            + r,
                            g0 * _rcp(1.0 + _exp(-g0)) * v0,
                            g1 * _rcp(1.0 + _exp(-g1)) * v1,
                        )
                if (c == 0) & (tid < S):  # routing record (debug / tests)
                    if shared | (tid == 0):
                        put(
                            mb("sel"), shared.select(tid, s_u) * MOE_SLOTS + slot, e_sel
                        )
                        put(
                            mb("prob"), shared.select(tid, s_u) * MOE_SLOTS + slot, prob
                        )

            stamp("ug", bid, 5)
            dn_route(load_bias())  # every sample's top-8, in wave-sized batches
            gpu.barrier()
            stamp("ug", bid, 6)
            cur = ug_job(0)
            stamp("ug", bid, 0)
            stage_xq(list(range(S)))
            stamp("ug", bid, 2)
            gpu.barrier()
            job0 = cur[:6]
            for k in range_constexpr(NF + 1):
                shared, s_u, slot, c, e_sel, prob, pre = cur
                if const_expr(k > 0):
                    stamp("ug", k * G + bid, 0)
                if const_expr(k < NF):
                    cur = ug_job(k + 1)
                acc = mma_units([fx.Float32(0.0) for _ in range(4)], pre)
                reduce_rows(2, acc, emit_out(UG_TILE * 2))
                stamp("ug", k * G + bid, 3)
                gpu.barrier()
                if const_expr(k == 0):
                    if tid < S * 2 * UG_TILE:
                        put(
                            mb("ugp"),
                            ((x_seg - NF * G) * SEG + seg) * S * 2 * UG_TILE + tid,
                            lds_ld(outs, tid),
                        )
                else:
                    ug_mid(shared, s_u, slot, c, e_sel, prob)
                stamp("ug", k * G + bid, 4)
            if seg == 0:  # sum the leftover tile's K-segments
                gpu.barrier()
                if tid < S * 2 * UG_TILE:
                    parts = getf_many(
                        [
                            (
                                (mb("ugp")),
                                ((x_seg - NF * G) * SEG + j) * S * 2 * UG_TILE + tid,
                            )
                            for j in range(SEG)
                        ]
                    )
                    tot_p = parts[0]
                    for j in range_constexpr(1, SEG):
                        tot_p = tot_p + parts[j]
                    lds_st(outs, tid, tot_p)
                gpu.barrier()
                ug_mid(*job0)
        else:
            # S > 1 (the router already quantized every sample's activation): this CTA's
            # tasks are software pipelined -- task k+1 is routed (by every wave on its
            # own) and its weights are in flight while task k computes
            UG_NT = (N_UG + G - 1) // G
            NSC = N_EXPERTS // 64
            u0 = start("ug")

            def ug_prep(k):
                u = fx.Int32(u0 + k * G)
                live = u < N_UG
                s_u, slot, c = ug_task(fx.min(u, N_UG - 1))
                bs = load_bias()
                raws = getf_many(
                    [
                        (mb("scores"), s_u * N_EXPERTS + lane + i * 64)
                        for i in range(NSC)
                    ]
                )
                e, w = route_top8(s_u, raws, bs)
                i_pk = fx.max(slot - 1, 0)
                e_sel = _uniform(
                    (slot == 0).select(fx.Int32(SHARED), read_lane_i32(e, i_pk))
                )
                prob = (slot == 0).select(
                    fx.Float32(1.0),
                    read_lane_i32(w.bitcast(fx.Int32), i_pk).bitcast(fx.Float32),
                )
                u_ug = ug_units(e_sel, c, live)
                return (
                    u,
                    live,
                    s_u,
                    slot,
                    c,
                    e_sel,
                    prob,
                    u_ug,
                    [u_ug(cc) for cc in range(UG_CPW // 2)],
                )

            cur = ug_prep(0)
            for k in range_constexpr(UG_NT):
                u, live, s_u, slot, c, e_sel, prob, u_ug, pre = cur
                if live:
                    stamp("ug", u, 0)
                    stage_xq([s_u])
                    stamp("ug", u, 2)
                gpu.barrier()
                if const_expr(k + 1 < UG_NT):
                    cur = ug_prep(k + 1)
                if live:
                    ug_finish(u, s_u, slot, c, e_sel, lambda: prob, u_ug, pre)

        # ======== 10. mid FP8 quant + expert down + route weighting + MoE TP reduce
        (x_out, positions, slot_mapping, sparse_kv_indptr, kv_cache, pe_cache, indices, rope_cos, rope_sin, g_in, g_q, g_kv, g_post, w_qkv_a, s_qkv_a, w_q_b, s_q_b, w_uk, s_uk, w_uv, s_uv, w_o, s_o, w_r, bias, w_ug, s_ug, w_dn, s_dn, scratch, sym, timeline_buf, block_table, req_ids, out_indices, ix_wk, ix_sk, ix_ww, ix_wq, ix_sq, ix_gk, ix_bk) = reload_args()  # fmt: skip
        if const_expr(RELOAD_ARGS and x_pushed):
            xin = (
                sym
                + fx.Int64(SY["xin"])
                + fx.Int64(peer_slot) * fx.Int64(SY["_xin_stride"])
            )
        # 2 row groups x (sample tile * 9 slots * 4) chunks: 4 waves per group.
        # S=8 is evaluated as two four-sample groups so its FP8 mid tile and
        # in-flight weight batches fit comfortably in LDS/VGPRs.
        DN_NKC = expert_inter // 64
        DN_CM = CM_NT if S != 4 else 0
        DN_R = (
            (DN_TILE + 15) // 16
        )  # 16-row groups touched by a tile (24-row tiles start at row 0 or 8 of one)
        DN_WPR = WAVES // DN_R
        DN_BATCH = down_prefetch_batch(S, native_fp4_mfma)
        DN_W_BYTES = HIDDEN * expert_inter // (2 if expert_mxfp4 else 1)
        DN_S_BYTES = (
            HIDDEN * (expert_inter // 32)
            if expert_mxfp4
            else HIDDEN // SCALE_BM * (expert_inter // 128) * 4
        )
        for t in range(start("down"), N_DN_TILES, G):
            t = fx.Int32(t)
            stamp("down", t, 0)
            if const_expr(S == 1):  # S > 1 routed before up/gate
                dn_route(load_bias())
            gpu.barrier()
            gu = wave // DN_WPR
            dn_rg = t * DN_TILE // 16
            dn_off = t * DN_TILE % 16
            # this lane's row, as a tile row; rows outside the tile load their lane ^ 8 twin
            # (same cache lines) and are dropped in the output
            dn_lr = gu * 16 + lane % 16 - dn_off
            dn_ln = ((dn_lr >= 0) & (dn_lr < DN_TILE)).select(lane, lane ^ 8)

            DN_NU = (dense_experts if dense_fp8 else S * MOE_SLOTS) * DN_NKC // 2
            DN_UPW = (DN_NU + DN_WPR - 1) // DN_WPR
            DN_BLK = S * MOE_SLOTS * expert_inter // 128

            def u_dn_dense(cc):
                """A dense MLP's 128-k chunk cc of this wave: one slice's weights for
                every row, each lane's column scaled by its row's mid block scale."""
                qu = (wave % DN_WPR) * DN_UPW + cc
                live = qu < DN_NU
                q = fx.min(qu, DN_NU - 1) * 2  # 64-k chunk over (slice, kc)
                d_sl, kc = q // DN_NKC, q % DN_NKC
                nr = None if DN_NU % DN_WPR == 0 else live
                wb = bo.create_buffer_resource_from_addr(
                    w_dn + fx.Int64(d_sl) * fx.Int64(DN_W_BYTES),
                    num_records_bytes=(
                        None
                        if nr is None
                        else nr.select(fx.Int32(DN_W_BYTES), fx.Int32(0))
                    ),
                )
                sb = bo.create_buffer_resource_from_addr(
                    s_dn + fx.Int64(d_sl) * fx.Int64(DN_S_BYTES),
                    num_records_bytes=(
                        None
                        if nr is None
                        else nr.select(fx.Int32(DN_S_BYTES), fx.Int32(0))
                    ),
                )
                qn = (n_sel() * MOE_SLOTS + d_sl) * DN_NKC + kc
                return unit_f8f8(
                    wb,
                    sb,
                    dn_rg + gu,
                    kc,
                    DN_NKC,
                    expert_inter,
                    qn * 16,
                    lambda: lds_ld(misc, qn // 2),
                    dn_ln,
                    shuffled=fp8_shuffled,
                    cm=DN_CM,
                )

            def u_dn(cc):  # cc: 128-k chunk of this wave
                if const_expr(dense_fp8):
                    return u_dn_dense(cc)
                qu = (wave % DN_WPR) * DN_UPW + cc
                live = qu < DN_NU
                unit = fx.min(qu, DN_NU - 1)
                q = unit * 2  # 64-k chunk index over (s, slot, kc)
                s_q = q // (MOE_SLOTS * DN_NKC)
                slot_q = (q // DN_NKC) % MOE_SLOTS
                kc = q % DN_NKC
                e = _uniform(lds_ld(keys, s_q * MOE_SLOTS + slot_q))
                wb = bo.create_buffer_resource_from_addr(
                    w_dn + fx.Int64(e) * fx.Int64(DN_W_BYTES),
                    num_records_bytes=(
                        None
                        if DN_NU % DN_WPR == 0
                        else live.select(fx.Int32(DN_W_BYTES), fx.Int32(0))
                    ),
                )
                sb = bo.create_buffer_resource_from_addr(
                    s_dn + fx.Int64(e) * fx.Int64(DN_S_BYTES),
                    num_records_bytes=(
                        None
                        if DN_NU % DN_WPR == 0
                        else live.select(fx.Int32(DN_S_BYTES), fx.Int32(0))
                    ),
                )

                def coef():  # mid block scale * route weight, only in this sample's column
                    return (lane % 16 == s_q).select(
                        _uniform_f32(lds_ld(misc, q // 2)), fx.Float32(0.0)
                    )

                if const_expr(expert_mxfp4):

                    def bf16_coef():
                        return (lane % 16 == s_q).select(
                            _uniform_f32(lds_ld(dnw, s_q * MOE_SLOTS + slot_q)),
                            fx.Float32(0.0),
                        )

                    if const_expr(native_fp4_mfma):
                        return unit_native_mxfp4(
                            wb,
                            sb,
                            dn_rg + gu,
                            unit % (expert_inter // 128),
                            expert_inter,
                            unit * 32,
                            unit * 4 + lane // 16,
                            bf16_coef,
                            dn_ln,
                        )
                    return unit_mxfp4_bf16(
                        wb,
                        sb,
                        dn_rg + gu,
                        unit % (expert_inter // 128),
                        expert_inter,
                        unit * 64,
                        bf16_coef,
                        dn_ln,
                    )
                return unit_f8f8(
                    wb,
                    sb,
                    dn_rg + gu,
                    kc,
                    DN_NKC,
                    expert_inter,
                    q * 16,
                    coef,
                    dn_ln,
                    shuffled=fp8_shuffled,
                    s_addr=s_dn + fx.Int64(e) * fx.Int64(DN_S_BYTES),
                    cm=DN_CM,
                )

            # Above 4 rows (FP8 experts) down streams each expert the step uses once:
            # a unit (group g, k half h) multiplies the expert's tile with every
            # column row's mid chunk at its slot of it, and the products go to LDS;
            # each column's sum is then taken over its units in the order and
            # the wave segments of the one-row-a-wave form, so it is the same bits.
            DEDUP = S > 4 and not expert_mxfp4 and not dense_experts
            SLOT0_DN = JOB_SLOT_OFF - JOB_OFF
            ROW_U = MOE_SLOTS * DN_KU  # a row's units
            assert not DEDUP or ROW_U % min(DN_UPW, ROW_U) == 0
            DD_UPW = (DN_KU * MOE_GROUPS + DN_WPR - 1) // DN_WPR
            prod = xs + DN_PROD_OFF

            def u_dd(m):
                """Unit m of this wave: (unit, live, the lane's row's slot, k half)."""
                i = (wave % DN_WPR) + m * DN_WPR
                n_groups = _uniform(lds_ld(jobs, 0))
                live = i < DN_KU * n_groups
                g = fx.min(i // DN_KU, n_groups - 1)
                h = i % DN_KU
                e = _uniform(lds_ld(jobs, 1 + g))
                sl = lds_ld(jobs, SLOT0_DN + g * S + n_sel())
                blk = (n_sel() * MOE_SLOTS + fx.max(sl, 0)) * DN_KU + h
                wb = bo.create_buffer_resource_from_addr(
                    w_dn + fx.Int64(e) * fx.Int64(DN_W_BYTES),
                    num_records_bytes=live.select(fx.Int32(DN_W_BYTES), fx.Int32(0)),
                )
                unit = unit_f8f8(
                    wb,
                    wb,
                    dn_rg + gu,
                    2 * h,
                    DN_NKC,
                    expert_inter,
                    blk // DN_KU * MID_STRIDE + h * 32,
                    lambda: lds_ld(misc, blk),
                    dn_ln,
                    shuffled=fp8_shuffled,
                    s_addr=s_dn + fx.Int64(e) * fx.Int64(DN_S_BYTES),
                    cm=DN_CM,
                )
                return unit, live, sl, h

            def dd_products(units):
                """MFMA each unit; store a used column's products (its 4 rows)."""
                for (fmt, wv, coef, bw), live, sl, h in units:
                    a = fx.Vector.from_elements(
                        [wv[hh][ee] for hh in range(2) for ee in range(4)], fx.Int32
                    )
                    bv = [
                        fx.ptr_load(xs + (bw + hh * 16), result_type=v4f).bitcast(
                            fx.Int32
                        )
                        for hh in range(2)
                    ]
                    b = fx.Vector.from_elements(
                        [bv[hh][ee] for hh in range(2) for ee in range(4)], fx.Int32
                    )
                    one = fx.Int32(127)
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
                    k = coef()
                    if live & (sl >= 0) & (lane % 16 < S):
                        fx.ptr_store(
                            fx.Vector.from_elements(
                                [c[ee] * k for ee in range(4)], fx.Float32
                            ),
                            prod
                            + (
                                (gu * S + lane % 16) * DN_PROD_STRIDE
                                + (sl * DN_KU + h) * 16
                                + (lane // 16) * 4
                            ),
                        )

            # FP8 experts: mid block b (a row's slot's 128 intermediates) is quantized
            # once, by wave k of down task t for b = t + k * N_DN_TILES, and every
            # task reads the FP8 blocks (a quarter of mid's bytes)
            if const_expr(not expert_mxfp4):
                assert DN_BLK <= WAVES * N_DN_TILES
                qb = t + wave * N_DN_TILES
                if qb < DN_BLK:
                    m0, m1 = get2(mb("mid"), qb * 128 + lane * 2)
                    q0, q1, qs = quant_scaled(m0, m1)
                    w8 = (
                        fx.Int32(
                            rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)
                        )
                        & 0xFFFF
                    )
                    word = w8 | (_xshfl(w8, 1) << 16)  # even lanes: bytes 2 lane .. + 3
                    word_n = _xshfl(word, 2)
                    if lane % 4 == 0:
                        put2(mb("midq"), qb * 32 + lane // 2, word, word_n)
                    if lane == 0:
                        put(mb("midqs"), qb, qs)
            # the routing is known before up/gate: the first weight batch loads
            # before the mid wait (above 4 rows only with the expert groups)
            if const_expr(DEDUP):
                pre = [u_dd(m) for m in range(min(DN_BATCH, DD_UPW))]
            elif const_expr(S <= 4):
                pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
            hint_wait(
                N_UG,
                lambda k: (
                    mb("mid"),
                    (
                        k // (MOE_SLOTS * I_PER_SLOT) * MOE_SLOTS
                        + (k // I_PER_SLOT) % MOE_SLOTS
                    )
                    * expert_inter
                    + (k % I_PER_SLOT) * UG_TILE
                    + UG_TILE
                    - 1,
                ),
                mark=("down", t),
            )
            if const_expr(expert_mxfp4):
                mids = get2_many(
                    [
                        (
                            mb("mid"),
                            fx.min(wave + b * WAVES, DN_BLK - 1) * 128 + lane * 2,
                        )
                        for b in range((DN_BLK + WAVES - 1) // WAVES)
                    ]
                )
                stamp("down", t, 2)
                for b in range_constexpr((DN_BLK + WAVES - 1) // WAVES):
                    blk = wave + b * WAVES
                    if blk < DN_BLK:
                        if const_expr(native_fp4_mfma):
                            q0, q1, qs = quant_mxfp8(mids[b][0], mids[b][1])
                            st_mxfp8(blk * 128 + lane * 2, q0, q1)
                            if lane % 16 == 0:
                                lds_st(
                                    misc,
                                    blk * 4 + lane // 16,
                                    qs.bitcast(fx.Float32),
                                )
                        else:
                            lds_st(
                                xs,
                                blk * 64 + lane,
                                bf16_pair(mids[b][0], mids[b][1]),
                            )
            else:
                # the FP8 blocks: two words (8 values) a load, the scales one a thread
                Q_LD = (DN_BLK * 16 + THREADS - 1) // THREADS
                got = poll(
                    [
                        (mb("midq"), fx.min(tid + i * THREADS, DN_BLK * 16 - 1) * 2, 2)
                        for i in range(Q_LD)
                    ]
                    + [(mb("midqs"), fx.min(tid, DN_BLK - 1), 1)]
                )
                stamp("down", t, 2)
                for i in range_constexpr(Q_LD):
                    j = tid + i * THREADS
                    if j < DN_BLK * 16:
                        for e in range_constexpr(2):
                            k = (j * 2 + e) * 4  # the word's first value
                            if const_expr(DEDUP):  # a (row, slot) block a stride
                                at = k // expert_inter * MID_STRIDE + f8_word(
                                    k % expert_inter
                                )
                            else:
                                at = f8_word(k)
                            lds_st(xs, at, got[i][e].bitcast(fx.Float32))
                if tid < DN_BLK:
                    lds_st(
                        misc,
                        tid,
                        got[Q_LD][0].bitcast(fx.Float32)
                        * lds_ld(dnw, tid // (expert_inter // 128)),
                    )
            gpu.barrier()

            def emit_dn(rl, n, v):
                if (rl >= dn_off) & (rl < dn_off + DN_TILE):
                    lds_st(outs, n * DN_TILE + rl - dn_off, v)

            if const_expr(DEDUP):
                dd_units = DN_KU * _uniform(lds_ld(jobs, 0))  # live units a row group
                cur = pre
                for b0 in range_constexpr(0, DD_UPW, DN_BATCH):
                    now = cur
                    if const_expr(b0 + DN_BATCH < DD_UPW):
                        cur = [
                            u_dd(m)
                            for m in range(
                                b0 + DN_BATCH, min(b0 + 2 * DN_BATCH, DD_UPW)
                            )
                        ]
                    # a batch past the step's expert groups has no products
                    if (wave % DN_WPR) + b0 * DN_WPR < dd_units:
                        dd_products(now)
                gpu.barrier()
                # a column's units are a wave's in the one-row-a-wave form when the
                # wave segment (DN_UPW units) covers a whole row, else halves of it
                SEG = min(DN_UPW, ROW_U)

                def dd_sum(t_):
                    rl = t_ % (DN_R * 16)
                    n = t_ // (DN_R * 16)
                    at = ((rl // 16) * S + n) * DN_PROD_STRIDE + rl % 16
                    tot = fx.Float32(0.0)
                    for s0 in range_constexpr(0, ROW_U, SEG):
                        part = fx.Float32(0.0)
                        for u in range_constexpr(s0, s0 + SEG):
                            part = part + lds_ld(prod, at + u * 16)
                        tot = tot + part
                    emit_dn(rl, n, tot)

                for i in range_constexpr((DN_R * 16 * S + THREADS - 1) // THREADS):
                    if tid + i * THREADS < DN_R * 16 * S:
                        dd_sum(tid + i * THREADS)
            else:
                if const_expr(S > 4):
                    pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
                acc = run_units(u_dn, DN_UPW, DN_BATCH, pre)
                reduce_rows(DN_R, acc, emit_dn)
            stamp("down", t, 3)
            gpu.barrier()

            def store_x(s, row, v0, v1):
                bo.buffer_store(
                    fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16),
                    _rsrc(x_out),
                    s * HIDDEN + row,
                )

            def residual_a(s, row):
                return bf2_f32(get(mb("a"), (s * HIDDEN + row) // 2))

            peer_reduce("ffn", t, residual_a, store_x, tile=DN_TILE)
            gpu.barrier()
            stamp("down", t, 4)
        guarded.__exit__(None, None, None)

    @flyc.jit
    def launch(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        positions: Int64,
        slot_mapping: Int64,
        sparse_kv_indptr: Int64,
        kv_cache: Int64,
        pe_cache: Int64,
        indices: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_in: Int64,
        g_q: Int64,
        g_kv: Int64,
        g_post: Int64,
        w_qkv_a: Int64,
        s_qkv_a: Int64,
        w_q_b: Int64,
        s_q_b: Int64,
        w_uk: Int64,
        s_uk: Int64,
        w_uv: Int64,
        s_uv: Int64,
        w_o: Int64,
        s_o: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        timeline_buf: Int64,
        step: Int64,
        block_table: Int64,
        req_ids: Int64,
        out_indices: Int64,
        ix_wk: Int64,
        ix_sk: Int64,
        ix_ww: Int64,
        ix_wq: Int64,
        ix_sq: Int64,
        ix_gk: Int64,
        ix_bk: Int64,
        rank: Int32,
        layer: Int32,
        bt_stride: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = keyed
        glm5_monokernel(
            h_in,
            x_out,
            cur_pos,
            positions,
            slot_mapping,
            sparse_kv_indptr,
            kv_cache,
            pe_cache,
            indices,
            rope_cos,
            rope_sin,
            g_in,
            g_q,
            g_kv,
            g_post,
            w_qkv_a,
            s_qkv_a,
            w_q_b,
            s_q_b,
            w_uk,
            s_uk,
            w_uv,
            s_uv,
            w_o,
            s_o,
            w_r,
            bias,
            w_ug,
            s_ug,
            w_dn,
            s_dn,
            scratch,
            sym,
            peers,
            timeline_buf,
            step,
            block_table,
            req_ids,
            out_indices,
            ix_wk,
            ix_sk,
            ix_ww,
            ix_wq,
            ix_sq,
            ix_gk,
            ix_bk,
            rank,
            layer,
            bt_stride,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    return launch
