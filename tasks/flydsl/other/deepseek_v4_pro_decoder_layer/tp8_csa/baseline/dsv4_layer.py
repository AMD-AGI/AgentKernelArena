# SPDX-License-Identifier: Apache-2.0
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 4cd75a2
# (ROCm/ATOM#2559), MIT License, Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v4/mono/kernels/layer.py
# Vendored for AgentKernelArena from sgl-amd-mono 52486703: imports made relative; the build key, build constants and launch values are spelled out
# instead of locals(); a fixed source tag and _key_dependencies replace kernel_sources; the scrub kernel
# (unused: the evaluator zeroes the mailboxes every call) is removed.
"""DeepSeek-V4 attention + MoE layer in ONE persistent launch per rank (TP8 decode).

V4 is shared-KV (MQA): ``wkv`` emits one ``HEAD_DIM`` vector per token, K and V
are the same tensor, and RoPE occupies its last ``ROPE_DIM`` lanes; there is no
nope/pe split and no absorbed ``W_UK``/``W_UV``.  ``split`` adds a per-head
softmax ``sink``; ``uv`` merges the splits and de-rotates the output's RoPE lanes;
``o`` is the grouped low-rank pair ``o_a`` then ``o_b``.

One launch of ``grid = 256 CTAs x 512 threads`` (one CTA per MI355X CU) runs the
whole layer body for this rank's TP shard::

    input RMSNorm -> q_a / kv projection -> q_a RMSNorm -> q_b (+ head RMS, RoPE)
      -> KV RMSNorm / RoPE / FP8 round-trip -> sliding-window KV ring publish
      -> gather-sparse split softmax (+ sink) -> merge -> inverse RoPE
      -> o_a (grouped low rank) -> o_b
      -> attention TP peer reduce + residual                       (sym_attn)
      -> post-attention RMSNorm -> router sqrt-softplus + expert activation
      -> flat top-6 -> 1 shared + 6 routed expert up/gate/clamped SwiGLU
      -> expert down + route weighting
      -> MoE TP peer reduce + residual -> x_out                    (sym_ffn)

The KV compressor (HCA, ratio 128) and the lightning indexer (CSA, ratio 4) add
stages; hyper-connections (``hc_mult > 1``) replace the plain residual.

Scheduling: task ``t`` of a stage runs on CTA ``(stage_base + t) % 256`` and
every CTA walks the stages in order.  No grid-wide barrier: dependencies only
point to earlier stages and all CTAs are co-resident, so every spin wait
makes progress.

Mailboxes are tagged pairs ``(value, tag)`` stored with device- (``sc1``) or
system-coherent (``sc0 sc1``) 8 / 16-byte stores; a consumer polls the payload
until the tag matches this launch's epoch, so a hand-off is one round trip.

GEMVs run on the matrix cores: ``packing.py`` arranges weights so one wave loads
16 rows x 64 k as one contiguous 1 KB, FP8 is widened exactly to bf16 and fed to
``mfma_f32_16x16x32_bf16`` with the samples as N; each 64-k partial is scaled by
its f32 block scale.  Weight loads are issued before waiting for inputs.

Cross-GPU: each rank pushes BF16 partial rows plus a tag into every peer's
symmetric buffer; every rank sums the 8 partials in rank order, so all ranks
produce bit-identical hidden states (and routing).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from . import mono_guard as guard
from .mono_ops import (
    rsrc,
    uniform,
)
from .mono_plan import (
    BLOCKS,
    THREADS,
    WAVES,
    key_tuple,
)
from .dsv4_config import (
    BLOCK_TOKENS,
    HC_EPS,
    HC_SINKHORN_ITERS,
    HEAD_DIM,
    HIDDEN,
    INTER,
    N_EXPERTS,
    O_GROUPS,
    O_LORA,
    Q_LORA,
    ROPE_DIM,
    SOFTMAX_SCALE,
    SWIGLU_LIMIT,
    TOP_K,
    WINDOW,
)
from .dsv4_common import (
    common_defs,
)
from .stage_attention import (
    attention_stages,
)
from .stage_ffn import (
    ffn_stages,
)
from .stage_hc import (
    hc_attn_stages,
)
from .stage_indexer import (
    index_query_stages,
    index_select_stages,
)
from .stage_qkv import (
    q_b_stage,
    qkv_stages,
)
from .dsv4_helpers import (
    mb,
    put,
    put2,
    put_bf,
    poll,
    get,
    getf,
    getf_many,
    get2_many,
    pre_poll,
    hint_wait,
    start,
    block_sums,
    block_sum,
    unit_fp8,
    unit_fp8mx,
    unit_bf16,
    unit_mxfp4_atom,
    mma_units,
    run_units,
    reduce_rows,
    emit_out,
    n_sel,
    rmsnorm_tail_ks,
    load_x_rmsnorm,
    stage_x_rmsnorm,
    stage_x_pairs,
    load_bias,
    peer_reduce,
    _store_pairs,
    _store_pairs_if,
    _scale_parked,
    _scale_parked_if,
    _stage_block_quant,
)
from .mono_sync import _spin
from .dsv4_layout import (
    LAYER_SLOTS,
    Q_B_TILE,
    QKV_A_TILE,
    ROUTER_TILE,
    ROW_TILE,
    SPLIT_KEYS,
    UG8,
    UG_TILE,
    dn_tile,
    hc_shape,
    layout,
    n_index,
    n_topk_parts,
    n_topk_trips,
    qkv_a_rows,
    stage_tasks,
    uv_tile,
)

_CURRENT_STREAM = fx.Stream(None)
# AgentKernelArena: upstream digests the traced modules' source files (kernel_sources); the
# vendored package reads no files, so the JIT key carries this fixed tag instead (bumped
# with every change to the vendored device code, so no FlyDSL disk cache serves an older build).
SOURCES = "sgl-amd-mono@52486703/deepseek_v4/gfx950+aka-vendored-v2"

def _key_dependencies():
    """Every helper the stages reach only through the context dict or ``LaunchState``,
    named here so that FlyDSL's JIT cache key (which follows module globals by name from
    the kernel) hashes their source too. AgentKernelArena addition: upstream covers them
    with the ``kernel_sources`` file digest instead."""
    return (
        mb,
        put,
        put2,
        put_bf,
        poll,
        get,
        getf,
        getf_many,
        get2_many,
        pre_poll,
        hint_wait,
        start,
        block_sums,
        block_sum,
        unit_fp8,
        unit_fp8mx,
        unit_bf16,
        unit_mxfp4_atom,
        mma_units,
        run_units,
        reduce_rows,
        emit_out,
        n_sel,
        rmsnorm_tail_ks,
        load_x_rmsnorm,
        stage_x_rmsnorm,
        stage_x_pairs,
        load_bias,
        peer_reduce,
        _store_pairs,
        _store_pairs_if,
        _scale_parked,
        _scale_parked_if,
        _stage_block_quant,
        _spin,
    )



def build_dsv4_kernel(
    S: int = 1,
    heads: int = 16,
    npes: int = 8,
    window: int = WINDOW,
    scale: float = SOFTMAX_SCALE,
    spec_steps: int = 0,
    hidden: int = HIDDEN,
    q_lora: int = Q_LORA,
    head_dim: int = HEAD_DIM,
    o_groups: int = O_GROUPS,
    o_lora: int = O_LORA,
    n_experts: int = N_EXPERTS,
    top_k: int = TOP_K,
    inter: int = INTER,
    swiglu_limit: float = SWIGLU_LIMIT,
    hc_mult: int = 1,
    hc_sinkhorn_iters: int = HC_SINKHORN_ITERS,
    hc_eps: float = HC_EPS,
    compress_ratio: int = 0,
    window_rows: int | None = None,
    n_keys: int | None = None,
    c_coff: int = 1,
    index_head_dim: int = 0,
    index_heads: int = 0,
    index_topk: int = 0,
    max_seq: int = 0,
    indexer_hadamard: bool = True,
    c_ring: int = 0,
    fault_site: int = 0,
    fault_after: int = 0,
):
    """Return the ``@flyc.jit`` launcher for one rank's whole V4 layer.

    ``window`` keys come per sample from a ring cache (-1 = unwritten); ``compress_ratio``
    > 0 appends the compressed entries (all for HCA, the indexer's top-k for CSA).
    ``spec_steps`` > 0 serves speculative verify steps of up to ``spec_steps`` + 1 rows a
    request: a request's rows are consecutive tokens, contiguous in the launch, and
    ``run_pos`` holds each row's index among them. ``c_ring`` is the rows of SGLang's
    compressor state ring (0: ``C_ROWS + spec_steps``); it holds at least the window
    and a step's draft positions, so draft rows never alias the window a later step
    reads after a rejection. ``fault_site`` / ``fault_after``: the test fault injection
    (``common/guard.py``).
    """
    # the build parameters, named explicitly (upstream: key_tuple(locals(), SOURCES))
    keyed = key_tuple(
        {
            "S": S,
            "heads": heads,
            "npes": npes,
            "window": window,
            "scale": scale,
            "spec_steps": spec_steps,
            "hidden": hidden,
            "q_lora": q_lora,
            "head_dim": head_dim,
            "o_groups": o_groups,
            "o_lora": o_lora,
            "n_experts": n_experts,
            "top_k": top_k,
            "inter": inter,
            "swiglu_limit": swiglu_limit,
            "hc_mult": hc_mult,
            "hc_sinkhorn_iters": hc_sinkhorn_iters,
            "hc_eps": hc_eps,
            "compress_ratio": compress_ratio,
            "window_rows": window_rows,
            "n_keys": n_keys,
            "c_coff": c_coff,
            "index_head_dim": index_head_dim,
            "index_heads": index_heads,
            "index_topk": index_topk,
            "max_seq": max_seq,
            "indexer_hadamard": indexer_hadamard,
            "c_ring": c_ring,
            "fault_site": fault_site,
            "fault_after": fault_after,
        },
        SOURCES,
    )
    assert heads % WAVES == 0 and heads <= 16, (
        "the split-attention mapping needs a whole number of wave-groups per head, heads <= 16"
    )
    assert window % SPLIT_KEYS == 0
    # what bounds the sample count: fail the build rather than hang on the GPU
    assert 1 <= S <= 16, (
        "n_sel / reduce_rows put the samples in the MFMA's 16 B columns"
    )
    assert S <= WAVES, (
        f"dn_route routes sample s on wave s, and there are {WAVES} waves"
    )
    assert S * (1 + top_k) <= SPLIT_KEYS, (
        f"dn_route packs S * MOE_SLOTS = {S * (1 + top_k)} expert ids into Smem.keys, "
        f"which the split stage sizes at SPLIT_KEYS = {SPLIT_KEYS}"
    )
    assert head_dim % 64 == 0, (
        "the score MFMA walks HEAD_DIM in 32-wide steps over 2 wave halves"
    )
    # too small a HEAD_DIM gives a wave no PV / gather work: an unfilled mailbox, a hang
    assert head_dim % (32 * WAVES) == 0, (
        f"head_dim must be a multiple of {32 * WAVES} for the PV MFMA's per-wave dim groups, got {head_dim}"
    )
    assert head_dim >= 128, "the KV gather needs at least one packed word per lane"
    assert (head_dim - ROPE_DIM) % 64 == 0, (
        "the KV row's FP8 round-trip blocks the nope part by 64"
    )
    assert heads % o_groups == 0, "o_a groups partition the concatenated heads"
    assert o_lora % ROW_TILE == 0 and (heads * head_dim // o_groups) % 64 == 0
    assert head_dim <= THREADS, "the cache / q_norm stages map one thread per head dim"
    assert (head_dim - ROPE_DIM) % 2 == 0, (
        "interleaved RoPE pairs must align to the nope boundary"
    )
    assert n_experts % 64 == 0, (
        "route_topk packs n_experts // 64 selection keys per lane"
    )
    assert n_experts <= 1 << 16, (
        "the packed selection key needs room for the score above the id field"
    )
    assert inter % UG_TILE == 0
    # shadow the module constants with this build's overrides (before any read of them)
    HIDDEN = hidden
    Q_LORA = q_lora
    HEAD_DIM = head_dim
    O_GROUPS = o_groups
    O_LORA = o_lora
    N_EXPERTS = n_experts
    TOP_K = top_k
    INTER = inter
    NOPE_DIM = HEAD_DIM - ROPE_DIM
    MOE_SLOTS = 1 + TOP_K
    SHARED_EXPERT = N_EXPERTS
    QKV_A_ROWS = (
        Q_LORA + HEAD_DIM
    )  # the FP8 rows; the compressors' BF16 rows follow (qkv_c)
    N_QKV_A = QKV_A_ROWS // QKV_A_TILE
    N_QKV_C = (
        qkv_a_rows(Q_LORA, HEAD_DIM, compress_ratio, c_coff, index_head_dim)
        - QKV_A_ROWS
    ) // QKV_A_TILE
    N_ROW_TILES = HIDDEN // ROW_TILE
    N_ROUTER = N_EXPERTS // ROUTER_TILE
    # the routed experts run A8W4, as the original decode: MXFP4 weights on MXFP8
    # activations, one E8M0 scale per 32-value block
    XQ_BLOCKS = HIDDEN // 32
    XQ_WAVES = (XQ_BLOCKS + N_ROUTER * 4 - 1) // (N_ROUTER * 4)
    assert XQ_WAVES <= WAVES
    # KV compression (CR == 0 compiles out); CSA's entries pool 2*CR tokens at stride CR (C_COFF = 2)
    CR = compress_ratio
    C_COFF = c_coff
    # the lightning indexer's own compressor (Hadamard, FP4); IHD == 0 compiles it out
    IHD = index_head_dim
    INDEXER_HADAMARD = indexer_hadamard
    if CR:
        assert BLOCK_TOKENS % CR == 0, (
            "a block holds a whole number of compressed entries"
        )
    # SGLang's two-pool fp8 unified KV (unified_kv_kernels/layout.py): a NoPE plane of
    # KV_ROW_BYTES per row (NOPE_DIM FP8 bytes, each 64-group's E8M0 byte twice,
    # padding) plus a bf16 RoPE plane, one row index addressing both
    KV_ROW_BYTES = 512
    assert (
        NOPE_DIM % 64 == 0 and HEAD_DIM - NOPE_DIM == ROPE_DIM and THREADS == HEAD_DIM
    )
    IW = C_COFF * IHD
    # Compressed entries are paged by SGLang's 256-token pages: entry e of sequence s
    # is KV row dest_rows[1, s] + block_tables[s][e // K_PB] * env_rows + e % K_PB
    # (the SWA ring's rows, then the page's K_PB entries). The indexer's FP4 key pool:
    # per page, codes [IHD / 32][K_PB][16 B] and E8M0 scales [IHD / 32][K_PB] with the
    # entry axis interleaved (byte (e % 16) * 4 + (e % K_PB) // 16).
    K_PB = BLOCK_TOKENS // CR if CR else 1
    IC_GRP_WORDS = K_PB * 16 // 4  # one 32-element group of one block, in dwords
    IC_BLK_WORDS = (IHD // 32) * IC_GRP_WORDS if IHD else 0
    IC_S_BLK = (IHD // 32) * K_PB if IHD else 0  # scale bytes of one block
    CW = C_COFF * HEAD_DIM  # kv (or score) floats of one state row
    # SGLang's state row is [kv | score] (CompressStatePool): the score pool is the kv
    # pool CW floats on, and both step 2 * CW floats a row
    C_ROW_W = 2 * CW
    I_ROW_W = 2 * IW
    # compressor state: a ring indexed by position (window at p: rows (p + 1 + i) % C_ROWS)
    C_ROWS = C_COFF * CR
    SPEC_STEPS = spec_steps
    # a request's rows in one launch, at most: its anchor and its drafted tokens
    RUN_MAX = min(SPEC_STEPS + 1, S)
    assert RUN_MAX == 1 or not CR or RUN_MAX <= C_ROWS, (
        "the in-launch tail of the window is at most its length"
    )
    # the ring the state is stored in: at least C_ROWS plus the K draft positions of a step
    C_RING = c_ring or C_ROWS + SPEC_STEPS
    assert not CR or C_RING >= C_ROWS + SPEC_STEPS, (
        f"a {C_RING}-row state ring cannot hold the window and {SPEC_STEPS} drafts"
    )
    OVERLAP = C_COFF > 1
    # state rows the pooling loop loads per trip
    CMP_CHUNK = max(c for c in range(1, 9) if C_ROWS % c == 0) if C_ROWS else 1
    # with RUN_MAX > 1 the ring part of the window is its first C_ROWS - RUN_MAX rows
    CMP_CHUNK_T = (
        max(c for c in range(1, 9) if (C_ROWS - RUN_MAX) % c == 0)
        if (C_ROWS and RUN_MAX > 1)
        else 1
    )
    # window and compressed entries share one cache, the compressed half at `window`
    CACHE_ROWS = window if window_rows is None else window_rows
    N_KEYS = window if n_keys is None else n_keys
    if CR:
        assert CACHE_ROWS > window, (
            "a compressing layer needs cache rows past the window"
        )
        assert HEAD_DIM <= THREADS, "the compressor maps one thread per channel"

    # hyper-connections (HC == 1 is a plain residual and compiles out)
    HC = hc_mult
    HC_TASKS, HC_KSLICE, HC_ROWS, HC_VALS = hc_shape(HC, HIDDEN)
    HC_MIX = (2 + HC) * HC if HC > 1 else 0
    # waves sharing the coefficient poll, one poll batch (POLL_MAX) of hcd tasks each
    HC_PW = min(WAVES, -(-HC_TASKS // 12)) if HC > 1 else 1
    HC_TPW = -(-HC_TASKS // HC_PW) if HC > 1 else 1
    HC_COEF = 2 * HC + HC * HC if HC > 1 else 0
    # comb lane = j * HC + k: XOR offsets walk a row (low bits) or a column (high bits)
    HC_ROW_OFFS = tuple(1 << b for b in range(HC.bit_length() - 1)) if HC > 1 else ()
    HC_COL_OFFS = tuple(HC << b for b in range(HC.bit_length() - 1)) if HC > 1 else ()
    HC_NKC = HC_KSLICE // 64 if HC > 1 else 0
    HC_RG = HC_ROWS // 16 if HC > 1 else 0
    HC_WPR = WAVES // HC_RG if HC > 1 else 0
    HC_NKC_FULL = (HC * HIDDEN) // 64 if HC > 1 else 0
    if HC > 1:
        assert HC & (HC - 1) == 0, (
            "hc_mult must be a power of two for the cross-lane Sinkhorn"
        )
        assert HC * HC <= 64, "the Sinkhorn matrix must fit one wave"
        assert HC_NKC % HC_WPR == 0, (
            "hc_pre chunks must divide over the waves of a row group"
        )

    # misc[:WAVES]: the KV row's per-wave FP8 scale bytes (put_kv_row); the mHC coefficients after
    HC_MISC = WAVES
    # `uv` stores one per-split weight in misc, so it must hold N_SPLIT of them
    misc_words = max(HC_MISC + S * max(HC_COEF, 1), n_keys // SPLIT_KEYS)
    H = heads
    W = npes
    G = BLOCKS
    SC, SY = layout(
        S,
        H,
        W,
        window,
        hidden=HIDDEN,
        q_lora=Q_LORA,
        head_dim=HEAD_DIM,
        o_groups=O_GROUPS,
        o_lora=O_LORA,
        hc_mult=HC,
        compress_ratio=CR,
        n_keys=N_KEYS,
        c_coff=C_COFF,
        index_head_dim=IHD,
        index_heads=index_heads,
        index_topk=index_topk,
        max_seq=max_seq,
        n_experts=N_EXPERTS,
        top_k=TOP_K,
        inter=INTER,
    )
    assert N_KEYS % SPLIT_KEYS == 0, (
        "the index list must be a whole number of key tiles"
    )
    N_SPLIT = N_KEYS // SPLIT_KEYS
    # split/merge sized per launch by the live keys (see live_splits)
    LIVE_SPLITS = bool(compress_ratio) and not index_head_dim and N_KEYS > window
    # the `uv` merge gives each split one thread: one wave while they fit, else the block
    UV_WIDE = N_SPLIT > THREADS // WAVES
    UV_CHUNK = max(c for c in range(1, 9) if N_SPLIT % c == 0)
    assert N_SPLIT <= THREADS, (
        f"{N_SPLIT} key splits exceeds the {THREADS} threads the block-wide `uv` merge has. "
        f"n_keys {N_KEYS} = window {window} + the compressed list, so this is a max_seq limit"
    )
    N_QB = H * HEAD_DIM // Q_B_TILE
    IH = index_heads if IHD else 0
    N_IQB = IH * IHD // Q_B_TILE
    N_IHG = -(-IH // 16)  # 16-head MFMA column groups of the score
    N_COMP = (max_seq // CR) if (IHD and CR) else 0
    N_INDEX = n_index(max_seq, compress_ratio, index_head_dim, index_topk)
    # a top-k thread takes TK_PER consecutive candidates per trip (two 16-byte loads)
    TK_PER = 4
    TK_BINS = 256  # 8-bit radix digit; 4 passes cover the key
    # bin copies per lane group, so near-identical top digits do not serialize the atomics
    TK_REP = 16
    TK_BC = (
        TK_BINS * TK_REP
    )  # the four words past the bins that broadcast a pass's result
    TK_PARTS = n_topk_parts(max_seq, compress_ratio, index_head_dim)
    # trips PER PART: part q takes every TK_PARTS-th trip, starting at q
    TK_TRIPS = n_topk_trips(max_seq, compress_ratio, index_head_dim)
    # the index list pads to a whole key tile; the surplus is -1
    N_ISEL = N_KEYS - window
    if IHD:
        assert window % SPLIT_KEYS == 0, (
            "a key tile must fall wholly inside or outside the window"
        )
        assert N_KEYS >= window + N_INDEX, (
            "the index list must hold the window and the pick"
        )
        assert IHD % 8 == 0 and N_COMP > 0
        # keeps a clamped group 16-byte aligned and wholly past n_live (so masked)
        assert N_COMP % TK_PER == 0, "the top-k reads whole groups of TK_PER candidates"
        # every part polls every other part's bins: two parts on one CTA would hang
        assert S * TK_PARTS <= BLOCKS, "each top-k part needs its own CTA"
        # 32 per thread covers a 1M context
        assert TK_TRIPS * TK_PER <= 32, (
            "the top-k's candidates per thread must fit in registers"
        )
        # its compaction keeps the parts' trip totals and bases past the bin words
        assert TK_PARTS * TK_TRIPS + 2 * TK_TRIPS <= TK_BC, "the top-k's bases must fit"
        assert TK_PARTS * TK_TRIPS <= THREADS, "a thread scans each (trip, part) total"
        # its compaction packs two per-part counts into one scanned word
        assert TK_TRIPS * TK_PER * THREADS < 1 << 16, "a part's counts must fit 16 bits"
        assert IHD == 128, "the indexer's Hadamard is written for a 128-wide head"
        assert K_PB == 64, (
            "the FP4 pool's scale interleave is written for 64 entries a block (4 runs of 16)"
        )
        assert IH % WAVES == 0, "one wave takes a whole index head"
        assert IHD - ROPE_DIM == 64, "rope must fall entirely in the head's second half"
    UV_TILE = uv_tile(S, H, HEAD_DIM)  # shadows the module minimum
    N_UV = H * HEAD_DIM // UV_TILE
    UV_PER_HEAD = HEAD_DIM // UV_TILE
    OA_K = H * HEAD_DIM // O_GROUPS  # one group's slice of the concatenated heads
    OA_PER_GROUP = O_LORA // ROW_TILE
    OB_K = O_GROUPS * O_LORA
    UG_PER_SLOT = INTER // UG8
    N_UG_TASKS = TOP_K * UG_PER_SLOT
    # route weight sum over lanes 0..TOP_K-1 (lanes >= TOP_K hold 0)
    _off, TOPK_SUM_OFFS = 1, []
    while _off < TOP_K:
        TOPK_SUM_OFFS.append(_off)
        _off *= 2
    TOPK_SUM_OFFS = tuple(TOPK_SUM_OFFS)
    # one KV tile: RoPE is inside the HEAD_DIM vector
    QK_DIM = HEAD_DIM
    QS = QK_DIM // 2 + 4
    KS = HEAD_DIM // 2 + 4
    KT_OFF = H * QS
    XN = max(S * HIDDEN // 2, KT_OFF + SPLIT_KEYS * KS)
    ON = S * max(ROW_TILE, QKV_A_TILE, UG_TILE * 2)
    DN_TILE = dn_tile(S, HIDDEN)
    N_DN_TILES = HIDDEN // DN_TILE

    st_args = {
        "window": window,
        "hidden": HIDDEN,
        "q_lora": Q_LORA,
        "head_dim": HEAD_DIM,
        "o_groups": O_GROUPS,
        "o_lora": O_LORA,
        "top_k": TOP_K,
        "inter": INTER,
        "hc_mult": HC,
        "compress_ratio": CR,
        "n_keys": N_KEYS,
        "c_coff": C_COFF,
        "index_head_dim": IHD,
        "index_heads": index_heads,
        "index_topk": index_topk,
        "max_seq": max_seq,
        "n_experts": N_EXPERTS,
    }
    base = {}
    # CTA placement: split is placed before q_b so its tasks land on CTAs freed by qkv_a
    tasks = dict(stage_tasks(S, H, **st_args))
    acc = 0
    for name in (
        "hcd_a",
        "hcc_a",
        "qkv_a",
        "qkv_c",
        "cache",
        "cmp",
        "i_cmp",
        "split",
        "q_b",
        "q_norm",
        "i_q_b",
        "i_q",
        "i_wp",
        "i_score",
        "i_topk",
        "uv",
        "o_a",
        "o_b",
        "hcd_f",
        "hcc_f",
        "router",
        "ug",
        "down",
    ):
        base[name] = acc % G
        acc += tasks[name]

    @fx.struct
    class Smem:
        x: fx.Array[fx.Float32, XN, 16]  # bf16 activations (pairs) / split q + KV tile
        out: fx.Array[fx.Float32, ON, 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        misc: fx.Array[fx.Float32, misc_words, 16]
        p: fx.Array[fx.Float32, H * SPLIT_KEYS, 16]
        keys: fx.Array[fx.Int32, SPLIT_KEYS, 16]
        dnw: fx.Array[fx.Float32, S * MOE_SLOTS, 16]  # expert-down route weights
        # radix-select bins, replicated, plus two words to broadcast the winning digit
        hist: fx.Array[fx.Int32, (TK_BC + 4) if IHD else 1, 16]

    # the stages read this build's constants from bc; ``keyed`` (the build parameters and
    # the source digest) is what the JIT cache key holds of them
    # the build constants the stages read (upstream: dict(locals()))
    bc = {
        "CMP_CHUNK": CMP_CHUNK,
        "CMP_CHUNK_T": CMP_CHUNK_T,
        "CR": CR,
        "CW": CW,
        "C_COFF": C_COFF,
        "C_RING": C_RING,
        "C_ROWS": C_ROWS,
        "C_ROW_W": C_ROW_W,
        "DN_TILE": DN_TILE,
        "G": G,
        "H": H,
        "HC": HC,
        "HC_COEF": HC_COEF,
        "HC_COL_OFFS": HC_COL_OFFS,
        "HC_KSLICE": HC_KSLICE,
        "HC_MISC": HC_MISC,
        "HC_MIX": HC_MIX,
        "HC_NKC": HC_NKC,
        "HC_NKC_FULL": HC_NKC_FULL,
        "HC_PW": HC_PW,
        "HC_RG": HC_RG,
        "HC_ROWS": HC_ROWS,
        "HC_ROW_OFFS": HC_ROW_OFFS,
        "HC_TASKS": HC_TASKS,
        "HC_TPW": HC_TPW,
        "HC_VALS": HC_VALS,
        "HC_WPR": HC_WPR,
        "HEAD_DIM": HEAD_DIM,
        "HIDDEN": HIDDEN,
        "IC_BLK_WORDS": IC_BLK_WORDS,
        "IC_GRP_WORDS": IC_GRP_WORDS,
        "IC_S_BLK": IC_S_BLK,
        "IH": IH,
        "IHD": IHD,
        "INDEXER_HADAMARD": INDEXER_HADAMARD,
        "INTER": INTER,
        "IW": IW,
        "I_ROW_W": I_ROW_W,
        "KS": KS,
        "KT_OFF": KT_OFF,
        "KV_ROW_BYTES": KV_ROW_BYTES,
        "K_PB": K_PB,
        "LIVE_SPLITS": LIVE_SPLITS,
        "MOE_SLOTS": MOE_SLOTS,
        "NOPE_DIM": NOPE_DIM,
        "N_COMP": N_COMP,
        "N_DN_TILES": N_DN_TILES,
        "N_EXPERTS": N_EXPERTS,
        "N_IHG": N_IHG,
        "N_INDEX": N_INDEX,
        "N_IQB": N_IQB,
        "N_ISEL": N_ISEL,
        "N_KEYS": N_KEYS,
        "N_QB": N_QB,
        "N_QKV_A": N_QKV_A,
        "N_QKV_C": N_QKV_C,
        "N_ROUTER": N_ROUTER,
        "N_ROW_TILES": N_ROW_TILES,
        "N_SPLIT": N_SPLIT,
        "N_UG_TASKS": N_UG_TASKS,
        "N_UV": N_UV,
        "OA_K": OA_K,
        "OA_PER_GROUP": OA_PER_GROUP,
        "OB_K": OB_K,
        "OVERLAP": OVERLAP,
        "O_GROUPS": O_GROUPS,
        "O_LORA": O_LORA,
        "QKV_A_ROWS": QKV_A_ROWS,
        "QK_DIM": QK_DIM,
        "QS": QS,
        "Q_LORA": Q_LORA,
        "RUN_MAX": RUN_MAX,
        "S": S,
        "SC": SC,
        "SHARED_EXPERT": SHARED_EXPERT,
        "SY": SY,
        "TK_BC": TK_BC,
        "TK_BINS": TK_BINS,
        "TK_PARTS": TK_PARTS,
        "TK_PER": TK_PER,
        "TK_REP": TK_REP,
        "TK_TRIPS": TK_TRIPS,
        "TOPK_SUM_OFFS": TOPK_SUM_OFFS,
        "TOP_K": TOP_K,
        "UG_PER_SLOT": UG_PER_SLOT,
        "UV_CHUNK": UV_CHUNK,
        "UV_PER_HEAD": UV_PER_HEAD,
        "UV_TILE": UV_TILE,
        "UV_WIDE": UV_WIDE,
        "W": W,
        "XQ_BLOCKS": XQ_BLOCKS,
        "XQ_WAVES": XQ_WAVES,
        "base": base,
        "hc_eps": hc_eps,
        "hc_sinkhorn_iters": hc_sinkhorn_iters,
        "scale": scale,
        "swiglu_limit": swiglu_limit,
        "window": window,
    }

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def dsv4_kernel(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        kv_cache: Int64,
        kv_rope: Int64,
        dest_rows: Int64,
        indices: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_in: Int64,
        g_q: Int64,
        g_kv: Int64,
        g_post: Int64,
        attn_sink: Int64,
        ape: Int64,
        g_ckv: Int64,
        kv_state: Int64,
        score_state: Int64,
        i_ape: Int64,
        g_ickv: Int64,
        i_kv_state: Int64,
        i_score_state: Int64,
        i_cache: Int64,
        hc_attn_fn: Int64,
        hc_attn_sb: Int64,
        hc_ffn_fn: Int64,
        hc_ffn_sb: Int64,
        w_qkv_a: Int64,
        s_qkv_a: Int64,
        w_qkv_c: Int64,
        w_q_b: Int64,
        s_q_b: Int64,
        w_i_q_b: Int64,
        s_i_q_b: Int64,
        i_w: Int64,
        w_o_a: Int64,
        s_o_a: Int64,
        w_o_b: Int64,
        s_o_b: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        w_sug: Int64,
        s_sug: Int64,
        w_sdn: Int64,
        s_sdn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        step: Int64,
        state_slots: Int64,
        state_slots_in: Int64,
        tok_ids: Int64,
        tid2eid: Int64,
        block_tables: Int64,
        i_cache_s: Int64,
        rank: Int32,
        layer: Int32,
        st_kv: Int32,
        st_i: Int32,
        use_hash: Int32,
        bt_stride: Int32,
        env_rows: Int32,
        run_pos: Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        lds = fx.SharedAllocator().allocate(Smem).peek()
        xs = lds.x.ptr
        outs = lds.out.ptr
        red = lds.red.ptr
        misc = lds.misc.ptr
        pl = lds.p.ptr
        keys = lds.keys.ptr
        dnw = lds.dnw.ptr
        hist = lds.hist.ptr
        ktile = xs + KT_OFF  # f32-typed view holding raw bf16 pairs
        v4f = fx.Vector.make_type(4, fx.Float32)

        r_h = rsrc(h_in)
        # this launch's tag: the step (word 0 of the epoch buffer, which the host moves on
        # after a step's last launch), unique per layer
        tag = (
            uniform(bo.buffer_load(rsrc(step), 0, vec_width=1, dtype=T.i32))
            * LAYER_SLOTS
            + layer
            + 1
        )
        r_pos = rsrc(cur_pos)

        def ld_pos(s):
            """Sample ``s``'s position (``s`` is wave-uniform)."""
            return uniform(bo.buffer_load(r_pos, s, vec_width=1, dtype=T.i32))

        def ld_run(s):
            """Sample ``s``'s index among its request's rows in this launch (0 for its
            first: the earlier ones are ``s - 1`` ..), from ``run_pos``."""
            return (
                s % RUN_MAX  # no run: the build without speculative rows
                if RUN_MAX == 1
                else uniform(bo.buffer_load(rsrc(run_pos), s, vec_width=1, dtype=T.i32))
            )

        r_dest = rsrc(dest_rows)

        def ld_dest(j, s):
            """Plane rows for sample ``s``, supplied by the pool: j=0 the row this
            token's KV goes to, j=1 the base of its compressed entries (see comp_row).
            """
            return uniform(bo.buffer_load(r_dest, j * S + s, vec_width=1, dtype=T.i32))

        def bt_block(s, e):
            """The physical block holding sequence ``s``'s compressed entry ``e``."""
            return fx.Int32(
                bo.buffer_load(
                    rsrc(block_tables),
                    s * bt_stride + e // K_PB,
                    vec_width=1,
                    dtype=T.i32,
                )
            )

        def comp_row(s, e):
            """Plane row of sequence ``s``'s compressed entry ``e`` (see K_PB)."""
            return ld_dest(1, s) + bt_block(s, e) * env_rows + e % K_PB

        # a step writes a sequence's rolling state at state_slots and reads its ring at
        # state_slots_in (SGLang's request slot, both)
        r_slot, r_slot_in = rsrc(state_slots), rsrc(state_slots_in)

        # serving pools exceed a buffer resource's 4 GB reach: row / slot bases are 64-bit
        def slot_rsrc(ptr, s, stride, slots=r_slot):
            """A resource at sample ``s``'s slot of an f32 rolling-state pool."""
            slot = uniform(bo.buffer_load(slots, s, vec_width=1, dtype=T.i32))
            return rsrc(ptr + fx.Int64(slot) * fx.Int64(stride) * 4)

        def row_rsrc(ptr, row, row_bytes):
            """A resource at plane row ``row`` (wave-uniform) of rows ``row_bytes`` wide."""
            return rsrc(ptr + fx.Int64(uniform(row)) * row_bytes)

        r_peers = rsrc(peers)
        # each wave sends to one peer
        pv = fx.Vector(
            bo.buffer_load(r_peers, fx.min(wave, W - 1) * 2, vec_width=2, dtype=T.i32)
        )
        peer_dst = (fx.Int64(uniform(pv[1])) << 32) | fx.Int64(
            fx.Uint32(uniform(pv[0]))
        )

        _ = keyed
        _ = _key_dependencies
        ctx = dict(bc)
        # the launch values the stages read (upstream: ctx.update(locals()))
        ctx.update(
            {
                "ape": ape,
                "attn_sink": attn_sink,
                "bias": bias,
                "bid": bid,
                "bt_block": bt_block,
                "comp_row": comp_row,
                "dnw": dnw,
                "g_ckv": g_ckv,
                "g_ickv": g_ickv,
                "g_in": g_in,
                "g_kv": g_kv,
                "g_post": g_post,
                "g_q": g_q,
                "hc_attn_fn": hc_attn_fn,
                "hc_attn_sb": hc_attn_sb,
                "hc_ffn_fn": hc_ffn_fn,
                "hc_ffn_sb": hc_ffn_sb,
                "hist": hist,
                "i_ape": i_ape,
                "i_cache": i_cache,
                "i_cache_s": i_cache_s,
                "i_kv_state": i_kv_state,
                "i_score_state": i_score_state,
                "i_w": i_w,
                "indices": indices,
                "keys": keys,
                "ktile": ktile,
                "kv_cache": kv_cache,
                "kv_rope": kv_rope,
                "kv_state": kv_state,
                "lane": lane,
                "ld_dest": ld_dest,
                "ld_pos": ld_pos,
                "ld_run": ld_run,
                "misc": misc,
                "outs": outs,
                "peer_dst": peer_dst,
                "pl": pl,
                "r_h": r_h,
                "r_slot_in": r_slot_in,
                "rank": rank,
                "red": red,
                "rope_cos": rope_cos,
                "rope_sin": rope_sin,
                "row_rsrc": row_rsrc,
                "s_dn": s_dn,
                "s_i_q_b": s_i_q_b,
                "s_o_a": s_o_a,
                "s_o_b": s_o_b,
                "s_q_b": s_q_b,
                "s_qkv_a": s_qkv_a,
                "s_sdn": s_sdn,
                "s_sug": s_sug,
                "s_ug": s_ug,
                "score_state": score_state,
                "scratch": scratch,
                "slot_rsrc": slot_rsrc,
                "st_i": st_i,
                "st_kv": st_kv,
                "sym": sym,
                "tag": tag,
                "tid": tid,
                "tid2eid": tid2eid,
                "tok_ids": tok_ids,
                "use_hash": use_hash,
                "v4f": v4f,
                "w_dn": w_dn,
                "w_i_q_b": w_i_q_b,
                "w_o_a": w_o_a,
                "w_o_b": w_o_b,
                "w_q_b": w_q_b,
                "w_qkv_a": w_qkv_a,
                "w_qkv_c": w_qkv_c,
                "w_r": w_r,
                "w_sdn": w_sdn,
                "w_sug": w_sug,
                "w_ug": w_ug,
                "wave": wave,
                "x_out": x_out,
                "xs": xs,
            }
        )
        # every wait runs under the guard: a deadline, the error words (words 1-5 of the
        # epoch buffer), and the peers' flags at the tail of each rank's peer buffer
        with guard.scope(
            guard.Guard(
                step,
                own=sym,
                peers=peers,
                rank=rank,
                npes=W,
                flags_offset=SY["_bytes"],
                fault_site=fault_site or None,
                fault_after=fault_after,
            )
        ):
            ctx.update(common_defs(ctx))
            ctx.update(hc_attn_stages(ctx))
            ctx.update(qkv_stages(ctx))
            ctx.update(index_query_stages(ctx))
            ctx.update(q_b_stage(ctx))
            ctx.update(index_select_stages(ctx))
            ctx.update(attention_stages(ctx))
            ffn_stages(ctx)

    @flyc.jit
    def launch_dsv4(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        kv_cache: Int64,
        kv_rope: Int64,
        dest_rows: Int64,
        indices: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_in: Int64,
        g_q: Int64,
        g_kv: Int64,
        g_post: Int64,
        attn_sink: Int64,
        ape: Int64,
        g_ckv: Int64,
        kv_state: Int64,
        score_state: Int64,
        i_ape: Int64,
        g_ickv: Int64,
        i_kv_state: Int64,
        i_score_state: Int64,
        i_cache: Int64,
        hc_attn_fn: Int64,
        hc_attn_sb: Int64,
        hc_ffn_fn: Int64,
        hc_ffn_sb: Int64,
        w_qkv_a: Int64,
        s_qkv_a: Int64,
        w_qkv_c: Int64,
        w_q_b: Int64,
        s_q_b: Int64,
        w_i_q_b: Int64,
        s_i_q_b: Int64,
        i_w: Int64,
        w_o_a: Int64,
        s_o_a: Int64,
        w_o_b: Int64,
        s_o_b: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        w_sug: Int64,
        s_sug: Int64,
        w_sdn: Int64,
        s_sdn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        step: Int64,
        state_slots: Int64,
        state_slots_in: Int64,
        tok_ids: Int64,
        tid2eid: Int64,
        block_tables: Int64,
        i_cache_s: Int64,
        rank: Int32,
        layer: Int32,
        st_kv: Int32,
        st_i: Int32,
        use_hash: Int32,
        bt_stride: Int32,
        env_rows: Int32,
        run_pos: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        dsv4_kernel(
            h_in,
            x_out,
            cur_pos,
            kv_cache,
            kv_rope,
            dest_rows,
            indices,
            rope_cos,
            rope_sin,
            g_in,
            g_q,
            g_kv,
            g_post,
            attn_sink,
            ape,
            g_ckv,
            kv_state,
            score_state,
            i_ape,
            g_ickv,
            i_kv_state,
            i_score_state,
            i_cache,
            hc_attn_fn,
            hc_attn_sb,
            hc_ffn_fn,
            hc_ffn_sb,
            w_qkv_a,
            s_qkv_a,
            w_qkv_c,
            w_q_b,
            s_q_b,
            w_i_q_b,
            s_i_q_b,
            i_w,
            w_o_a,
            s_o_a,
            w_o_b,
            s_o_b,
            w_r,
            bias,
            w_ug,
            s_ug,
            w_dn,
            s_dn,
            w_sug,
            s_sug,
            w_sdn,
            s_sdn,
            scratch,
            sym,
            peers,
            step,
            state_slots,
            state_slots_in,
            tok_ids,
            tid2eid,
            block_tables,
            i_cache_s,
            rank,
            layer,
            st_kv,
            st_i,
            use_hash,
            bt_stride,
            env_rows,
            run_pos,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    # LLVM's VectorCombine (foldShuffleToIdentity) went exponential on an earlier up/gate; kept off
    return flyc.compile[{"llvm_options": {"disable-vector-combine": True}}](launch_dsv4)
