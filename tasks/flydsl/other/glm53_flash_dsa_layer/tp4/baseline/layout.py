# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm at ca328e297b (Apache-2.0):
# vllm/models/glm5next/amd/ops/mono/ffn/layout.py, itself adapted from
# ROCm/aiter#6173 at c39b56c36 (Apache-2.0 License),
# Copyright (c) 2025 FlyDSL Project Contributors:
# aiter/ops/flydsl/kernels/glm5_mono/glm/layout.py

# Vendored from sgl-project/sgl-amd-mono at 52486703fbf27e77f05b573e4d931d4eec87118d
# (sgl_amd_mono/glm53_flash/gfx950/layout.py); AKA: the guard import is package-relative.
"""Compile-time storage layout and CTA schedule of the GLM-5.3-Flash layer launch."""

from .config import (
    HC_MULT,
    HIDDEN,
    INDEX_DIM,
    INDEX_HEADS,
    KPOOL,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    Q_LORA,
    TOP_K,
    TOPK_TOKENS,
    V_DIM,
    VOCAB,
    FlashShard,
)
from .tiles import (
    BLOCKS,
    ROUTER_TILE,
    ROW_TILE,
    WAVES,
)

# rows of a projection task: one MFMA tile, or one tile a wave when K is 128
IN_TILE = 16
FGB_TILE = 16 * WAVES
O_TILE = 16
# a dense up/gate task: 16 gate rows and the same 16 up rows, the whole K
DUG_TILE = 16
# an LM-head tile: 16 vocabulary rows, the whole K
HEAD_TILE = 16

# sparse attention: q_a | kv_a (FP8) then wk | weights_proj | kpool gate (bf16)
QKV_TILE = 16
N_QKV_FP8 = (Q_LORA + KV_LORA) // QKV_TILE
INDEX_ROWS = INDEX_DIM + INDEX_HEADS + INDEX_DIM
N_QKV = N_QKV_FP8 + INDEX_ROWS // QKV_TILE
QB_TILE = 32
UK_TILE = 128
UV_TILE = 64
POOLS_PER_TASK = 256
# a row attends to its 2048 selected pool tokens plus the tail past its last complete
# pool (up to KPOOL - 1 tokens), as SGLang's top-k appends it; a row of at most
# MAX_KEYS tokens attends to every token
MAX_KEYS = TOPK_TOKENS + KPOOL - 1
# split attention: one task 64 selected keys x up to 16 heads (the 16 columns of
# its MFMAs); the merge sums the first BASE_SPLITS partials in two halves, then the
# tail's split. The tail splits (split BASE_SPLITS of each row and head group) are
# tasks 0 .., then the regular splits
SPLIT_KEYS = 64
BASE_SPLITS = TOPK_TOKENS // SPLIT_KEYS
N_SPLIT = (MAX_KEYS + SPLIT_KEYS - 1) // SPLIT_KEYS
SEL_STRIDE = N_SPLIT * SPLIT_KEYS
assert N_SPLIT == BASE_SPLITS + 1
# per-step row table of a sparse-attention launch (int32 [S, ROW_WORDS])
ROW_POS, ROW_SLOT, ROW_POOL_SLOT, ROW_TAIL_SLOT, ROW_BT = range(5)
ROW_WORDS = 8
# entries of a sparse-attention layer's int64 argument table
DSA_TAB = (
    "w_qkv",
    "s_qkv",
    "w_idx",
    "w_gate",
    "g_qa",
    "g_kv",
    "ln_w",
    "ln_b",
    "ape",
    "w_qb",
    "s_qb",
    "w_iq",
    "w_uk",
    "w_kvb",
    "w_o",
    "s_o",
    "kv_cache",
    "idx_cache",
    "tail_k",
    "tail_g",
    "idx_bt_group",
    "kv_block_elems",
    "kv_block_tokens",
    "kv_row_elems",
    "idx_block_bytes",
    "idx_block_pools",
    "tail_block_elems",
)
# entries of the int64 step-input table of a launch from the step's tokens: the token
# ids (int32 words an id), this rank's embedding rows and the vocabulary range they
# hold, the step's positions, MLA slots and request rows (words an entry), its row
# count, and the KDA slot and row tables the step's later launches read
STEP_TAB = (
    "ids",
    "ids_words",
    "emb",
    "vocab_lo",
    "vocab_hi",
    "positions",
    "positions_words",
    "out_cache_loc",
    "loc_words",
    "req_pool_indices",
    "rpi_words",
    "n_rows",
    "slots",
    "rows",
)
# the step-input table's int32 entries (the rest are pointers)
STEP_COUNTS = frozenset(
    name for name in STEP_TAB if name.endswith(("_words", "_lo", "_hi", "_rows"))
)

# a box: each consumer CTA's own 128-byte lines, a word a producer in a line's first
# 16 words (one s_load_dwordx16)
BOX_LINE = 128
BOX_WORDS = 16


def box_lines(producers: int) -> int:
    return -(-producers // BOX_WORDS)


N_ROW_TILES = HIDDEN // ROW_TILE
# mix projection outputs (pre, post, comb) plus one sum of squares a tile
N_MIX = HC_MULT * (2 + HC_MULT)
PMIX = N_MIX + 1
N_ROUTER = N_EXPERTS // ROUTER_TILE
XQ_GROUPS = HIDDEN // 32
assert N_EXPERTS % ROUTER_TILE == 0


def dn_tile(samples: int) -> int:
    """Rows per expert-down / FFN-reduce task."""
    return 32 if samples == 1 or samples > 8 else HIDDEN // BLOCKS


# the grouped MoE's rows a weighted-sum task (and its TP reduce tile); a unit's rows
GROUPED_SUM_TILE = 16
UNIT_ROWS = 16


def grouped_units_max(samples: int) -> int:
    """A bound on the grouped MoE's units (an expert's picks, 16 a unit)."""
    routed = samples * TOP_K
    return min(N_EXPERTS, routed) + routed // UNIT_ROWS + (samples + 15) // 16


def sample_wave_batches(samples: int) -> int:
    return (samples + WAVES - 1) // WAVES


def ug_tasks(inter: int) -> int:
    """Routed up/gate tasks: 32 intermediate values of one routed slot for half of the
    rows."""
    return TOP_K * (inter // 32) * 2


def down_prefetch_batch(samples: int) -> int:
    return 9 if samples <= 8 else 3


def uv_tiles(samples: int) -> int:
    """V tiles of one (row, head) a uv task covers: two at 8 rows, so a CTA merges
    the head's split partials once."""
    return 2 if samples > 4 else 1


def down_x_words(samples: int, inter: int) -> int:
    return samples * MOE_SLOTS * inter // 4


def _align(size: int, alignment: int = 256) -> int:
    return (size + alignment - 1) // alignment * alignment


def layout(
    samples: int,
    npes: int,
    inter: int,
    kda: bool = False,
    dsa: bool = False,
    max_pools: int = 0,
    dense: bool = False,
    head: bool = False,
    x_in: bool = False,
    grouped: bool = False,
):
    """Byte offsets of the per-rank scratch and the symmetric peer buffer (with
    ``x_in`` a region for the ranks' partials of the layer input; with ``grouped``
    the expert-major MoE's per-pick down partials)."""
    from .guard import FLAG_BYTES

    pair_bytes = 8
    shard = FlashShard(npes)
    if head:
        # the contracted rows (bf16 pairs) and their tiles' sums of squares; every
        # rank's logits (bf16 pairs, [S, VOCAB]) in the symmetric buffer, in two slots
        # by launch parity: a peer's next tail never writes the slot a slow rank
        # still gathers
        scratch, offset = {}, 0
        for name, size in (
            ("hc", samples * HIDDEN // 2 * pair_bytes),
            ("hss", samples * N_ROW_TILES * pair_bytes),
        ):
            scratch[name] = offset
            offset += _align(size)
        scratch["_bytes"] = offset
        logits = _align(samples * VOCAB // 2 * pair_bytes)
        return scratch, {
            "logits": 0,
            "_logits_stride": logits,
            "attn": 0,
            "ffn": 0,
            "_part_stride": 0,
            "flags": 2 * logits,
            "_bytes": 2 * logits + FLAG_BYTES,
        }
    items = []
    if kda or dsa:
        items = [
            ("rnew_a", samples * HC_MULT * HIDDEN // 2 * pair_bytes),
            ("pmix_a", samples * N_ROW_TILES * PMIX * pair_bytes),
            ("a_in", samples * HIDDEN // 2 * pair_bytes),
            ("box_a_in", BLOCKS * box_lines(samples) * BOX_LINE),
            ("mix_a", samples * HC_MULT * (1 + HC_MULT) * pair_bytes),
        ]
    if kda:
        items += [
            ("proj", samples * shard.kda_in_rows * pair_bytes),
            ("gates", samples * 2 * shard.kda_proj * pair_bytes),
            ("o_in", samples * shard.kda_proj // 2 * pair_bytes),
            ("box_o_in", BLOCKS * box_lines(samples * shard.kda_heads) * BOX_LINE),
        ]
    if dsa:
        H = shard.mla_heads
        items += [
            ("qa", samples * Q_LORA * pair_bytes),
            ("kva", samples * KV_LORA * pair_bytes),
            ("ik", samples * INDEX_DIM * pair_bytes),
            ("iw", samples * INDEX_HEADS * pair_bytes),
            ("ig", samples * INDEX_DIM * pair_bytes),
            ("q_an", samples * Q_LORA // 2 * pair_bytes),
            ("q_nope", samples * H * NOPE_DIM // 2 * pair_bytes),
            ("iq", samples * INDEX_HEADS * INDEX_DIM // 2 * pair_bytes),
            ("iq8", samples * INDEX_HEADS * INDEX_DIM // 4 * pair_bytes),
            ("iqw", samples * INDEX_HEADS * pair_bytes),
            ("q_lat", samples * H * KV_LORA // 2 * pair_bytes),
            ("kv_ready", samples * pair_bytes),
            ("iscores", samples * max(max_pools, 1) * pair_bytes),
            ("tok_sel", samples * SEL_STRIDE * pair_bytes),
            ("sp_acc", samples * N_SPLIT * H * KV_LORA // 2 * pair_bytes),
            ("sp_m", samples * N_SPLIT * H * pair_bytes),
            ("sp_l", samples * N_SPLIT * H * pair_bytes),
            ("o_in", samples * H * V_DIM // 2 * pair_bytes),
        ]
    items += [
        ("rnew", samples * HC_MULT * HIDDEN // 2 * pair_bytes),
        ("pmix", samples * N_ROW_TILES * PMIX * pair_bytes),
        ("a", samples * HIDDEN // 2 * pair_bytes),
        ("box_a", BLOCKS * box_lines(samples) * BOX_LINE),
    ]
    if dense:
        # the dense MLP's mid (bf16 pairs)
        items += [("dmid", samples * shard.dense_inter // 2 * pair_bytes)]
    else:
        items += [
            ("scores", samples * N_EXPERTS * pair_bytes),
            ("xq", samples * HIDDEN // 4 * pair_bytes),
            ("xqs", samples * XQ_GROUPS * pair_bytes),
            ("sel", samples * MOE_SLOTS * pair_bytes),
            ("prob", samples * MOE_SLOTS * pair_bytes),
            # the shared expert's mid (f32); the routed slots' mid as MXFP8 (4 values
            # a word) and its E8M0 block exponents, indexed (row, slot) like the
            # shared one's
            ("mid", samples * inter * pair_bytes),
            ("midq", samples * MOE_SLOTS * inter // 4 * pair_bytes),
            ("midqs", samples * MOE_SLOTS * (inter // 32) * pair_bytes),
        ]
        if grouped:
            # each (row, slot)'s down output (f32), before its route weight
            items += [("dpart", samples * MOE_SLOTS * HIDDEN * pair_bytes)]
    assert len({name for name, _ in items}) == len(items), "mailbox names"
    offset, scratch = 0, {}
    for name, size in items:
        scratch[name] = offset
        offset += _align(size)
    scratch["_bytes"] = offset

    part = npes * samples * HIDDEN * pair_bytes
    region = 2 * part
    # the attention partials, the FFN (MoE or MLP) partials and, with ``x_in``, the
    # layer input's partials: a region each, so no two exchanges of one launch share
    # a word
    names = ("attn", "ffn", "xin") if x_in else ("attn", "ffn")
    symmetric = {name: k * region for k, name in enumerate(names)}
    end = len(names) * region
    symmetric.update(
        {
            "_part_stride": part,
            # the guard's error flags: rank r writes word r of every rank's flags
            "flags": end,
            "_bytes": end + FLAG_BYTES,
        }
    )
    return scratch, symmetric


def stage_tasks(
    samples: int,
    inter: int,
    npes: int = 1,
    kda: bool = False,
    dsa: bool = False,
    dense: bool = False,
    head: bool = False,
    grouped: bool = False,
):
    """``(stage name, CTA slots)`` pairs in execution order: a stage's first task runs
    on the CTA after the previous stages' slots. ``iscore`` is sized per row at run
    time; its count here only places the next stage. The tail splits run ahead of
    the regular splits on their CTAs and take no slots, and the split keeps the slots
    of 8-head tasks (tasks of 16 heads use the first half). The grouped MoE's up/gate
    and down tasks are counted at run time and split over every CTA."""
    shard = FlashShard(npes)
    if head:
        return [("hpost", N_ROW_TILES), ("lmhead", BLOCKS), ("gather", BLOCKS)]
    attention = []
    if dsa:
        H = shard.mla_heads
        attention = [
            ("aseam", N_ROW_TILES),
            ("agate", samples),
            ("qkv_a", N_QKV),
            ("qnorm", samples),
            ("cache", samples),
            ("q_b", H * NOPE_DIM // QB_TILE),
            ("index_q", BLOCKS),
            ("iq_quant", INDEX_HEADS),
            ("uk", H * KV_LORA // UK_TILE),
            ("iscore", samples * 8),
            ("iselect", samples),
            ("split", samples * (H // WAVES) * BASE_SPLITS),
            ("uv", samples * H * (V_DIM // UV_TILE) // uv_tiles(samples)),
            ("oproj", HIDDEN // O_TILE),
        ]
    if kda:
        attention = [
            ("aseam", N_ROW_TILES),
            ("agate", samples),
            # the attention gate's CTAs run none of its tasks
            ("inproj", (shard.kda_in_rows + IN_TILE - 1) // IN_TILE + samples),
            ("fgb", 2 * shard.kda_proj // FGB_TILE),
            ("kda", samples * shard.kda_heads),
            ("oproj", HIDDEN // O_TILE),
        ]
    if dense:
        return attention + [
            ("seam", N_ROW_TILES),
            ("gate", samples),
            ("dug", shard.dense_inter // DUG_TILE),
            ("down", BLOCKS),
        ]
    if grouped:
        return attention + [
            ("seam", N_ROW_TILES),
            ("gate", samples),
            ("router", samples * N_ROUTER),
            ("ug", BLOCKS),
            ("down", BLOCKS),
            ("dsum", HIDDEN // GROUPED_SUM_TILE),
        ]
    return attention + [
        ("seam", N_ROW_TILES),
        ("gate", samples),
        ("router", samples * N_ROUTER),
        ("ug", (ug_tasks(inter) + BLOCKS - 1) // BLOCKS * BLOCKS),
        ("down", HIDDEN // dn_tile(samples)),
    ]
