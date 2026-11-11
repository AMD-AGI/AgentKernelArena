# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Adapted from ROCm/aiter#6173 at 56d6390def (Apache-2.0):
# aiter/ops/flydsl/kernels/glm5_mono/glm/layout.py
# Modified for AgentKernelArena (the guard's flag size imported package-relative).

"""Compile-time storage layout and CTA schedule for the GLM-5 MonoKernel."""

from .config import (
    HIDDEN,
    INTER,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    QKV_A_ROWS,
    TOP_K,
    V_DIM,
)
from .guard import FLAG_BYTES
from .tiles import (
    BLOCKS,
    Q_B_TILE,
    QKV_A_TILE,
    ROUTER_TILE,
    ROW_TILE,
    UG_TILE,
    UK_TILE,
    UV_TILE,
    WAVES,
)

INDEX_HEADS = 32
INDEX_DIM = 128
# LM-head rows a tail task computes; also the NextN entry's rows (x columns) a task
HEAD_TILE = 16
# the NextN entry's pointer-table entries: this rank's eh_proj rows, the enorm and
# hnorm weights, and (from the tokens) this rank's embedding rows
EH_ARGS = (8, 9, 10, 11)


def eh_parts(eh_tiles: int) -> int:
    """The NextN entry's K parts a tile: four (two of each of the e and h_prev halves)
    when their tasks fit one round of CTAs, else the two halves."""
    return 4 if 4 * eh_tiles <= BLOCKS else 2


def eh_base(eh_tiles: int, with_indexer: bool) -> int:
    """The NextN entry's first CTA. With the indexer, the CTA past the front's index
    projection tiles, which are the grid's last CTAs, so CTA 0: those tiles' weights go
    out at the launch start, not behind an entry task. Else the last CTAs."""
    if with_indexer:
        return 0
    return max(BLOCKS - eh_parts(eh_tiles) * eh_tiles, 0)


INDEX_Q_ROWS = INDEX_HEADS * INDEX_DIM
INDEX_TILE = 16
INDEX_KEYS_PER_TASK = 64
# a row's top-k runs on up to this many slices (CTAs), with staging room for this many
# candidates of the threshold's prefix
INDEX_SEL_SLICES = 64
INDEX_CAND_STAGE_CAP = 16384
# radix digits of the sliced top-k; a row has a slice per this many keys (rounded up),
# at most INDEX_SEL_SLICES
INDEX_SEL_DIGITS = 6
INDEX_SEL_SLICE_KEYS = 1024


def select_slices(bound, minimum=min, maximum=max):
    """(slices, keys a slice) of a row of ``bound`` keys: runs of the row's keys in
    position order, none empty (one for an empty row). The kernel calls it on traced
    Int32 with ``fx.min`` and ``fx.max``, the CPU test on ints."""
    n = minimum(
        (bound + INDEX_SEL_SLICE_KEYS - 1) // INDEX_SEL_SLICE_KEYS, INDEX_SEL_SLICES
    )
    n = maximum(n, 1)
    return n, (bound + n - 1) // n


DCP_SUMMARY_PAIRS = KV_LORA // 2 + 2


def split_acc_head(head_group, lane_group: int, element: int):
    """Map one split-PV MFMA row to its local attention head."""

    return head_group * WAVES + lane_group * 4 + element


def split_score_column(wave: int, lane: int):
    """Locate this wave's head column in a 16x16 MFMA score tile."""

    return wave + 16 * ((lane % 16) // 4)


def fp8_kv_upper_pair_lane(lane):
    return (lane & -4) + 2


def fp8_pe_upper_pair_lane(lane):
    return (lane & -2) + 1


N_QKV_A = QKV_A_ROWS // QKV_A_TILE
# the front splits each q_a / kv_a tile's K: the first FRONT_K_BIG on the tile's own CTA,
# the rest on one of the FRONT_SMALL_CTAS CTAs past them (two tiles' rests a CTA)
FRONT_K_BIG = 4096
FRONT_SMALL_CTAS = BLOCKS - N_QKV_A
N_ROW_TILES = HIDDEN // ROW_TILE
N_ROUTER = N_EXPERTS // ROUTER_TILE
XQ_BLOCKS = HIDDEN // 128
XQ_GROUPS = HIDDEN // 32
XQ_WAVES = (XQ_BLOCKS + N_ROUTER - 1) // N_ROUTER
assert XQ_WAVES * 4 <= WAVES


def dn_tile(samples: int, expert_mxfp4: bool = False) -> int:
    """Return rows per expert-down/FFN-reduce task for the tuned schedule."""

    return 32 if samples == 1 or (samples > 4 and expert_mxfp4) else HIDDEN // BLOCKS


def sparse_keys_per_task(samples: int, heads: int = WAVES) -> int:
    """Use narrower sparse tiles when samples or attention heads fill LDS."""

    return 32 if heads > 2 * WAVES else 64


def sample_wave_batches(samples: int) -> int:
    return (samples + WAVES - 1) // WAVES


def ug_task_rounds(inter: int) -> int:
    return (inter + BLOCKS - 1) // BLOCKS


def down_prefetch_batch(samples: int, native_fp4_mfma: bool = False) -> int:
    if samples <= 4:
        return 9
    if native_fp4_mfma:
        return 4 if samples <= 6 else 3
    return 8


def down_x_words(
    samples: int, inter: int, expert_mxfp4: bool, native_fp4_mfma: bool = False
) -> int:
    return (
        samples * MOE_SLOTS * inter // (4 if native_fp4_mfma or not expert_mxfp4 else 2)
    )


def dcp_summary_index(source, sample, tile, item, samples, n_uv):
    return ((source * samples + sample) * n_uv + tile) * DCP_SUMMARY_PAIRS + item


def dcp_uv_owner(tile, output_heads, rank):
    return tile // (V_DIM // UV_TILE) // output_heads == rank


def dcp_local_uv_tile(tile, output_heads):
    return tile % (output_heads * V_DIM // UV_TILE)


def ug_split(samples: int, inter: int = INTER):
    """Return the balanced up/gate leftover split for batches two and four."""

    full_tiles, remainder = divmod((samples * TOP_K + 1) * (inter // UG_TILE), BLOCKS)
    if samples not in (2, 4) or remainder == 0 or BLOCKS % remainder:
        return None
    segments = BLOCKS // remainder
    if (HIDDEN // 128) % segments or (HIDDEN // 128) // segments > WAVES // 2:
        return None
    return full_tiles, segments


def _align(size: int, alignment: int = 256) -> int:
    return (size + alignment - 1) // alignment * alignment


def layout(
    samples: int,
    heads: int,
    npes: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    inter: int = INTER,
    output_heads: int | None = None,
    dcp_size: int = 1,
    native_fp4_mfma: bool = False,
    tokens: bool = False,
    head_vocab: int = 0,
    eh_tiles: int = 0,
):
    """Return byte offsets for per-rank scratch and symmetric peer buffers.

    ``tokens``: also the layer input's rows, which the token's owner rank pushes
    (``xin``, two slots by launch parity). ``eh_tiles``: the NextN entry's rows of
    ``xin`` instead, which every rank pushes its ``eh_tiles`` 16-column tiles of, and
    the partial sums of every K part but a tile's first (``ehp``); with ``tokens``
    the embedding rows go to ``ein`` (as ``xin``). ``head_vocab``: the tail launch's
    buffers instead, the gathered bf16 logits of a ``head_vocab`` vocabulary (two
    slots)."""
    if head_vocab:
        part = samples * head_vocab * 4
        return {"_bytes": 0}, {
            "logits": 0,
            "_logits_stride": part,
            "flags": 2 * part,
            "_bytes": 2 * part + FLAG_BYTES,
        }
    split_count = sparse_attention_topk // sparse_keys_per_task(samples, heads)
    output_heads = heads if output_heads is None else output_heads
    pair_bytes = 8
    items = [
        ("q_a", samples * Q_LORA * pair_bytes),
        ("q_a2", samples * Q_LORA * pair_bytes),
        ("kv_a2", samples * (KV_LORA + PE_DIM) * pair_bytes),
        ("q_an", samples * Q_LORA // 2 * pair_bytes),
        ("kv_a", samples * (KV_LORA + PE_DIM) * pair_bytes),
        ("kvnew", samples * KV_LORA * pair_bytes),
        ("penew", samples * PE_DIM * pair_bytes),
        ("q_nope", samples * heads * NOPE_DIM * pair_bytes),
        ("q_pe", samples * heads * PE_DIM * pair_bytes),
        ("q_lat", samples * heads * KV_LORA * pair_bytes),
        ("sp_acc", samples * split_count * heads * KV_LORA * pair_bytes),
        ("sp_m", samples * split_count * heads * pair_bytes),
        ("sp_l", samples * split_count * heads * pair_bytes),
        ("o", samples * output_heads * V_DIM * pair_bytes),
        ("a", samples * HIDDEN * pair_bytes),
        ("scores", samples * N_EXPERTS * pair_bytes),
        ("xq", samples * HIDDEN // 4 * pair_bytes),
        (
            "xqs",
            samples * (XQ_GROUPS if native_fp4_mfma else XQ_BLOCKS) * pair_bytes,
        ),
        ("sel", samples * MOE_SLOTS * pair_bytes),
        ("prob", samples * MOE_SLOTS * pair_bytes),
        ("mid", samples * MOE_SLOTS * inter * pair_bytes),
        # mid's FP8 blocks (four values a pair) and their scales
        ("midq", samples * MOE_SLOTS * inter // 4 * pair_bytes),
        ("midqs", samples * MOE_SLOTS * inter // 128 * pair_bytes),
        ("ugp", BLOCKS * samples * 2 * UG_TILE * pair_bytes),
        ("xqd", samples * HIDDEN * 4),
    ]
    if with_indexer:
        items += [
            ("index_k", samples * INDEX_DIM * pair_bytes),
            # each row's new key as cached: its FP8 value words, then its scale
            ("index_k8", samples * (INDEX_DIM // 4 + 1) * pair_bytes),
            ("index_w", samples * INDEX_HEADS * pair_bytes),
            ("index_q", samples * INDEX_Q_ROWS // 2 * pair_bytes),
            ("index_qh", samples * INDEX_Q_ROWS // 2 * pair_bytes),
            ("index_scores", samples * index_max_seq * pair_bytes),
            ("indices", samples * sparse_attention_topk * pair_bytes),
            # sliced top-k: per row, digit and slice the slice's 64 counts (two a
            # pair); per row the candidates (key, cache slot) under the threshold's prefix
            (
                "sel_hist",
                samples * INDEX_SEL_DIGITS * INDEX_SEL_SLICES * 32 * pair_bytes,
            ),
            ("sel_cand", samples * INDEX_CAND_STAGE_CAP * 2 * pair_bytes),
            # per row and scoring run, its two slice segments' first-digit counts (64
            # bins each, two a pair)
            ("score_hist", samples * BLOCKS * 64 * pair_bytes),
        ]
    if eh_tiles:
        items += [
            (
                "ehp",
                (eh_parts(eh_tiles) - 1) * eh_tiles * samples * HEAD_TILE * pair_bytes,
            )
        ]

    offset, scratch = 0, {}
    for name, size in items:
        scratch[name] = offset
        offset += _align(size)
    scratch["_bytes"] = offset

    part = npes * samples * HIDDEN * pair_bytes
    region = 2 * part
    dcp_part = (
        dcp_size * samples * (heads * V_DIM // UV_TILE) * DCP_SUMMARY_PAIRS * pair_bytes
        if dcp_size > 1
        else 0
    )
    symmetric = {
        "attn": 0,
        "ffn": region,
        "dcp": 2 * region,
        # the guard's error flags: rank r writes word r of every rank's flags
        "flags": 2 * region + 2 * dcp_part,
        "_part_stride": part,
        "_dcp_part_stride": dcp_part,
        "_bytes": 2 * region + 2 * dcp_part + FLAG_BYTES,
    }
    if tokens or eh_tiles:
        symmetric["xin"] = symmetric["_bytes"]
        symmetric["_xin_stride"] = samples * HIDDEN * 4
        symmetric["_bytes"] += 2 * samples * HIDDEN * 4
    if tokens and eh_tiles:
        symmetric["ein"] = symmetric["_bytes"]
        symmetric["_bytes"] += 2 * samples * HIDDEN * 4
    return scratch, symmetric


def stage_tasks(
    samples: int,
    heads: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    expert_mxfp4: bool = False,
    inter: int = INTER,
    tokens: bool = False,
    head_rows: int = 0,
    eh_tiles: int = 0,
):
    """Return ``(stage name, task count)`` pairs in execution order; with ``tokens``
    led by the embedding rows' push, with ``eh_tiles`` by the NextN entry (each tile's
    K parts, ``eh_parts``). With ``head_rows`` (this rank's LM-head rows) the tail's
    instead."""

    if head_rows:
        return [("lmhead", head_rows // HEAD_TILE), ("gather", BLOCKS)]
    tasks = [("embed", N_ROW_TILES)] if tokens else []
    tasks += [("eh", eh_parts(eh_tiles) * eh_tiles)] if eh_tiles else []
    tasks += [
        # every CTA: a tile's K head, two tiles' K rests or, with the indexer, an index
        # K or weights projection tile
        ("qkv_a", BLOCKS),
        ("q_norm", samples),
        ("cache", samples),
    ]
    if with_indexer:
        tasks += [("index_key", samples)]
    tasks += [
        ("q_b", heads * (NOPE_DIM + PE_DIM) // Q_B_TILE),
    ]
    if with_indexer:
        # index_qh: each row's query, rotated and Hadamard-transformed, 8 heads a task
        tasks += [
            ("index_q", INDEX_Q_ROWS // INDEX_TILE),
            ("index_qh", samples * INDEX_HEADS // 8),
        ]
    tasks += [("uk", heads * KV_LORA // UK_TILE)]
    if with_indexer:
        tasks += [
            # one task a CTA: its runs of the requests' keys
            ("index_score", BLOCKS),
            ("index_slice", samples * INDEX_SEL_SLICES),
        ]
    tasks += [
        (
            "split",
            samples
            * (heads // WAVES)
            * (sparse_attention_topk // sparse_keys_per_task(samples, heads)),
        ),
        ("uv", samples * (heads * V_DIM // UV_TILE)),
        ("o", N_ROW_TILES),
        ("router", samples * N_ROUTER),
        (
            "ug",
            (
                BLOCKS
                if samples == 1
                else samples
                * max(
                    BLOCKS,
                    ((MOE_SLOTS * inter // UG_TILE + BLOCKS - 1) // BLOCKS) * BLOCKS,
                )
                if samples <= 4
                # above 4 rows: rounds of 8-intermediate tiles of the shared expert
                # and of up to every row's routed experts
                else ((samples * TOP_K + 1) * (inter // 8) + BLOCKS - 1)
                // BLOCKS
                * BLOCKS
            ),
        ),
        ("down", HIDDEN // dn_tile(samples, expert_mxfp4)),
    ]
    return tasks


def stage_bases(
    samples: int,
    heads: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    expert_mxfp4: bool = False,
    inter: int = INTER,
):
    """Return ``{stage name: base CTA}``: task ``t`` of a stage runs on CTA
    ``(base + t) % BLOCKS``."""

    tasks = dict(
        stage_tasks(
            samples,
            heads,
            sparse_attention_topk,
            with_indexer,
            index_max_seq,
            expert_mxfp4,
            inter=inter,
        )
    )
    base, acc = {}, 0
    # the front takes every CTA (a tile's K head, or two tiles' K rests); q_norm (with
    # each row's cache task) and q_b follow on the first CTAs, split past q_b's (its
    # last few on the q_norm CTAs); the cache keeps one CTA of the sequence after
    # split, so uk (on the q_b CTAs it waits on anyway) and uv keep their CTAs
    for name in ("qkv_a", "q_norm", "q_b", "split", "cache", "uk", "uv"):
        base[name] = acc % BLOCKS
        acc += 1 if name == "cache" else tasks[name]
    base["cache"] = base["q_norm"]
    # o from split's first CTA: its tasks fill the split CTAs, free once the
    # attention ends
    base["o"] = base["split"]
    # the MoE stages keep the CTAs of a front of whole tiles
    acc = N_QKV_A + ((INDEX_DIM + INDEX_HEADS) // QKV_A_TILE if with_indexer else 0)
    for name in ("q_norm", "q_b", "split", "uk", "uv", "o", "router", "ug", "down"):
        if name in ("router", "ug", "down"):
            base[name] = acc % BLOCKS
        acc += tasks[name] + (1 if name == "split" else 0)
    if with_indexer:
        # q_b occupies exactly half the grid.  Put the second half of index-Q on
        # the complementary CTAs while q_b CTAs reuse their normalized q_lora
        # tile for the first half, instead of serializing two index-Q tiles on
        # every q_b CTA.
        base["index_q"] = (base["q_b"] + tasks["q_b"]) % BLOCKS
        # on the first CTAs of index_q's own tiles, after their tile
        base["index_qh"] = base["index_q"]
        # each row's index key on its q_norm CTA, after the CTA's index-query tile
        base["index_key"] = base["q_norm"]
        # one scoring task a CTA, numbered by its CTA
        base["index_score"] = 0
        # the selection's slices past uk's CTAs
        base["index_slice"] = (base["uk"] + tasks["uk"]) % BLOCKS
    return base
