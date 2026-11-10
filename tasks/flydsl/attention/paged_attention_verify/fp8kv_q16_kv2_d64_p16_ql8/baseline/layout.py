# SPDX-License-Identifier: Apache-2.0
"""Where the Muse Spark launches put their tasks and data, per step width.

Plain Python, evaluated at build time: which CTA runs each stage's tasks, the
scratch regions and the LDS sizes. Kept free of FlyDSL so it is testable without
a GPU (``test/unit/test_muse_spark_layout.py``).
"""

from .plan import BLOCKS, THREADS, WAVES, pair_layout

TRACE_SLOTS = 16  # stage stamps a CTA (``device.stamp``), an int64 pair each
TRACE_LAYERS = 64  # stamp slots a layer: a trace region holds every layer's last launch
from .config import (
    HEAD_DIM,
    HIDDEN,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    Q_K,
    QKVG_ROWS,
)

# ---------------------------------------------------------------- K1 (pre_attn)
ROWS = 16  # projection rows an MFMA tile: the MFMA's M
TILE = 16  # tokens an MFMA B operand holds: a token tile
K_STEP = 32  # the bf16 MFMA's K
G_ROWS = 144  # projection rows a GEMV task: nine MFMA tiles
G_TILES = G_ROWS // ROWS  # 9
K_PARTS = 16  # a task covers one sixteenth of K: 16 groups x 16 parts = one task a CTA, 147 KB each
PARTS = K_PARTS  # partial slots a (token, row)
K_PART = HIDDEN // K_PARTS  # 512
K_PER_WAVE = K_PART // WAVES  # 64: each wave's contiguous K range of a part
K_STEPS = K_PER_WAVE // K_STEP  # 2 MFMAs a wave a (row tile, token tile)
STEP_BATCH = K_STEPS  # one batch: a task's weights all in flight (72 VGPRs)
GEMV_GROUPS = QKVG_ROWS // G_ROWS  # 16 row groups
GEMV_TASKS = GEMV_GROUPS * K_PARTS  # 256 tasks: exactly one a CTA
SECOND_TASKS = GEMV_TASKS - BLOCKS  # 0
HEAD_TASKS = LOCAL_Q_HEADS + 2 * LOCAL_KV_HEADS  # 20: a CTA a q, k or v head
GATE_TASKS = Q_K // HEAD_DIM  # 16: the gate rows' part sums, 64 rows a task
HT0 = 32  # the first head task's CTA
GT0 = HT0 + HEAD_TASKS  # the first gate task's CTA
QT0 = (
    GT0 + GATE_TASKS
)  # the first CTA with a second GEMV task (past the head / gate CTAs)
ELEMS = 4  # a head task lane's dims: 16 lanes x 4 = HEAD_DIM, two RoPE pairs
TOKEN_LANES = 16
TOKENS_PER_PASS = THREADS // TOKEN_LANES  # 32 tokens a head / gate pass
RED_FLOATS = WAVES * 64 * 4  # one (row tile, token tile)'s MFMA accumulators in LDS
assert (
    QKVG_ROWS % G_ROWS == 0
    and K_STEPS % STEP_BATCH == 0
    and GEMV_GROUPS * G_ROWS == QKVG_ROWS
)
assert QT0 + SECOND_TASKS <= BLOCKS and 0 <= SECOND_TASKS <= BLOCKS
assert MAX_TOKENS % TILE == 0


def token_tiles(s: int) -> int:
    """The token tiles of a step of ``s`` rows (a GEMV accumulator each)."""
    return -(-s // TILE)


def token_passes(s: int) -> int:
    """The head / gate task passes of a step: TOKENS_PER_PASS tokens a pass."""
    return -(-s // TOKENS_PER_PASS)


def k1_scratch_layout() -> dict:
    """K1's scratch regions, name -> (byte offset, bytes), sized for MAX_TOKENS:
    the K parts' fp32 partials (a pair a (part, token, row)) and, in a merged
    launch, a flag a token for its x."""
    return pair_layout(
        [
            ("k1_part", PARTS * MAX_TOKENS * QKVG_ROWS),
            ("k1_xflag", MAX_TOKENS),
            ("k1_trace", TRACE_LAYERS * BLOCKS * TRACE_SLOTS),
        ],
        align=256,
    )


def scratch_bytes(layout: dict) -> int:
    return max(off + nbytes for off, nbytes in layout.values())


def k1_lds_floats(s: int) -> int:
    """K1's LDS floats at width ``s``: a reduction row set a row tile (the token tiles
    go one at a time)."""
    return G_TILES * RED_FLOATS


def k1_lds_bytes(s: int) -> int:
    return 4 * k1_lds_floats(s)


# ---------------------------------------------------------------- K-attn (attention)
GQA = 8  # query heads a KV head on a rank
ATTN_CHUNK = 256  # keys a chunk of the task loop
ATTN_MAX_PARTS = 128  # partitions a request at most (the kernel sizes ``part`` to fit)
ATTN_RING_DEPTH = 2  # chunks of K / V a wave keeps in flight (LDS DMA, 4 KB a chunk)
ATTN_IDS_CHUNKS = 128  # chunks whose page ids a task stages in LDS at once (2048 ids)
ATTN_MERGE_GROUPS = (
    8  # level-1 merge jobs a (request, KV head, row tile) when it has more partitions
)
ATTN_Q_PITCH = HEAD_DIM + 4  # LDS pitches (bf16 elements), 16 B aligned, bank-spread
ATTN_V_PITCH = 40  # keys: a 32-key V^T sub-block's row
ATTN_PART_ROW = (
    4 + HEAD_DIM
)  # fp32 words a partial row: (m, l, pad, pad, O[64]), O 16 B aligned
ATTN_KEY_SPLITS = WAVES  # the waves share a task's keys
ATTN_KEYS_PER_WAVE = ATTN_CHUNK // ATTN_KEY_SPLITS  # 32
ATTN_KEY_TILES = ATTN_KEYS_PER_WAVE // 16  # 2
ATTN_V_STEPS = ATTN_KEYS_PER_WAVE // 32  # 32-key PV steps a wave: 1
ATTN_P_PITCH = ATTN_KEYS_PER_WAVE + 8  # bf16 elements, 16 B aligned
assert GQA * LOCAL_KV_HEADS == LOCAL_Q_HEADS


def attn_rows(q_len: int) -> int:
    """Rows a (request, KV head): the request's q_len rows x GQA heads."""
    return q_len * GQA


def kattn_scratch_layout(tokens: int, q_len: int) -> dict:
    """K-attn's scratch (a "pair" is 8 bytes): the tasks' partial rows ((request, head,
    partition, row) x ATTN_PART_ROW fp32 words) and a flag a task, (request, head,
    first row tile, partition): the tasks of one partition's row tiles finish apart,
    so each has its own flag; the level-2 records
    ((request, head, row tile, group, row16) x ATTN_PART_ROW words) and a flag each."""
    R = tokens // q_len
    rows = attn_rows(q_len)
    tiles = (rows + 15) // 16
    part_words = R * LOCAL_KV_HEADS * ATTN_MAX_PARTS * rows * ATTN_PART_ROW
    l2_words = R * LOCAL_KV_HEADS * tiles * ATTN_MERGE_GROUPS * 16 * ATTN_PART_ROW
    return pair_layout(
        [
            ("attn_part", part_words // 2),
            ("attn_pflag", R * LOCAL_KV_HEADS * tiles * ATTN_MAX_PARTS),
            ("attn_l2", l2_words // 2),
            ("attn_l2flag", R * LOCAL_KV_HEADS * tiles * ATTN_MERGE_GROUPS),
            ("attn_trace", TRACE_LAYERS * BLOCKS * TRACE_SLOTS),
        ],
        align=256,
    )


def kattn_lds_bytes(nt: int) -> int:
    """The LDS of a task of ``nt`` row tiles: Q and the merge jobs' reduction, then the
    larger of the chunk loop's K / V ring and page ids, and the key-split tree (they
    alias)."""
    qs = nt * 16 * ATTN_Q_PITCH * 2
    red = THREADS * 10 * 4
    ring = (
        WAVES * ATTN_RING_DEPTH * (2 * 2 * 16 * HEAD_DIM + 64)
    )  # (K, V) x 2 pages x 16 keys x 64 fp8 a chunk, the V skew
    ids = ATTN_IDS_CHUNKS * 16 * 4
    comb = WAVES * nt * 16 * ATTN_PART_ROW * 4
    return qs + red + max(ring + ids, comb)
