# SPDX-License-Identifier: Apache-2.0
"""The wide-row layer launch's tuned GEMM configurations, by row bucket.

AKA: copied verbatim from sgl-amd-mono ``sgl_amd_mono/kimi_k31/wide.py`` (the constant tables and
``cfg_rows``); the rest of that module (SGLang glue: buffers, the step loop, the routed experts) is not
part of the launch and is left out.
"""

TP, H, SL, E, NF, LAT = 8, 7168, 896, 896, 2096, 3584
REQ_BUCKETS = (
    2,
    4,
    8,
    16,
    32,
    48,
)  # requests a bucket holds; its rows are L x requests
SUPPORTED_L = (6, 8)
BUCKETS = tuple(8 * r for r in REQ_BUCKETS)
CFGS = {  # 8-token bucket -> (up, in_proj, o_proj, front, shared down): bm:bn:wm:wn:ks; an L-token
    # bucket of the same requests takes the same configs
    16: (
        "16:128:1:8:8",
        "16:128:1:8:4",
        "16:128:1:8:4",
        "16:128:1:8:8",
        "16:128:1:8:1",
    ),
    32: (
        "32:128:2:4:8",
        "32:128:2:4:4",
        "32:128:2:4:2",
        "32:128:2:4:8",
        "32:128:2:4:1",
    ),
    64: ("64:128:2:4:8", "64:128:2:4:4", "64:64:2:4:2", "64:128:2:4:8", "64:64:2:4:1"),
    128: (
        "128:32:4:2:8",
        "128:128:2:4:4",
        "64:64:2:4:1",
        "128:128:2:4:8",
        "64:64:2:4:1",
    ),
    256: (
        "128:32:4:2:8",
        "128:128:2:4:4",
        "128:64:2:4:1",
        "128:128:2:4:8",
        "128:64:2:4:1",
    ),
    384: (
        "128:32:4:2:4",
        "192:128:4:2:2",
        "192:64:4:2:1",
        "128:128:2:4:4",
        "128:64:2:4:1",
    ),
}
# bucket rows -> MoE front config, where the CFGS one is a poor fit: 17 column tiles x the row tiles x ks must stay
# within BLOCKS (one round), and a row tile past m is wasted work
FRONT_ROWS = {192: "64:128:2:4:4", 256: "96:128:2:4:4", 288: "96:128:2:4:4"}
# bucket rows -> in_proj config, by the same rule (50 column tiles x row tiles x ks within BLOCKS, no padded row tile)
INP_ROWS = {96: "96:128:2:4:4", 192: "192:128:4:2:4", 256: "128:128:2:4:2"}
# bucket rows -> latent up / shared down configs, by the same rule (o_proj: padded rows measured no cost, kept)
UP_ROWS = {
    192: "96:64:2:4:8",
    256: "128:64:4:2:8",
    288: "96:64:2:4:4",
    384: "96:128:2:4:8",
}
SDOWN_ROWS = {288: "96:128:2:4:1", 384: "192:64:4:2:1"}
NARROW = 8  # rows up to this run Kimi-K3's launches (the wide launch starts at 2 requests above 8 rows)


def cfg_rows(b: int, L: int) -> str:
    """The CFGS entry of an L-token bucket of b rows."""
    assert b % L == 0 and b // L in REQ_BUCKETS, (b, L)
    up, inp, oproj, front, sdown = CFGS[8 * (b // L)]
    return (
        UP_ROWS.get(b, up),
        INP_ROWS.get(b, inp),
        oproj,
        FRONT_ROWS.get(b, front),
        SDOWN_ROWS.get(b, sdown),
    )
