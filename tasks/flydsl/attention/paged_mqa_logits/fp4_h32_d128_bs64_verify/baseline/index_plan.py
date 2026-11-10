# SPDX-License-Identifier: Apache-2.0
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 68506823, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v41/mono/index_plan.py (the FP4 plane's parts), by way of
# sgl-project/sglang#43984 at 134919476c
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/deepseek_v41/gfx950/indexer/index_plan.py).
# Modified for AgentKernelArena: the constants and the per-token chunk count are kept; the host
# walk plan ``fill_walk_plan`` (numpy; this baseline builds the plan on the device, walk_plan.py)
# and the REINDEX split helpers (this task scores FULL layers) are not.
"""The FP4 scorer's tasks of a step, defined once.

The kernel calls these on traced Int32; only non-negative operands reach ``//`` and ``%``, where
Python's floor and the device's truncation agree. A token's ``bound`` is the columns its scorer
may read.
"""

from .plan import cdiv, sel

CHUNK = 128  # columns a score task: 16 tiles of 8 rows
TOPK = 512
TOPK_BLOCKS = 2048
BLOCK_ROWS = 8  # a candidate block, and an index tile (an FP4 plane's page)
STEP = 64  # keys an FP4 walk step: a wave's 4 MFMA tiles, 8 pages
WALK_ROWS = 6  # query rows an FP4 walk scores off each key load
HEADS = 32  # the indexer's heads (``index_n_heads``)
DIM = 128  # an index head's dims (``index_head_dim``)


def scored(bound):
    """Whether the scorer runs for a token.

    At ``bound <= TOPK`` SGLang's top-k takes every column and reads no logits."""
    return bound > TOPK


def score_chunks(bound):
    """A token's scored chunks; chunk ch covers columns [CHUNK ch, CHUNK (ch + 1))."""
    return sel(scored(bound), cdiv(bound, CHUNK), 0)
