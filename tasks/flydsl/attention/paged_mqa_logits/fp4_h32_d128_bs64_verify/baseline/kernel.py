# SPDX-License-Identifier: Apache-2.0
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1-Flash's FP4 lightning-indexer logits of one DSpark verify step (K2i) in the arena
run() interface.

The device code is sgl-amd-mono's K2i (``k2i.py`` / ``score_fp4.py``, vendored at 52486703): one
grid of 256 CTAs x 512 threads, each scoring the task of the step's walk plan (``walk_plan.py``, a
FlyDSL port of the plugin's one-CTA Triton planner). Host work mirrors ``K2iRunner.walk_plan`` and
``K2iRunner.score`` (``indexer/runner.py``) for one FULL index-source layer: the walk plan from the
step's device bounds and owners, then the scorer on SGLang's per-row page table (an entry is a
64-row page, 2 ** 3 of the scorer's virtual 8-row pages), launched on the current stream.

Differences from the runner, all host-side: the plan is a fresh buffer of this call (the runner
keeps one per compress ratio and step), and the logits are the caller's [rows, ctx] tensor (the
runner's is [rows, page_table width * 64] in a preallocated pool), whose columns at or past each
row's bound, which K2i does not write (SGLang's top-k never reads them), the task defines as -inf:
``tail_fill.py`` writes exactly those, so every element is written once.
"""

import torch

from .k2i import MAX_ROWS, build_k2i
from .plan import BLOCKS
from .tail_fill import blocks, build_tail_fill
from .walk_plan import build_walk_plan

PAGE_SHIFT = 3  # a page-table entry: a 64-row page, 8 virtual 8-row pages


def run(*, q_fp4, q_scale, weights, kv_cache, kv_scale, page_table, bounds, owners, logits) -> None:
    """q_fp4 [rows, 32, 64] uint8 (E2M1 pairs); q_scale [rows, 1, 4, 16, 4] uint8 (E8M0); weights
    [rows, 32] bfloat16; kv_cache [pages, 1, 4, 64, 16] and kv_scale [pages, 1, 4, 64] uint8
    (SGLang's FP4 index plane); page_table [rows, pages_per_request] int32; bounds and owners [rows]
    int32; logits [rows, ctx] float32."""
    rows = q_fp4.shape[0]
    assert 0 < rows <= MAX_ROWS and page_table.stride(1) == 1 and logits.stride(1) == 1
    stream = torch.cuda.current_stream()
    plan = torch.empty(BLOCKS, 4, dtype=torch.int32, device=q_fp4.device)
    build_walk_plan()(bounds.data_ptr(), owners.data_ptr(), plan.data_ptr(), rows, stream=stream)
    # -inf from each row's bound on (K2i writes the columns below it)
    width = logits.shape[1]
    build_tail_fill()(bounds.data_ptr(), logits.data_ptr(), logits.stride(0), rows, blocks(width), stream=stream)
    build_k2i()(
        q_fp4.data_ptr(),
        q_scale.data_ptr(),
        weights.data_ptr(),
        kv_cache.data_ptr(),
        kv_scale.data_ptr(),
        plan.data_ptr(),
        page_table.data_ptr(),
        page_table.stride(0),
        page_table.shape[1],
        owners.data_ptr(),
        PAGE_SHIFT,
        bounds.data_ptr(),
        logits.data_ptr(),
        logits.stride(0),
        stream=stream,
    )
