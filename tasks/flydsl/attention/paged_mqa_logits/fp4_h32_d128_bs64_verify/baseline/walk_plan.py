# SPDX-License-Identifier: Apache-2.0
# Written for AgentKernelArena: a FlyDSL port of the Triton ``_walk_plan_kernel`` of sgl-amd-mono @
# 52486703 (sgl_amd_mono/deepseek_v41/gfx950/indexer/walk_plan.py, "From sgl-project/sglang#43984 at
# 134919476c"), which lays out ATOM's host ``index_plan.fill_walk_plan`` (ROCm/ATOM at 68506823, MIT
# License, Copyright (C) 2026, Advanced Micro Devices, Inc.) in one CTA. A FlyDSL baseline may not
# launch Triton, so the plan is built here in FlyDSL, with the same result: the Triton kernel's
# linear scan for the smallest feasible width is a bisection over the same range (feasibility is
# monotone in the width), and its per-piece reductions are lane gathers over a lead row's next
# WALK_ROWS - 1 rows (a piece has at most WALK_ROWS rows). A CPU emulation of this lane program
# was checked to equal fill_walk_plan on 2900 random steps (upstream test_walk_plan's distribution
# and this task's shapes).
"""The FULL-layer walk plan on the device, graph safe: one launch, no host reads.

``plan`` [BLOCKS, 4] gets (lead, span, ch0, ch1) a CTA -- chunks [ch0, ch1) of a piece, the
``span`` rows from ``lead``: at most WALK_ROWS adjacent rows of one owner (``owners[t]`` is the
first row of row t's request), scored off each key load -- and zeros past the last task. A piece's
chunks are split evenly at the smallest width that fits BLOCKS CTAs. One wave: lane t is row t.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64

from . import buffer_ops as bo
from . import index_plan as ip
from .k2i import MAX_ROWS, SOURCES, kernel_symbol
from .ops import ld_i32, rsrc
from .plan import BLOCKS

_STREAM = fx.Stream(None)
LANES = 64  # one wave; a step has at most MAX_ROWS < LANES rows
SEARCH_STEPS = 16  # bisection steps: the feasible-width range spans fewer than 2 ** 16 widths
# LDS words: per piece (in lead order) its tasks' end, task count, chunks, first row and span
END_AT, K_AT, NEED_AT, FIRST_AT, SPAN_AT = (i * LANES for i in range(5))


@functools.lru_cache(maxsize=None)
def build_walk_plan():
    """The walk plan's launch for a step of ``n`` rows (a runtime argument, 1 <= n <= MAX_ROWS)."""
    name = kernel_symbol("dsv41_mono_walk_plan", c=BLOCKS)
    keyed = (("ctas", BLOCKS), ("sources", SOURCES))

    @fx.struct
    class PlanLds:
        words: fx.Array[fx.Int32, 5 * LANES, 16]

    @flyc.kernel(name=name, known_block_size=[LANES, 1, 1])
    def walk_plan(bounds: Int64, owners: Int64, plan: Int64, n: Int32):
        _ = keyed
        lds = fx.SharedAllocator().allocate(PlanLds).peek().words.ptr
        add, top = fx.ReductionOp.ADD, fx.ReductionOp.MAX
        one, zero = fx.Int32(1), fx.Int32(0)
        t = fx.thread_idx.x
        row = t < n
        tc = fx.min(t, n - 1)
        bound = row.select(ld_i32(bounds, tc), zero)
        owner = row.select(ld_i32(owners, tc), fx.Int32(-1))
        prev = (row & (t > 0)).select(ld_i32(owners, fx.max(tc - 1, 0)), fx.Int32(-1))
        run_start = row & ((t == 0) | (owner != prev))
        run_first = fx.coop.warp_inclusive_scan(run_start.select(t, zero), top)
        lead = row & (run_start | (((t - run_first) % ip.WALK_ROWS) == 0))
        lead_i = lead.select(one, zero)
        piece = fx.coop.warp_inclusive_scan(lead_i, add) - 1
        chunks = (row & (bound > ip.TOPK)).select((bound + (ip.CHUNK - 1)) // ip.CHUNK, zero)

        # a lead's piece: the lead and the rows after it up to the next lead or the step's end
        span, need, open_ = one, chunks, one
        for d in range_constexpr(1, ip.WALK_ROWS):
            src = fx.min(t + d, LANES - 1)
            member = (open_ != 0) & ((t + d) < n) & (fx.shuffle_idx(lead_i, src, LANES) == 0)
            span = span + member.select(one, zero)
            need = fx.max(need, member.select(fx.shuffle_idx(chunks, src, LANES), zero))
            open_ = member.select(one, zero)
        need = lead.select(need, zero)  # a piece's chunks on its lead lane, 0 elsewhere
        pieces = fx.coop.warp_reduce((need > 0).select(one, zero), add)  # live pieces
        npieces = fx.coop.warp_reduce(lead_i, add)
        total = fx.coop.warp_reduce(need, add)

        # the smallest width whose tasks fit in BLOCKS; ceil(total / (BLOCKS - pieces)) fits, as
        # each ceil adds less than 1
        lo = fx.max((total + (BLOCKS - 1)) // BLOCKS, one)
        room = fx.max(BLOCKS - pieces, one)
        hi = fx.max((total + room - 1) // room, lo)
        for _ in range_constexpr(SEARCH_STEPS):
            mid = (lo + hi) // 2
            tasks = fx.coop.warp_reduce((need + mid - 1) // mid, add)
            go = lo < hi
            hi = (go & (tasks <= BLOCKS)).select(mid, hi)
            lo = (go & (tasks > BLOCKS)).select(mid + 1, lo)
        k = (need + lo - 1) // lo  # the piece's tasks (0 for a dead piece and off the leads)
        end = fx.coop.warp_inclusive_scan(k, add)
        total_k = fx.coop.warp_reduce(k, add)
        if lead:
            fx.ptr_store(end, lds + (END_AT + piece))
            fx.ptr_store(k, lds + (K_AT + piece))
            fx.ptr_store(need, lds + (NEED_AT + piece))
            fx.ptr_store(t, lds + (FIRST_AT + piece))
            fx.ptr_store(span, lds + (SPAN_AT + piece))
        gpu.barrier()

        # CTA j's piece: the first whose tasks end past j (a dead piece ends where it starts)
        ends = [fx.ptr_load(lds + (END_AT + p)) for p in range(MAX_ROWS)]
        out = rsrc(plan)
        for q in range_constexpr(BLOCKS // LANES):
            j = t + q * LANES
            c = zero
            for p in range_constexpr(MAX_ROWS):
                c = c + ((fx.Int32(p) < npieces) & (ends[p] <= j)).select(one, zero)
            c = fx.min(c, npieces - 1)
            k_c = fx.ptr_load(lds + (K_AT + c))
            need_c = fx.ptr_load(lds + (NEED_AT + c))
            i = j - (fx.ptr_load(lds + (END_AT + c)) - k_c)
            per = fx.max(k_c, one)
            task = [
                fx.ptr_load(lds + (FIRST_AT + c)),
                fx.ptr_load(lds + (SPAN_AT + c)),
                i * need_c // per,
                (i + 1) * need_c // per,
            ]
            has = j < total_k
            bo.buffer_store(
                fx.Vector.from_elements([has.select(v, zero) for v in task], fx.Int32), out, j * 4
            )

    @flyc.jit
    def launch(bounds: Int64, owners: Int64, plan: Int64, n: Int32, stream: fx.Stream = _STREAM):
        _ = keyed
        walk_plan(bounds, owners, plan, n).launch(grid=(1,), block=(LANES,), stream=stream)

    return launch
