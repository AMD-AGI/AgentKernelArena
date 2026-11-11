# SPDX-License-Identifier: Apache-2.0
"""Tile GEMMs fused with the column-slice collectives (one launch): the all-gather -> GEMM half.

AG -> GEMM (in_proj): every rank pushes its x slice [m, k / TP] into every rank's ag region (plain 16 B
system-scope stores; CTA group j feeds rank j), the last CTA to finish (a local arrival counter) flags
every rank, and a GEMM task starts once the owners of its K range have flagged; it reads x from this
rank's ag region. A task is a [bm, bn] output tile over one of ``ks`` K slices (``tile_mma.mma_tile``).

Merge mode "linbf": packed-bf16 atomics into a zeroed bf16 out (``tile.atomic_lin``), so the K-slice
order follows timing and out is not bitwise deterministic for ks > 2.

AKA (vendored from sgl-amd-mono 52486703, ``kimi_k31/gfx950/stages/gemm_coll.py``: ``_ag_push``,
``_ag_wait`` and ``ag_gemm``). Modifications:

* the launch takes its row count ``rows`` (a kernel argument, 1 <= rows <= m) where upstream bakes the
  build's ``m`` into the push loop, the task count, the x-load clamp and the output masks; the build's
  ``m`` (the row bucket) still sets the peer layout (``coll.context`` / ``coll.flags_offset``). A launch
  with rows == m does exactly the upstream build's work, and one build serves every row count of its
  bucket;
* only the production-relevant "linbf" merge with ks > 1 is kept ("part" and the ks = 1 bf16 store, and
  with them ``dev_emit``, are dropped), with ``phases`` = 3 (push, wait and GEMM; the timing-only
  ``phases`` = 1 is dropped) and without the ``stress`` delays (and the ``delay`` kernel argument);
* the arrival counter's ``fx.inttoptr`` + ``fx.llvm.atomic_add`` (FlyDSL >= 0.3.4) is
  ``compat.global_atomic_add``, the same ``atomicrmw add`` (agent scope, monotonic);
* the build key lists the parameters by name (upstream ``key_tuple(locals(), ...)``; ``locals`` is not
  allowed in candidate code);
* the GEMM -> reduce-scatter half (``gemm_rs`` and its helpers) is not vendored.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from . import coll, guard
from .compat import global_atomic_add
from .ops import CM_SYS, load_ptr64, rsrc
from .plan import BLOCKS, THREADS, key_tuple
from .tile import atomic_lin
from .tile_mma import DEPTH, RW, mma_tile

SOURCES = "aka-flat"  # AKA: no source digest (FlyDSL hashes the flat package's helpers)

TP = coll.TP


@flyc.jit
def _ag_push(c, xs, m, sl, cnt):
    """Push this rank's x slice [m, sl] to every rank's ag region [m, TP sl] (plain
    16 B stores; CTA group j feeds rank j); the last CTA to finish flags every rank."""
    grp = BLOCKS // TP
    j = c["bid"] // grp
    dst = c["ag_of"](j)
    q8 = sl // 8
    h = TP * sl
    r_x = rsrc(xs)
    for u in range((c["bid"] % grp) * THREADS + c["tid"], m * q8, grp * THREADS):
        row, q = u // q8, u % q8
        v = fx.Vector(
            bo.buffer_load(r_x, (row * sl + 8 * q) // 2, vec_width=4, dtype=T.i32)
        )
        bo.buffer_store(
            v,
            rsrc(dst.value),
            (row * h + c["rank"] * sl + 8 * q) // 2,
            cache_modifier=CM_SYS,
        )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if c["tid"] == 0:
        # AKA: upstream fx.llvm.atomic_add on an fx.inttoptr pointer (FlyDSL >= 0.3.4)
        old = global_atomic_add(cnt, fx.Int32(1), syncscope="agent")
        if (old + 1) % BLOCKS == 0:
            for p in range_constexpr(TP):
                c["put"](c["fl_of"](p), c["rank"], fx.Int32(1))


@flyc.jit
def _ag_wait(c, j0, nown):
    """Wait for the flags of owners j0 .. j0 + nown (their slices have landed here)."""
    if c["tid"] < nown:
        c["poll"]([(c["fl_own"], j0 + c["tid"], 1)])
    gpu.barrier()


def ag_gemm(
    m,
    n,
    k,
    bm=128,
    bn=128,
    wm=2,
    wn=4,
    ks=4,
    depth=DEPTH,
    launches=1,
):
    """out[rows, n] = AG(x slices) @ W^T in one launch: every rank pushes its x slice
    [rows, k / TP] (``xs``), and a GEMM task starts once the owners of its K range
    have landed, reading x from the ag region. ``out``: bf16 [rows, n], zeroed before
    the launch (linbf). ``cnt``: one int32, zero before the launch. ``m``: the row
    bucket the build is laid out for (rows <= m, a launch argument).
    ``slot_i`` / ``launches``: the launch's place in its step (``coll.context``)."""
    # AKA: the build parameters by name (upstream: locals(), which the candidate rules do not allow)
    params = dict(m=m, n=n, k=k, bm=bm, bn=bn, wm=wm, wn=wn, ks=ks, depth=depth, launches=launches)
    keyed = key_tuple(params, SOURCES)
    key = (m, n, k, bm, bn, wm, wn, ks, depth, launches)
    if key in _AG_BUILDS:
        return _AG_BUILDS[key]
    sl = k // TP
    kt = k // ks
    assert ks > 1 and k % ks == 0 and kt % 64 == 0 and (kt % sl == 0 or sl % kt == 0)
    nown = max(kt // sl, 1)
    tn = (n + bn - 1) // bn
    fo = coll.flags_offset(m, k, "flag")

    @fx.struct
    class AbLds:
        xs: fx.Array[fx.Int32, 2 * bm * RW, 16]
        ws: fx.Array[fx.Int32, 2 * bn * RW, 16]

    @fx.struct
    class CLds:
        c: fx.Array[fx.Float32, bm * (bn + 4), 16]

    @fx.union
    class Smem:
        ab: AbLds
        c: CLds

    name = f"k31_ag_gemm_m{m}_n{n}_k{k}_t{bm}x{bn}_w{wm}x{wn}_ks{ks}_d{depth}_linbf_p3_l{launches}_rows"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def ag_gemm_kernel(
        xs: Int64,
        w: Int64,
        out: Int64,
        cnt: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot_i: Int32,
        rows: Int32,
    ):
        own = load_ptr64(peers, rank)
        with guard.scope(
            guard.Guard(
                epoch, own=own, peers=peers, rank=rank, npes=TP, flags_offset=fo
            )
        ):
            tid, bid = fx.thread_idx.x, fx.block_idx.x
            st = fx.SharedAllocator().allocate(Smem)
            lds, cl = st.ab.peek(), st.c.peek().c.ptr
            c = coll.context(
                peers, rank, epoch, slot_i, m, k, "flag", tid, bid, launches
            )
            _ag_push(c, xs, rows, sl, cnt)
            x = c["ag_own"].value
            # AKA: the launch's row tiles (upstream: (m + bm - 1) // bm of the build's m)
            ntasks = ((rows + (bm - 1)) // bm) * (tn * ks)
            for task in range(bid, ntasks, BLOCKS):
                sk = task % ks
                tile = task // ks
                m0 = (tile // tn) * bm
                n0 = (tile % tn) * bn
                _ag_wait(c, (sk * kt) // sl, nown)
                lin = atomic_lin(out, n, n0, rows, n, True)
                mma_tile(
                    tid,
                    x,
                    k,
                    rows,
                    m0,
                    w,
                    k,
                    n,
                    n0,
                    sk * kt,
                    kt,
                    bm,
                    bn,
                    wm,
                    wn,
                    lds.xs.ptr,
                    lds.ws.ptr,
                    None,
                    depth,
                    lin,
                    cl,
                )

    @flyc.jit
    def launch(
        xs: Int64,
        w: Int64,
        out: Int64,
        cnt: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot_i: Int32,
        rows: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):  # noqa: B008
        _ = keyed
        ag_gemm_kernel(xs, w, out, cnt, peers, rank, epoch, slot_i, rows).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    _AG_BUILDS[key] = launch
    return launch


_AG_BUILDS: dict = {}
