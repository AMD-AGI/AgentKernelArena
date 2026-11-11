# SPDX-License-Identifier: Apache-2.0
"""Column-slice collectives of the wide-row launch.

The residual stream is column-sharded: rank r owns columns [r sl, (r + 1) sl)
of an [m, H] activation, sl = H / TP. Three collectives move slices:

* RS: every rank holds a partial [m, H]; the owner of slice j sums the TP
  partials of its columns -> [m, sl].
* AG: every owner holds its [m, sl]; every rank gets the [m, H].
* AR (two-shot): RS, then AG of the reduced slice.

Two formats:

* "tag": each bf16 pair travels next to the launch tag (``Mailbox``), so a
  consumer polls exactly the pairs it reads and needs no flag. Ties or beats
  "flag" at every measured size (8 .. 384 rows x 7168) despite twice the bytes.
* "flag": plain 16 B stores into the peer's uncached memory, drained with
  ``s_waitcnt vmcnt(0)`` before a tagged flag a (chunk, source); CTA b handles
  chunk b of every slice on every rank, and a consumer waits on its chunk's
  flags. Half the link bytes, for when the link is shared with overlapped work.
  The vmcnt drain orders the stores before the flag on gfx950 (acked uncached
  writes), which is measured, not documented: ``coll_bench.py --stress``.

Both reduce in f32 in rank order, so the result does not depend on timing.
Each region is double-buffered by the launch parity: a fast rank may push the
next launch's slices while a slow peer still polls this launch's, and the next
tag would never match that poll.

AKA: only the peer-region layout (``layout`` / ``peer_bytes`` / ``flags_offset``) and the region context
(``context`` / ``region_context`` / ``_Context``) are vendored, unchanged: ``gemm_coll.ag_gemm`` pushes and
polls through them. The collectives themselves (tag / flag RS, AG, AR), ``_delay`` and the standalone
builder (test/kernels/kimi_k31/coll_bench.py) are not used by this launch.
"""

import flydsl.expr as fx

from . import guard
from .ops import (
    ld_i32,
    load_ptr64,
    sfresh64,
    sload64,
    uniform,
)
from .plan import BLOCKS
from .sync import Mailbox, preg, shift
from .tag import step_tag

TP = 8
KINDS = ("rs", "ag", "ar")
FORMATS = ("tag", "flag")


def layout(m, h, fmt, blocks=BLOCKS):
    """Byte offsets of one parity half: {"rs", "ag", "fl"} and the half's size.
    "tag" stores a 4 B tag next to every bf16 pair (2x the payload)."""
    eb = 4 if fmt == "tag" else 2  # bytes an element in a region
    rs = TP * m * (h // TP) * eb
    ag = m * h * eb
    fl = 2 * TP * blocks * 8 if fmt == "flag" else 0
    half = (rs + ag + fl + 255) // 256 * 256
    return {"rs": 0, "ag": rs, "fl": rs + ag}, half


def peer_bytes(m, h, fmt, blocks=BLOCKS):
    """The peer buffer a standalone collective needs: two halves + the guard flags."""
    return 2 * layout(m, h, fmt, blocks)[1] + guard.FLAG_BYTES


def flags_offset(m, h, fmt, blocks=BLOCKS):
    return 2 * layout(m, h, fmt, blocks)[1]


# ------------------------------------------------------------ the context


def context(peers, rank, epoch, slot, m, h, fmt, tid, bid, launches=1, parity="launch"):
    """The collective's mailbox, regions (this launch's parity half) and ids.

    A step bumps the epoch once and runs ``launches`` launches, slot 0 .. on one
    peer buffer. "launch" parity alternates the half every launch (global launch
    index e * launches + slot), so a rank that races into the next launch writes
    the other half; launch n + 2 may reuse n's half because every launch has an
    exchange in which each rank waits for every other, so no rank can reach n + 2
    before all have left n. "epoch" (the K3 tag scheme's) shares one half within a
    step: only for the test that shows the hazard."""
    offs, half = layout(m, h, fmt)
    return region_context(
        peers, rank, epoch, slot, offs, half, tid, bid, launches, parity
    )


def region_context(
    peers, rank, epoch, slot, offs, half, tid, bid, launches=1, parity="launch"
):
    """``context`` for any named peer regions (``offs``: name -> offset in a half):
    ``<name>_of(p)`` is rank p's region, ``<name>_own`` this rank's."""
    assert parity in ("launch", "epoch")
    mb = Mailbox(step_tag(epoch, slot))
    e = uniform(ld_i32(epoch, 0))
    bit = (e * launches + slot) % 2 if parity == "launch" else e % 2
    par = fx.Int64(bit) * half

    def base_of(p):
        # read from the peers table where used (scalar load through an opaque copy of the table address):
        # bases formed once at the entry stay live, or spilled, across the whole launch
        return sload64(sfresh64(peers) + fx.Int64(p) * 8)

    def region(name):
        def of(p):
            base = base_of(p) if isinstance(p, int) else load_ptr64(peers, p)
            return shift(preg(base, offs[name], name), par)

        return of

    c = _Context(
        {
            "tid": tid,
            "bid": bid,
            "rank": rank,
            "put": mb.put,
            "put_words": mb.put_words,
            "poll": mb.poll,
        }
    )
    for name in offs:
        c[f"{name}_of"] = region(name)
        c.lazy[f"{name}_own"] = lambda name=name: shift(
            preg(base_of(rank), offs[name], name), par
        )
    return c


class _Context(dict):
    """A region context whose ``<name>_own`` addresses are formed at each use (``lazy``), not at the entry."""

    def __init__(self, *a):
        super().__init__(*a)
        self.lazy = {}

    def __getitem__(self, k):
        if not dict.__contains__(self, k) and k in self.lazy:
            return self.lazy[k]()
        return super().__getitem__(k)

    def __contains__(self, k):
        return dict.__contains__(self, k) or k in self.lazy

    def get(self, k, default=None):
        return self[k] if k in self else default
