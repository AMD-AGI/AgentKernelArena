# SPDX-License-Identifier: Apache-2.0
"""Deadlines, sticky error words and the stop of a failed launch (RFC #43949).

Each wait on another workgroup or rank has a deadline below the server's watchdog:
half of ``--watchdog-timeout``, at most 60 s, and SGLang refuses one under 5 s, so a
normal delay on one rank is not a failure. On a timeout the wait sets a sticky error
word, flags every peer, fences and stops the launch (a store to the null page), and
with it the process; a wait that sees a failure (its GPU's words or a peer's flag)
stops too. A launch never runs on after a failed wait, since its data-dependent
addresses would come from words that never arrived, so a finished step's tokens are
correct with nothing checked on the host. A pending wait checks the flags every
``CHECK_POLLS`` polls and its deadline on the 100 MHz ``s_memrealtime`` clock from the
first check; the guard reads the deadline and the peers' addresses only on those
checks and on a failure.

The guard block lives in the launch's epoch buffer (int32 words):

====  ======================================================================
word  meaning
====  ======================================================================
0     the epoch
1-2   unused here: the deadline is the compile-time ``DEADLINE_TICKS``
3     ``deadline``: 1 + the site of the first wait on this GPU that timed out
4     ``peer``: set when a wait saw a failure elsewhere
5     ``fault``: unused here
====  ======================================================================

A rank's error flags are ``FLAG_WORDS`` words at ``flags_offset`` of each rank's
peer buffer: rank r writes word r of every rank's flags, its own too.
"""

import contextlib
from typing import Optional

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from . import buffer_ops as bo
from .compat import gptr, store
from .ops import CM_DEV, CM_SYS, load_ptr64, rsrc
from .ops import clock as clock

GUARD_DEADLINE = 1  # int64 at words 1-2
ERR_DEADLINE = 3
ERR_PEER = 4
ERR_FAULT = 5
ERROR_WORDS = (ERR_DEADLINE, ERR_PEER, ERR_FAULT)
GUARD_WORDS = 8  # the epoch buffer's words before the per-CTA marks

FLAG_WORDS = 8  # one a rank; TP 8 at most
FLAG_BYTES = 256  # the peer buffer's flags region, padded to a cache line pair

CLOCK_HZ = 100_000_000  # s_memrealtime: the constant 100 MHz reference clock
# a wait's check compares time in 2**TICK_SHIFT ticks (2.56 us), as 32-bit scalars
TICK_SHIFT = 8
# a longer deadline wraps in that check, and a wait fails at its first check
MAX_DEADLINE_TICKS = (1 << 31 << TICK_SHIFT) - 1


def deadline_ticks(seconds: float) -> int:
    ticks = int(seconds * CLOCK_HZ)
    if ticks > MAX_DEADLINE_TICKS:
        raise ValueError(
            f"a {seconds} s wait deadline does not fit the wait's 32-bit check; "
            f"use at most {MAX_DEADLINE_TICKS // CLOCK_HZ} s"
        )
    return ticks


# every wait's deadline: SGLang sets half of --watchdog-timeout, at most 60 s
DEADLINE_TICKS = deadline_ticks(20.0)


def stop_launch():
    """Stop the launch with a volatile store to the null page.

    The fault at address 0x0 tells a deliberate stop apart from a real one, and the
    runtime tears the queue down. Unlike ``llvm.trap`` it asks nothing of the kernel's
    ABI, so a launch that never fails pays nothing for it."""
    store(gptr(0), fx.Int32(1), words=1, volatile=True)


class Guard:
    """One launch's deadline and error words, and optionally the peers' flags.

    ``epoch``: the launch's epoch buffer address. ``own``: this rank's peer
    buffer base and ``peers`` the table of every rank's (traced), with ``rank``,
    ``npes`` and ``flags_offset``; without them a failure stays on this GPU (a
    launch that never waits on peers). ``stop=False`` keeps a failed launch
    running.
    """

    def __init__(
        self,
        epoch,
        own=None,
        peers=None,
        rank=None,
        npes=1,
        flags_offset=0,
        stop=True,
    ):
        assert (own is None) == (peers is None), (
            "own and peers come together: a rank reads the flags its peers write"
        )
        self.epoch = epoch
        self.own = own
        self.peers = peers
        self.has_peers = peers is not None
        self.rank = rank
        self.npes = npes
        self.flags_offset = flags_offset
        self.stop = stop
        self._sites = 0

    def deadline(self):
        """The deadline in clock ticks."""
        return fx.Int64(DEADLINE_TICKS)

    def scalar_deadline(self):
        """The deadline >> ``TICK_SHIFT``."""
        return fx.Int32(DEADLINE_TICKS >> TICK_SHIFT)

    def next_site(self) -> int:
        self._sites += 1
        return self._sites

    def _word(self, i):
        return fx.Int32(
            bo.buffer_load(
                rsrc(self.epoch), i, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
            )
        )

    def flags(self):
        """This rank's flags as its peers wrote them (system scope); 0 when clean."""
        if not self.has_peers:
            return fx.Int32(0)
        acc = fx.Int32(0)
        # a buffer load carries at most 4 dwords: TP 8 reads its flags in two
        for c0 in range_constexpr(0, self.npes, 4):
            n = min(4, self.npes - c0)
            w = bo.buffer_load(
                rsrc(self.own + fx.Int64(self.flags_offset)),
                c0,
                vec_width=n,
                dtype=T.i32,
                cache_modifier=CM_SYS,
            )
            if const_expr(n == 1):
                acc = acc | fx.Int32(w)
            else:
                v = fx.Vector(w)
                for p in range_constexpr(n):
                    acc = acc | v[p]
        return acc

    def stopped(self):
        """Whether this GPU already failed or saw a peer fail: later waits return at once."""
        own = self._word(ERR_DEADLINE) | self._word(ERR_PEER) | self._word(ERR_FAULT)
        return (own | self.flags()) != 0

    def fail(self, site):
        """This GPU's wait at ``site`` passed its deadline: set the error word, flag
        every peer, fence, and stop the launch."""
        bo.buffer_store(
            fx.Int32(site + 1), rsrc(self.epoch), ERR_DEADLINE, cache_modifier=CM_DEV
        )
        if self.has_peers:
            for p in range_constexpr(self.npes):
                bo.buffer_store(
                    fx.Int32(1),
                    rsrc(load_ptr64(self.peers, p) + fx.Int64(self.flags_offset)),
                    self.rank,
                    cache_modifier=CM_SYS,
                )
        rocdl.s_waitcnt(vmcnt=0)
        if const_expr(self.stop):
            stop_launch()

    def note_peer(self):
        """A wait saw a failure elsewhere: stop the launch (the failed rank flagged
        everyone), or without the stop keep a sticky ``peer`` word."""
        if const_expr(self.stop):
            stop_launch()
        else:
            old = self._word(ERR_PEER)
            bo.buffer_store(
                (old == 0).select(fx.Int32(1), old),
                rsrc(self.epoch),
                ERR_PEER,
                cache_modifier=CM_DEV,
            )


_CURRENT: list[Guard] = []


@contextlib.contextmanager
def scope(guard: Optional[Guard]):
    """Make ``guard`` the one the waits traced inside this block take, so a kernel
    builds its guard once at the top."""
    _CURRENT.append(guard)
    try:
        yield guard
    finally:
        _CURRENT.pop()


def current() -> Optional[Guard]:
    return _CURRENT[-1] if _CURRENT else None
