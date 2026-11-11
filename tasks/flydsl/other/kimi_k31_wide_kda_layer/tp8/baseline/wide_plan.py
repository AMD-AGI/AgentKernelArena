# SPDX-License-Identifier: Apache-2.0
"""Build-time planning of the wide-row launch: peer / scratch regions and stage checks.

A launch is a list of stages that every CTA walks in order, each stage's tasks
spread over the CTAs (task t on CTA t % BLOCKS). Stages do not meet at a grid
barrier: a task waits only for the words it reads (mailbox pairs, or flags after
plain stores), so a CTA can start stage s + 1 while others are still in stage s.
That is deadlock-free when every CTA is resident (one CTA a CU, BLOCKS = CUs)
and every wait is on a stage at or before the waiter's.

The peer buffer holds two halves of the launch's regions (``coll.context``
alternates them by launch); that is enough only if the launch has an exchange in
which each rank waits for every other (``Plan.check``).
"""

import dataclasses

from . import guard
from .plan import BLOCKS

ALIGN = 256


# AKA: frozen dataclasses instead of the upstream ``msgspec.Struct`` (not a candidate dependency)
@dataclasses.dataclass(frozen=True)
class Region:
    name: str
    nbytes: int
    peer: bool  # in the peer buffer (both halves), else in the local scratch


@dataclasses.dataclass(frozen=True)
class Stage:
    name: str
    tasks: int
    waits_on: tuple = ()  # names of the stages whose words its tasks wait for
    all_ranks: bool = False  # a task waits for a word from every rank


class Plan:
    """Regions and stages of one launch build; offsets are fixed at build."""

    def __init__(self):
        self.regions: list[Region] = []
        self.stages: list[Stage] = []

    def region(self, name, nbytes, peer=True):
        self.regions.append(Region(name, nbytes, peer))

    def stage(self, name, tasks, waits_on=(), all_ranks=False):
        self.stages.append(Stage(name, tasks, tuple(waits_on), all_ranks))

    def _layout(self, peer):
        offs, off = {}, 0
        for r in self.regions:
            if r.peer == peer:
                offs[r.name] = off
                off += (r.nbytes + ALIGN - 1) // ALIGN * ALIGN
        return offs, off

    @property
    def peer_offsets(self):
        """(name -> byte offset in a half, the half's bytes)."""
        return self._layout(True)

    @property
    def scratch_offsets(self):
        return self._layout(False)

    def peer_bytes(self):
        return 2 * self.peer_offsets[1] + guard.FLAG_BYTES

    def flags_offset(self):
        return 2 * self.peer_offsets[1]

    def scratch_bytes(self):
        return max(self.scratch_offsets[1], ALIGN)

    def check(self):
        """Waits only point backwards, and one stage exchanges with every rank."""
        seen = set()
        for s in self.stages:
            for w in s.waits_on:
                assert w in seen or w == s.name, (
                    f"stage {s.name} waits on {w}, which runs later"
                )
            seen.add(s.name)
        assert any(s.all_ranks for s in self.stages), (
            "no stage waits for every rank: two buffer halves would not separate consecutive launches"
        )
        return self

    def describe(self):
        po, ph = self.peer_offsets
        so, sh = self.scratch_offsets
        lines = [
            f"peer half {ph / 1e6:.2f} MB, scratch {sh / 1e6:.2f} MB, {BLOCKS} CTAs"
        ]
        lines += [
            f"  stage {s.name:10s} {s.tasks:5d} tasks  waits {','.join(s.waits_on) or '-'}"
            f"{'  (all ranks)' if s.all_ranks else ''}"
            for s in self.stages
        ]
        return "\n".join(lines)
