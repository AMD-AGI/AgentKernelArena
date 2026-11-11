# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/minimax_m31/sync.py). Modified for
# AgentKernelArena (FlyDSL 0.3.2 runtime, whose bundled LLVM predates the image the kernel was written
# for): the monotonic loads and stores are llvm-dialect atomic loads / stores (FlyDSL 0.3.2 has no
# fx.generic_load / fx.generic_store / fx.AtomicOrdering), the 16-bit store is an i16 atomic store of
# the bf16 bits; ``load128``'s ``llvm.amdgcn.av.load.b128`` becomes the buffer load with the
# agent-scope cache bit (sc1) that the intrinsic lowers to; the ``amdgcn-av:none`` MMRA fences
# become the waits they lower to (``s_waitcnt vmcnt(0)`` before a release) next to a workgroup-scope
# fence that keeps the compiler's order; the LDS atomics take llvm ``atomicrmw`` directly (no
# fx.llvm on 0.3.2). ``put_release_all`` / ``acquire_system`` (unused by the FFN kernel) are dropped.
"""Tagged hand-offs of one launch, its waits bounded by the plugin's guard.

The contract, in the LLVM "AMDGPU Memory Model" (llvm/docs/AMDGPUMemoryModel.md,
AMM) and AMDGPUUsage "Execution Barriers":

- A pair is one naturally aligned 64-bit atomic, (value | tag << 32),
  ``monotonic`` at its space's scope (``agent`` for this GPU's scratch, system
  for the peer heap). An atomic access is never split, so a load that sees a
  pair's tag sees that store's value. A triple pair (``put_triple``, the
  attention seam) is the same atomic with three bf16 in bits 0 .. 47 under a
  16-bit tag that cycles 1 .. 65535 over its parity's launches; the launch that
  ends a cycle clears its receive half (``ffn.clear_cycle``), so a stale pair
  never carries a current tag.
- Data a pair announces is stored *store-available* and loaded *load-visible*
  at agent scope (agent atomics of 16 to 64 bits; 128-bit loads with the
  agent-scope cache bit; ``store_dev`` / ``load_dev``).
- Happens-before from producer to consumer: the producer's wait for its stores
  before the pair (``publish``), the consumer's poll that saw it (``acquire``),
  and within a CTA ``s_barrier`` (``gpu.barrier``). Upstream these are
  ``amdgcn-av:none`` fences, which synchronize without any cache write-back or
  invalidate and lower to waits only; here they are those waits, with a
  workgroup-scope fence (no cache maintenance outside tgsplit mode) so the
  compiler moves no access across them.

The GFX942 workgroup release waits ``lgkmcnt(0)`` only outside tgsplit mode, so a
publishing CTA also has every wave wait for its own stores before the barrier
(``cta_sync(drain=True)``).

A poll waits as every plugin wait does (``sync._spin`` under the launch's
``guard.scope``): past the deadline, or once any rank failed, the launch stops.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from . import buffer_ops as bo
from .ops import CM_DEV, rsrc
from .sync import Space, _spin

POLL_MAX = 12

_SCOPE = {Space.SCRATCH: "agent", Space.PEER: None}
_MONOTONIC = _llvm.AtomicOrdering.monotonic


def _ptr(addr, i, bits=64):
    """The llvm global pointer (address space 1) to element i (of ``bits``) of address ``addr``."""
    at = addr + fx.Int64(i) * (bits // 8)
    return _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(at))).result


def _atomic_store(ptr, value, nbytes, scope):
    """``store atomic monotonic`` at ``scope`` (None: system scope)."""
    kw = {"syncscope": scope} if scope is not None else {}
    _llvm.StoreOp(as_ir_value(value), ptr, alignment=nbytes, ordering=_MONOTONIC, **kw)


def _atomic_load(ty, ptr, nbytes, scope):
    """``load atomic monotonic`` at ``scope`` (None: system scope); the raw result value."""
    kw = {"syncscope": scope} if scope is not None else {}
    return _llvm.LoadOp(ty, ptr, alignment=nbytes, ordering=_MONOTONIC, **kw).result


def store64(addr, i, word, scope):
    _atomic_store(_ptr(addr, i), fx.Int64(word), 8, scope)


def load64(addr, i, scope):
    return fx.Int64(_atomic_load(ir.IntegerType.get_signless(64), _ptr(addr, i), 8, scope))


def store32(addr, i, word, scope):
    _atomic_store(_ptr(addr, i, 32), fx.Int32(word), 4, scope)


def load32(addr, i, scope):
    return fx.Int32(_atomic_load(ir.IntegerType.get_signless(32), _ptr(addr, i, 32), 4, scope))


def store16(addr, i, v, scope):
    """The bf16 ``v`` at element i of ``addr``: a 16-bit atomic store of its bits."""
    bits = _llvm.bitcast(ir.IntegerType.get_signless(16), as_ir_value(v))
    _atomic_store(_ptr(addr, i, 16), bits, 2, scope)


def load128(addr, i, scope):
    """Dwords i .. i + 3 (i a multiple of 4): one load-visible at ``scope`` (agent: the sc1 cache
    bit, as upstream's ``llvm.amdgcn.av.load.b128`` at agent scope lowers)."""
    assert scope == "agent", scope
    return fx.Vector(bo.buffer_load(rsrc(addr), i, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV))


def store_dev(addr, i, words):
    """Dwords i .. of this GPU's scratch := ``words`` (1 or 2, i aligned),
    store-available at agent scope."""
    if const_expr(len(words) == 1):
        store32(addr, i, words[0], "agent")
    else:
        store64(addr + fx.Int64(i) * 4, 0, pack(words[0], words[1]), "agent")


def load_dev(addr, i, n=1):
    """Dwords i .. i + n of this GPU's scratch (n in {1, 2, 4}, i aligned),
    load-visible at agent scope."""
    if const_expr(n == 4):
        return load128(addr, i, "agent")
    if const_expr(n == 2):
        w = load64(addr + fx.Int64(i) * 4, 0, "agent")
        return fx.Vector.from_elements([fx.Int32(w), fx.Int32(w >> 32)], fx.Int32)
    return load32(addr, i, "agent")


def _fence(ordering, scope):
    _llvm.FenceOp(ordering, syncscope=scope)


def sync_release(scope):
    """Upstream's ``amdgcn-av:none`` release fence: the wait for this thread's stores (agent scope;
    nothing at wavefront scope), and a fence that keeps the compiler's order."""
    if const_expr(scope == "wavefront"):
        _fence(_llvm.AtomicOrdering.release, "wavefront")
    else:
        rocdl.s_waitcnt(vmcnt=0)
        _fence(_llvm.AtomicOrdering.release, "workgroup")


def sync_acquire(scope):
    """Upstream's ``amdgcn-av:none`` acquire fence: no cache invalidate, the compiler's order kept."""
    _fence(_llvm.AtomicOrdering.acquire, "wavefront" if scope == "wavefront" else "workgroup")


def lds_atomic(op, ptr, value, scope):
    """Monotonic ``atomicrmw`` ``op`` of an Int32 on the LDS word at ``ptr`` (fx.llvm.atomic_* of
    FlyDSL >= 0.3.4); the word's previous value."""
    return fx.Int32(_llvm.atomicrmw(op, ptr.llvm_ptr, as_ir_value(fx.Int32(value)), _MONOTONIC,
                                    syncscope=scope, alignment=4))


def lds_atomic_max(ptr, value, scope="workgroup"):
    return lds_atomic(_llvm.AtomicBinOp.max, ptr, value, scope)


def lds_atomic_or(ptr, value, scope="workgroup"):
    return lds_atomic(_llvm.AtomicBinOp._or, ptr, value, scope)


def lds_load_monotonic(ptr, scope):
    """``load atomic monotonic`` of the Int32 LDS word at ``ptr``."""
    return fx.Int32(_llvm.LoadOp(ir.IntegerType.get_signless(32), ptr.llvm_ptr, alignment=4,
                                 ordering=_MONOTONIC, syncscope=scope).result)


@flyc.jit
def wave_sync(count, target):
    """Every lane of the wave: what each lane did before (its acquires)
    happens before what any lane does after. Each lane releases, adds one to
    the wave's LDS ``count``, and acquires once it reads ``target``; the read
    value comes after every lane's add in the counter's order (a release
    sequence), so it synchronizes with every lane's release."""
    sync_release("wavefront")
    lds_atomic(_llvm.AtomicBinOp.add, count, fx.Int32(1), "wavefront")
    n = lds_load_monotonic(count, "wavefront")
    while n < target:
        n = lds_load_monotonic(count, "wavefront")
    sync_acquire("wavefront")


def cta_sync(drain=False):
    """``gpu.barrier``: workgroup release fence, ``s_barrier``, acquire fence
    (HIP's ``__syncthreads``). ``drain``: every wave's stores complete first."""
    if const_expr(drain):
        rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()


def pack(word, tag):
    return (fx.Int64(tag) << 32) | fx.Int64(fx.Uint32(fx.Int32(word)))


class Mailbox:
    """Tagged pairs of one launch (``tag``), every address an ``Addr``."""

    def __init__(self, tag, tag16=None):
        self.tag = tag
        # a triple pair's 16-bit tag (``put_triple``), if the launch has one
        self.tag16 = tag16

    def put(self, a, i, v):
        """Pair i of ``a`` := (v, tag), monotonic: a value no other data hangs on."""
        bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
        store64(a.value, i, pack(bits, self.tag), _SCOPE[a.space])

    def put_words(self, a, i, words):
        for k in range_constexpr(len(words)):
            store64(a.value, i + k, pack(words[k], self.tag), _SCOPE[a.space])

    def publish(self, a, i, v):
        """Pair i of ``a`` := (v, tag) after a synchronizing release: what
        happened before it (the CTA's stores past a ``cta_sync(drain=True)``,
        pairs it acquired) happens before an acquire of the pair."""
        sync_release(_SCOPE[a.space])
        self.put(a, i, v)

    def acquire(self, space):
        """After a poll of published pairs, before what they announce."""
        sync_acquire(_SCOPE[space])

    def put_triple(self, a, i, bits48):
        """Pair i of ``a`` := (48 value bits | tag16 << 48), monotonic."""
        word = (bits48 & ((1 << 48) - 1)) | (fx.Int64(self.tag16) << 48)
        store64(a.value, i, word, _SCOPE[a.space])

    def poll(self, specs, batch=POLL_MAX, triples=False):
        """Pairs ``specs`` = [(Addr, pair index, npairs in {1, 2})], all of one
        space, re-loaded together until every tag is this launch's. Returns the
        value words; ``triples``: of triple pairs (``put_triple``), both
        words."""
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], batch, triples) + self.poll(
                specs[batch:], batch, triples
            )
        spaces = {a.space for a, _, _ in specs}
        assert len(spaces) == 1, f"one poll batch spans {spaces}"
        scope = _SCOPE[spaces.pop()]
        tag = self.tag16 if const_expr(triples) else self.tag

        def high_tag(w):
            return ((w >> 16) & 0xFFFF) if const_expr(triples) else w

        def load_all():
            words = []
            for a, i, n in specs:
                for q in range(n):
                    w = load64(a.value, fx.Int32(i) + q, scope)
                    words += [fx.Int32(w), fx.Int32(w >> 32)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = high_tag(v[1]) != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (high_tag(v[e]) != tag)
            return bad

        v = _spin(load_all, pending)
        outs, e = [], 0
        for _, _, n in specs:
            if const_expr(triples):
                outs.append([(v[e + 2 * q], v[e + 2 * q + 1]) for q in range(n)])
            else:
                outs.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs
