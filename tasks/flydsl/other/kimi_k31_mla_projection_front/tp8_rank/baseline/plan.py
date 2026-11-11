# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/execution.py
# atom/mono/plan/layout.py
# atom/mono/plan/build_key.py
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/common/plan.py). Modified for
# AgentKernelArena: only the execution model, the mailbox layout and the build-key flattening the
# Kimi-K3 launch uses are kept. ``kernel_sources`` (which hashes the imported source files on disk)
# is replaced by a constant tag: every helper of this launch sits in the kernel's own directory,
# which FlyDSL's JIT cache key already covers. The TP shards and msgspec Structs are not kept.
"""Plain-Python planning of the mono kernels: execution model, mailbox layouts and build keys."""

import dataclasses
import enum

# ---------------------------------------------------------------- execution model
BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64


def first_task(bid, base):
    """Return CTA ``bid``'s first task of a stage placed from CTA ``base``."""
    return (bid - base % BLOCKS + BLOCKS) % BLOCKS


# ---------------------------------------------------------------- mailbox layouts
PAIR_BYTES = 8  # a (value, tag) pair


def pair_layout(items, align: int = PAIR_BYTES, start: int = 0) -> dict:
    """Lay out ``items`` (name, pairs) in order: name -> (byte offset, bytes), ``align``-aligned."""
    out, off = {}, start
    for name, pairs in items:
        off = (off + align - 1) // align * align
        out[name] = (off, pairs * PAIR_BYTES)
        off += pairs * PAIR_BYTES
    return out


# ---------------------------------------------------------------- build keys
_SCALARS = (int, float, bool, str, enum.Enum, type(None))
# the constant that stands in for sgl-amd-mono's per-source digest (see the header)
SOURCES = "aka-vendored-52486703"


def kernel_sources(module: str) -> str:
    """The build-key source tag of a kernel module (a constant here; see the header)."""
    return SOURCES


def _key_pairs(prefix: str, items):
    for name, value in items:
        if isinstance(value, dict):
            yield from _key_pairs(f"{prefix}{name}.", value.items())
        elif dataclasses.is_dataclass(value):
            # asdict keeps the field order and turns nested dataclasses into dicts (no computed getattr)
            yield from _key_pairs(f"{prefix}{name}.", dataclasses.asdict(value).items())
        elif isinstance(value, _SCALARS):
            yield f"{prefix}{name}", value
        else:
            raise TypeError(f"{prefix}{name} = {value!r}: a build parameter must be a scalar, or a"
                            " dataclass of them, to reach the JIT cache key")


def key_tuple(key, sources: str) -> tuple:
    """Every build parameter in ``key`` (a dataclass) as a (name, value) pair, then ("sources",
    ``sources``): FlyDSL leaves dataclasses a kernel closes over out of its key."""
    items = list(dataclasses.asdict(key).items())
    return (*_key_pairs("", items), ("sources", sources))
