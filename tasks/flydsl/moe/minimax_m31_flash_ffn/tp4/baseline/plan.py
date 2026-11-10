# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/execution.py
# atom/mono/plan/layout.py
# atom/mono/plan/build_key.py
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/common/plan.py). Modified for
# AgentKernelArena: reduced to what the MiniMax-M3.1 FFN kernel uses (the execution model,
# ``pair_layout`` and ``key_tuple``); no msgspec, and no source digest (``kernel_sources`` read the
# installed package's files; FlyDSL hashes each jit and kernel function's own source itself).
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


def _key_pairs(prefix: str, items):
    for name, value in items:
        if dataclasses.is_dataclass(value):
            yield from _key_pairs(f"{prefix}{name}.", [(f.name, getattr(value, f.name))
                                                       for f in dataclasses.fields(value)])
        elif isinstance(value, _SCALARS):
            yield f"{prefix}{name}", value
        else:
            raise TypeError(
                f"{prefix}{name} = {value!r}: a build parameter must be a scalar, or a"
                " dataclass of them, to reach the JIT cache key"
            )


def key_tuple(key) -> tuple:
    """Every build parameter of the build dataclass ``key`` as a (name, value) pair; a nested
    dataclass's fields are flattened, as FlyDSL leaves dataclasses and dicts a kernel closes over out
    of its key."""
    return tuple(_key_pairs("", [(f.name, getattr(key, f.name)) for f in dataclasses.fields(key)]))
