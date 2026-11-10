# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/execution.py
# atom/mono/plan/layout.py
# atom/mono/plan/arith.py
# atom/mono/plan/shard.py
# atom/mono/plan/build_key.py
"""Plain-Python planning of the mono kernels: execution model, layouts, TP shards and build keys."""

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


# ---------------------------------------------------------------- arithmetic
def sel(pred, a, b):
    """Return a if pred else b, as ``select`` on traced values."""
    if isinstance(pred, bool):
        return a if pred else b
    out = pred.select(a, b)
    traced = [x for x in (a, b) if not isinstance(x, int)]
    return type(traced[0])(out) if traced else out


def imin(a, b):
    return sel(a < b, a, b)


def i32(x):
    """``x``, asserted inside the device's Int32 range when it is a Python int."""
    if isinstance(x, int):
        assert -(2**31) <= x < 2**31, x
    return x


def cdiv(a, b):
    """ceil(a / b) for a >= 0, b > 0."""
    return i32((a + b - 1) // b)


# ---------------------------------------------------------------- TP shards
def padded(n: int, align: int) -> int:
    return cdiv(n, align) * align


# ---------------------------------------------------------------- build keys
_SCALARS = (int, float, bool, str, enum.Enum, type(None))
def _fields(value) -> list:
    return [(f.name, getattr(value, f.name)) for f in dataclasses.fields(value)]


def _key_pairs(prefix: str, items):
    for name, value in items:
        if dataclasses.is_dataclass(value):
            yield from _key_pairs(f"{prefix}{name}.", _fields(value))
        elif isinstance(value, _SCALARS):
            yield f"{prefix}{name}", value
        else:
            raise TypeError(
                f"{prefix}{name} = {value!r}: a build parameter must be a scalar, or a"
                " Struct or dataclass of them, to reach the JIT cache key"
            )


def key_tuple(key, sources: str) -> tuple:
    """Every build parameter in ``key`` as a (name, value) pair, then ("sources", ``sources``).

    ``key`` is a build Struct, or a builder's ``locals()`` before its first
    assignment; a nested Struct's or dataclass's fields are flattened, as FlyDSL
    leaves Structs, dataclasses and dicts a kernel closes over out of its key."""
    items = key.items() if isinstance(key, dict) else _fields(key)
    return (*_key_pairs("", items), ("sources", sources))
