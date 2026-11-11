# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/execution.py
# atom/mono/plan/build_key.py
"""Plain-Python planning of the mono kernels: execution model and build keys.

AKA: only the execution model and ``key_tuple`` are kept (the layouts, arithmetic and TP-shard helpers are
not used by this launch); ``kernel_sources`` (a digest of the package's source files) is dropped, as
FlyDSL hashes the helpers of a kernel's own directory and this package is flat.
"""

import dataclasses
import enum

# ---------------------------------------------------------------- execution model
BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64


# ---------------------------------------------------------------- build keys
_SCALARS = (int, float, bool, str, enum.Enum, type(None))


def _fields(value) -> list:
    # asdict keeps the field order and turns nested dataclasses into dicts (no computed getattr)
    return list(dataclasses.asdict(value).items())


def _key_pairs(prefix: str, items):
    for name, value in items:
        if isinstance(value, dict):
            yield from _key_pairs(f"{prefix}{name}.", value.items())
        elif dataclasses.is_dataclass(value):
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
