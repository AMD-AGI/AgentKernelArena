# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/execution.py
# atom/mono/plan/build_key.py
# Vendored from sgl-project/sgl-amd-mono at 52486703fbf27e77f05b573e4d931d4eec87118d
# (sgl_amd_mono/common/plan.py). AKA: the execution-model constants and ``key_tuple`` over a
# dataclass build key; ``kernel_sources`` (which hashes the imported source files on disk) is
# replaced by the kernel module's fixed digest, and the msgspec shard and layout helpers are
# left out.
"""Plain-Python planning of the mono kernels: execution model and build keys."""

import dataclasses
import enum

# ---------------------------------------------------------------- execution model
BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64

_SCALARS = (int, float, bool, str, enum.Enum, type(None))


def _key_pairs(prefix, value):
    for field in dataclasses.fields(value):
        item = getattr(value, field.name)
        if dataclasses.is_dataclass(item):
            yield from _key_pairs(f"{prefix}{field.name}.", item)
        elif isinstance(item, _SCALARS):
            yield f"{prefix}{field.name}", item
        else:
            raise TypeError(f"{prefix}{field.name} = {item!r}: a build parameter must be a scalar")


def key_tuple(key, sources: str) -> tuple:
    """Every build parameter in the dataclass ``key`` as a (name, value) pair, then
    ("sources", ``sources``): the kernel closes over it, so it reaches FlyDSL's JIT key."""
    return (*_key_pairs("", key), ("sources", sources))
