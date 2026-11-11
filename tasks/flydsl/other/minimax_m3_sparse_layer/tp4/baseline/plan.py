# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/execution.py
# atom/mono/plan/build_key.py
"""Plain-Python planning of the mono kernels: the execution model and build keys.

Vendored from sgl-project/sgl-amd-mono at 52486703 (sgl_amd_mono/common/plan.py), reduced to what the
MiniMax-M3 kernel uses: the execution model and ``key_tuple`` (without the source digest, which read the
installed package's files: FlyDSL hashes each jit and kernel function's own source itself). ``kernel_symbol``
comes from sgl_amd_mono/common/epoch.py and its ``kernel_signature`` from ROCm/aiter
aiter/ops/flydsl/kernels/kernels_common.py at 1662d806fac304fc11dab6dc1125acce012172f1 (Apache-2.0,
FlyDSL Project Contributors).
"""

import dataclasses
import enum

# ---------------------------------------------------------------- execution model
BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64


def first_task(bid, base):
    """Return CTA ``bid``'s first task of a stage placed from CTA ``base``."""
    return (bid - base % BLOCKS + BLOCKS) % BLOCKS


# ---------------------------------------------------------------- build keys
_SCALARS = (int, float, bool, str, enum.Enum, type(None))


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
            raise TypeError(
                f"{prefix}{name} = {value!r}: a build parameter must be a scalar, or a"
                " dataclass of them, to reach the JIT cache key"
            )


def key_tuple(key: dict) -> tuple:
    """Every build parameter in ``key`` as a (name, value) pair; a nested dataclass's fields are flattened,
    as FlyDSL leaves dataclasses and dicts a kernel closes over out of its key."""
    return tuple(_key_pairs("", key.items()))


def kernel_signature(**params: object) -> str:
    """Render build parameters into a kernel-name suffix (booleans as 0 / 1, no hyphens)."""
    parts = [f"{name}{int(value) if isinstance(value, bool) else value}" for name, value in params.items()]
    return "_".join(parts).replace("-", "_")


def kernel_symbol(stem, **params):
    """Kernel symbol that profilers demangle to ``aiter::<stem>_<params>``."""
    name = f"{stem}_{kernel_signature(**params)}"
    return f"_ZN5aiter{len(name)}{name}E"
