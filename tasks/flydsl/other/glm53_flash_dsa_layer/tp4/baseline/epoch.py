# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from sgl-project/sgl-amd-mono at 52486703fbf27e77f05b573e4d931d4eec87118d
# (sgl_amd_mono/common/epoch.py). AKA: only ``kernel_symbol`` and ``spin_until``; the epoch
# marks (device-advanced epochs) are not used by this kernel; ``kernel_signature`` (AITER's
# aiter/ops/flydsl/kernels/kernels_common.py) is a local stand-in with the same role.
"""The launch's kernel symbol and its spin wait."""

from .sync import _spin


def kernel_signature(**params):
    """The parameter part of a kernel symbol: ``name value`` pairs joined by underscores."""
    return "_".join(f"{k}{int(v) if isinstance(v, bool) else v}" for k, v in params.items())


def kernel_symbol(stem, **params):
    """Kernel symbol that profilers demangle to ``aiter::<stem>_<params>``."""
    name = f"{stem}_{kernel_signature(**params)}"
    return f"_ZN5aiter{len(name)}{name}E"


def spin_until(load, pending):
    """Poll ``load()``, sleeping between, until ``pending`` clears; return the value."""
    return _spin(load, pending)
