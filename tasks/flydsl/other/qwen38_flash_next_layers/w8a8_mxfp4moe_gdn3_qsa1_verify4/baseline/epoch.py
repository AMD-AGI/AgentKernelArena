# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernel symbol of a mono kernel build."""


def kernel_signature(**params) -> str:
    """Build parameters as a kernel-name suffix (booleans as 0/1, no hyphens)."""
    parts = [f"{name}{int(value) if isinstance(value, bool) else value}" for name, value in params.items()]
    return "_".join(parts).replace("-", "_")


def kernel_symbol(stem, **params):
    """Kernel symbol that profilers demangle to ``aiter::<stem>_<params>``."""
    name = f"{stem}_{kernel_signature(**params)}"
    return f"_ZN5aiter{len(name)}{name}E"
