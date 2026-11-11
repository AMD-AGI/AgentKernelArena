# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for AgentKernelArena: only ``spin_until`` of sgl-amd-mono 52486703 sgl_amd_mono/common/epoch.py,
# the part the GLM-5.3 kernel imports (the epoch_begin / epoch_end helpers and kernel_symbol are unused).
"""The launch epoch's wait helper."""
from .sync import _spin


def spin_until(load, pending):
    """Poll ``load()``, sleeping between, until ``pending`` clears; return the value."""
    return _spin(load, pending)
