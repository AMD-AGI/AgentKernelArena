# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL port of the a16w16 GEMM operator -- starter stub.

``forge-rewrite-by-flydsl`` replaces this file with a FlyDSL implementation.

The module must expose one factory, named by ``builder_symbol`` in the task's
workload.json (the same value KernelForge passes to the driver as
``KERNELFORGE_REWRITE_BUILDER_SYMBOL``, and the only name the harness looks up)::

    build_<operator>_module(m, n, k) -> launch

    launch(a, b) -> out              # bf16 [m, n], computed as a @ b.T

where ``a`` is [m, k] bf16, ``b`` is [n, k] bf16, and the builder is called once
per scored case so all shape-dependent work stays outside the timed region.

The name is per-operator, so this stub deliberately defines no factory rather
than hardcoding one: ``scripts/forge_driver.py`` owns the operator definition,
the tensor layouts and where the baseline implementation lives, and the factory
is written here by the port session.

While the factory is absent the harness scores the operator's own baseline
(``aiter.tuned_gemm.gemm_a16w16``), which is what Arena measures before the
agent runs.
"""

from __future__ import annotations
