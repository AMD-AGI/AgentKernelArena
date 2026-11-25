# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Server-container start-up hook of the AgentX workload capture (on ``PYTHONPATH`` only when enabled).

Installs the shape hooks named by ``AKA_SHAPE_HOOKS`` (see ``../shape_hooks.py``) in every Python process
of the server, scheduler subprocesses included, then runs the ``sitecustomize`` this one shadows.
"""
import importlib.machinery
import importlib.util
import os
import sys


def _install() -> None:
    if not os.environ.get("AKA_SHAPE_HOOKS", "").strip() or "aka_shape_hooks" in sys.modules:
        return
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shape_hooks.py")
    try:
        spec = importlib.util.spec_from_file_location("aka_shape_hooks", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["aka_shape_hooks"] = module
        spec.loader.exec_module(module)
        module.install()
    except Exception as exc:  # never keep the server from starting
        print(f"[shape_hooks] not installed: {exc!r}", file=sys.stderr, flush=True)


def _chain() -> None:
    """Run the next ``sitecustomize`` on ``sys.path`` (e.g. the distribution's), which this module shadows."""
    here = os.path.dirname(os.path.abspath(__file__))
    rest = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is None or spec.loader is None or os.path.abspath(spec.origin or "") == os.path.abspath(__file__):
        return
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"[shape_hooks] {spec.origin}: {exc!r}", file=sys.stderr, flush=True)


_install()
_chain()
