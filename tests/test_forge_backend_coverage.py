"""Every backend a task declares must be one KernelForge actually serves.

The companion to `test_cheatsheet_language_coverage.py`, guarding the same class
of failure one layer down. There, an unregistered `repository_language` made the
agent never start and the run report a speedup of exactly 1.0. Here, a backend
KernelForge does not register does not stop anything: upstream maps the unknown
name onto flydsl and says nothing, so the run starts, finishes, and reports a
plausible number reached under the wrong expertise prompt. Nothing in the logs
connects the two, which is worse than the crash.

`_resolve_kernel_backend` refuses such a name at launch. These tests make sure
the task tree stays inside what it will accept, so a task added with a backend
nobody serves is a red test rather than a wasted GPU-day.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import yaml

from agents.forge.launch_agent import (
    _DELIBERATE_BACKEND_ALIASES,
    _infer_backend,
    _resolve_kernel_backend,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The union of what the two KernelForge layouts register. Only a test fixture:
# the launcher itself reads the registry from the installed package, precisely so
# this list cannot become the thing that decides what runs. Kept here so the
# suite is meaningful in a checkout with no KernelForge installed.
KNOWN_BACKENDS = {
    "aiter",
    "ck",
    "flydsl",
    "fusion",
    "gluon",
    "hip",
    "hipblaslt",
    "intellikit",  # pre-merge standalone only
    "triton",
}


def _declared_backends() -> dict[str, list[str]]:
    """Map each backend the task tree infers to the tasks that infer it."""
    backends: dict[str, list[str]] = {}
    for path in (PROJECT_ROOT / "tasks").rglob("config.yaml"):
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(cfg, dict):
            continue
        try:
            backend = _infer_backend(cfg)
        except ValueError:
            continue  # a task that cannot resolve one is the other test's problem
        backends.setdefault(backend, []).append(path.parent.name)
    return backends


def test_every_task_resolves_to_a_backend_forge_will_accept():
    declared = _declared_backends()
    assert declared, "no task configs found; the walk is broken, not the tree"

    unserved: dict[str, list[str]] = {}
    for backend, tasks in declared.items():
        if backend in KNOWN_BACKENDS or backend in _DELIBERATE_BACKEND_ALIASES:
            continue
        unserved[backend] = sorted(tasks)[:5]

    assert not unserved, (
        "these tasks declare a backend KernelForge does not serve and no "
        f"deliberate alias covers: {unserved}. Register it upstream, or add an "
        "entry to _DELIBERATE_BACKEND_ALIASES with the evidence for the "
        "substitution."
    )


@pytest.mark.parametrize("declared", sorted(_DELIBERATE_BACKEND_ALIASES))
def test_each_alias_points_at_a_backend_that_exists(declared):
    """An alias onto another unserved name would just relocate the problem."""
    assert _DELIBERATE_BACKEND_ALIASES[declared] in KNOWN_BACKENDS


@pytest.mark.parametrize("declared", sorted(_DELIBERATE_BACKEND_ALIASES))
def test_each_alias_is_still_needed(declared):
    """Delete the entry once upstream registers the backend for real.

    An alias left in place after upstream catches up would keep sending the
    substitute forever, silently, which is the behaviour being removed.
    """
    assert declared not in KNOWN_BACKENDS, (
        f"{declared} is served now; drop it from _DELIBERATE_BACKEND_ALIASES so "
        "the real backend is used"
    )


def test_the_tilelang_alias_is_what_the_launcher_actually_sends(monkeypatch):
    """Ties the tree-level guard to the value that reaches the CLI."""
    # By module object, not by dotted string: agents.forge re-exports the
    # launch_agent *function*, so the string form resolves to that, not here.
    monkeypatch.setattr(
        sys.modules["agents.forge.launch_agent"],
        "_installed_kernel_backends",
        lambda: KNOWN_BACKENDS,
    )
    assert (
        _resolve_kernel_backend("tilelang-fellow", logging.getLogger(__name__))
        == "flydsl"
    )
