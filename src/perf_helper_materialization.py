# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Materialize the canonical serving helpers into schema-v2 serving workspaces.

The end-to-end serving tasks under ``tasks/e2e/`` keep a small
``scripts/evaluate.py`` that imports ``_aka_serving``. A copied run workspace
receives the canonical client, measurement and bundle modules beside it, so the
task never imports AgentKernelArena's ``src`` package at runtime. Serving recipe
tasks instead commit the exact bootstrap stub below; host preparation supplies
their immutable client before any action runs.

Schema-v3 kernel tasks need nothing from this module: their evaluator
(``_aka_eval/``, including the canonical timing helper) is written by
``src.task_contract.render_workspace``.
"""

from __future__ import annotations

import logging
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SERVING_RUNNER = Path("scripts/evaluate.py")
SERVING_RECIPE_RUNNER_STUB = '''"""The Docker runner prepares this recipe before any task action executes."""
raise RuntimeError(
    "This serving recipe has no prepared runtime. Select it with the normal Docker "
    "runner and declare its immutable binding in serving_runtime.bindings."
)
'''

# Materialized name -> canonical source, relative to the repository root.
SERVING_HELPERS = {
    "_aka_serving.py": "src/tools/perf/serving.py",
    "_aka_measurement.py": "src/measurement.py",
    "_aka_serving_bundle.py": "src/tools/perf/serving_bundle.py",
}
# Multi-scenario clients (``_aka_serving_backends``) also need these.
SERVING_BACKEND_HELPERS = {
    "_aka_serving_backends.py": "src/tools/perf/serving_backends.py",
    "_aka_serving_scenarios.py": "src/tools/perf/serving_scenarios.py",
    "_aka_serving_request_plan.py": "src/tools/perf/serving_request_plan.py",
}
GENERATED_HELPER_NAMES = frozenset(SERVING_HELPERS) | frozenset(SERVING_BACKEND_HELPERS)


def _materialize_file(path: Path, content: str, materialized: list[Path]) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    materialized.append(path)


def materialize_perf_helpers_in_workspace(
    workspace: Path,
    logger: logging.Logger | None = None,
    root: Path = ROOT,
) -> list[Path]:
    """Copy the canonical serving helpers beside an ``_aka_serving`` runner.

    The operation is idempotent and returns the files it wrote. A workspace
    whose ``scripts/evaluate.py`` does not import ``_aka_serving`` (a serving
    recipe stub, or any other task) is left unchanged.
    """

    log = logger or logging.getLogger(__name__)
    workspace = Path(workspace)
    materialized: list[Path] = []

    runner = workspace / SERVING_RUNNER
    text = runner.read_text() if runner.is_file() else ""
    if "_aka_serving" in text:
        helpers = dict(SERVING_HELPERS)
        if "_aka_serving_backends" in text:
            helpers.update(SERVING_BACKEND_HELPERS)
        for name, source in helpers.items():
            _materialize_file(runner.parent / name, (root / source).read_text(), materialized)

    if materialized:
        log.info(
            "Materialized canonical serving helper(s) in workspace: %s",
            [str(path.relative_to(workspace)) for path in materialized],
        )
    return materialized
