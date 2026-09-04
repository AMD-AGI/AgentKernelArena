# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Fill a SIKL solution document from a scored rewrite task workspace.

A rewrite task carries the workload schema's unfilled ``kernel-forges`` solution
slot as ``solution.json``. After Arena has scored the task this module fills the
slot's empty fields from the run and writes the result into the task workspace,
next to ``task_result.yaml``.

It runs in post-processing rather than in the agent launcher because the
launcher returns before ``evaluate_kernel``: at that point the only evidence
available is KernelForge's own port verdict, which is what the loop uses to
decide KEEP/REVERT and explicitly not what the task is scored on. Everything
recorded here comes from Arena's own centralized evaluation.

The filled document is a run artifact, not a publication: syncing it into the
workload-schema bundle is a separate, deliberate step. Writing into another
repository from inside the evaluation loop would put a side effect outside the
run directory, race between the per-GPU workers of a parallel run, and let a
re-run silently overwrite a previously published solution.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any

import yaml

# The workload schema's SupportedLanguages enum has no FlyDSL member, so this
# value does not validate against sikl's Solution model as it stands. It is
# recorded because a FlyDSL port is what these tasks produce; reconciling the
# enum is a schema-side decision.
SOLUTION_LANGUAGE = "flydsl"

SOLUTION_FILENAME = "solution.json"
TASK_RESULT_FILENAME = "task_result.yaml"


def _defines_symbol(source: str, symbol: str) -> bool:
    """Report whether a module defines ``symbol`` at top level.

    Parsing rather than importing: the candidate builds FlyDSL kernels and post
    processing has no business running device code.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol
        for node in tree.body
    )


def _describe(task_result: dict[str, Any], definition: str, case_count: int) -> str:
    """Summarize what the run actually established about this implementation."""
    speedup = task_result.get("speedup_ratio")
    speedup_text = (
        f"{float(speedup):.4f}x mean per-case speedup"
        if isinstance(speedup, (int, float))
        else "no comparable speedup"
    )
    return (
        f"FlyDSL rewrite of {definition}, produced by KernelForge's "
        "forge-rewrite-by-flydsl pipeline and scored by AgentKernelArena. "
        f"Compilation {'passed' if task_result.get('pass_compilation') else 'failed'}, "
        f"correctness {'passed' if task_result.get('pass_correctness') else 'failed'} "
        f"against the workload's fp32 reference over {case_count} cases; "
        f"{speedup_text} against the production implementation under CUDA-graph "
        "replay. The entry point is the shape factory the rewrite protocol "
        "mandates: it takes the case shape and returns the launch callable, so a "
        "consumer that calls the entry point directly with the operator's tensors "
        "needs an adapter."
    )


def fill_solution(workspace: Path, task_source: Path, logger: logging.Logger) -> Path | None:
    """Fill one workspace's solution slot; return the path written, or None."""
    template_path = task_source / SOLUTION_FILENAME
    if not template_path.is_file():
        logger.info("forge_rewrite: %s ships no %s; nothing to fill",
                    task_source.name, SOLUTION_FILENAME)
        return None

    result_path = workspace / TASK_RESULT_FILENAME
    if not result_path.is_file():
        logger.warning("forge_rewrite: %s has no %s; skipping solution backfill",
                       workspace.name, TASK_RESULT_FILENAME)
        return None

    workload = json.loads((workspace / "workload.json").read_text())
    builder_symbol = str(workload["builder_symbol"])
    port_path = workspace / "kernel.py"
    if not port_path.is_file():
        logger.warning("forge_rewrite: %s has no kernel.py; skipping solution backfill",
                       workspace.name)
        return None

    port_source = port_path.read_text()
    if not _defines_symbol(port_source, builder_symbol):
        # The workspace still holds the task's stub, so the run produced no
        # implementation. Publishing it would file a placeholder as a solution.
        logger.info(
            "forge_rewrite: %s left kernel.py without %s (no port); skipping "
            "solution backfill", workspace.name, builder_symbol,
        )
        return None

    with result_path.open() as handle:
        task_result = yaml.safe_load(handle) or {}

    solution = json.loads(template_path.read_text())
    solution["spec"]["language"] = SOLUTION_LANGUAGE
    solution["spec"]["entry_point"] = f"kernel.py::{builder_symbol}"
    solution["sources"] = [{"path": "kernel.py", "content": port_source}]
    solution["description"] = _describe(
        task_result, str(solution["definition"]), len(workload["cases"])
    )

    destination = workspace / SOLUTION_FILENAME
    destination.write_text(json.dumps(solution, indent=2) + "\n")
    logger.info("forge_rewrite: wrote %s (%s)", destination, solution["name"])
    return destination


def backfill_solutions(workspace_paths: list[str], logger: logging.Logger) -> None:
    """Fill the solution slot of every rewrite workspace that produced a port."""
    arena_root = Path(__file__).resolve().parents[2]
    for raw in workspace_paths:
        workspace = Path(raw)
        try:
            result_path = workspace / TASK_RESULT_FILENAME
            if not result_path.is_file():
                continue
            with result_path.open() as handle:
                task_result = yaml.safe_load(handle) or {}
            task_name = str(task_result.get("task_name") or "")
            if not task_name:
                logger.warning("forge_rewrite: %s records no task_name; skipping "
                               "solution backfill", workspace.name)
                continue
            fill_solution(workspace, arena_root / "tasks" / task_name, logger)
        except Exception:
            # Backfill is a reporting step: a failure here must not lose the run.
            logger.error("forge_rewrite: solution backfill failed for %s",
                         workspace, exc_info=True)
