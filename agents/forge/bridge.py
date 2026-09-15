"""KernelForge stdout adapter for the public task command protocol.

Only this provider adapter consumes KERNELFORGE_* variables. Task runners see
their ordinary declared paths and ARENA_EVAL_PHASE, through run_action().
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import statistics
import tempfile
from urllib.parse import quote

from src.task_execution import run_action
from src.task_spec import resolve_task_path
from agents.forge.task_context import TaskContext, bounded_spec
from agents.forge.bundles import copy_workspace, install_candidate


def load_plan(path: Path) -> dict:
    plan = json.loads(Path(path).read_text())
    if plan.get("version") != 1:
        raise ValueError("Unsupported Forge bridge plan")
    return plan


def bound_candidate_root(plan: dict, engine_root: Path) -> Path:
    if plan["workflow"] == "optimize":
        return engine_root
    raw = os.environ.get("KERNELFORGE_REWRITE_CANDIDATE_KERNEL")
    if not raw:
        raise ValueError("Rewrite driver received no current candidate binding")
    # Upstream can carry an absolute master path into a copied lane. Rebind its
    # relative attempt path to this driver's actual workspace, never the master.
    path = Path(raw)
    master = Path(plan["engine_root"])
    relative = path.relative_to(master) if path.is_absolute() else path
    if len(relative.parts) < 3 or relative.parts[0] != ".forge_rewrite":
        raise ValueError("Rewrite candidate is not in an attempt workspace")
    attempt = resolve_task_path(engine_root, Path(*relative.parts[:2]).as_posix(), must_exist=True)
    expected = resolve_task_path(attempt, plan["anchor"])
    if relative.as_posix() != expected.relative_to(engine_root).as_posix():
        raise ValueError("Rewrite candidate binding differs from the declared anchor")
    return attempt


@contextmanager
def evaluation_workspace(context: TaskContext, plan: dict, engine_root: Path, role: str):
    # Every invocation gets a private build tree. Candidate edits cannot change
    # the reference, harness, or a concurrently measured baseline.
    with tempfile.TemporaryDirectory(prefix="evaluate-", dir=Path(plan["template"]).parent) as temporary:
        root = Path(temporary) / "task"
        template = context.baseline_workspace if role == "baseline" else Path(plan["template"])
        copy_workspace(template, root)
        if role == "candidate":
            install_candidate(context.spec, bound_candidate_root(plan, engine_root), root)
        yield root


def execute(plan: dict, engine_root: Path, *, role: str, action: str):
    context = TaskContext.load(plan["context"])
    with evaluation_workspace(context, plan, engine_root, role) as root:
        phases = ["compile"] if action == "compile" else ["compile", action]
        if role == "candidate" and action == "performance":
            phases = ["compile", "correctness", "performance"]
        result = None
        for step in phases:
            executed = run_action(bounded_spec(context.spec, plan["deadline_unix"]), root,
                                  role=role, action=step, phase="candidate_evaluation",
                                  manifest=context.manifest)
            result = executed.result
            if not result.passed:
                raise RuntimeError(f"{role}.{step}: {result.reason}")
        return result


def timings(result) -> dict[str, float]:
    # URL encoding is injective even for spaces, %, and Unicode. Replacing
    # spaces with underscores would collapse distinct manifest case IDs.
    return {quote(row["test_case_id"], safe=""): row["execution_time_ms"] for row in result.cases}


def emit_timings(result) -> None:
    values = timings(result)
    for case, elapsed in values.items():
        print(f"case_ms: {case} {elapsed:.12g}")
    # Upstream accepts mean_ms (deprecated spelling), and KEEP uses the mean of
    # per-case baseline/candidate ratios. Do not falsely label a mean a median.
    print(f"mean_ms: {statistics.fmean(values.values()):.12g}")


def run(plan_path: str | Path, engine_root: str | Path, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--bench-mode", action="store_true")
    modes.add_argument("--ref-bench-mode", action="store_true")
    modes.add_argument("--profile-run", action="store_true")
    modes.add_argument("--compile-only", action="store_true")
    parser.add_argument("--mode", default="full")
    # Upstream smoke/stability stages supply these hints. They must not reduce
    # task-owned samples, cases, or warmups.
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iters", type=int)
    args = parser.parse_args(argv)
    if args.profile_run:
        print('capability: {"profile": "unsupported", "reason": "no public task profiling action"}')
        return 2
    role = "baseline" if args.ref_bench_mode else "candidate"
    action = "performance" if args.bench_mode or args.ref_bench_mode else "compile" if args.compile_only else "correctness"
    try:
        result = execute(load_plan(Path(plan_path)), Path(engine_root).resolve(strict=True), role=role, action=action)
        if action == "performance":
            emit_timings(result)
        else:
            print("allclose: True")
        return 0
    except Exception as exc:
        print("allclose: False")
        print(f"arena_error: {type(exc).__name__}: {exc}")
        return 1


def render_driver(plan_path: Path, arena_root: Path) -> str:
    return f'''#!/usr/bin/env python3
# Generated by the Forge adapter; the task supplies no Forge-specific driver.
# --ref-bench-mode calls the separate baseline; --bench-mode calls the candidate.
import sys
from pathlib import Path
sys.path.insert(0, {str(arena_root)!r})
from agents.forge.bridge import run_managed as run
if __name__ == "__main__":
    raise SystemExit(run({str(plan_path)!r}, Path(__file__).resolve().parent, sys.argv[1:]))
'''


def run_managed(plan_path, engine_root, argv=None):
    # Each individual check needs cleanup, not only the enclosing campaign.
    # An upstream stage timeout can kill its bridge while task compilers/GPU
    # workers have moved into separate process groups.
    from agents.forge.process_tree import managed_children
    with managed_children():
        return run(plan_path, engine_root, argv)
