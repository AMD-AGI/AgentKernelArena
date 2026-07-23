# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Generic KernelForge driver that reuses an AgentKernelArena task's own
correctness and performance measurement.

Instead of re-implementing (or hardcoding) precision/perf parsing per kernel,
this adapter calls Arena's OWN evaluation functions and translates their results
into the KernelForge driver contract consumed by
``kernel_agents.mcp_server.tools.{test,bench}``:

  * correctness mode (default / --mode smoke|stability|determinism):
        src.evaluator.evaluate_compilation(...)   # rebuild the edited source
        src.evaluator.evaluate_correctness(...) -> "allclose: True/False"
  * bench mode (--bench-mode):
        src.evaluator.evaluate_compilation(...)   # rebuild the edited source
        src.performance.measure_performance(...) -> "mean_ms: <mean-of-cases>"
        (arithmetic mean across test cases, matching Arena's own evaluator
        aggregation — deliberately a MEAN, not a median, so the label is honest)

Both modes first re-run the task's ``compile_command`` (when it has one) so the
edited kernel's build artifacts are regenerated before it is validated or timed.
Forge edits source in place between iterations; for tasks whose compile step is
separate from correctness/bench (e.g. a HIP runner that builds an executable in
``--compile`` and only executes it in ``--correctness`` / ``--benchmark``),
skipping the rebuild would validate and benchmark the STALE pre-edit binary.
JIT-only tasks (no ``compile_command``) skip this step harmlessly.

Reusing Arena's functions means the adapter is task-type aware and not tied to a
single report filename: Arena already searches multiple candidate report files
(``build/performance_report.json``, ``perf/benchmark_results.json``, ...) and
falls back to stdout parsing, per task type. So this one driver works for any
task Arena itself can score — no per-kernel driver and no hardcoded filename.

This module is a LIBRARY: it takes the task paths (workspace / task config /
Arena repo root) as explicit arguments — NO environment variables. The Arena
forge launcher generates a tiny ``forge_driver.py`` shim with those paths baked
in that calls :func:`run`.
"""

import argparse
import statistics
import sys
from pathlib import Path

import yaml


def _load_task_config(task_config: str) -> dict:
    with open(task_config, "r") as f:
        return yaml.safe_load(f) or {}


def _rebuild_edited_source(workspace: str, task_config_data: dict) -> tuple[bool, str | None]:
    """Regenerate build artifacts from the CURRENT (agent-edited) source.

    Returns ``(ok, error)``. Tasks with no ``compile_command`` are JIT-only, so
    there is nothing to prebuild — that is a benign no-op returning ``(True, None)``.
    Only a real ``compile_command`` that FAILS yields ``(False, error)``; callers
    must then skip correctness/bench (they would otherwise run a stale binary).
    """
    if not task_config_data.get("compile_command"):
        return True, None
    from src.evaluator import evaluate_compilation

    return evaluate_compilation(Path(workspace), task_config_data)


def do_correctness(workspace: str, task_config: str, arena_root: str = "") -> int:
    """Rebuild the edited source, then reuse Arena's correctness evaluation."""
    try:
        from src.evaluator import evaluate_correctness
    except Exception as e:  # noqa: BLE001
        print("allclose: False")
        print(f"error: cannot import Arena evaluator (arena_root={arena_root!r}): {e}")
        return 0

    task_config_data = _load_task_config(task_config)

    # Recompile first so correctness validates the EDITED kernel, not a stale
    # pre-edit binary (tasks whose --compile step is separate only execute the
    # existing artifact in --correctness).
    compiled, comp_err = _rebuild_edited_source(workspace, task_config_data)
    if not compiled:
        print("allclose: False")
        if comp_err:
            print(comp_err[-1500:])
        return 0

    passed, err = evaluate_correctness(Path(workspace), task_config_data)
    if passed:
        print("allclose: True")
    else:
        print("allclose: False")
        if err:
            print(err[-1500:])
    return 0


def do_bench(workspace: str, task_config: str, arena_root: str = "") -> int:
    """Rebuild the edited source, then reuse Arena's performance measurement."""
    try:
        from src.performance import measure_performance
    except Exception as e:  # noqa: BLE001
        print(f"error: cannot import Arena performance module (arena_root={arena_root!r}): {e}")
        return 1

    task_config_data = _load_task_config(task_config)

    # Recompile first so we time the EDITED kernel, not a stale pre-edit binary
    # (tasks whose --compile step is separate only execute the existing artifact
    # in --benchmark). A failed rebuild must not silently benchmark old artifacts.
    compiled, comp_err = _rebuild_edited_source(workspace, task_config_data)
    if not compiled:
        print("error: recompilation of edited source failed before benchmark")
        if comp_err:
            print(comp_err[-1500:])
        return 1

    # is_baseline=False: bench whatever kernel is currently in the tree (forge
    # benches the pristine kernel and each edited version the same way).
    cases = measure_performance(Path(workspace), task_config_data, is_baseline=False)
    times = [c.execution_time_ms for c in cases
             if getattr(c, "execution_time_ms", None) and c.execution_time_ms > 0]
    if not times:
        print("error: Arena measure_performance returned no usable timing")
        return 1

    # Aggregate the per-case times into the single wall time the forge-loop uses
    # for its keep/revert decision. This is the ARITHMETIC MEAN across cases,
    # matching Arena's own evaluator (sum/len over cases), so forge optimizes the
    # same quantity Arena scores. Reported as ``mean_ms:`` (not ``median_ms:``)
    # so the label reflects the statistic actually computed.
    agg = statistics.mean(times)
    print(f"mean_ms: {agg:.6f}")
    return 0


def run(workspace: str, task_config: str, arena_root: str = "", argv: list[str] | None = None) -> int:
    """Driver entry point. All task paths are passed explicitly (no env).

    Args:
        workspace: dir where the task commands run (the kernel lives here).
        task_config: path to the task's config.yaml.
        arena_root: AgentKernelArena repo root, prepended to sys.path so Arena's
            ``src`` package (evaluator / performance) is importable.
        argv: driver args from KernelForge (``--shape`` / ``--mode`` /
            ``--bench-mode`` / ...); defaults to ``sys.argv[1:]``.
    """
    if arena_root and arena_root not in sys.path:
        sys.path.insert(0, arena_root)

    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="default")   # task harness owns its shapes
    ap.add_argument("--mode", default="full")        # all modes -> task correctness
    ap.add_argument("--bench-mode", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    args, _unknown = ap.parse_known_args(argv)

    if args.bench_mode:
        return do_bench(workspace, task_config, arena_root)
    return do_correctness(workspace, task_config, arena_root)
