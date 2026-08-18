#!/usr/bin/env python3
"""forge-loop measurement driver (generic, delegates to the task's task_runner).

Why this file exists
--------------------
The Arena forge launcher (``agents/forge/launch_agent.py``) prefers a
task-shipped ``scripts/forge_driver.py`` and copies it VERBATIM to the run
workspace root as ``forge_driver.py``. If it is absent, the launcher generates a
generic shim that delegates to ``arena_task_adapter`` — which does NOT implement
``--profile-run`` — so forge-loop's pre-loop "task preparation" then spends an
LLM agent (minutes) authoring/repairing a profiling-capable driver every run.

Shipping this file makes forge-loop's driver preflight pass on the first check
(correctness + graph-timed bench + profiling), so task preparation is skipped
entirely and no per-run driver authoring can fail.

How it satisfies the contract without per-kernel reimplementation
-----------------------------------------------------------------
Every image_kernel ``scripts/task_runner.py`` in this suite exposes the same
canonical entry points: ``_configure`` / ``_torch`` / ``_make(case)``
/ ``_run(inputs)`` / ``run_correctness()`` / ``run_performance()`` (the latter
writes ``build/performance_report.json`` and times under a CUDA/HIP graph via
``_benchmark_cuda_graph_or_events``). This driver REUSES those, so it measures
exactly the same op — and per-operator correctness reference — that Arena scores.
No kernel-specific math is duplicated here, which is why the file is identical
across the tasks that share this task_runner shape.

Contract implemented (forge-loop runs ``python forge_driver.py <args>`` and reads
only stdout — see kernel_agents.mcp_server.tools.{test,bench} and
kernel_agents.loop.task_preparer.DRIVER_CONTRACT_SPEC):

  * Correctness  ``--mode <smoke|stability|determinism|full>``
        runs the task's own ``run_correctness()`` (per-operator reference +
        tolerance) and prints ``allclose: True/False``. We deliberately do NOT
        print an ``SNR: <db> dB`` line: these quantized / attention ops sit below
        forge's default 30 dB SNR gate, so emitting SNR would fail even the
        PRISTINE kernel. ``allclose`` is a valid contract fallback (test tool:
        SNR preferred, allclose otherwise).

  * Benchmark    ``--warmup <n> --iters <n> --bench-mode``
        runs the task's own ``run_performance()`` (graph-timed) and then, from
        the ``build/performance_report.json`` it wrote, prints
        ``case_ms: <case_id> <ms>`` for every case plus a single ``mean_ms: <ms>``
        aggregate (arithmetic mean across cases, matching Arena's evaluator). The
        real CUDA/HIP-graph replays satisfy forge-loop's graph probe.

  * Profiling    ``--profile-run``
        builds ONE case's inputs (``_make(case)``) and launches
        ONLY the target region (``_run``): a few warmups to settle Triton JIT, a
        couple of profiled launches, one synchronize, exit 0. No timing printed.

This file is the correctness ORACLE and perf MEASURER; forge-loop never edits it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path


def _import_task_runner():
    """Import the task's own harness (scripts/task_runner.py), robustly.

    In a real run the launcher copies this file to ``<workspace>/forge_driver.py``
    while task_runner stays at ``<workspace>/scripts/task_runner.py``; when run
    in place from the task dir both files sit in ``scripts/``. Search both.
    """
    here = Path(__file__).resolve().parent
    for cand in (here / "scripts", here, here.parent / "scripts"):
        tr = cand / "task_runner.py"
        if tr.is_file():
            spec = importlib.util.spec_from_file_location("_forge_task_runner", tr)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_forge_task_runner"] = mod
            spec.loader.exec_module(mod)
            return mod, cand
    raise RuntimeError(f"task_runner.py not found near {here}")


def _report_path(tr) -> Path:
    return Path(tr.WORKSPACE) / "build" / "performance_report.json"


def _run_correctness(tr) -> int:
    """Delegate to the task's own per-operator correctness; map to allclose."""
    ok = True
    try:
        tr.run_correctness()  # asserts / raises on any failing case
    except Exception as exc:  # noqa: BLE001 - any failure is a correctness fail
        ok = False
        print(f"# correctness failed: {type(exc).__name__}: {str(exc)[:300]}")
    print(f"allclose: {ok}")
    return 0


def _run_bench(tr) -> int:
    """Delegate to the task's graph-timed run_performance(); expose per-case ms."""
    tr.run_performance()  # writes build/performance_report.json, graph-timed
    rows = json.loads(_report_path(tr).read_text())
    expected_ids = [str(case.get("id") or "") for case in tr.CASES]
    if not isinstance(rows, list):
        print("error: performance report must be a list", file=sys.stderr)
        return 1
    timings = []
    seen_ids = set()
    try:
        for row in rows:
            cid = str(row.get("test_case_id") or "")
            ms = float(row.get("execution_time_ms"))
            if not cid or cid in seen_ids or not math.isfinite(ms) or ms <= 0:
                raise ValueError(f"invalid or duplicate performance case {cid!r}")
            seen_ids.add(cid)
            timings.append((cid, ms))
    except (AttributeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    missing = [case_id for case_id in expected_ids if case_id not in seen_ids]
    unexpected = [case_id for case_id, _ in timings if case_id not in expected_ids]
    if missing or unexpected:
        print(
            f"error: incomplete performance suite: missing={missing}, unexpected={unexpected}",
            file=sys.stderr,
        )
        return 1
    for cid, ms in timings:
        print(f"case_ms: {cid.replace(' ', '_')} {ms:.6f}")
    times = [ms for _, ms in timings]
    print(f"mean_ms: {sum(times) / len(times):.6f}")
    return 0


def _pick_profile_case(tr) -> dict:
    cases = {c["id"]: c for c in tr.CASES}
    # Prefer the slowest case from a prior complete performance run.
    rp = _report_path(tr)
    if rp.is_file():
        try:
            rows = json.loads(rp.read_text())
            best = max(rows, key=lambda r: float(r.get("execution_time_ms") or 0))
            if best.get("test_case_id") in cases:
                return cases[best["test_case_id"]]
        except Exception:  # noqa: BLE001 - best-effort fallback
            pass
    return tr.CASES[-1]


def _run_profile(tr) -> int:
    torch = tr._torch()
    case = _pick_profile_case(tr)
    inputs = tr._make(case)
    for _ in range(5):          # settle Triton JIT / autotune selection
        tr._run(inputs)
    torch.cuda.synchronize()
    for _ in range(3):          # profiled launches (profiler replays per group)
        tr._run(inputs)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generic image_kernel forge driver")
    parser.add_argument("--mode", default="full")       # all modes -> task correctness
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    tr, _scripts_dir = _import_task_runner()
    tr._configure()  # arch env + make the seeded (agent-editable) repo importable

    import torch
    if not torch.cuda.is_available():
        print("error: ROCm GPU (gfx950) is required (torch.cuda.is_available() is False)",
              file=sys.stderr)
        return 1

    if args.profile_run:
        return _run_profile(tr)
    if args.bench_mode:
        return _run_bench(tr)
    return _run_correctness(tr)


if __name__ == "__main__":
    sys.exit(main())
