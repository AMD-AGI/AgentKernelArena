#!/usr/bin/env python3
"""KernelForge driver backed by this task's canonical task runner.

The profile mode prepares one full performance case and launches only the target
operator. It never invokes compilation, correctness, references, or benchmarking.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path


def _import_task_runner():
    here = Path(__file__).resolve().parent
    for candidate in (here / "scripts", here, here.parent / "scripts"):
        task_runner_path = candidate / "task_runner.py"
        if task_runner_path.is_file():
            spec = importlib.util.spec_from_file_location(
                "_forge_task_runner", task_runner_path
            )
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            sys.modules["_forge_task_runner"] = module
            spec.loader.exec_module(module)
            return module
    raise RuntimeError(f"task_runner.py not found near {here}")


def _report_path(task_runner) -> Path:
    return Path(task_runner.WORKSPACE) / "build" / "performance_report.json"


def _run_correctness(task_runner) -> int:
    passed = True
    try:
        task_runner.run_correctness()
    except Exception as error:  # noqa: BLE001 - any task failure is incorrect
        passed = False
        print(f"# correctness failed: {type(error).__name__}: {str(error)[:300]}")
    print(f"allclose: {passed}")
    return 0


def _run_bench(task_runner) -> int:
    task_runner.run_performance()
    rows = json.loads(_report_path(task_runner).read_text())
    expected_ids = [str(case.get("id") or "") for case in task_runner.CASES]
    if not isinstance(rows, list):
        print("error: performance report must be a list", file=sys.stderr)
        return 1

    timings = []
    seen_ids = set()
    try:
        for row in rows:
            case_id = str(row.get("test_case_id") or "")
            elapsed_ms = float(row.get("execution_time_ms"))
            if (
                not case_id
                or case_id in seen_ids
                or not math.isfinite(elapsed_ms)
                or elapsed_ms <= 0
            ):
                raise ValueError(f"invalid or duplicate performance case {case_id!r}")
            seen_ids.add(case_id)
            timings.append((case_id, elapsed_ms))
    except (AttributeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    missing = [case_id for case_id in expected_ids if case_id not in seen_ids]
    unexpected = [
        case_id for case_id, _ in timings if case_id not in expected_ids
    ]
    if missing or unexpected:
        print(
            f"error: incomplete performance suite: "
            f"missing={missing}, unexpected={unexpected}",
            file=sys.stderr,
        )
        return 1

    for case_id, elapsed_ms in timings:
        print(f"case_ms: {case_id.replace(' ', '_')} {elapsed_ms:.6f}")
    print(f"mean_ms: {sum(value for _, value in timings) / len(timings):.6f}")
    return 0


def _pick_profile_case(task_runner) -> dict:
    cases = {case["id"]: case for case in task_runner.CASES}
    report_path = _report_path(task_runner)
    if report_path.is_file():
        try:
            rows = json.loads(report_path.read_text())
            slowest = max(
                rows, key=lambda row: float(row.get("execution_time_ms") or 0)
            )
            case_id = slowest.get("test_case_id")
            if case_id in cases:
                return cases[case_id]
        except Exception:  # noqa: BLE001 - report reuse is best effort
            pass
    return task_runner.CASES[-1]


def _run_profile(task_runner) -> int:
    torch = task_runner._torch()
    inputs = task_runner._make(
        _pick_profile_case(task_runner),
        correctness=False,
    )
    for _ in range(5):
        task_runner._run(inputs)
    torch.cuda.synchronize()
    for _ in range(3):
        task_runner._run(inputs)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CK image-kernel Forge driver")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    task_runner = _import_task_runner()
    task_runner._configure()

    import torch

    if not torch.cuda.is_available():
        print(
            "error: ROCm GPU (gfx950) is required "
            "(torch.cuda.is_available() is False)",
            file=sys.stderr,
        )
        return 1

    if args.profile_run:
        return _run_profile(task_runner)
    if args.bench_mode:
        return _run_bench(task_runner)
    return _run_correctness(task_runner)


if __name__ == "__main__":
    sys.exit(main())
