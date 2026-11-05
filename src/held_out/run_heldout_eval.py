# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Held-out evaluation: re-run finished candidates on inputs they never saw.

An optimizing agent can observe its task's inputs: the evaluator derives two
seeds per workload case from the task and case IDs, and an agent may run the
evaluator as often as it likes. A held-out evaluation therefore re-runs a
finished, accepted schema-v3 candidate on the same task with new inputs:

* every workload case gets two new seeds, derived from a salt the framework
  draws for this evaluation (``secrets.token_hex``; ``--salt`` reproduces one);
* optional extra in-domain cases, from ``<cases-dir>/<task id>/held_out_cases.json``
  (``{"cases": [...]}`` in the ``workloads.json`` case format), are added. They
  must name exactly the task's ``var`` axes within their declared domains and
  differ from every existing case; seeds are never supplied by the file.

Per task workspace of a completed run:

1. The original run must be framework-complete (its completion record matches
   its report and candidate sources), its candidate accepted, and the current
   scoring runtime identical to the captured one. Nothing in the original run
   is written.
2. ``<output>/workspace`` is the session's frozen initial workspace (evaluator,
   contract, baseline, reference and task files exactly as materialized) with
   the evaluated candidate files (``solution/``) copied in; their digests must
   equal the report's ``evaluated_candidate_sources``.
3. Only ``_aka_eval/contract.json`` changes: held-out seeds for every case plus
   the extra cases.
4. The trusted evaluator runs ``validate-task`` (whose case list must equal the
   held-out contract's), then baseline and candidate ``compile`` and
   ``correctness``, then ``performance`` in alternating fresh processes B, C,
   B, C, averaged per case, as in the final evaluation. After every action the
   harness guard and the candidate digests are rechecked. Action evidence is
   kept under ``<output>/actions/``.
5. ``generalization_status`` combines baseline and candidate correctness:
   ``both_pass``, ``candidate_regression`` (only the baseline is correct),
   ``candidate_improvement`` (only the candidate is correct) or ``both_fail``;
   ``evaluation_error`` marks missing evidence or a violated protocol. Only
   ``both_pass`` with consistent timing methods and workloads receives a
   held-out speedup (geometric mean) and score, by the normal scoring rules.

Run it inside the scoring runtime of the original run (``make docker-shell``):

    python3 -m src.held_out.run_heldout_eval \\
        --run-dir workspace_MI355X_codex/run_20261105_120000 \\
        [--cases-dir held_out_cases/] [--salt HEX] [--tasks TASK_ID ...] \\
        [--output-suffix _heldout]

Results go to ``<run-dir><suffix>/``: one ``heldout_task_result.yaml`` per
task and ``heldout_summary.yaml``. An existing output directory is never
reused. This workflow runs no evaluation tools and makes no sanitizer claims.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import statistics
import sys

import yaml

from src.evaluator import _mean_cases
from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.score import score
from src.task_contract import EVAL_DIR
from src.task_execution import TaskExecutionError, run_action
from src.task_protocol import CaseManifest, performance_cases
from src.task_run import _state_directory, task_run_is_complete
from src.task_runtime import bind_session_runtime
from src.task_session import TaskSession, _snapshot
from src.task_spec import TaskSpec, resolve_task_path
from src.testcases import (
    analyze_benchmark_method_consistency, analyze_workload_consistency, calculate_average_speedup,
)

CASES_FILE = "held_out_cases.json"
QUADRANTS = ("both_pass", "candidate_regression", "candidate_improvement", "both_fail")
_CASE_KEYS = {"id", "axes", "scalars", "checks", "source", "description"}
_CHECKS = (["correctness"], ["correctness", "performance"])


def heldout_seeds(salt: str, task_id: str, case_id: str) -> list[int]:
    """Two seeds per case from this evaluation's salt, in the evaluator's seed range."""
    seeds = []
    for index in range(2):
        digest = hashlib.sha256(f"aka-v3-heldout|{salt}|{task_id}|{case_id}|{index}".encode()).digest()
        seeds.append(int.from_bytes(digest[:4], "big") & 0x7FFFFFFF)
    return seeds


def _identity(case: dict) -> str:
    return json.dumps([case["axes"], case.get("scalars", {})], sort_keys=True)


def _extra_case(contract: dict, case, taken: set[str], identities: set[str]) -> dict:
    """Validate one extra case against the trusted resolved contract."""
    if not isinstance(case, dict) or set(case) - _CASE_KEYS or not {"id", "axes", "checks"} <= set(case):
        raise ValueError("A held-out case has id, axes, checks and optional scalars, source, description")
    case_id = case["id"]
    if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", case_id) or case_id in taken:
        raise ValueError(f"Held-out case ids must be new and match [A-Za-z0-9_.-]: {case_id!r}")
    var = {name: axis for name, axis in contract["axes"].items() if axis["kind"] == "var"}
    axes = case["axes"]
    if not isinstance(axes, dict) or set(axes) != set(var):
        raise ValueError(f"Held-out case {case_id} must name exactly the var axes {sorted(var)}")
    for name, value in axes.items():
        axis = var[name]
        if (type(value) is not int or not axis["min"] <= value <= axis["max"]
                or value % axis.get("multiple_of", 1)):
            raise ValueError(f"Held-out case {case_id}: {name}={value!r} is outside its declared domain")
    scalars = case.get("scalars", {})
    declared = {name: item["dtype"] for name, item in contract["inputs"].items() if item["kind"] == "scalar"}
    if not isinstance(scalars, dict) or set(scalars) - set(declared):
        raise ValueError(f"Held-out case {case_id} names unknown scalars")
    for name, value in scalars.items():
        dtype = declared[name]
        valid = (type(value) is bool if dtype == "bool" else type(value) is int if dtype.startswith("int")
                 else type(value) in (int, float) and math.isfinite(value))
        if not valid:
            raise ValueError(f"Held-out case {case_id}: scalar {name}={value!r} is not a {dtype}")
    if case["checks"] not in _CHECKS:
        raise ValueError(f"Held-out case {case_id}: checks must be one of {list(_CHECKS)}")
    row = {"id": case_id, "axes": dict(axes), "scalars": dict(scalars), "checks": list(case["checks"])}
    if _identity(row) in identities:
        raise ValueError(f"Held-out case {case_id} repeats the axes and scalars of another case")
    return row


def heldout_contract(contract: dict, *, salt: str, extra_cases=()) -> dict:
    """The resolved contract with held-out seeds for every case and validated extra cases."""
    result = copy.deepcopy(contract)
    taken = {case["id"] for case in contract["cases"]}
    identities = {_identity(case) for case in contract["cases"]}
    for case in extra_cases:
        row = _extra_case(contract, case, taken, identities)
        taken.add(row["id"])
        identities.add(_identity(row))
        result["cases"].append(row)
    seen = {seed for case in contract["cases"] for seed in case["seeds"]}
    for case in result["cases"]:
        case["seeds"] = heldout_seeds(salt, contract["task_id"], case["id"])
        if seen & set(case["seeds"]) or case["seeds"][0] == case["seeds"][1]:
            raise ValueError("Held-out seeds collide with each other or the run's seeds; choose another salt")
        seen.update(case["seeds"])
    return result


def _load_extra_cases(cases_file: Path | None) -> list:
    if cases_file is None or not cases_file.is_file():
        return []
    data = json.loads(cases_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) - {"cases", "description"} or not isinstance(data.get("cases"), list):
        raise ValueError(f"{cases_file.name} must be an object with a cases list")
    return data["cases"]


def _prepare_workspace(session: TaskSession, destination: Path, evaluated: dict[str, str]) -> None:
    """Copy the frozen initial workspace, then put the evaluated candidate files in place."""
    _snapshot(session.baseline_workspace, destination)
    for edit in session.spec.candidate.editable:
        path = resolve_task_path(destination, edit.path)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif os.path.lexists(path):
            path.unlink()
    for relative, digest in evaluated.items():
        if digest == "missing":
            continue
        source = resolve_task_path(session.workspace, relative, must_exist=True)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=False)
    if session._candidate_sources(allow_missing=True, root=destination) != evaluated:
        raise ValueError("The held-out workspace does not hold the evaluated candidate")


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")


def classify(baseline_correct: bool, candidate_correct: bool) -> str:
    if baseline_correct:
        return "both_pass" if candidate_correct else "candidate_regression"
    return "candidate_improvement" if candidate_correct else "both_fail"


def evaluate_task(workspace: Path, output: Path, *, salt: str, cases_file: Path | None = None,
                  logger: logging.Logger | None = None) -> dict:
    """Evaluate one finished task workspace on held-out inputs; never raises for a task failure."""
    log = logger or logging.getLogger(__name__)
    workspace = Path(workspace).resolve(strict=True)
    output = Path(output).resolve()
    if output.is_relative_to(workspace) or workspace.is_relative_to(output):
        raise ValueError("Held-out output must be separate from the original workspace")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "task_name": workspace.name, "heldout": True, "salt": salt, "extra_case_ids": [],
        "baseline_pass_compilation": False, "baseline_pass_correctness": False,
        "candidate_pass_compilation": False, "candidate_pass_correctness": False,
        "baseline_execution_time": 0.0, "candidate_execution_time": 0.0,
        "generalization_status": "evaluation_error", "speedup_ratio": 0.0,
        "benchmark_method_consistent": False, "benchmark_method_mismatches": [],
        "workload_consistent": False, "score": 0.0,
        "original_run_speedup_ratio": 0.0, "original_run_score": 0.0, "error": None,
    }
    try:
        state = _state_directory(workspace)
        completion = json.loads((state / "completion.json").read_text())
        task_id = completion["task_id"]
        result["task_name"] = task_id
        if not task_run_is_complete(workspace, task_id, completion["agent"]):
            raise ValueError("The original run lacks a matching framework completion record")
        report = yaml.safe_load((workspace / "task_result.yaml").read_text())
        if report.get("candidate_accepted") is not True:
            raise ValueError("Held-out evaluation requires an accepted candidate")
        result.update(original_run_speedup_ratio=report.get("speedup_ratio", 0.0),
                      original_run_score=report.get("score", 0.0))
        spec = TaskSpec.from_mapping(json.loads((state / "task_spec.json").read_text()), task_id=task_id)
        if spec.contract is None:
            raise ValueError("Held-out evaluation applies to schema-v3 kernel tasks")
        session = TaskSession.load(spec, workspace, state, log, read_only=True)
        if not (state / "runtime_identity.json").is_file():
            raise ValueError("The original run has no captured scoring runtime identity")
        result["runtime_identity"] = bind_session_runtime(session)
        evaluated = report.get("evaluated_candidate_sources")
        if not isinstance(evaluated, dict) or session.candidate_source_evidence()["sources"] != evaluated:
            raise ValueError("The workspace no longer holds the evaluated candidate")

        heldout = output / "workspace"
        _prepare_workspace(session, heldout, evaluated)
        contract_path = heldout / EVAL_DIR / "contract.json"
        original = json.loads(contract_path.read_text())
        contract = heldout_contract(original, salt=salt, extra_cases=_load_extra_cases(cases_file))
        result["extra_case_ids"] = [case["id"] for case in contract["cases"][len(original["cases"]):]]
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

        guard = snapshot_workspace_harness(heldout, task_spec=spec)
        evidence = output / "actions"
        evidence.mkdir()
        count = 0

        def execute(role: str, action: str, manifest: CaseManifest | None = None):
            nonlocal count
            count += 1
            record = evidence / f"{count:02d}-{role}-{action}.json"
            phase = "candidate_evaluation" if role == "candidate" else "task_validation"
            try:
                executed = run_action(spec, heldout, role=role, action=action, phase=phase,
                                      manifest=manifest, logger=log)
            except TaskExecutionError as exc:
                _write_json(record, {"phase": phase, "role": role, "action": action, "error": str(exc),
                                     "commands": [asdict(c) for c in exc.commands]})
                raise
            _write_json(record, {"phase": phase, "invocation_id": executed.invocation_id,
                                 "result": executed.result.to_mapping(),
                                 "commands": [asdict(c) for c in executed.commands]})
            verify_workspace_harness(guard, logger=log)
            if session._candidate_sources(allow_missing=True, root=heldout) != evaluated:
                raise ValueError(f"{role} {action} changed the candidate")
            return executed.result

        task = execute("task", "validate-task")
        if not task.passed:
            raise ValueError(f"validate-task failed: {task.reason}")
        manifest = CaseManifest.from_result(task)
        if ({row["test_case_id"]: list(row["checks"]) for row in manifest.cases}
                != {case["id"]: case["checks"] for case in contract["cases"]}):
            raise ValueError("The evaluator did not report the held-out contract's cases")

        failures = []
        for role in ("baseline", "candidate"):
            compiled = execute(role, "compile", manifest)
            result[f"{role}_pass_compilation"] = compiled.passed
            if not compiled.passed:
                failures.append(f"{role} compile: {compiled.reason}")
                continue
            correct = execute(role, "correctness", manifest)
            result[f"{role}_pass_correctness"] = correct.passed
            if not correct.passed:
                failures.append(f"{role} correctness: {correct.reason}")
        baseline_ok, candidate_ok = result["baseline_pass_correctness"], result["candidate_pass_correctness"]
        result["generalization_status"] = classify(baseline_ok, candidate_ok)

        if baseline_ok and candidate_ok:
            # Fresh alternating processes, as in the final evaluation, so drift
            # cannot favor either side.
            rounds = {"baseline": [], "candidate": []}
            for role in ("baseline", "candidate", "baseline", "candidate"):
                performance = execute(role, "performance", manifest)
                if not performance.passed:
                    failures.append(f"{role} performance: {performance.reason}")
                    break
                rounds[role].append(performance_cases(performance))
            if len(rounds["baseline"]) == len(rounds["candidate"]) == 2:
                baseline_cases, candidate_cases = _mean_cases(rounds["baseline"]), _mean_cases(rounds["candidate"])
                for role, cases in (("baseline", baseline_cases), ("candidate", candidate_cases)):
                    result[f"{role}_execution_time"] = sum(c.execution_time_ms for c in cases) / len(cases)
                consistent, mismatches = analyze_benchmark_method_consistency(baseline_cases, candidate_cases, log)
                workloads, _ = analyze_workload_consistency(baseline_cases, candidate_cases, log)
                result.update(benchmark_method_consistent=consistent, benchmark_method_mismatches=mismatches,
                              workload_consistent=workloads)
                if consistent and workloads:
                    result["speedup_ratio"] = calculate_average_speedup(baseline_cases, candidate_cases, log)
        result["error"] = "; ".join(failures) or None
        result["score"] = score(
            result["candidate_pass_compilation"], candidate_ok, result["baseline_execution_time"],
            result["candidate_execution_time"], result["speedup_ratio"],
            benchmark_method_consistent=result["benchmark_method_consistent"] and result["workload_consistent"])
        if not task_run_is_complete(workspace, task_id, completion["agent"]):
            raise ValueError("The original run changed during held-out evaluation")
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", generalization_status="evaluation_error",
                      speedup_ratio=0.0, score=0.0, benchmark_method_consistent=False)
        log.error("Held-out evaluation of %s failed: %s", result["task_name"], result["error"])
    (output / "heldout_task_result.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    return result


def discover_workspaces(run_dir: Path) -> list[tuple[Path, str]]:
    """Framework-completed schema-v3 task workspaces of a run and their task IDs."""
    found = []
    for workspace in sorted(path for path in run_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
        state = _state_directory(workspace)
        try:
            completion = json.loads((state / "completion.json").read_text())
            declaration = json.loads((state / "task_spec.json").read_text())
        except (OSError, ValueError):
            continue
        if "contract" in declaration:
            found.append((workspace, completion["task_id"]))
    return found


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "geomean": 0.0, "median": 0.0}
    return {"count": len(values), "geomean": round(math.exp(statistics.fmean(map(math.log, values))), 4),
            "median": round(statistics.median(values), 4)}


def write_summary(output_dir: Path, results: list[dict], salt: str, logger: logging.Logger) -> dict:
    """Write heldout_summary.yaml: quadrants, conditional correctness and speedups."""
    quadrants = {name: 0 for name in (*QUADRANTS, "evaluation_error")}
    for row in results:
        quadrants[row["generalization_status"]] += 1
    baseline_correct = [row for row in results if row["generalization_status"] in ("both_pass", "candidate_regression")]
    retained = sum(row["generalization_status"] == "both_pass" for row in baseline_correct)
    both_pass = [row for row in results if row["generalization_status"] == "both_pass"]
    summary = {
        "total_tasks": len(results),
        "salt": salt,
        "quadrant_counts": quadrants,
        "conditional_correctness": {
            "description": "P(candidate correct on held-out inputs | baseline correct on them)",
            "baseline_correct_count": len(baseline_correct),
            "candidate_also_correct_count": retained,
            "rate_pct": round(100 * retained / len(baseline_correct), 1) if baseline_correct else 0.0,
        },
        "heldout_speedup": _stats([row["speedup_ratio"] for row in both_pass if row["speedup_ratio"] > 0]),
        "original_run_speedup": _stats([row["original_run_speedup_ratio"] for row in both_pass
                                        if row["original_run_speedup_ratio"] > 0]),
        "per_task": [{
            "task_name": row["task_name"], "generalization_status": row["generalization_status"],
            "heldout_speedup": round(row["speedup_ratio"], 4),
            "original_run_speedup": round(row["original_run_speedup_ratio"], 4),
            "heldout_score": round(row["score"], 4), "original_run_score": round(row["original_run_score"], 4),
            "error": row["error"],
        } for row in results],
    }
    (output_dir / "heldout_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    logger.info("Held-out quadrants: %s; conditional correctness %s%%; held-out speedup %s",
                quadrants, summary["conditional_correctness"]["rate_pct"], summary["heldout_speedup"])
    return summary


def _logger(log_file: Path) -> logging.Logger:
    log = logging.getLogger("heldout_eval")
    log.setLevel(logging.INFO)
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_file)):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Held-out evaluation of finished schema-v3 candidates")
    parser.add_argument("--run-dir", required=True, type=Path, help="completed run directory")
    parser.add_argument("--cases-dir", type=Path, default=None,
                        help=f"optional extra in-domain cases: <dir>/<task id>/{CASES_FILE}")
    parser.add_argument("--salt", default=None,
                        help="seed salt to reproduce an evaluation (default: a fresh random salt)")
    parser.add_argument("--tasks", nargs="*", default=None, help="task IDs to evaluate (default: all)")
    parser.add_argument("--output-suffix", default="_heldout",
                        help="suffix of the output directory beside --run-dir")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.output_suffix):
        parser.error("--output-suffix may only contain letters, numbers, dot, underscore, and dash")
    if args.salt is not None and not re.fullmatch(r"[0-9a-f]{8,64}", args.salt):
        parser.error("--salt must be 8 to 64 lowercase hexadecimal digits")
    run_dir = args.run_dir.resolve(strict=True)
    output_dir = run_dir.parent / (run_dir.name + args.output_suffix)
    if os.path.lexists(output_dir):
        parser.error(f"{output_dir} exists; choose another --output-suffix")
    output_dir.mkdir(parents=True)
    salt = args.salt or secrets.token_hex(16)
    logger = _logger(output_dir / "heldout_eval.log")
    logger.info("Run %s -> %s (salt %s)", run_dir, output_dir, salt)

    results = []
    for workspace, task_id in discover_workspaces(run_dir):
        if args.tasks and task_id not in args.tasks:
            continue
        cases_file = args.cases_dir / task_id / CASES_FILE if args.cases_dir else None
        logger.info("Evaluating %s", task_id)
        results.append(evaluate_task(workspace, output_dir / workspace.name, salt=salt,
                                     cases_file=cases_file, logger=logger))
    write_summary(output_dir, results, salt, logger)
    if not results:
        logger.error("No completed schema-v3 task workspace was evaluated")
    return 0 if results and all(row["generalization_status"] != "evaluation_error" for row in results) else 1


if __name__ == "__main__":
    sys.exit(main())
