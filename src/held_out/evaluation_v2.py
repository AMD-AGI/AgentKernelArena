"""Held-out evaluation using the original v2 session and public task actions.

Shape injections are evaluator-maintained test changes, never an optimizer
capability. Their numerical semantics still require review. This module binds
the candidate, baseline, runtime, manifest and commands used for each result.
"""
from __future__ import annotations

import ast
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path

import yaml

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_execution import TaskExecutionError, run_action
from src.task_protocol import CaseManifest, performance_cases
from src.task_run import _state_directory, task_run_is_complete
from src.task_runtime import bind_session_runtime
from src.task_session import TaskSession, _snapshot
from src.task_spec import load_task_spec, resolve_task_path
from src.score import score
from src.testcases import (
    analyze_benchmark_method_consistency, calculate_average_speedup, save_performance_results,
)

from .injection import apply_all_injections


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _implementation(workspace: Path, session: TaskSession) -> dict:
    """Allow case edits in a colocated harness, but preserve its implementation."""
    result = {}
    for edit in session.spec.candidate.editable:
        path = resolve_task_path(workspace, edit.path)
        if not path.exists():
            result[edit.path] = "missing"
            continue
        if edit.scope == "symbols":
            tree = ast.parse(path.read_text())
            initial = session.harness.initial_symbols.get(edit.path, frozenset())
            nodes = [node for node in tree.body if (
                isinstance(node, (ast.Import, ast.ImportFrom)) or
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and
                (node.name in edit.symbols or edit.allow_new_helpers and node.name not in initial)
            )]
            result[edit.path] = hashlib.sha256(
                ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False).encode()
            ).hexdigest()
        else:
            paths = path.rglob("*") if edit.scope == "tree" else (path,)
            for source in paths:
                relative = source.relative_to(workspace).as_posix()
                resolve_task_path(workspace, relative)
                if source.is_file() and not set(source.relative_to(workspace).parts[:-1]) & {
                    "__pycache__", ".pytest_cache", ".git", "build",
                }:
                    result[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return result


def _case_input(row: dict) -> str:
    # Renaming an existing case does not make its input held out.
    return json.dumps({key: row.get(key) for key in ("shape", "dtype", "params")}, sort_keys=True)


def evaluate_task_v2(original_workspace: Path, output_workspace: Path,
                     heldout_config: dict, logger: logging.Logger) -> dict:
    original_workspace = original_workspace.resolve(strict=True)
    output_workspace = output_workspace.resolve()
    if (output_workspace.is_relative_to(original_workspace)
            or original_workspace.is_relative_to(output_workspace)):
        raise ValueError("Held-out output must be separate from the original workspace")
    # Existing experiment artifacts are never overwritten by a rerun.
    output_workspace.mkdir(parents=True, exist_ok=False)
    result = {
        "task_name": original_workspace.name, "heldout": True, "task_schema_version": 2,
        "opt_pass_compilation": False, "opt_pass_correctness": False, "opt_execution_time": 0.0,
        "orig_heldout_pass_compilation": False, "orig_heldout_pass_correctness": False,
        "orig_heldout_execution_time": 0.0, "generalization_status": "evaluation_error",
        "speedup_ratio": 0.0, "benchmark_method_consistent": False,
        "benchmark_method_mismatches": [], "score": 0.0,
        "original_run_pass_correctness": False, "original_run_speedup_ratio": 0.0,
        "original_run_score": 0.0, "error": None,
    }
    try:
        state = _state_directory(original_workspace)
        descriptor = json.loads((state / "session.json").read_text())
        completion = json.loads((state / "completion.json").read_text())
        task_id = descriptor["task_id"]
        result["task_name"] = task_id
        if not task_run_is_complete(original_workspace, task_id, completion["agent"]):
            raise ValueError("Original v2 run lacks matching completion and source evidence")
        run_result = yaml.safe_load((original_workspace / "task_result.yaml").read_text())
        if run_result.get("candidate_accepted") is not True:
            raise ValueError("Held-out evaluation requires an accepted original candidate")
        result.update(original_run_pass_correctness=run_result["pass_correctness"],
                      original_run_speedup_ratio=run_result["speedup_ratio"],
                      original_run_score=run_result["score"])
        spec = load_task_spec(original_workspace / "config.yaml", task_id=task_id)
        session = TaskSession.load(spec, original_workspace, state, logger)
        if not (state / "runtime_identity.json").is_file():
            raise ValueError("Original run has no captured scoring runtime identity")
        runtime = bind_session_runtime(session)
        if session.manifest is None:
            raise ValueError("Original run has no independently captured task manifest")
        result["runtime_identity"] = runtime
        _json(output_workspace / "injections.json", heldout_config)
        _json(output_workspace / "original_completion.json", completion)

        orig_ws, opt_ws = output_workspace / "orig", output_workspace / "opt"
        # The saved baseline includes image sources and generation stubs exactly
        # as materialized for this run. Never restore from today's task checkout.
        _snapshot(session.baseline_workspace, orig_ws)
        _snapshot(original_workspace, opt_ws)
        for workspace in (orig_ws, opt_ws):
            before = _implementation(workspace, session)
            for injection in heldout_config.get("injections", []):
                target = resolve_task_path(workspace, injection["file"], must_exist=True)
                if target == workspace / "config.yaml":
                    raise ValueError("Held-out injections cannot change the task contract")
            if not apply_all_injections(workspace, heldout_config, logger):
                raise ValueError(f"Held-out injection failed in {workspace.name}")
            if _implementation(workspace, session) != before:
                raise ValueError("Held-out injections changed candidate implementation or imports")

        guards = {ws: snapshot_workspace_harness(ws, task_spec=spec) for ws in (orig_ws, opt_ws)}
        implementations = {ws: _implementation(ws, session) for ws in (orig_ws, opt_ws)}
        evidence_dir = output_workspace / "actions"
        evidence_dir.mkdir()
        action_count = 0

        def verify() -> None:
            for ws in (orig_ws, opt_ws):
                verify_workspace_harness(guards[ws], logger=logger)
                if _implementation(ws, session) != implementations[ws]:
                    raise ValueError(f"Held-out action changed implementation in {ws.name}")

        def execute(ws, role, action, manifest=None):
            nonlocal action_count
            verify()
            action_count += 1
            record = evidence_dir / f"{action_count:02d}-{role}-{action}.json"
            phase = "candidate_evaluation" if role == "candidate" else "task_validation"
            try:
                executed = run_action(spec, ws, role=role, action=action, phase=phase,
                                      manifest=manifest, logger=logger)
            except TaskExecutionError as exc:
                _json(record, {"phase": phase, "role": role, "action": action,
                               "error": str(exc), "commands": [asdict(c) for c in exc.commands]})
                raise
            _json(record, {"phase": phase, "workspace": ws.name,
                           "invocation_id": executed.invocation_id,
                           "result": executed.result.to_mapping(),
                           "commands": [asdict(c) for c in executed.commands]})
            verify()
            return executed

        task = execute(orig_ws, "task", "validate-task")
        session._verify_initial_state(task)
        manifest = CaseManifest.from_result(task.result)
        old_inputs = {_case_input(row) for row in session.manifest.cases}
        overlap = [row["test_case_id"] for row in manifest.cases
                   if "performance" in row["checks"] and _case_input(row) in old_inputs]
        if overlap:
            raise ValueError(f"Held-out performance cases reuse original inputs: {overlap}")
        _json(output_workspace / "manifest.json", {"cases": manifest.cases})

        timings = {}
        action_failures = []
        for ws, role, prefix in ((orig_ws, "baseline", "orig_heldout"), (opt_ws, "candidate", "opt")):
            compile_result = execute(ws, role, "compile", manifest).result
            result[f"{prefix}_pass_compilation"] = compile_result.passed
            if not compile_result.passed:
                action_failures.append(f"{role} compile: {compile_result.reason}")
                continue
            correctness = execute(ws, role, "correctness", manifest).result
            result[f"{prefix}_pass_correctness"] = correctness.passed
            if not correctness.passed:
                action_failures.append(f"{role} correctness: {correctness.reason}")
                continue
            performance = execute(ws, role, "performance", manifest).result
            if not performance.passed:
                raise ValueError(f"{role} performance failed: {performance.reason}")
            timings[role] = performance_cases(performance)
            result[f"{prefix}_execution_time"] = sum(
                row.execution_time_ms for row in timings[role]) / len(timings[role])
            save_performance_results(timings[role], ws, f"{role}_perf.yaml", logger)

        orig_correct, opt_correct = result["orig_heldout_pass_correctness"], result["opt_pass_correctness"]
        result["generalization_status"] = (
            "both_pass" if orig_correct and opt_correct else
            "opt_regression" if orig_correct else "opt_improvement" if opt_correct else "both_fail"
        )
        result["error"] = "; ".join(action_failures) or None
        if orig_correct and opt_correct:
            consistent, mismatches = analyze_benchmark_method_consistency(
                timings["baseline"], timings["candidate"], logger, require_complete_match=True)
            result.update(benchmark_method_consistent=consistent, benchmark_method_mismatches=mismatches)
            if consistent:
                result["speedup_ratio"] = calculate_average_speedup(
                    timings["baseline"], timings["candidate"], logger, require_complete_match=True)
        result["score"] = score(
            result["opt_pass_compilation"], opt_correct, result["orig_heldout_execution_time"],
            result["opt_execution_time"], result["speedup_ratio"],
            benchmark_method_consistent=result["benchmark_method_consistent"])
        verify()
        if not task_run_is_complete(original_workspace, task_id, completion["agent"]):
            raise ValueError("Original run changed during held-out evaluation")
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", generalization_status="evaluation_error",
                      speedup_ratio=0.0, score=0.0, benchmark_method_consistent=False)
        logger.error("Held-out v2 evaluation failed: %s", result["error"])
    (output_workspace / "heldout_task_result.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    return result
