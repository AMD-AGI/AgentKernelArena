"""V2 task orchestration shared by every optimization integration."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

import yaml

from .eval_tools.config import EvalToolsConfig, merge_task_tool_config
from .evaluator import evaluate_task_session
from .harness_guard import verify_workspace_harness
from .preprocessing import setup_workspace
from .runtime_env import build_subprocess_env
from .task_execution import CommandEvidence, _run_process
from .task_runtime import bind_session_runtime
from .task_session import TaskSession
from .task_spec import TaskSpec, load_task_spec, resolve_task_path


def _json_file(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _result_digest(workspace: Path) -> str:
    report = yaml.safe_load((workspace / "task_result.yaml").read_text())
    if not isinstance(report, dict):
        raise ValueError("Final task report must be a mapping")
    return hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _state_directory(workspace: Path) -> Path:
    return workspace.parent / ".task-sessions" / workspace.name


def _completion_record(session: TaskSession, agent_name: str) -> dict:
    evidence = session.candidate_source_evidence()
    record = {"version": 1, "task_id": session.spec.task_id, "agent": agent_name,
              "result_sha256": _result_digest(session.workspace), "candidate_sources": evidence["sources"]}
    if evidence["error"]:
        record["candidate_source_error"] = evidence["error"]
    return record


def task_run_is_complete(workspace: Path, expected_task_name: str, agent_name: str) -> bool:
    """A report's mere existence is not a completed framework evaluation."""
    try:
        state = _state_directory(workspace)
        completion = json.loads((state / "completion.json").read_text())
        spec = TaskSpec.from_mapping(json.loads((state / "task_spec.json").read_text()), task_id=expected_task_name)
        session = TaskSession.load(spec, workspace, state, read_only=True)
        return completion == _completion_record(session, agent_name)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, yaml.YAMLError):
        return False


@contextmanager
def _agent_environment(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_exports(session: TaskSession, harness, logger: logging.Logger) -> list[dict]:
    declarations = session.spec.to_mapping().get("exports", [])
    results = []
    report_path = session.workspace / "task_result.yaml"
    finalized_bytes = report_path.read_bytes()
    original_candidate = session._candidate_sources(allow_missing=True)
    for declaration in declarations:
        record = {"format": declaration["format"], "output": declaration["output"],
                  "status": "FAIL", "error": None}
        try:
            output = resolve_task_path(session.workspace, declaration["output"])
            relative = output.relative_to(session.workspace).as_posix()
            if (relative in harness.digests
                    or relative in {"task_result.yaml", "validation_report.yaml", ".validation_complete"}
                    or any(edit.contains(relative) for edit in session.spec.candidate.editable)):
                raise ValueError("Export output overlaps protected input, candidate, or final report")
            # Preserve previous artifacts while requiring this invocation to
            # freshly produce the declared file.
            before = (output.stat().st_mtime_ns, output.stat().st_size) if output.is_file() else None
            candidate_before = session._candidate_sources()
            env = {key: value for key, value in build_subprocess_env().items() if not key.startswith("ARENA_")}
            env.update(ARENA_FINAL_RESULT_PATH=str(report_path), ARENA_EXPORT_PATH=str(output),
                       ARENA_TASK_CONFIG_PATH=str(session.workspace / "config.yaml"))
            command = tuple(declaration["command"])
            python = env.get("AGENT_KERNEL_ARENA_PYTHON")
            if python and command[0] in {"python", "python3"}:
                command = (python,) + command[1:]
            started = time.monotonic()
            try:
                executed = _run_process(command, session.workspace, env, declaration.get("timeout_s", 60))
            except subprocess.TimeoutExpired as exc:
                # _run_process kills the process group and attaches its drained
                # output to the timeout. Keep the same evidence shape as actions.
                record["command"] = asdict(CommandEvidence(
                    command, -signal.SIGKILL, exc.stdout or "", exc.stderr or "", time.monotonic() - started))
                record["timed_out"] = True
                raise
            # Artifact and source checks can raise even after the process has
            # completed. Retain its diagnostics before inspecting those outputs.
            record["command"] = asdict(executed)
            resolve_task_path(session.workspace, declaration["output"], must_exist=True)
            after = (output.stat().st_mtime_ns, output.stat().st_size) if output.is_file() else None
            passed = executed.returncode == 0 and after is not None and after != before
            error = None if passed else "Exporter failed or did not produce a fresh declared artifact"
            record.update(status="PASS" if passed else "FAIL", error=error)
            if session._candidate_sources() != candidate_before:
                record.update(status="FAIL", error="Exporter modified the evaluated candidate", candidate_unchanged=False)
        except Exception as exc:
            record.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        if report_path.is_symlink() or not report_path.is_file() or report_path.read_bytes() != finalized_bytes:
            if report_path.is_symlink():
                report_path.unlink()
            elif report_path.is_dir():
                # Preserve unexpected exporter output for diagnosis while
                # restoring the report owned by the evaluator.
                preserved = session.state_directory / f"export-invalid-report-{uuid.uuid4().hex}"
                report_path.rename(preserved)
                record["invalid_report_artifact"] = preserved.name
            report_path.write_bytes(finalized_bytes)
            record.update(status="FAIL", error="Exporter attempted to modify the framework-finalized result")
        try:
            verify_workspace_harness(harness, logger=logger)
            session.verify_baseline_sources()
        except Exception as exc:
            record.update(status="FAIL", error=f"Exporter changed protected task state: {exc}",
                          protected_state_unchanged=False)
        evidence = session.candidate_source_evidence()
        if evidence["error"] or evidence["sources"] != original_candidate:
            record.update(status="FAIL", error=evidence["error"] or "Exporter modified the evaluated candidate",
                          candidate_unchanged=False)
        results.append(record)
    return results


def validate_task_session(session: TaskSession, *, eval_config: dict, task_config_dir: str,
                          agent_launcher) -> dict:
    """Provide identical trusted initial evidence to main and quality_loop."""
    from agents.task_validator.report_schema import finalize_report, validation_report_is_complete

    bind_session_runtime(session)
    if session.initial_validation is None:
        session.validate_initial()
    context_path = session.state_directory / "validation_context.json"
    trusted_context = session.validation_context()
    eval_config.update(_task_id=session.spec.task_id, _task_validation_context=str(context_path))
    eval_config["_task_validation_request_id"] = uuid.uuid4().hex
    environment = {"ARENA_EVAL_PHASE": "task_validation", "ARENA_VALIDATION_CONTEXT": str(context_path)}
    error = None
    try:
        with _agent_environment(environment):
            agent_launcher(eval_config=eval_config, task_config_dir=task_config_dir, workspace=str(session.workspace))
    except Exception as exc:
        error = f"Validator launcher failed: {type(exc).__name__}: {exc}"
    error = "; ".join(filter(None, [error, eval_config.get("_task_validation_backend_error")])) or None
    if (session.workspace / "validation_report.yaml").exists() and not validation_report_is_complete(session.workspace):
        error = "; ".join(filter(None, [error, "Validator returned an altered or incomplete finalized report"]))
    try:
        session.verify_candidate_harness()
        session.verify_baseline_sources()
    except (ValueError, RuntimeError, OSError) as exc:
        error = "; ".join(filter(None, [error, f"Validation changed protected task state: {exc}"]))
    return finalize_report(
        session.workspace, expected_task_name=session.spec.task_id, trusted_task_evidence=trusted_context,
        validation_request_id=eval_config.get("_task_validation_request_id"),
        framework_error=error, task_schema_version=2)


def run_task_v2(*, eval_config: dict, agent, agent_launcher, task_name: str,
                task_config_dir: str, run_directory: Path, timestamp: str,
                logger: logging.Logger) -> tuple[bool, Path]:
    """Preserve agent failures and score the delivered files independently."""
    spec = load_task_spec(Path(task_config_dir), task_id=task_name)
    config = spec.to_mapping()
    merge_task_tool_config(EvalToolsConfig.from_mapping(eval_config), config)
    workspace = setup_workspace(task_config_dir, run_directory, timestamp, logger, task_name=task_name)
    state = _state_directory(workspace)
    if state.exists():
        session = TaskSession.load(spec, workspace, state, logger)
    else:
        from .task_materialization import verify_original_materialization

        verify_original_materialization(workspace)
        session = TaskSession.create(spec, workspace, state, logger)
    bind_session_runtime(session)
    initial = session.initial_validation or session.validate_initial()
    harness = session.harness
    agent_config = {**eval_config, "_task_id": task_name,
                    "_task_validation_context": str(state / "validation_context.json")}
    environment = {"ARENA_EVAL_PHASE": "candidate_evaluation",
                   "ARENA_VALIDATION_CONTEXT": str(state / "validation_context.json")}
    if initial.accepted:
        environment["ARENA_TASK_CONTEXT"] = str(session.agent_context_path)
    if agent.value == "task_validator":
        validate_task_session(session, eval_config=agent_config, task_config_dir=task_config_dir,
                              agent_launcher=agent_launcher)
        from agents.task_validator.report_schema import validation_report_is_complete
        return validation_report_is_complete(workspace), workspace

    agent_result = {"status": "NOT_RUN", "error": None, "duration_s": 0.0,
                    "candidate_changed": None}
    if initial.accepted:
        source_before = session.candidate_source_evidence()
        started = time.monotonic()
        try:
            with _agent_environment(environment):
                agent_launcher(eval_config=agent_config, task_config_dir=task_config_dir, workspace=str(workspace))
            agent_result["status"] = "COMPLETED"
        except Exception as exc:
            agent_result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            logger.warning("Agent execution failed; evaluating retained candidate: %s", exc)
        agent_result["duration_s"] = time.monotonic() - started
        source_after = session.candidate_source_evidence()
        if source_before["error"] is None and source_after["error"] is None:
            agent_result["candidate_changed"] = source_before["sources"] != source_after["sources"]
    _json_file(state / f"agent-{uuid.uuid4().hex}.json", agent_result)
    result = evaluate_task_session(
        session, eval_config={**eval_config, "agent": {**eval_config.get("agent", {}), "template": agent.value}},
        logger=logger, result_metadata={"agent_execution": agent_result})
    accepted = all(result.get(key) is True for key in (
        "pass_compilation", "pass_correctness", "pass_tool_gate", "benchmark_method_consistent", "workload_consistent"))
    accepted = accepted and result.get("best_optimized_execution_time", 0) > 0
    exports = []
    if accepted:
        exports = _run_exports(session, harness, logger)
        _json_file(state / "exports.json", {"exports": exports})
    result.update(candidate_accepted=accepted, exports=exports,
                  delivery_status=("COMPLETE" if all(row["status"] == "PASS" for row in exports)
                                   else "INCOMPLETE") if accepted else "NOT_ACCEPTED")
    if any(row.get("candidate_unchanged") is False or row.get("protected_state_unchanged") is False
           for row in exports):
        result["candidate_accepted"] = False
    (workspace / "task_result.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    _json_file(state / "completion.json", _completion_record(session, agent.value))
    return True, workspace
