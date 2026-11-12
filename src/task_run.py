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
                from .process_control import bounded_timeout
                deadline = os.environ.get("ARENA_TASK_DEADLINE")
                limit = bounded_timeout(declaration.get("timeout_s", 60), float(deadline) if deadline else None)
                executed = _run_process(command, session.workspace, env, limit)
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


def audit_task_session(session: TaskSession, *, eval_config: dict, task_config_dir: str,
                       agent_launcher, audit_request_id: str | None = None) -> dict:
    """Run the QA auditor on identical trusted initial evidence (main and quality_loop).

    The auditor writes the itemized report. The framework then verifies that
    neither the audit nor its model review changed protected task state or the
    captured evidence; any such change, or a missing or altered report, adds a
    blocking QA-AUDIT finding.
    """
    from agents.qa_auditor.report import build_report, load_report, write_report
    from .task_materialization import load_materialization_record

    bind_session_runtime(session)
    if session.initial_validation is None:
        session.validate_initial()
    context_path = session.state_directory / "validation_context.json"
    candidate_before = session.candidate_source_evidence()

    def evidence_fingerprints():
        files = {}
        for path in session.state_directory.glob("*.json"):
            if path.name.startswith("qa_"):
                continue
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("Validation evidence must remain a regular controller-owned file")
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return files

    evidence_before = evidence_fingerprints()
    request_id = audit_request_id or uuid.uuid4().hex
    eval_config.update(_task_id=session.spec.task_id, _task_validation_context=str(context_path),
                       _task_validation_request_id=request_id)
    environment = {"ARENA_EVAL_PHASE": "task_validation", "ARENA_VALIDATION_CONTEXT": str(context_path)}
    errors = []
    try:
        with _agent_environment(environment):
            agent_launcher(eval_config=eval_config, task_config_dir=task_config_dir, workspace=str(session.workspace))
    except Exception as exc:
        errors.append(f"Auditor launcher failed: {type(exc).__name__}: {exc}")
    try:
        session.verify_candidate_harness()
        session.verify_baseline_sources()
        if session.candidate_source_evidence() != candidate_before:
            raise RuntimeError("The audit changed the initial candidate after captured GPU checks")
        if evidence_fingerprints() != evidence_before:
            raise RuntimeError("The audit changed captured action or context evidence")
    except (ValueError, RuntimeError, OSError) as exc:
        errors.append(f"Audit changed protected task state: {exc}")
    report = load_report(session.workspace)
    if report is None or report.get("audit_request_id") != request_id or report.get("task_id") != session.spec.task_id:
        errors.append("The auditor returned no complete report for this request")
        report = None
    if errors:
        record = load_materialization_record(session.workspace)
        report = build_report(task_id=session.spec.task_id,
                              findings=(report or {}).get("findings", []),
                              task_package_sha256=record.get("task_package_sha256"),
                              target_language=session.spec.target_language, request_id=request_id,
                              evidence=(report or {}).get("evidence", {}), framework_errors=errors)
        write_report(session.workspace, report, state=session.state_directory)
    return report


def _budget_path(run_directory: Path, task_name: str) -> Path:
    return run_directory / ".budgets" / (hashlib.sha256(task_name.encode()).hexdigest() + ".json")


def run_task_v2(*, eval_config: dict, agent, agent_launcher, task_name: str,
                task_config_dir: str, run_directory: Path, timestamp: str,
                logger: logging.Logger) -> tuple[bool, Path]:
    """Open the persisted task budget, then run the shared v2 lifecycle under it."""
    from .measurement import measurement_kind
    from .run_budget import budget_policy, open_budget
    from .task_contract import audit_target_language, is_v3_task, target_language_for

    target_language = None
    if is_v3_task(Path(task_config_dir)):
        # An audit qualifies the task itself: its candidate starts as the
        # baseline whenever the baseline is written in a target language.
        target_language = (audit_target_language(Path(task_config_dir), task_id=task_name)
                           if agent.value == "qa_auditor" else target_language_for(eval_config, task_name))
        if target_language is None:
            raise ValueError(f"Task {task_name} needs a target_language from the run config")
    eval_config["_target_language"] = target_language
    spec = load_task_spec(Path(task_config_dir), task_id=task_name, target_language=target_language)
    from .agent_sandbox import resolve as resolve_sandbox, sandbox_mode
    # Decided, and for agent_sandbox: landlock refused, before any workspace,
    # baseline or model work is spent.
    mode = sandbox_mode(eval_config)
    confine, unconfined_reason = resolve_sandbox(mode, agent.value, measurement=measurement_kind(spec.to_mapping()),
                                                 world_size=_world_size(spec))
    if unconfined_reason:
        logger.info("Agent sandbox not applied to %s: %s", task_name, unconfined_reason)
    sandbox_plan = {"mode": mode, "confine": confine, "reason": unconfined_reason}
    policy = budget_policy(eval_config)
    if policy and "final_evaluation_pairs" in policy and measurement_kind(spec.to_mapping()) != "serving":
        raise ValueError("final_evaluation_pairs applies only to serving tasks")
    if policy is None and measurement_kind(spec.to_mapping()) == "serving":
        # Refuse before any workspace, baseline or model work is spent.
        raise ValueError("Serving tasks require budget.task_wall_time_s and budget.final_evaluation_reserve_s")
    from .candidate_verification import policy as review_policy, bind_policy
    review = review_policy(eval_config, serving=measurement_kind(spec.to_mapping()) == 'serving')
    if agent.value != 'qa_auditor':
        bind_policy(_budget_path(run_directory, task_name).with_suffix('.verification.json'), review)
    record = open_budget(_budget_path(run_directory, task_name), policy)
    arguments = dict(eval_config=eval_config, agent=agent, agent_launcher=agent_launcher,
                     task_name=task_name, task_config_dir=task_config_dir, run_directory=run_directory,
                     timestamp=timestamp, logger=logger, spec=spec, budget_record=record,
                     sandbox_plan=sandbox_plan)
    if record is None:
        return _run_task_v2(**arguments)
    with _agent_environment({"ARENA_TASK_DEADLINE": str(record["deadline_epoch"])}):
        return _run_task_v2(**arguments)


def _world_size(spec: TaskSpec) -> int:
    """GPUs one evaluation of the task uses; multi-GPU tasks record it in their contract."""
    return (spec.to_mapping().get("contract") or {}).get("world_size", 1)


def _sweep_before_final_evaluation(config: dict, state: Path, logger: logging.Logger) -> dict:
    """Stop agent leftovers in the dedicated worker container and record the outcome.

    Serving measurements share the GPU and host network with this container, so
    a surviving agent-started server or script would distort the final A/B.
    """
    from .measurement import measurement_kind
    from .process_control import dedicated_container, sweep_foreign_processes

    if not dedicated_container():
        record = {"status": "unavailable", "reason": "not running in a dedicated framework container"}
    else:
        try:
            record = sweep_foreign_processes()
        except (OSError, RuntimeError) as exc:
            record = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    record["required"] = measurement_kind(config) == "serving"
    _json_file(state / "process_sweep.json", record)
    if record.get("terminated"):
        logger.warning("Stopped %d leftover process(es) before final evaluation: %s",
                       len(record["terminated"]), [(row["pid"], row["comm"]) for row in record["terminated"]])
    if record.get("survivors"):
        logger.error("Leftover process(es) survived the sweep: %s", record["survivors"])
    return record


def _run_task_v2(*, eval_config: dict, agent, agent_launcher, task_name: str,
                task_config_dir: str, run_directory: Path, timestamp: str,
                logger: logging.Logger, spec: TaskSpec, budget_record: dict | None,
                sandbox_plan: dict) -> tuple[bool, Path]:
    """Preserve agent failures and score the delivered files independently."""
    config = spec.to_mapping()
    from .measurement import measurement_kind, serving_pair_count
    if measurement_kind(config) == "serving":
        serving_pair_count(config, (eval_config.get("budget") or {}).get("final_evaluation_pairs"))
    merge_task_tool_config(EvalToolsConfig.from_mapping(eval_config), config)
    from .preprocessing import get_task_workspace_path
    retained = get_task_workspace_path(run_directory, task_name, timestamp)
    expired_resume = (budget_record is not None and budget_record['deadline_epoch'] <= time.time()
                      and _state_directory(retained).is_dir())
    # Loading a prior session to write a failed final result is bookkeeping, not
    # new setup/search/GPU work. Action deadlines below remain expired.
    workspace = retained if expired_resume else setup_workspace(
        task_config_dir, run_directory, timestamp, logger, task_name=task_name,
        target_language=spec.target_language)
    state = _state_directory(workspace)
    if state.exists():
        session = TaskSession.load(spec, workspace, state, logger)
    else:
        from .task_materialization import verify_original_materialization

        verify_original_materialization(workspace)
        session = TaskSession.create(spec, workspace, state, logger)
    bind_session_runtime(session)
    initial = session.initial_validation or session.validate_initial()
    from .measurement import measurement_kind
    if initial.accepted and measurement_kind(config) == "serving":
        from .run_budget import reserve_from_validation
        from .candidate_verification import policy as review_policy
        review = None if agent.value == 'qa_auditor' else review_policy(eval_config, serving=True)
        budget_record = reserve_from_validation(budget_record, session,
            verification_seconds=review['timeout_seconds'] if review else 0)
        _json_file(_budget_path(run_directory, task_name), budget_record)
    harness = session.harness
    agent_config = {**eval_config, "_task_id": task_name,
                    "_task_validation_context": str(state / "validation_context.json")}
    environment = {"ARENA_EVAL_PHASE": "candidate_evaluation",
                   "ARENA_VALIDATION_CONTEXT": str(state / "validation_context.json")}
    if initial.accepted:
        environment["ARENA_TASK_CONTEXT"] = str(session.agent_context_path)
    if agent.value == "qa_auditor":
        from .run_budget import agent_budget
        # Qualification review uses the remaining wall time; there is no later
        # optimization search for which it must reserve another evaluation.
        review_budget = {**budget_record, "effective_reserve_s": 0} if budget_record else None
        agent_config = agent_budget(agent_config, review_budget)
        audit_task_session(session, eval_config=agent_config, task_config_dir=task_config_dir,
                           agent_launcher=agent_launcher)
        from agents.qa_auditor.report import qa_report_is_complete
        return qa_report_is_complete(workspace), workspace

    agent_result = {"status": "NOT_RUN", "error": None, "duration_s": 0.0,
                    "candidate_changed": None}
    sandbox = None
    if initial.accepted:
        source_before = session.candidate_source_evidence()
        if sandbox_plan["confine"]:
            from .agent_sandbox import prepare as prepare_sandbox
            sandbox = prepare_sandbox(sandbox_plan["mode"], workspace=workspace, state_directory=state,
                                      baseline_workspace=session.baseline_workspace)
            environment.update(sandbox.environment)
        started = time.monotonic()
        try:
            from .run_budget import agent_budget
            agent_config = agent_budget(agent_config, budget_record)
            with _agent_environment(environment):
                agent_launcher(eval_config=agent_config, task_config_dir=task_config_dir, workspace=str(workspace))
            agent_result["status"] = "COMPLETED"
        except TimeoutError as exc:
            agent_result.update(status="BUDGET_EXHAUSTED", error=str(exc))
            logger.info("Search budget exhausted; evaluating the retained candidate")
        except Exception as exc:
            agent_result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            logger.warning("Agent execution failed; evaluating retained candidate: %s", exc)
        agent_result["duration_s"] = time.monotonic() - started
        source_after = session.candidate_source_evidence()
        if source_before["error"] is None and source_after["error"] is None:
            agent_result["candidate_changed"] = source_before["sources"] != source_after["sources"]
        if sandbox is not None:
            agent_result["sandbox"] = sandbox.record()
        elif sandbox_plan["reason"]:
            from .agent_sandbox import not_applied_record
            agent_result["sandbox"] = not_applied_record(sandbox_plan["mode"], sandbox_plan["reason"])
    _json_file(state / f"agent-{uuid.uuid4().hex}.json", agent_result)
    sweep = _sweep_before_final_evaluation(config, state, logger)
    result = evaluate_task_session(
        session, eval_config={**eval_config, "agent": {**eval_config.get("agent", {}), "template": agent.value}},
        logger=logger, result_metadata={"agent_execution": agent_result, "process_sweep": sweep})
    accepted = all(result.get(key) is True for key in (
        "pass_compilation", "pass_correctness", "pass_tool_gate", "benchmark_method_consistent", "workload_consistent"))
    accepted = accepted and result.get("pass_verification", True) is True
    accepted = accepted and result.get("best_optimized_execution_time", 0) > 0
    if "sandbox" in agent_result:
        result["agent_sandbox"] = agent_result["sandbox"]
    if sandbox is not None:
        # Fail closed: an applied sandbox that no agent launch attested taints the run.
        accepted = accepted and agent_result["sandbox"]["status"] == "enforced"
    # Leftover processes taint any measurement; serving tasks additionally need
    # a container the framework could sweep at all.
    if sweep["status"] == "leftovers" or (sweep["required"] and sweep["status"] != "clean"):
        accepted = False
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
    # A resumed invocation may do no new search while retaining an earlier
    # submission. Report its change against the original, not this invocation.
    original = session.candidate_source_evidence(initial=True)
    evaluated = result.get('evaluated_candidate_sources')
    changed = (evaluated != original['sources']
               if evaluated is not None and original['error'] is None else None)
    result['candidate_changed_from_initial'] = changed
    result['optimization_outcome'] = ('unchanged' if changed is False else
        'accepted_change' if changed is True and result['candidate_accepted'] else
        'rejected_change' if changed is True else 'incomplete')
    if result.get('candidate_verification', {}).get('implementation_changed') is False:
        result['optimization_outcome'] = 'unchanged_computation' if result['candidate_accepted'] else 'not_accepted'
    (workspace / "task_result.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    _json_file(state / "completion.json", _completion_record(session, agent.value))
    return True, workspace
