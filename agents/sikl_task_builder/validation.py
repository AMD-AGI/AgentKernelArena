"""Runtime checks and QA-auditor validation with content-bound evidence.

A draft is materialized as an ordinary schema-v3 workspace for the language
the QA auditor qualifies (the baseline's, so the candidate path receives an
exact copy of the baseline as its positive control). The framework's
TaskSession runs the seven initial actions and ``audit_task_session`` runs the
QA auditor on that captured evidence; only a current, framework-finalized
``qa_report.yaml`` with status PASS accepts a draft.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path

from src.runtime_env import build_subprocess_env

from .config import Config
from .execution import run_process
from .materialize import TASK_FILE, task_digest, task_tree

LOG = logging.getLogger(__name__)
CHECK_MODES = {"validate-task": ("task", "validate-task"), "source-check": ("baseline", "correctness"),
               "compile": ("candidate", "compile"), "correctness": ("candidate", "correctness"),
               "performance": ("candidate", "performance")}


def copy_task_files(draft: Path, workspace: Path) -> None:
    # Validate the same file set that installation will publish. In particular,
    # stale bytecode or draft build products cannot substitute for source code.
    workspace.mkdir(parents=True, exist_ok=False)
    for relative in task_tree(draft):
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(draft / relative, target)


def runtime_identity(config: Config) -> dict:
    if os.environ.get("AGENT_KERNEL_ARENA_DOCKER") != "1":
        raise RuntimeError("Run GPU generation/validation through make docker-sikl-task-builder")
    import torch
    from src.preprocessing import _resolve_gfx_arch
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("Compatible ROCm GPU is unavailable")
    properties = torch.cuda.get_device_properties(0)
    arch = getattr(properties, "gcnArchName", "").split(":")[0]
    if arch != _resolve_gfx_arch(config.target_gpu_model):
        raise RuntimeError(f"Runtime architecture {arch} does not match {config.target_gpu_model}")
    return {"torch": torch.__version__, "rocm": torch.version.hip, "gpu": properties.name,
            "arch": arch, "image": os.environ.get("AGENT_KERNEL_ARENA_IMAGE", "unknown")}


def task_id_of(draft: Path, config: Config) -> str:
    """The stable Arena ID the installed task will have; drafts live outside tasks/."""
    return config.arena_task_id(draft.name)


def qa_eval_config(config: Config) -> dict:
    """Run settings for the QA auditor; the validator section selects its model review."""
    review = {key: value for key, value in config.validator.items() if key in ("backend", "model", "effort")}
    review.setdefault("backend", "codex")
    return {"target_gpu_model": config.target_gpu_model,
            "agent": {"template": "qa_auditor", **review, "timeout_seconds": config.agent_timeout,
                      "controls_timeout_seconds": config.command_timeout}}


def _evaluate(workspace: Path, role: str, action: str, artifacts: Path, timeout: float) -> tuple[dict, object]:
    env = build_subprocess_env()
    env["ARENA_EVAL_PHASE"] = "task_validation"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    python = os.environ.get("AGENT_KERNEL_ARENA_PYTHON") or sys.executable
    argv = [python, "-B", "-m", "_aka_eval.evaluate", *(["validate-task"] if role == "task" else [role, action])]
    result = run_process(argv, workspace, artifacts / f"{role}-{action}.log", timeout, env)
    if result["timed_out"]:
        return result, None
    from src.task_protocol import parse_command_result
    try:
        parsed = parse_command_result(Path(result["log"]).read_text(), role=role, action=action,
                                      returncode=result["exit_code"])
    except ValueError as error:
        result.update(ok=False, protocol_error=str(error))
        return result, None
    result["result"] = parsed.to_mapping()
    result["ok"] = result["ok"] and parsed.passed
    return result, parsed


def render_check_workspace(draft: Path, artifacts: Path, task_id: str) -> tuple[Path, str]:
    """A rendered v3 workspace for the auditor's language, without framework session state."""
    from src.task_contract import audit_target_language, render_workspace
    package, workspace = (artifacts / "package").resolve(), (artifacts / "workspace").resolve()
    copy_task_files(draft, package)
    copy_task_files(draft, workspace)
    language = audit_target_language(package / TASK_FILE, task_id=task_id)
    render_workspace(package, workspace, task_id=task_id, target_language=language)
    return workspace, language


def check_task(draft: Path, artifacts: Path, config: Config, mode: str, timeout=None) -> dict:
    """One evaluator action, checked against the task's own validate-task manifest (diagnostic)."""
    if mode not in CHECK_MODES:
        raise ValueError(f"Unknown check: {mode}")
    runtime_identity(config)
    artifacts.mkdir(parents=True, exist_ok=True)
    task_id = task_id_of(draft, config)
    workspace, language = render_check_workspace(draft, artifacts, task_id)
    limit = timeout or config.command_timeout
    manifest_run, manifest = _evaluate(workspace, "task", "validate-task", artifacts, limit)
    role, action = CHECK_MODES[mode]
    if mode == "validate-task" or manifest is None or not manifest.passed:
        return {**manifest_run, "mode": mode, "target_language": language}
    result, parsed = _evaluate(workspace, role, action, artifacts, limit)
    if parsed is not None:
        from src.task_protocol import CaseManifest
        try:
            CaseManifest.from_result(manifest).validate(parsed)
        except ValueError as error:
            result.update(ok=False, protocol_error=str(error))
    return {**result, "mode": mode, "target_language": language}


def _findings(report: dict | None) -> list[dict]:
    return [{"check": item.get("id"), "severity": item.get("severity"), "detail": item.get("detail", ""),
             "fix": item.get("fix", ""), "evidence": item.get("evidence", {})}
            for item in (report or {}).get("findings", []) if isinstance(item, dict)]


def validate_task(draft: Path, artifacts: Path, config: Config, timeout=None) -> dict:
    environment = runtime_identity(config)
    validation_id = uuid.uuid4().hex
    root = (artifacts / validation_id).resolve()
    package = root / "package" / draft.name
    source_digest = task_digest(draft)
    copy_task_files(draft, package)
    package_files = task_tree(package)
    task_id = task_id_of(draft, config)
    # The framework owns every execution, its manifest and the pre-review
    # context. One bounded subprocess covers materialization, actions and audit.
    request = root / "audit_request.json"
    request.write_text(json.dumps({"package": str(package), "workspace": str(root / "workspace"),
                                   "session": str(root / "session"), "config": config.mapping(),
                                   "task_id": task_id, "validation_id": validation_id}))
    formal = run_process(
        [sys.executable, "-m", "agents.sikl_task_builder.validation", str(request)],
        Path(__file__).resolve().parents[2], root / "auditor.log",
        timeout or config.max_task_seconds, build_subprocess_env(),
    )
    from agents.qa_auditor.report import load_report, report_is_current
    from src.task_materialization import MaterializationError, load_materialization_record
    workspace = root / "workspace"
    report = load_report(workspace) if workspace.is_dir() else None
    try:
        package_sha = load_materialization_record(workspace)["task_package_sha256"]
    except (MaterializationError, KeyError, OSError):
        package_sha = None
    complete = bool(report is not None and package_sha
                    and report.get("audit_request_id") == validation_id
                    and report.get("task_id") == task_id
                    and report_is_current(report, package_sha))
    checks = {}
    for path in sorted((root / "session").glob("action-*.json")):
        record = json.loads(path.read_text())
        outcome = record.get("result") or {}
        key = f"{outcome.get('role', 'unknown')}/{outcome.get('action', path.stem)}"
        checks[key] = {"ok": outcome.get("status") == "PASS", "result": outcome,
                       "execution_error": record.get("execution_error")}
    after = task_tree(package)
    changed = sorted(p for p in set(package_files) | set(after) if package_files.get(p) != after.get(p))
    unchanged = not changed and task_digest(draft) == source_digest
    status = (report or {}).get("status", "FAIL") if complete else "FAIL"
    result = {"validation_id": validation_id, "task_id": task_id, "task_digest": source_digest,
              "environment": environment, "commands": checks, "formal_process": formal,
              "report_complete": complete, "overall_status": status,
              "blocking": (report or {}).get("blocking") if complete else None,
              "warnings": (report or {}).get("warnings") if complete else None,
              "diagnostics": _findings(report if complete else None),
              "changed_files": changed, "workspace": str(workspace),
              "ok": bool(formal["ok"] and complete and unchanged and status == "PASS")}
    (root / "evidence.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def prepare_audit_workspace(request: dict):
    """Materialize the draft package as the workspace the QA auditor qualifies."""
    from src.task_contract import audit_target_language
    from src.task_materialization import materialize_task_workspace
    from src.task_spec import load_task_spec
    task_file = Path(request["package"]) / TASK_FILE
    language = audit_target_language(task_file, task_id=request["task_id"])
    spec = load_task_spec(task_file, task_id=request["task_id"], target_language=language)
    workspace = materialize_task_workspace(spec, task_file, Path(request["workspace"]), LOG)
    return spec, task_file, workspace


def _launch_auditor(*, eval_config, task_config_dir, workspace):
    from agents.qa_auditor.launch_agent import launch_agent
    return launch_agent(eval_config, task_config_dir, workspace)


def _main():
    from src.task_run import audit_task_session
    from src.task_session import TaskSession
    request = json.loads(Path(sys.argv[1]).read_text())
    config = Config(**request["config"])
    logging.basicConfig(level=logging.INFO)
    spec, task_file, workspace = prepare_audit_workspace(request)
    session = TaskSession.create(spec, workspace, Path(request["session"]), LOG)
    report = audit_task_session(session, eval_config=qa_eval_config(config),
                                task_config_dir=str(task_file), agent_launcher=_launch_auditor,
                                audit_request_id=request["validation_id"])
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
