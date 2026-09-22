"""Real runtime checks, formal task validation and content-bound evidence."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path

import yaml

from src.perf_helper_materialization import materialize_perf_helpers_in_workspace
from src.runtime_env import build_subprocess_env

from .config import Config
from .execution import run_process
from .materialize import task_digest, task_tree


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


def check_task(draft: Path, artifacts: Path, config: Config, mode: str, timeout=None) -> dict:
    if mode not in {"compile", "correctness", "performance", "source-check"}:
        raise ValueError(f"Unknown check: {mode}")
    runtime_identity(config)
    artifacts.mkdir(parents=True, exist_ok=True)
    workspace = (artifacts / "workspace").resolve()
    copy_task_files(draft, workspace)
    materialize_perf_helpers_in_workspace(workspace)
    role, action = ("baseline", "correctness") if mode == "source-check" else ("candidate", mode)
    env = build_subprocess_env()
    env["ARENA_EVAL_PHASE"] = "task_validation"
    result = run_process([sys.executable, "scripts/task_runner.py", role, action], workspace,
                         artifacts / f"{mode}.log", timeout or config.command_timeout, env)
    if not result["timed_out"]:
        from src.task_protocol import parse_command_result, CaseManifest, ActionResult
        from .bundle import case_manifest
        try:
            parsed = parse_command_result(Path(result["log"]).read_text(), role=role,
                                          action=action, returncode=result["exit_code"])
            contract = json.loads((workspace / "scripts/workload.json").read_text())
            manifest = ActionResult("task", "validate-task", "PASS", tuple(case_manifest(
                contract["definition"], contract["rows"])))
            CaseManifest.from_result(manifest).validate(parsed)
            result["result"] = parsed.to_mapping()
            result["ok"] = result["ok"] and parsed.passed
        except ValueError as error:
            result.update(ok=False, protocol_error=str(error))
    return result


def validate_task(draft: Path, artifacts: Path, config: Config, timeout=None) -> dict:
    environment = runtime_identity(config)
    validation_id = uuid.uuid4().hex
    root = (artifacts / validation_id).resolve()
    workspace = root / "workspace"
    source_digest = task_digest(draft)
    copy_task_files(draft, workspace)
    materialize_perf_helpers_in_workspace(workspace)
    runtime_files = task_tree(workspace)
    # The v2 framework owns all seven executions, their manifests and the
    # pre-review context. One bounded subprocess covers actions plus review.
    request = root / "validator_request.json"
    task_id = config.arena_task_id(json.loads((draft / "scripts/provenance.json").read_text())["definition"])
    request.write_text(json.dumps({"workspace": str(workspace), "config": config.mapping(),
                                   "task_id": task_id, "validation_id": validation_id}))
    formal = run_process(
        [sys.executable, "-m", "agents.sikl_task_builder.validation", str(request)],
        Path(__file__).resolve().parents[2], root / "validator.log",
        timeout or config.max_task_seconds, build_subprocess_env(),
    )
    from agents.task_validator.report_schema import validation_report_is_complete
    complete = validation_report_is_complete(workspace)
    report_path = workspace / "validation_report.yaml"
    report = yaml.safe_load(report_path.read_text()) if complete else {}
    complete = bool(complete and report.get("validation_schema_version") == 4
                    and report.get("task_schema_version") == 2
                    and report.get("task_name") == task_id
                    and report.get("validation_request_id") == validation_id)
    command_ok = complete and report.get("initial_validation_gate") == "PASS"
    checks = {}
    for path in sorted((root / "session").glob("action-*.json")):
        record = json.loads(path.read_text())
        outcome = record.get("result") or {}
        key = f"{outcome.get('role', 'unknown')}/{outcome.get('action', path.stem)}"
        checks[key] = {"ok": outcome.get("status") == "PASS", "result": outcome,
                       "execution_error": record.get("execution_error")}
    after = task_tree(workspace)
    changed = [p for p, digest in runtime_files.items() if after.get(p) != digest]
    added_code = [p for p in after if p not in runtime_files and
                  (p.startswith(("scripts/", "source/")) or p.endswith((".py", ".so", ".pth")))]
    unchanged = not changed and not added_code and task_digest(draft) == source_digest
    diagnostics = [
        {"check": name, "status": check.get("status"), "details": check.get("details", ""),
         "evidence": check.get("evidence", [])}
        for name, check in report.get("checks", {}).items() if check.get("status") != "PASS"
    ]
    result = {"validation_id": validation_id, "task_digest": source_digest,
              "environment": environment, "commands": checks, "formal_process": formal,
              "report_complete": complete, "overall_status": report.get("overall_status", "FAIL"),
              "diagnostics": diagnostics,
              "changed_files": changed + added_code, "workspace": str(workspace),
              "ok": bool(command_ok and formal["ok"] and complete and unchanged and
                         report.get("overall_status") == "PASS")}
    (root / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _main():
    from agents.task_validator.launch_agent import launch_agent
    request = json.loads(Path(sys.argv[1]).read_text())
    config = Config(**request["config"])
    workspace = Path(request["workspace"])
    logging.basicConfig(level=logging.INFO)
    from src.task_spec import load_task_spec
    from src.task_session import TaskSession
    from src.task_run import validate_task_session
    spec = load_task_spec(workspace / "config.yaml", task_id=request["task_id"])
    session = TaskSession.create(spec, workspace, workspace.parent / "session")
    report = validate_task_session(
        session,
        eval_config={"target_gpu_model": config.target_gpu_model,
                     "agent": {"template": "task_validator", **config.validator}},
        task_config_dir=str(workspace / "config.yaml"), agent_launcher=launch_agent,
        validation_request_id=request["validation_id"],
    )
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
