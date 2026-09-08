"""Real runtime checks, formal task validation and content-bound evidence."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
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
    return run_process([sys.executable, "scripts/task_runner.py", "--mode", mode], workspace,
                       artifacts / f"{mode}.log", timeout or config.command_timeout,
                       build_subprocess_env())


def validate_task(draft: Path, artifacts: Path, config: Config, timeout=None) -> dict:
    environment = runtime_identity(config)
    validation_id = uuid.uuid4().hex
    root = (artifacts / validation_id).resolve()
    workspace = root / "workspace"
    source_digest = task_digest(draft)
    copy_task_files(draft, workspace)
    materialize_perf_helpers_in_workspace(workspace)
    runtime_files = task_tree(workspace)
    deadline = time.monotonic() + (timeout or config.max_task_seconds)
    checks = {}
    # Independent command outcomes cannot be replaced by a model-written PASS.
    for mode in ("source-check", "compile", "correctness", "performance"):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            checks[mode] = {"ok": False, "timed_out": True}
            break
        checks[mode] = run_process(
            [sys.executable, "scripts/task_runner.py", "--mode", mode], workspace,
            root / f"{mode}.log", min(config.command_timeout, remaining), build_subprocess_env(),
        )
        if not checks[mode]["ok"]:
            break
    command_ok = len(checks) == 4 and all(c["ok"] for c in checks.values())
    formal = {"ok": False}
    if command_ok and deadline > time.monotonic():
        # Use a subprocess to enforce the campaign deadline even if the formal
        # validator expands its own backend timeout to cover command budgets.
        request = root / "validator_request.json"
        request.write_text(json.dumps({"workspace": str(workspace), "config": config.mapping()}))
        formal = run_process(
            [sys.executable, "-m", "agents.sikl_task_builder.validation", str(request)],
            Path(__file__).resolve().parents[2], root / "validator.log",
            deadline - time.monotonic(), build_subprocess_env(),
        )
    from agents.task_validator.report_schema import validation_report_is_complete
    complete = validation_report_is_complete(workspace)
    report_path = workspace / "validation_report.yaml"
    report = yaml.safe_load(report_path.read_text()) if complete else {}
    after = task_tree(workspace)
    changed = [p for p, digest in runtime_files.items() if after.get(p) != digest]
    added_code = [p for p in after if p not in runtime_files and
                  (p.startswith(("scripts/", "source/")) or p.endswith((".py", ".so", ".pth")))]
    unchanged = not changed and not added_code and task_digest(draft) == source_digest
    result = {"validation_id": validation_id, "task_digest": source_digest,
              "environment": environment, "commands": checks, "formal_process": formal,
              "report_complete": complete, "overall_status": report.get("overall_status", "FAIL"),
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
    launch_agent({"target_gpu_model": config.target_gpu_model,
                  "agent": {"template": "task_validator", **config.validator}},
                 str(workspace / "config.yaml"), str(workspace))


if __name__ == "__main__":
    _main()
