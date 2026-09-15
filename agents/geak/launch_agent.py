"""One schema-v2 integration using GEAK's upstream multi-agent Workflow engine."""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

import yaml

from agents import register_agent
from src.harness_guard import snapshot_workspace_harness
from src.task_execution import _run_process
from src.task_spec import load_task_spec
from .bridge import (Bridge, TaskContext, candidate_files, copy_task, digests,
                     remaining_budget, snapshot_mapping, write_json)
from .compatibility import prepare_engine


def load_options(eval_config: dict) -> dict:
    options = yaml.safe_load(Path(__file__).with_name("agent_config.yaml").read_text())
    overrides = eval_config.get("agent", {})
    if not isinstance(overrides, dict):
        raise ValueError("agent must be a mapping")
    for key in options:
        if key in overrides:
            options[key] = overrides[key]
    for key in ("budget", "deep_cost", "timeout_seconds"):
        if type(options[key]) is not int or options[key] <= 0:
            raise ValueError(f"agent.{key} must be a positive integer")
    value = options["min_improve"]
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("agent.min_improve must be finite and nonnegative")
    if options["model"] is not None and (not isinstance(options["model"], str) or not options["model"].strip()):
        raise ValueError("agent.model must be null or a nonempty string")
    if options["effort"] not in {"low", "medium", "high", "xhigh", "max", "ultracode"}:
        raise ValueError("Unsupported GEAK effort")
    options["gpu_ids"] = logical_gpu_ids(eval_config)
    return options


def logical_gpu_ids(eval_config: dict) -> str:
    if os.environ.get("AGENT_KERNEL_ARENA_HOST_GPU_ID") is not None:
        return "0"
    for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        if os.environ.get(key):
            return ",".join(str(i) for i, _ in enumerate(os.environ[key].split(",")))
    value = eval_config.get("gpu_ids", "0")
    text = ",".join(str(item) for item in value) if isinstance(value, (tuple, list)) else str(value)
    if not text or any(not part.isdigit() for part in text.split(",")):
        raise ValueError("GEAK requires process-visible numeric GPU IDs")
    return text


def _descendants(pid: int) -> set[int]:
    children: dict[int, set[int]] = {}
    for entry in Path("/proc").glob("[0-9]*/stat"):
        try:
            # comm can contain whitespace and parentheses; ppid follows the final ')'.
            fields = entry.read_text().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), set()).add(int(entry.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    found, queue = set(), [pid]
    while queue:
        for child in children.get(queue.pop(), set()) - found:
            found.add(child)
            queue.append(child)
    return found


def _run_engine(bridge: Bridge, python: str) -> int:
    worker = Path(__file__).with_name("engine_worker.py")
    # Authentication is inherited, never serialized into argv, prompts or job data.
    with subprocess.Popen([python, str(worker), str(bridge.job_path)], cwd=bridge.root,
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, start_new_session=True) as process:
        try:
            return process.wait(timeout=bridge.remaining())
        finally:
            # Public runners can start new sessions; kill descendants as well as
            # the SDK process group when the hard deadline terminates the worker.
            descendants = _descendants(process.pid)
            for pid in descendants:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def prepare_job(context: TaskContext, run_dir: Path, options: dict, *,
                deadline: float, deadline_epoch: float) -> Bridge:
    remaining_budget(deadline)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "context.json", context.raw)
    job = {"version": 1, "context": str(run_dir / "context.json"), "options": options,
           "deadline_monotonic": deadline, "deadline_epoch": int(deadline_epoch),
           "harness": snapshot_mapping(snapshot_workspace_harness(context.workspace, task_spec=context.spec)),
           "baseline_harness": snapshot_mapping(snapshot_workspace_harness(context.baseline, task_spec=context.spec)),
           "original_sources": digests(candidate_files(context.spec, context.workspace)),
           "baseline_sources": digests(candidate_files(context.spec, context.baseline))}
    job_path = run_dir / "job.json"
    write_json(job_path, job)
    bridge = Bridge(job_path)
    bridge.remaining()
    bridge.eval_dir.mkdir()
    copy_task(context.workspace, bridge.eval_dir / "original", remaining=bridge.remaining)
    copy_task(context.workspace, bridge.eval_dir / "workspace", remaining=bridge.remaining)
    # GEAK's engineers exchange git patches in private copies. Do not initialize
    # a repository in the Arena workspace or touch the user's global git config.
    private = bridge.eval_dir / "workspace"
    for args in (("init", "-q"), ("add", "-A"),
                 ("-c", "user.name=Arena GEAK", "-c", "user.email=geak@arena.invalid",
                  "-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty", "-m", "Arena initial candidate")):
        result = _run_process(("git", *args), private, os.environ.copy(), bridge.remaining())
        if result.returncode:
            raise RuntimeError("Cannot initialize GEAK's private candidate repository")
    bridge.action("baseline", "compile")
    job["baseline_performance"] = bridge.action("baseline", "performance").to_mapping()
    write_json(job_path, job)
    bridge.remaining()
    return Bridge(job_path)


@register_agent("geak")
def launch_agent(eval_config: dict, task_config_dir: str, workspace: str) -> str:
    options = load_options(eval_config)
    started = time.monotonic()
    deadline = started + options["timeout_seconds"]
    deadline_epoch = time.time() + options["timeout_seconds"]
    context_path = os.environ.get("ARENA_TASK_CONTEXT")
    if not context_path:
        raise ValueError("GEAK requires the framework's ARENA_TASK_CONTEXT")
    context = TaskContext.load(Path(context_path))
    if context.workspace != Path(workspace).resolve(strict=True):
        raise ValueError("GEAK workspace does not match ARENA_TASK_CONTEXT")
    if load_task_spec(Path(task_config_dir), task_id=context.spec.task_id).to_mapping() != context.spec.to_mapping():
        raise ValueError("GEAK task config does not match ARENA_TASK_CONTEXT")
    checkout_value = os.environ.get("GEAK_HOME")
    if not checkout_value and os.environ.get("GEAK_V4_WORKFLOW_DIR"):
        checkout_value = str(Path(os.environ["GEAK_V4_WORKFLOW_DIR"]).parent)
    if not checkout_value:
        raise ValueError("Set GEAK_HOME to the pinned upstream checkout")
    checkout = Path(checkout_value).resolve(strict=True)
    python = os.environ.get("GEAK_PYTHON") or sys.executable
    options["claude_cli_path"] = os.environ.get("GEAK_CLAUDE_BIN") or shutil.which("claude")
    if not options["claude_cli_path"]:
        raise RuntimeError("GEAK requires Claude Code with the dynamic Workflow tool")
    from .compatibility import verify_upstream

    verify_upstream(checkout, remaining=lambda: remaining_budget(deadline))
    run_dir = context.workspace.parent / ("." + context.workspace.name + "_geak") / uuid.uuid4().hex
    logging.getLogger(__name__).info("GEAK artifacts: %s", run_dir)
    status = {"status": "FAILED", "delivery": "NOT_ATTEMPTED", "arena_acceptance": "PENDING"}
    try:
        bridge = prepare_job(context, run_dir, options, deadline=deadline, deadline_epoch=deadline_epoch)
        bridge.job["engine"] = prepare_engine(checkout, bridge, python=python, options=options)
        write_json(bridge.job_path, bridge.job)
        code = _run_engine(bridge, python)
        result_path = run_dir / "engine_result.json"
        raw_engine = json.loads(result_path.read_text()) if result_path.is_file() else {}
        engine = {"status": raw_engine.get("status") if raw_engine.get("status") in
                  {"accepted", "flagged", "author_failed", "no_baseline", "FAILED"} else "MISSING",
                  "workflow_completed": raw_engine.get("workflow_completed") is True}
        runtime = raw_engine.get("runtime")
        if isinstance(runtime, dict):
            engine["runtime"] = {key: runtime[key] for key in (
                "requested_model", "sdk_version", "cli_version", "init_model",
                "assistant_models", "workflow_models") if key in runtime}
        status["engine"] = engine
        # A full canonical candidate includes the author seed, unlike a patch
        # relative to its first commit. Preserve correct no-gain implementations.
        status["delivery"] = "CHECKING"
        delivery = bridge.deliver(bridge.eval_dir / "workspace")
        status.update(delivery="DELIVERED", candidate=delivery)
        if code != 0 or engine.get("status") != "accepted" or engine.get("workflow_completed") is not True:
            raise RuntimeError("GEAK engine failed; any checked candidate delivery is recorded separately")
        status["status"] = "COMPLETED"
        return json.dumps(status, sort_keys=True)
    except Exception as exc:
        status["error_type"] = type(exc).__name__
        if status["delivery"] == "CHECKING":
            status["delivery"] = "FAILED"
        raise RuntimeError(f"GEAK failed ({type(exc).__name__}); see its delivery.json artifact") from None
    finally:
        if run_dir.is_dir():
            status["elapsed_s"] = time.monotonic() - started
            try:
                status["retained_workspace_sources"] = digests(candidate_files(context.spec, context.workspace))
            except Exception as exc:
                status["retained_source_error_type"] = type(exc).__name__
            write_json(run_dir / "delivery.json", status)
