"""Run native head-kernel checks without an agent, inside the Docker runner.

These artifacts are direct execution evidence, never framework validation or
baseline-versus-candidate scores. Task commands own correctness and complete
benchmark-case validation; this coordinator preserves their contracts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time

import yaml

from src.perf_helper_materialization import materialize_perf_helpers_in_workspace
from src.preprocessing import _validate_task_symlinks
from src.scripts.top5_head_kernels import REPO_ROOT, plan_run
from src.tools.runtime_image_identity import verify_identity

PHASES = ("compile", "correctness", "performance")
FRAMEWORK = {"framework_task_validator": "NOT_RUN", "framework_PASS_claimed": False}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def copy_task(source: Path, destination: Path, repo: Path) -> dict:
    """Keep editable source aliases bound to the copied source, including on NFS."""
    _validate_task_symlinks(source, set())

    def ignore(directory, names):
        excluded = {name for name in names if name == "__pycache__" or name.endswith(".pyc")}
        if Path(directory) == source:
            excluded.update(set(names) & {"build", "validation_report.yaml", "task_result.yaml",
                                         "direct-native-reports",
                                         *(phase + "_report.json" for phase in PHASES),
                                         *(phase + ".stdout" for phase in PHASES),
                                         *(phase + ".stderr" for phase in PHASES)})
        return excluded

    shutil.copytree(source, destination, symlinks=True, copy_function=shutil.copy2, ignore=ignore)
    _validate_task_symlinks(destination, set())
    helpers = materialize_perf_helpers_in_workspace(destination, root=repo)
    identities = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.suffix != ".pt":
            identities[path.relative_to(destination).as_posix()] = {
                **fingerprint(path), **({"symlink": str(path.readlink())} if path.is_symlink() else {})}
    return {"files": identities,
            "materialized_perf_helpers": [p.relative_to(destination).as_posix() for p in helpers]}


def stop_process_tree(process: subprocess.Popen) -> None:
    """Also stop native workers that create their own sessions inside Docker."""
    parents = {}
    for status in Path("/proc").glob("[0-9]*/stat"):
        try:
            # comm can contain spaces and parentheses; fields after its final ')'
            # start with state, ppid. No host PID namespace is mounted by the runner.
            fields = status.read_text().rsplit(")", 1)[1].split()
            parents[int(status.parent.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    descendants = {process.pid}
    while True:
        children = {pid for pid, parent in parents.items() if parent in descendants}
        if children <= descendants:
            break
        descendants.update(children)
    for pid in descendants - {process.pid}:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_command(command: str, workspace: Path, stdout, stderr, timeout: float) -> dict:
    started = time.monotonic()
    process = subprocess.Popen(["bash", "-lc", command], cwd=workspace,
                               stdout=stdout, stderr=stderr, start_new_session=True)
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        # Also stop descendants after an interrupted coordinator or timed-out shell.
        if process.poll() is None or timed_out:
            stop_process_tree(process)
    return {"command": command, "returncode": process.returncode,
            "timed_out": timed_out, "elapsed_seconds": time.monotonic() - started}


def verify_task(source: Path, workspace: Path, repo: Path) -> dict:
    result = {"schema": "aka-direct-task-verification-v1", **FRAMEWORK,
              "task": source.relative_to(repo / "tasks").as_posix(),
              "workspace": workspace.relative_to(repo).as_posix(),
              "status": "preparing", "phases": [],
              "compile_scope": "Python AST and fixed ABI; device execution occurs in later phases"}
    report = workspace.parent / (workspace.name + ".direct.json")
    write_json(report, result)
    try:
        result["source_identity"] = copy_task(source, workspace, repo)
        config = yaml.safe_load((workspace / "config.yaml").read_text())
        for phase in PHASES:
            commands = config.get(phase + "_command")
            timeout = config.get(phase + "_timeout", 3600)
            if not isinstance(commands, list) or not commands or any(
                    not isinstance(command, str) or not command for command in commands):
                raise ValueError(f"{phase}_command must be a nonempty list of commands")
            if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                    or not math.isfinite(timeout) or timeout <= 0):
                raise ValueError(f"Invalid {phase}_timeout")
            entry = {"phase": phase, "status": "running", "timeout_seconds": timeout,
                     "commands": [], "started_utc": datetime.now(timezone.utc).isoformat(),
                     "stdout": phase + ".stdout", "stderr": phase + ".stderr"}
            result["phases"].append(entry)
            result["status"] = "running"
            write_json(report, result)
            deadline = time.monotonic() + timeout
            with (workspace / entry["stdout"]).open("xb") as stdout, \
                    (workspace / entry["stderr"]).open("xb") as stderr:
                for command in commands:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        entry["status"] = "timeout"
                        break
                    execution = run_command(command, workspace, stdout, stderr, remaining)
                    entry["commands"].append(execution)
                    if execution["timed_out"] or execution["returncode"] != 0:
                        entry["status"] = "timeout" if execution["timed_out"] else "command_failed"
                        break
            native = workspace / "build" / (phase + "_report.json")
            if native.is_file() and not native.is_symlink():
                retained = workspace / "direct-native-reports" / native.name
                retained.parent.mkdir(exist_ok=True)
                shutil.copy2(native, retained)
                entry["native_report"] = {"path": retained.relative_to(workspace).as_posix(),
                                          "original_path": native.relative_to(workspace).as_posix(),
                                          **fingerprint(retained)}
                try:
                    payload = json.loads(native.read_text())
                    entry["native_status"] = payload.get("status")
                except (ValueError, AttributeError):
                    entry["native_status"] = "invalid_json"
            if entry["status"] == "running":
                entry["status"] = ("native_phase_succeeded" if entry.get("native_status") == "ok"
                                   else "native_report_failed_or_missing")
            if entry["status"] != "native_phase_succeeded":
                result["status"] = entry["status"]
                write_json(report, result)
                return result
            write_json(report, result)
        result["status"] = "all_native_phases_succeeded"
    except Exception as error:
        result.update(status="verification_error", error_type=type(error).__name__, error=str(error))
    write_json(report, result)
    return result


def verify(config: Path, repo: Path = REPO_ROOT) -> tuple[int, Path]:
    repo = repo.resolve()
    plan = plan_run(config, repo)
    if os.environ.get("AGENT_KERNEL_ARENA_DOCKER") != "1":
        raise ValueError("Use top5_head_kernels.py verify through the standard Docker runner")
    image_id = os.environ.get("AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID", "")
    config_digest = os.environ.get("AGENT_KERNEL_ARENA_DOCKER_CONFIG_DIGEST", "")
    attestation = json.loads(os.environ.get("AGENT_KERNEL_ARENA_DOCKER_IDENTITY", "{}"))
    inspected_identity = verify_identity(plan["image"], plan["expected_image_id"], {
        "Id": image_id,
        "RepoDigests": json.loads(os.environ.get("AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS", "[]")),
        "Descriptor": attestation.get("descriptor"),
    })
    if (os.environ.get("AGENT_KERNEL_ARENA_DOCKER_IMAGE") != plan["image"]
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)
            or (plan["expected_image_id"] and plan["expected_image_id"] != config_digest)
            or (plan["expected_image_id"] and inspected_identity["verified_config_digest"] != config_digest)
            or os.environ.get("AGENT_KERNEL_ARENA_HEAD_KERNEL_VALIDATION_RUNTIME", "")
            != (plan.get("validation_runtime") or "")):
        raise ValueError("Selected Docker identity does not match the task runtime plan")
    run = Path(tempfile.mkdtemp(prefix="workspace_direct_verification_", dir=repo))
    identity = {name: os.environ.get(name) for name in (
        "AGENT_KERNEL_ARENA_DOCKER_IMAGE", "AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID",
        "AGENT_KERNEL_ARENA_DOCKER_CONFIG_DIGEST", "AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID_ROLE",
        "AGENT_KERNEL_ARENA_DOCKER_IDENTITY",
        "AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS", "AGENT_KERNEL_ARENA_GPU_ARCH",
        "AGENT_KERNEL_ARENA_HOST_GPU_ID", "AGENT_KERNEL_ARENA_HEAD_KERNEL_VALIDATION_RUNTIME",
        "AITER_JIT_DIR", "FLYDSL_RUNTIME_CACHE_DIR", "TVM_FFI_DISABLE_TORCH_C_DLPACK")}
    summary = {"schema": "aka-direct-verification-v1", **FRAMEWORK, "plan": plan,
               "runtime_identity": identity, "status": "running", "tasks": []}
    path = run / "direct-verification.json"
    write_json(path, summary)
    print(f"Direct verification evidence: {path.relative_to(repo)}", flush=True)
    for index, selector in enumerate(plan["tasks"]):
        source = (repo / "tasks" / selector).resolve()
        summary["tasks"].append(verify_task(source, run / f"{index:03d}-{source.name}", repo))
        write_json(path, summary)
    succeeded = all(task["status"] == "all_native_phases_succeeded" for task in summary["tasks"])
    summary["status"] = "all_native_phases_succeeded" if succeeded else "direct_verification_failed"
    write_json(path, summary)
    return (0 if succeeded else 1), path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_name", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        code, _ = verify(args.config_name)
        return code
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
