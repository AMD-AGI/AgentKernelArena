"""Host-owned clean action executor for serving tasks.

The agent has no Docker socket. This narrow service accepts declared actions,
reconstructs task inputs from a trusted template and copies only candidate
files. Every action runs in a fresh pinned container without agent credentials.
It is an integrity boundary, not a sandbox for hostile privileged processes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import socketserver
import subprocess
import time
import signal
import re
import math
import uuid

import yaml

from .measurement import measurement_kind
from .task_spec import load_task_spec, resolve_task_path
from .tasks import get_task_config


def validate_runtime_lock(lock: dict) -> None:
    if type(lock.get("version")) is not int or lock["version"] != 1:
        raise ValueError("Unsupported serving runtime lock version")
    for key in ("gpu_count", "minimum_final_evaluation_s"):
        if type(lock.get(key)) is not int or lock[key] <= 0:
            raise ValueError(f"Runtime lock requires a positive integer {key}")
    dependencies = lock.get("dependencies")
    if not isinstance(dependencies, dict):
        raise ValueError("Runtime lock dependencies must be a mapping")
    for name, source in [("model", lock.get("model")), *dependencies.items()]:
        revision = source.get("revision") if isinstance(source, dict) else None
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"Runtime source {name} requires a fixed 40-character commit revision")


def selected_serving_tasks(config_path: Path, root: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    tasks = {}
    for selector in config.get("tasks", []):
        tasks.update(get_task_config(str(root / "tasks"), None if selector == "all" else selector))
    selected = {}
    for task_id, path in tasks.items():
        data = yaml.safe_load(Path(path).read_text())
        if measurement_kind(data) != "serving":
            continue
        spec = load_task_spec(Path(path), task_id=task_id)
        lock_path = resolve_task_path(Path(path).parent, data["evaluation"]["measurement"]["runtime_lock"], must_exist=True)
        lock = json.loads(lock_path.read_text())
        validate_runtime_lock(lock)
        image = lock.get("image", "")
        if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
            raise ValueError("Serving runtime image must use an immutable registry digest")
        selected[task_id] = (Path(path).parent, spec, lock)
    if selected and len(selected) != len(tasks):
        raise ValueError("Run serving tasks separately from kernel tasks")
    if len({v[2]["image"] for v in selected.values()}) > 1:
        raise ValueError("Selected serving tasks require different runtime images; split the experiment")
    return selected


def gpu_groups(config: dict, available: list[str]) -> list[list[str]]:
    """Group indices address the scheduler-provided pool, never other GPUs."""
    resources = config.get("resources", {})
    if not isinstance(resources, dict) or set(resources) - {"gpu_groups"}:
        raise ValueError("resources only supports gpu_groups")
    groups = resources.get("gpu_groups", [[i] for i in range(len(available))])
    if not isinstance(groups, list) or not groups:
        raise ValueError("gpu_groups must be nonempty")
    used = set()
    for group in groups:
        if not isinstance(group, list) or not group:
            raise ValueError("GPU groups must be nonempty lists")
        for index in group:
            if type(index) is not int or index < 0 or index >= len(available) or index in used:
                raise ValueError("GPU groups overlap or escape the allocated device pool")
            used.add(index)
    return [[available[i] for i in group] for group in groups]


def action_client(*, task_id: str, workspace: Path, role: str, action: str,
                  phase: str, timeout: float) -> dict:
    endpoint = os.environ.get("AKA_SERVING_SOCKET")
    if not endpoint:
        raise RuntimeError("Serving actions require the host-owned clean runtime service")
    request = {"task_id": task_id, "workspace": str(workspace), "role": role,
               "action": action, "phase": phase, "timeout": timeout}
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(timeout + 30)
        client.connect(endpoint)
        client.sendall((json.dumps(request) + "\n").encode())
        response = json.loads(client.makefile("rb").readline(64 * 1024 * 1024))
    if "error" in response:
        raise RuntimeError(response["error"])
    return response


class CleanExecutor:
    def __init__(self, tasks: dict, root: Path, artifacts: Path, devices: str, model_cache: Path):
        self.root, self.artifacts, self.devices, self.model_cache = root.resolve(), artifacts, devices, model_cache
        self.tasks = {}
        self.active_container = None
        if artifacts.resolve().is_relative_to(self.root):
            raise ValueError("Trusted serving artifacts must be outside the agent-mounted checkout")
        artifacts.mkdir(parents=True, exist_ok=True)
        from .perf_helper_materialization import materialize_perf_helpers_in_workspace
        # Freeze before launching any optimization process.
        for task_id, (source, spec, lock) in tasks.items():
            if lock["gpu_count"] != len(devices.split(",")):
                raise ValueError("Allocated GPU group does not match the locked workload")
            template = artifacts / "templates" / task_id
            shutil.copytree(source, template)
            materialize_perf_helpers_in_workspace(template, root=self.root)
            self.tasks[task_id] = template, spec, lock

    def execute(self, request: dict) -> dict:
        template, spec, lock = self.tasks[request["task_id"]]
        role, action, phase = request["role"], request["action"], request["phase"]
        if phase not in {"task_validation", "candidate_evaluation"}:
            raise ValueError("Invalid evaluation phase")
        declared = spec.action(role, action)
        if len(declared.commands) != 1:
            raise ValueError("Clean serving actions require one task-owned runner command")
        requested_timeout = float(request["timeout"])
        if not math.isfinite(requested_timeout):
            raise ValueError("Action timeout must be finite")
        timeout = min(requested_timeout, declared.timeout_s)
        if timeout <= 0:
            raise TimeoutError("Serving action deadline exhausted")
        supplied = Path(request["workspace"])
        # Ordinary Docker workers mount this checkout at /workspace.
        relative = supplied.relative_to("/workspace")
        source = (self.root / relative).resolve(strict=True)
        if not source.is_relative_to(self.root) or source.is_relative_to(self.root / "tasks"):
            raise ValueError("Action workspace must be an experiment directory inside the checkout")
        # A resumed session must still refer to the same trusted task bytes.
        # Baseline requests carry the original frozen workspace, not candidate code.
        if role == "baseline":
            for path in template.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    original = source / path.relative_to(template)
                    if not original.is_file() or original.read_bytes() != path.read_bytes():
                        raise ValueError("Frozen baseline differs from the current trusted template; use a new experiment")
        action_id = uuid.uuid4().hex
        directory = self.artifacts / action_id
        shutil.copytree(template, directory)
        if phase == "candidate_evaluation" and role == "candidate":
            from .harness_guard import snapshot_workspace_harness, verify_workspace_harness
            snapshot = snapshot_workspace_harness(directory, task_spec=spec)
            for edit in spec.candidate.editable:
                if edit.scope == "tree":
                    raise ValueError("Serving source submission requires explicit file/symbol boundaries")
                src = resolve_task_path(source, edit.path, must_exist=True)
                dst = resolve_task_path(directory, edit.path)
                if not src.is_file() or src.stat().st_size > 16 * 1024 * 1024:
                    raise ValueError("Invalid candidate source file")
                dst.write_bytes(src.read_bytes())
            verify_workspace_harness(snapshot, discard_added=False)
        from .prepare_serving_runtime import verify
        identity = {key: lock[key] for key in ("model", "dependencies")}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        runtime = self.model_cache / key
        verify(runtime, identity)
        model = runtime / "model"
        container_name = "aka-eval-" + action_id
        command = ["docker", "run", "--rm", "--init", "--name", container_name,
                   "--network=host", "--device=/dev/kfd", "--device=/dev/dri",
                   "--group-add=video", "--security-opt=seccomp=unconfined", "--shm-size=16g",
                   "--tmpfs", f"/tmp/aiter_configs:rw,uid={os.getuid()},gid={os.getgid()},mode=1777",
                   "--user", f"{os.getuid()}:{os.getgid()}",
                   "-v", f"{directory}:/task", "-v", f"{model}:/task/model:ro",
                   "-v", f"{runtime}:/task/runtime:ro",
                   "-w", "/task", "-e", "HOME=/tmp", "-e", "HF_HUB_OFFLINE=1",
                   "-e", "USER=aka-worker", "-e", "LOGNAME=aka-worker",
                   "-e", "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
                   "-e", "TORCH_EXTENSIONS_DIR=/tmp/extensions",
                   "-e", "TRITON_CACHE_DIR=/tmp/triton", "-e", "AITER_JIT_DIR=/tmp/aiter",
                   "-e", "FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl", "-e", "PYTHONPYCACHEPREFIX=/tmp/pycache",
                   "-e", f"ROCR_VISIBLE_DEVICES={self.devices}",
                   "-e", "HIP_VISIBLE_DEVICES=" + ",".join(map(str, range(lock["gpu_count"]))),
                   "-e", f"ARENA_EVAL_PHASE={phase}", lock["image"], *declared.commands[0]]
        device_groups = {path.stat().st_gid for path in [Path("/dev/kfd"), *Path("/dev/dri").glob("render*")] if path.exists()}
        for group in sorted(device_groups):
            command[2:2] = ["--group-add", str(group)]
        self.active_container = container_name
        started = time.monotonic()
        try:
            stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                try:
                    result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout)
                    returncode = result.returncode
                except subprocess.TimeoutExpired:
                    returncode = -signal.SIGKILL
                    stderr.write("\nServing action exceeded its deadline\n")
            response = {"argv": list(declared.commands[0]), "returncode": returncode,
                        "stdout": stdout_path.read_text(errors="replace"), "stderr": stderr_path.read_text(errors="replace"),
                        "elapsed_s": time.monotonic() - started}
            (directory / "command_evidence.json").write_text(json.dumps(response, indent=2))
            return response
        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=30)
            self.active_container = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["image", "groups", "serve"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.mode == "groups":
        for group in gpu_groups(config, args.devices.split(",")):
            print(",".join(group))
        return
    tasks = selected_serving_tasks(args.config, args.root)
    if args.mode == "image":
        if tasks:
            print(next(iter(tasks.values()))[2]["image"])
        return
    cache = Path(os.environ.get("AKA_SERVING_CACHE", str(Path.home() / ".cache/aka-serving")))
    executor = CleanExecutor(tasks, args.root, args.artifacts, args.devices, cache)

    def terminate(signum, frame):
        if executor.active_container:
            subprocess.run(["docker", "rm", "-f", executor.active_container], capture_output=True, timeout=30)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                request = json.loads(self.rfile.readline(1024 * 1024))
                response = executor.execute(request)
            except Exception as exc:
                response = {"error": f"{type(exc).__name__}: {exc}"}
            self.wfile.write((json.dumps(response) + "\n").encode())

    with socketserver.UnixStreamServer(str(args.socket), Handler) as server:
        os.chmod(args.socket, 0o600)
        server.serve_forever()


if __name__ == "__main__":
    main()
