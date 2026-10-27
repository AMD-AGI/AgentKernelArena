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
from pathlib import Path, PurePosixPath
import shutil
import socket
import socketserver
import select
import subprocess
import time
import signal
import re
import math
import uuid

import yaml

from .measurement import lock_limits, measurement_kind
from .task_spec import load_task_spec, resolve_task_path
from .tasks import get_task_config

_CLEANUP_TIMEOUT_S = 90.0
_CLEANUP_POLL_S = 0.25


def _remove_action_container(name: str, evidence: Path) -> dict:
    """Bound removal retries and prove absence even if the rm client failed.

    Killing the attached Docker client does not synchronously remove its
    container. A timed-out removal command may still have taken effect; only
    a successful, empty exact-name query establishes that another GPU action
    may start. Unknown or lingering state remains a failure.
    """
    started = time.monotonic()
    deadline = started + _CLEANUP_TIMEOUT_S
    proof = {"container": name, "status": "IN_PROGRESS", "attempts": []}
    def save_proof():
        # The action container can write its task mount. Never follow a link
        # it may have left at the host-owned evidence filename.
        temporary = evidence.with_name(evidence.name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("x") as handle:
            json.dump(proof, handle, indent=2)
            handle.write("\n")
        temporary.replace(evidence)
    while not proof["attempts"] or time.monotonic() < deadline:
        attempt = {}
        for action, argv in (
            ("remove", ["docker", "rm", "-f", name]),
            ("inspect", ["docker", "ps", "-aq", "--filter", f"name=^/{name}$"]),
        ):
            try:
                result = subprocess.run(argv, capture_output=True, text=True,
                    timeout=min(30 if action == "remove" else 5,
                                max(0.1, deadline - time.monotonic())))
                attempt[action] = {"returncode": result.returncode,
                    "stdout": result.stdout[-1024:], "stderr": result.stderr[-1024:]}
            except (OSError, subprocess.SubprocessError) as exc:
                attempt[action] = {"error": f"{type(exc).__name__}: {exc}"}
        proof["attempts"].append(attempt)
        inspected = attempt["inspect"]
        if inspected.get("returncode") == 0 and not inspected["stdout"].strip():
            proof["status"] = "PASS"
        proof["elapsed_s"] = time.monotonic() - started
        save_proof()
        if proof["status"] == "PASS":
            return proof
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(_CLEANUP_POLL_S, remaining))
    proof["status"] = "FAIL"
    proof["elapsed_s"] = time.monotonic() - started
    save_proof()
    raise RuntimeError("Serving container absence could not be established")


def validate_runtime_lock(lock: dict) -> None:
    if type(lock.get("version")) is not int or lock["version"] != 1:
        raise ValueError("Unsupported serving runtime lock version")
    for key in ("gpu_count", "minimum_final_evaluation_s"):
        if type(lock.get(key)) is not int or lock[key] <= 0:
            raise ValueError(f"Runtime lock requires a positive integer {key}")
    if "server_seed" in lock and (type(lock["server_seed"]) is not int or not 0 <= lock["server_seed"] < 2**31):
        raise ValueError("Runtime server_seed must be a nonnegative 31-bit integer")
    dependencies = lock.get("dependencies")
    if not isinstance(dependencies, dict):
        raise ValueError("Runtime lock dependencies must be a mapping")
    for name, source in [("model", lock.get("model")), *dependencies.items()]:
        revision = source.get("revision") if isinstance(source, dict) else None
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"Runtime source {name} requires a fixed 40-character commit revision")
    lock_limits(lock)


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
               "action": action, "phase": phase, "timeout": timeout,
               "deadline_epoch": time.time() + timeout}
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(timeout + 30)
        client.connect(endpoint)
        client.sendall((json.dumps(request) + "\n").encode())
        response = json.loads(client.makefile("rb").readline(64 * 1024 * 1024))
    if "error" in response:
        raise RuntimeError(response["error"])
    return response


def runtime_mounts(runtime: Path, lock: dict) -> list[str]:
    """Mount verified host assets directly, including shared-directory symlinks.

    A symlink inside a bind mount retains its host absolute target, which need
    not exist inside the container. Mount only the declared, verified asset
    roots at their task paths; never expose their host parent directories.
    """
    mounts = ["-v", f"{(runtime/'model').resolve(strict=True)}:/task/model:ro"]
    for name in lock['dependencies']:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError('Invalid runtime dependency mount name')
        source = (runtime/name).resolve(strict=True)
        if not source.is_dir():
            raise ValueError('Runtime dependency must be a directory')
        mounts.extend(["-v", f"{source}:/task/runtime/{name}:ro"])
    if lock.get('assets'):
        mounts.extend(['-v', f"{(runtime/'assets').resolve(strict=True)}:/task/assets:ro"])
    return mounts


class CleanExecutor:
    def __init__(self, tasks: dict, root: Path, artifacts: Path, devices: str, model_cache: Path,
                 *, container_root: str = "/workspace"):
        self.root, self.artifacts, self.devices, self.model_cache = root.resolve(), artifacts, devices, model_cache
        self.container_root = PurePosixPath(container_root)
        if not self.container_root.is_absolute() or ".." in self.container_root.parts:
            raise ValueError("Container checkout root must be an absolute path without traversal")
        self.tasks = {}
        self.active_container = None
        self.cleanup_error = None
        from .prepare_serving_runtime import VerifiedRuntime
        self.runtime_verifier = VerifiedRuntime()
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

    def execute(self, request: dict, *, cancelled=lambda: False) -> dict:
        if self.cleanup_error:
            raise RuntimeError(self.cleanup_error)
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
        deadline = request.get("deadline_epoch", time.time() + timeout)
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError("Action deadline must be finite")
        timeout = min(timeout, deadline - time.time())
        if timeout <= 0:
            raise TimeoutError("Serving action deadline exhausted")
        started = time.monotonic()
        def check_request():
            if cancelled():
                raise ConnectionAbortedError("Serving action caller disconnected")
            if time.monotonic() - started >= timeout:
                raise TimeoutError("Serving action deadline exhausted")
        check_request()
        supplied = PurePosixPath(request["workspace"])
        # Only the trusted host launcher selects the worker's checkout mount.
        if ".." in supplied.parts:
            raise ValueError("Action workspace must not contain path traversal")
        relative = supplied.relative_to(self.container_root)
        if not relative.parts or relative.parts[0] == "tasks":
            raise ValueError("Action workspace must be an experiment directory inside the checkout")
        action_id = uuid.uuid4().hex
        directory = self.artifacts / action_id
        from .serving_snapshot import workspace_at, read_file_at, read_tree_at
        with workspace_at(self.root, relative) as source_fd:
            # Compare/read through anchored descriptors, including all ancestors.
            if role == "baseline":
                for path in template.rglob("*"):
                    if path.is_file() and "__pycache__" not in path.parts:
                        expected = path.read_bytes()
                        original = read_file_at(source_fd, path.relative_to(template), max_bytes=len(expected))
                        if original != expected:
                            raise ValueError("Frozen baseline differs from the current trusted template; use a new experiment")
            shutil.copytree(template, directory)
            if phase == "candidate_evaluation" and role == "candidate":
                from .harness_guard import snapshot_workspace_harness, verify_workspace_harness
                snapshot = snapshot_workspace_harness(directory, task_spec=spec)
                limit = lock_limits(lock)["max_candidate_source_bytes"]
                for edit in spec.candidate.editable:
                    check_request()
                    dst = resolve_task_path(directory, edit.path)
                    if edit.scope == "tree":
                        if edit.path != lock.get("candidate_bundle", {}).get("root"):
                            raise ValueError("Serving trees must be declared as the kernel bundle root")
                        files = read_tree_at(source_fd, edit.path, max_bytes=limit)
                        shutil.rmtree(dst)
                        dst.mkdir()
                        for name, data in files.items():
                            target = dst / name
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(data)
                        # Enforce the same language/file contract on the frozen
                        # tree; no subsequent read reopens the untrusted source.
                        from .tools.perf.serving_bundle import source_files
                        source_files(dst, max_bytes=limit)
                    else:
                        dst.write_bytes(read_file_at(source_fd, edit.path, max_bytes=limit))
                verify_workspace_harness(snapshot, discard_added=False)
        check_request()
        from .prepare_serving_runtime import asset_identity
        identity = asset_identity(lock)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        runtime = self.model_cache / key
        self.runtime_verifier.verify(runtime, identity)
        check_request()
        container_name = "aka-eval-" + action_id
        command = ["docker", "run", "--rm", "--init", "--name", container_name,
                   "--network=host", "--device=/dev/kfd", "--device=/dev/dri",
                   "--group-add=video", "--security-opt=seccomp=unconfined", "--shm-size=16g",
                   "--tmpfs", f"/tmp/aiter_configs:rw,uid={os.getuid()},gid={os.getgid()},mode=1777",
                   "--user", f"{os.getuid()}:{os.getgid()}",
                   "-v", f"{directory}:/task", *runtime_mounts(runtime, lock),
                   "-w", "/task", "-e", "HOME=/tmp", "-e", "HF_HUB_OFFLINE=1",
                   "-e", "USER=aka-worker", "-e", "LOGNAME=aka-worker",
                   "-e", "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
                   "-e", "TORCH_EXTENSIONS_DIR=/tmp/extensions",
                   "-e", "TRITON_CACHE_DIR=/tmp/triton", "-e", "AITER_JIT_DIR=/tmp/aiter",
                   "-e", "FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl", "-e", "PYTHONPYCACHEPREFIX=/tmp/pycache",
                   "-e", f"ROCR_VISIBLE_DEVICES={self.devices}",
                   "-e", "HIP_VISIBLE_DEVICES=" + ",".join(map(str, range(lock["gpu_count"]))),
                   "-e", f"ARENA_EVAL_PHASE={phase}",
                   "--entrypoint", declared.commands[0][0], lock["image"], *declared.commands[0][1:]]
        device_groups = {path.stat().st_gid for path in [Path("/dev/kfd"), *Path("/dev/dri").glob("render*")] if path.exists()}
        for group in sorted(device_groups):
            command[2:2] = ["--group-add", str(group)]
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Serving action deadline exhausted while verifying runtime assets")
        self.active_container = container_name
        try:
            stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                try:
                    with subprocess.Popen(command, stdout=stdout, stderr=stderr) as process:
                        try:
                            while process.poll() is None:
                                check_request()
                                try:
                                    process.wait(timeout=min(0.2, max(0.01, timeout - (time.monotonic() - started))))
                                except subprocess.TimeoutExpired:
                                    pass
                            returncode = process.returncode
                        except (TimeoutError, ConnectionAbortedError) as exc:
                            process.kill()
                            process.wait()
                            returncode = -signal.SIGKILL
                            stderr.write(f"\n{exc}\n")
                        except BaseException:
                            process.kill()
                            process.wait()
                            raise
                except OSError as exc:
                    returncode = 127
                    stderr.write(f"\nUnable to launch serving action: {exc}\n")
            response = {"argv": list(declared.commands[0]), "returncode": returncode,
                        "stdout": stdout_path.read_text(errors="replace"), "stderr": stderr_path.read_text(errors="replace"),
                        "elapsed_s": time.monotonic() - started}
            (directory / "command_evidence.json").write_text(json.dumps(response, indent=2))
            return response
        finally:
            try:
                _remove_action_container(container_name, directory / "cleanup_evidence.json")
                self.active_container = None
            except Exception as exc:
                self.cleanup_error = "Serving cleanup failed; refusing further GPU actions"
                raise RuntimeError(self.cleanup_error) from exc


def request_handler(executor):
    """Serialize GPU actions while cancelling work whose socket owner has exited."""
    class Handler(socketserver.StreamRequestHandler):
        def disconnected(self):
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                return bool(readable) and not self.connection.recv(1, socket.MSG_PEEK)
            except OSError:
                return True

        def handle(self):
            self.connection.settimeout(30)
            try:
                request = json.loads(self.rfile.readline(1024 * 1024))
                response = executor.execute(request, cancelled=self.disconnected)
            except Exception as exc:
                response = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                self.wfile.write((json.dumps(response) + "\n").encode())
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["config-image", "image", "groups", "verifier", "serve"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--container-root", default="/workspace")
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text()) or {}
    if not isinstance(config, dict):
        raise ValueError("Run config must be a mapping")
    if args.mode == "config-image":
        if "docker_image" in config:
            image = config["docker_image"]
            if not isinstance(image, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@:-]*", image):
                raise ValueError("docker_image must be a nonempty Docker image reference")
            print(image)
        return
    if args.mode == "groups":
        for group in gpu_groups(config, args.devices.split(",")):
            print(",".join(group))
        return
    tasks = selected_serving_tasks(args.config, args.root)
    if args.mode == 'verifier':
        from .candidate_verification import policy
        if config.get('agent', {}).get('template') != 'task_validator':
            review = policy(config, serving=bool(tasks))
            if review:
                print(review['backend'])
        return
    if args.mode == "image":
        if tasks:
            print(next(iter(tasks.values()))[2]["image"])
        return
    cache = Path(os.environ.get("AKA_SERVING_CACHE", str(Path.home() / ".cache/aka-serving")))
    executor = CleanExecutor(tasks, args.root, args.artifacts, args.devices, cache,
                             container_root=args.container_root)

    def terminate(signum, frame):
        if executor.active_container:
            subprocess.run(["docker", "rm", "-f", executor.active_container], capture_output=True, timeout=30)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    with socketserver.UnixStreamServer(str(args.socket), request_handler(executor)) as server:
        os.chmod(args.socket, 0o600)
        server.serve_forever()


if __name__ == "__main__":
    main()
