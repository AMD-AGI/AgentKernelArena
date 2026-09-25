"""Run upstream Apex and deliver only validated candidate source to Arena."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import yaml

from agents import register_agent
from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness

from .bundle import prepare_delivery, read_json
from .contract import build_task, load_context, regular_file
from .process import run_worker
from .runtime import runtime_environment


def load_options(eval_config: dict) -> dict:
    options = yaml.safe_load(Path(__file__).with_name("agent_config.yaml").read_text())
    overrides = {key: value for key, value in eval_config.get("agent", {}).items() if key != "template"}
    unknown = overrides.keys() - options.keys()
    if unknown:
        raise ValueError(f"Unsupported Apex options: {sorted(unknown)}")
    options.update(overrides)
    if options["backend"] not in {"codex", "claude", "cursor"}:
        raise ValueError("Apex backend must be codex, claude or cursor")
    for key in ("max_iterations", "max_turns", "timeout_seconds"):
        if type(options[key]) is not int or options[key] <= 0:
            raise ValueError(f"Apex {key} must be a positive integer")
    for key in ("model", "effort"):
        if options[key] is not None and (not isinstance(options[key], str) or not options[key].strip()):
            raise ValueError(f"Apex {key} must be a nonempty string or null")
    return options


def copy_workspace(source: Path, target: Path) -> None:
    # Reject links instead of following them outside the task package.
    count = size = 0
    for parent, directories, files in os.walk(source):
        directories[:] = [name for name in directories if name not in {".git", "__pycache__"}]
        for name in directories + files:
            if (Path(parent) / name).is_symlink():
                raise ValueError("Apex task copies cannot contain symlinks")
        for name in files:
            path = regular_file(source, (Path(parent) / name).relative_to(source).as_posix())
            count += 1
            size += path.stat().st_size
            if count > 20_000 or size > 2 * 1024**3:
                raise ValueError("Apex task copy exceeds the file or size limit")
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", "__pycache__"))


def initialize_snapshot(root: Path) -> None:
    # This local commit identifies a materialized task snapshot, not a published
    # source revision. No remote operation is performed.
    commands = [
        ["init", "-q"], ["add", "--force", "."],
        ["-c", "user.name=Arena task snapshot", "-c", "user.email=task@arena.invalid",
         "-c", "commit.gpgsign=false", "commit", "-q", "-m", "Capture materialized task input"],
        ["remote", "add", "origin", "https://github.com/AMD-AGI/AgentKernelArena.git"],
    ]
    for args in commands:
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, timeout=30)


def install_candidate(workspace: Path, original: dict[str, bytes], candidate: dict[str, bytes]) -> None:
    if any(regular_file(workspace, name).read_bytes() != data for name, data in original.items()):
        raise ValueError("Arena source changed while Apex was running")
    installed = []
    try:
        for name, data in candidate.items():
            path = regular_file(workspace, name)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            try:
                temporary.chmod(path.stat().st_mode & 0o777)
                temporary.replace(path)
                installed.append(name)
            finally:
                temporary.unlink(missing_ok=True)
    except BaseException:
        for name in installed:
            (workspace / name).write_bytes(original[name])
        raise


@register_agent("apex")
def launch_agent(eval_config: dict, task_config_dir: str, workspace: str) -> str:
    options = load_options(eval_config)
    deadline = time.monotonic() + options["timeout_seconds"]
    workspace_path = Path(workspace).resolve(strict=True)
    spec, context = load_context(workspace_path, Path(task_config_dir))
    env = runtime_environment()
    arch = os.environ.get("AGENT_KERNEL_ARENA_GPU_ARCH") or os.environ.get("PYTORCH_ROCM_ARCH")
    if not arch or not arch.startswith("gfx") or ":" in arch:
        raise ValueError("Apex requires the Docker runner's detected GPU architecture")
    # Sibling artifacts cannot be delivered or scored as candidate source.
    artifacts = Path(tempfile.mkdtemp(prefix=workspace_path.name + "-apex-", dir=workspace_path.parent))
    source = artifacts / "input"
    copy_workspace(workspace_path, source)
    initialize_snapshot(source)
    harness = snapshot_workspace_harness(workspace_path, task_spec=spec)
    original = {edit.path: regular_file(workspace_path, edit.path).read_bytes() for edit in spec.candidate.editable}
    job_path = artifacts / "job.json"
    results = artifacts / "results"
    results.mkdir()
    task = build_task(spec, source, results, job_path, options, arch)
    job = {"context": context, "source": str(source), "task": task, "result": str(results / "result.json")}
    job_path.write_text(json.dumps(job, indent=2) + "\n")
    code = run_worker([sys.executable, str(Path(__file__).with_name("worker.py")), str(job_path)],
                      cwd=artifacts, env=env, deadline=deadline, log=artifacts / "process.log")
    verify_workspace_harness(harness, discard_added=False)
    if code:
        raise RuntimeError(f"Apex optimizer failed (exit {code}); inspect {artifacts / 'process.log'}")
    if time.monotonic() >= deadline:
        raise TimeoutError("Apex invocation deadline expired before delivery")
    candidate = prepare_delivery(read_json(results / "result.json"), task=task, original=original,
                                 artifacts=results, source=source, harness=harness)
    if time.monotonic() >= deadline:
        raise TimeoutError("Apex invocation deadline expired during delivery validation")
    install_candidate(workspace_path, original, candidate)
    logging.getLogger(__name__).info("Apex delivered %d source files; artifacts: %s", len(candidate), artifacts)
    return f"Apex delivered {len(candidate)} source files for independent Arena evaluation."
