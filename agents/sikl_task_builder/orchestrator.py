"""Generate, repair, validate and install SIKL tasks with resumable evidence."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from . import BUILDER_VERSION
from .backend import generate
from .bundle import ImportProblem, fingerprint, inspect_bundle
from .config import Config
from .materialize import check_contract, materialize_task, task_digest, task_tree
from .validation import runtime_identity, validate_task

REPO = Path(__file__).resolve().parents[2]
LOG = logging.getLogger(__name__)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve_paths(config: Config, repo: Path = REPO):
    source = (repo / Path(config.input_dir).expanduser()).resolve()
    output = (repo / config.output_dir).resolve()
    artifacts = (repo / config.artifact_root).resolve()
    if not output.is_relative_to(repo / "tasks") or output == repo / "tasks":
        raise ValueError("output_dir must be below this repository's tasks/ directory")
    if not artifacts.is_relative_to(repo) or artifacts == repo or artifacts.is_relative_to(repo / "tasks"):
        raise ValueError("artifact_root must be below the repository, outside tasks/")
    if any(a.is_relative_to(b) for a, b in ((source, artifacts), (artifacts, source),
                                           (source, output), (output, source))):
        raise ValueError("Input, output and artifact trees must not overlap")
    return source, output, artifacts


def implementation_digest():
    return task_digest(Path(__file__).parent)


def install_task(draft: Path, output: Path, task_id: str, evidence: dict) -> Path:
    if not evidence.get("ok") or evidence.get("task_digest") != task_digest(draft):
        raise ImportProblem("stale_validation", "Task changed or has no accepted validation")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / task_id
    if destination.exists():
        if task_digest(destination) == evidence["task_digest"]:
            return destination
        raise ImportProblem("output_conflict", f"Existing task differs; refusing overwrite: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=".sikl-install-", dir=output))
    try:
        # Copy exactly the accepted files, excluding command-generated caches.
        for relative in task_tree(draft):
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(draft / relative, target)
        if task_digest(temporary) != evidence["task_digest"]:
            raise ImportProblem("stale_validation", "Task changed during installation")
        # Serialize installation across different campaigns and never replace
        # an existing task, including an empty directory created by another run.
        with (output / ".sikl-install.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if destination.exists():
                raise ImportProblem("output_conflict", str(destination))
            temporary.rename(destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run(config: Config, run_id: str | None = None, *, repo: Path = REPO,
        generator=generate, validator=validate_task, runtime=runtime_identity) -> dict:
    source, output, artifacts = resolve_paths(config, repo)
    environment = runtime(config)
    tasks = inspect_bundle(source, config.selections)
    if config.tasks:
        missing = set(config.tasks) - {t.task_id for t in tasks}
        if missing:
            raise ValueError(f"Unknown task selectors: {sorted(missing)}")
        tasks = [t for t in tasks if t.task_id in config.tasks]
    bundle_files = task_tree(source)
    identity = fingerprint({"config": config.mapping(), "builder": BUILDER_VERSION,
                            "implementation": implementation_digest(), "runtime": environment,
                            "tasks": {t.task_id: t.digest for t in tasks},
                            "bundle_files": bundle_files})
    resuming = run_id is not None
    run_id = run_id or (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("Invalid run ID")
    root = artifacts / run_id
    if resuming:
        if not root.is_dir():
            raise ValueError(f"Run does not exist: {run_id}")
    else:
        root.mkdir(parents=True, exist_ok=False)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resuming:
            state = json.loads((root / "state.json").read_text())
            if state["identity"] != identity:
                raise ValueError("Resume refused: input, configuration, builder or runtime changed")
        else:
            shutil.copytree(source, root / "bundle")
            state = {"run_id": run_id, "identity": identity, "environment": environment, "tasks": {}}
            atomic_json(root / "config.json", config.mapping())
            atomic_json(root / "state.json", state)
        frozen = task_digest(root / "bundle")
        if task_tree(root / "bundle") != bundle_files or json.loads((root / "config.json").read_text()) != config.mapping():
            raise ImportProblem("source_changed", "Run snapshot or saved config differs from source")
        manifest = {"config": (root / "config.json").read_bytes(),
                    "implementation": implementation_digest()}
        def verify_sources():
            if (task_digest(root / "bundle") != frozen or
                    (root / "config.json").read_bytes() != manifest["config"] or
                    implementation_digest() != manifest["implementation"]):
                raise ImportProblem("source_changed", "Generator changed source data, config or builder code")
        for task in tasks:
            record = state["tasks"].setdefault(task.task_id, {"state": "discovered", "attempts": 0, "elapsed": 0.0})
            draft = root / "drafts" / task.task_id
            task_artifacts = root / "evidence" / task.task_id
            if record["state"] == "installed":
                installed = output / task.task_id
                if not installed.is_dir() or task_digest(installed) != record["task_digest"]:
                    raise ImportProblem("installed_changed", f"Installed task changed: {task.task_id}")
                continue
            if record["state"] in {"failed", "needs_spec", "platform_deferred"}:
                continue
            started = time.monotonic()
            elapsed_before = record["elapsed"]
            # Persist an absolute deadline before invoking external processes.
            # Crashes/restarts (and time spent paused) cannot reset the budget.
            record.setdefault("deadline_at", time.time() + config.max_task_seconds)
            deadline = started + max(0, record["deadline_at"] - time.time())
            LOG.info("Building %s (%d cases)", task.task_id, len(task.rows))
            try:
                if not draft.exists():
                    materialize_task(task, config, draft)
                while record["attempts"] <= config.max_repair_attempts and time.monotonic() < deadline:
                    verify_sources()
                    attempt = record["attempts"]
                    record["attempts"] += 1
                    record["state"] = "generating" if attempt == 0 else "repairing"
                    atomic_json(root / "state.json", state)
                    generation = generator(config, draft, root, task.task_id, record.get("feedback", {}),
                                           task_artifacts / f"generator_{attempt}.log", deadline - time.monotonic())
                    verify_sources()
                    record["state"] = "checking"
                    contract = check_contract(task, config, draft)
                    feedback = {"contract": contract, "generation": generation}
                    if generation.get("ok") and contract["ok"] and time.monotonic() < deadline:
                        record["state"] = "validating"
                        record["elapsed"] = elapsed_before + time.monotonic() - started
                        atomic_json(root / "state.json", state)
                        evidence = validator(draft, task_artifacts / "validation", config,
                                             timeout=deadline - time.monotonic())
                        verify_sources()
                        # A validator workspace must not replace the generator's
                        # draft, and input repairs invalidate previous evidence.
                        contract = check_contract(task, config, draft)
                        feedback["validation"] = evidence
                        if evidence.get("ok") and contract["ok"]:
                            destination = install_task(draft, output, task.task_id, evidence)
                            record.update(state="installed", task_digest=evidence["task_digest"],
                                          destination=str(destination), validation_id=evidence["validation_id"])
                            break
                    record["feedback"] = feedback
                    record["elapsed"] = elapsed_before + time.monotonic() - started
                    atomic_json(root / "state.json", state)
                if record["state"] != "installed":
                    previous = record.get("feedback", {}).get("validation", {}).get("commands", {})
                    record["state"] = "needs_spec" if not previous.get("source-check", {"ok": True})["ok"] else "failed"
            except Exception as exc:
                code = getattr(exc, "code", "execution_error")
                record.update(state="platform_deferred" if code == "platform_deferred" else "failed",
                              error={"code": code, "message": str(exc)})
                LOG.exception("Task %s failed", task.task_id)
                if code == "source_changed":
                    raise
            finally:
                record["elapsed"] = elapsed_before + time.monotonic() - started
                atomic_json(root / "state.json", state)
        state["ok"] = all(t["state"] == "installed" for t in state["tasks"].values())
        atomic_json(root / "state.json", state)
        atomic_json(root / "summary.json", state)
        return state
