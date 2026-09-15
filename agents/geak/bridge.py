"""GEAK's CLI bridge to public Arena v2 actions; never imports task modules."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid
from typing import Callable

# This is an agent-side command, not a file installed into task packages.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.harness_guard import WorkspaceSnapshot, verify_workspace_harness
from src.task_execution import TaskExecutionError, run_action
from src.task_protocol import CaseManifest, RESULT_PREFIX, parse_command_result
from src.task_spec import TaskSpec, resolve_task_path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _task_diagnostic(value: object) -> str | None:
    """Bound validated runner reasons; never forward raw command output."""
    if not isinstance(value, str):
        return None
    for key, secret in os.environ.items():
        if len(secret) >= 8 and re.search(r"TOKEN|SECRET|PASSWORD|API_KEY", key, re.I):
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"sk-(?:ant-)?[A-Za-z0-9_-]{10,}", "[REDACTED]", value)
    return value[:1024]


class _ActionFailure(TaskExecutionError):
    def __init__(self, diagnostic: dict):
        super().__init__(f"GEAK public runner failed: {diagnostic['role']}.{diagnostic['action']}")
        self.diagnostic = diagnostic


@dataclass(frozen=True)
class TaskContext:
    spec: TaskSpec
    workspace: Path
    baseline: Path
    manifest: CaseManifest
    raw: dict

    @classmethod
    def load(cls, path: Path) -> "TaskContext":
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or type(raw.get("version")) is not int or raw["version"] != 1:
            raise ValueError("GEAK requires ARENA_TASK_CONTEXT version 1")
        spec = TaskSpec.from_mapping(raw["task_config"], task_id=raw["task_id"])
        if spec.candidate.language not in {"hip", "triton", "flydsl"}:
            raise ValueError("GEAK supports declared hip, triton, and flydsl candidates")
        workspace = Path(raw["workspace"]).resolve(strict=True)
        baseline = Path(raw["baseline_workspace"]).resolve(strict=True)
        if (not workspace.is_dir() or not baseline.is_dir() or
                baseline.is_relative_to(workspace) or workspace.is_relative_to(baseline)):
            raise ValueError("GEAK requires a separate framework-frozen baseline workspace")
        result = parse_command_result(RESULT_PREFIX + json.dumps(raw["manifest"]),
                                      role="task", action="validate-task", returncode=0)
        metadata = result.metadata or {}
        records = metadata.get("commands", [metadata])
        states = {item["candidate_state"] for item in records
                  if isinstance(item, dict) and "candidate_state" in item}
        if states != {spec.candidate.initial_state}:
            raise ValueError("Context candidate state does not match the declaration")
        return cls(spec, workspace, baseline, CaseManifest.from_result(result), raw)


def candidate_files(spec: TaskSpec, root: Path, *, complete: bool = False) -> dict[str, Path]:
    files = {}
    for edit in spec.candidate.editable:
        resolve_task_path(root, edit.path)
        path = root / edit.path
        if any(part.is_symlink() for part in (path, *path.parents) if part != root and part.is_relative_to(root)):
            raise ValueError("Candidate paths must not traverse symlinks")
        if edit.scope == "tree":
            paths = list(path.rglob("*")) if path.is_dir() else []
        else:
            paths = [path]
        found = False
        for item in paths:
            if not item.is_file():
                continue
            relative = item.relative_to(root).as_posix()
            resolve_task_path(root, relative, must_exist=True)
            if any(part.is_symlink() for part in (item, *item.parents)
                   if part != root and part.is_relative_to(root)):
                raise ValueError("Candidate artifacts must be regular files, not symlinks")
            files[relative] = item
            found = found or item.stat().st_size > 0
        if complete and not found:
            raise ValueError(f"Missing or empty candidate declaration: {edit.path}")
    if complete:
        for entry in spec.candidate.entrypoints:
            if not resolve_task_path(root, entry.file, must_exist=True).is_file():
                raise ValueError("Missing declared candidate entrypoint")
    return files


def digests(files: dict[str, Path]) -> dict[str, str]:
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}


def remaining_budget(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if not math.isfinite(deadline) or remaining <= 0:
        raise TimeoutError("GEAK shared deadline exhausted")
    return remaining


_COPY_CHUNK_SIZE = 1024 * 1024


def copy_file(source: Path, destination: Path, *, remaining: Callable[[], float],
              overwrite: bool = False) -> None:
    """Bound even a single large file; check after every read/write and metadata copy."""
    remaining()
    with source.open("rb") as reader, destination.open("wb" if overwrite else "xb") as writer:
        while True:
            remaining()
            chunk = reader.read(_COPY_CHUNK_SIZE)
            remaining()
            if not chunk:
                break
            writer.write(chunk)
            remaining()
    remaining()
    shutil.copystat(source, destination)
    remaining()


def copy_tree(source: Path, destination: Path, *, remaining: Callable[[], float],
              ignore_names: frozenset[str] = frozenset()) -> None:
    """Copy into a fresh tree with per-directory, per-file and per-chunk checks.

    Partial private artifacts remain on timeout; callers never continue into
    git, model calls or delivery. Escaping/cyclic links are not materialized.
    """
    remaining()
    origin = source.resolve(strict=True)

    def copy_directory(directory: Path, target: Path, ancestors: frozenset[Path]) -> None:
        remaining()
        resolved = directory.resolve(strict=True)
        if not resolved.is_relative_to(origin) or resolved in ancestors:
            raise ValueError("Copy source contains an escaping or cyclic directory link")
        target.mkdir(parents=True, exist_ok=False)
        with os.scandir(directory) as entries:
            for entry in entries:
                remaining()
                if entry.name in ignore_names:
                    continue
                path = Path(entry.path)
                if not path.resolve(strict=True).is_relative_to(origin):
                    raise ValueError("Copy source escapes its source tree")
                if entry.is_dir():
                    copy_directory(path, target / entry.name, ancestors | {resolved})
                elif entry.is_file():
                    copy_file(path, target / entry.name, remaining=remaining)
                else:
                    raise ValueError("Copy source must contain regular files and directories")
        remaining()
        shutil.copystat(directory, target)
        remaining()

    copy_directory(source, destination, frozenset())
    remaining()


def copy_task(source: Path, destination: Path, *, remaining: Callable[[], float]) -> None:
    # Preserve task-relative layouts, including multifile/image materializations.
    # Build outputs are copied too: only VCS and interpreter caches are excluded.
    copy_tree(source, destination, remaining=remaining, ignore_names=frozenset({".git", "__pycache__"}))


def snapshot_mapping(snapshot: WorkspaceSnapshot) -> dict:
    return {"digests": snapshot.digests,
            "initial_symbols": {k: sorted(v) for k, v in snapshot.initial_symbols.items()}}


class Bridge:
    def __init__(self, job_path: Path):
        self.job_path = job_path.resolve(strict=True)
        self.job = json.loads(self.job_path.read_text())
        self.context = TaskContext.load(Path(self.job["context"]))
        self.spec = self.context.spec
        self.root = self.job_path.parent
        self.eval_dir = self.root / "eval"
        self.deadline = float(self.job["deadline_monotonic"])
        if not math.isfinite(self.deadline):
            raise ValueError("GEAK deadline must be finite")
        self.logger = logging.getLogger("geak.public_runner")
        self.logger.addHandler(logging.NullHandler())
        self.logger.propagate = False

    def remaining(self) -> float:
        return remaining_budget(self.deadline)

    def _guard(self, root: Path, key: str = "harness") -> None:
        saved = self.job[key]
        verify_workspace_harness(WorkspaceSnapshot(
            root, saved["digests"], self.spec,
            {k: frozenset(v) for k, v in saved["initial_symbols"].items()}))

    def baseline_unchanged(self) -> None:
        self._guard(self.context.baseline, "baseline_harness")
        if digests(candidate_files(self.spec, self.context.baseline)) != self.job["baseline_sources"]:
            raise ValueError("Framework-frozen baseline sources changed")

    def candidate_root(self, path: Path) -> Path:
        root = path.resolve(strict=True)
        if not root.is_dir() or root == self.eval_dir or not root.is_relative_to(self.eval_dir):
            raise ValueError("Candidate checks require a private GEAK evaluation workspace")
        self._guard(root)
        return root

    @contextmanager
    def device_lock(self):
        # GEAK fans engineers out; their public measurements must not overlap.
        with (self.root / "device.lock").open("a") as stream:
            while True:
                self.remaining()
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(min(0.05, self.remaining()))
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def action(self, role: str, action: str, candidate: Path | None = None):
        with self.device_lock():
            return self._action(role, action, candidate)

    def _action(self, role: str, action: str, candidate: Path | None):
        self.baseline_unchanged()
        root = self.context.baseline if role == "baseline" else self.candidate_root(candidate or Path.cwd())
        before = digests(candidate_files(self.spec, root))
        selected = self.spec.action(role, action)
        # All commands, actions, GEAK calls, and final delivery consume ONE budget.
        bounded = replace(self.spec, actions=tuple(
            replace(item, timeout_s=min(item.timeout_s, self.remaining()))
            if item == selected else item for item in self.spec.actions))
        try:
            phase = "task_validation" if role == "baseline" else "candidate_evaluation"
            executed = run_action(bounded, root, role=role, action=action,
                                  phase=phase, manifest=self.context.manifest,
                                  logger=self.logger)
        finally:
            self.baseline_unchanged()
            self._guard(root, "baseline_harness" if role == "baseline" else "harness")
            if digests(candidate_files(self.spec, root)) != before:
                raise ValueError("Public runner changed implementation sources")
        # Keep validated envelopes, never raw subprocess output or environment secrets.
        result = executed.result
        record = {"invocation_id": executed.invocation_id, "role": role, "action": action, "phase": phase,
                  "status": result.status, "cases": [
                      {k: row[k] for k in ("test_case_id", "status", "execution_time_ms", "benchmark_method")
                       if k in row} for row in result.cases],
                  "sources": before, "elapsed_s": sum(c.elapsed_s for c in executed.commands)}
        if not result.passed:
            record["diagnostic"] = {
                "role": role, "action": action, "reason": _task_diagnostic(result.reason),
                "cases": [{"test_case_id": row["test_case_id"],
                           "reason": _task_diagnostic(row.get("reason"))}
                          for row in result.cases if row["status"] == "FAIL"][:5],
            }
        write_json(self.root / "checks" / (executed.invocation_id + ".json"), record)
        if not result.passed:
            raise _ActionFailure(record["diagnostic"])
        return result

    def materialize(self, source: Path, destination: Path) -> dict:
        source = self.candidate_root(source)
        destination = destination.resolve()
        if (not destination.is_relative_to(self.eval_dir) or destination.is_relative_to(source)
                or source.is_relative_to(destination)):
            raise ValueError("GEAK copies must stay in separate private evaluation directories")
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise ValueError("GEAK materialization requires a fresh destination")
            destination.rmdir()
        self.remaining()
        copy_task(source, destination, remaining=self.remaining)
        # Engineers and verifiers need an isolated git history to exchange diffs.
        if (source / ".git").is_dir():
            copy_tree(source / ".git", destination / ".git", remaining=self.remaining)
        self._guard(destination)
        self.remaining()
        return {"status": "COPIED"}

    def check(self, candidate: Path, *, performance: bool) -> dict:
        root = self.candidate_root(candidate)
        candidate_files(self.spec, root, complete=True)
        self.action("candidate", "compile", root)
        self.action("candidate", "correctness", root)
        if not performance:
            return {"correctness": "pass"}
        baseline = self.job["baseline_performance"]["cases"]
        result = self.action("candidate", "performance", root)
        bases = {row["test_case_id"]: row for row in baseline}
        rows = []
        for row in result.cases:
            base = bases[row["test_case_id"]]
            if base["benchmark_method"] != row["benchmark_method"]:
                raise ValueError("Baseline/candidate timing methods differ")
            rows.append({"name": row["test_case_id"], "baseline_ms": base["execution_time_ms"],
                         "optimized_ms": row["execution_time_ms"],
                         "speedup": base["execution_time_ms"] / row["execution_time_ms"]})
        geomean = math.exp(sum(math.log(row["speedup"]) for row in rows) / len(rows))
        return {"correctness": "pass", "per_case": rows, "speedup_geomean": geomean,
                "speedup_arithmetic": sum(row["speedup"] for row in rows) / len(rows)}

    def deliver(self, candidate: Path) -> dict:
        root = self.candidate_root(candidate)
        measured = self.check(root, performance=True)
        self.remaining()
        self._guard(self.context.workspace)
        before = candidate_files(self.spec, self.context.workspace)
        if digests(before) != self.job["original_sources"]:
            raise ValueError("Arena candidate changed outside GEAK artifact delivery")
        files = candidate_files(self.spec, root, complete=True)
        payload = {name: (path.read_bytes(), path.stat().st_mode & 0o777) for name, path in files.items()}
        # All paths and protected regions were checked before the first write.
        for name in set(before) | set(payload):
            resolve_task_path(self.context.workspace, name)
        for name, (content, mode) in payload.items():
            destination = resolve_task_path(self.context.workspace, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex)
            temporary.write_bytes(content)
            temporary.chmod(mode)
            temporary.replace(destination)
        for name in before.keys() - payload.keys():
            before[name].unlink()
        self._guard(self.context.workspace)
        return {"status": "DELIVERED", "files": sorted(payload),
                "sources": digests(candidate_files(self.spec, self.context.workspace)),
                "agent_measurement": measured, "arena_acceptance": "PENDING"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("operation", choices=("compile", "correctness", "performance", "baseline", "validate", "copy"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--source", type=Path)
    args = parser.parse_args(argv)
    try:
        bridge = Bridge(args.job)
        if args.operation == "copy":
            if args.source is None:
                raise ValueError("copy requires --source")
            result = bridge.materialize(args.source, args.workspace)
        elif args.operation == "baseline":
            bridge.action("baseline", "compile")
            result = bridge.action("baseline", "performance").to_mapping()
        elif args.operation == "compile":
            result = bridge.action("candidate", "compile", args.workspace).to_mapping()
        else:
            result = bridge.check(args.workspace, performance=args.operation != "correctness")
        if args.operation == "validate":
            result = {**result, "validation_status": "accepted", "applied_to_original": "false",
                      "director_verified_speedup_geomean": result["speedup_geomean"],
                      "director_verified_speedup_arithmetic": result["speedup_arithmetic"],
                      "timing_basis": "arena_public_runner",
                      "final_patch": str(bridge.eval_dir / "final_patch.diff")}
            write_json(bridge.eval_dir / "director_validation.json", result)
        print("GEAK_ARENA_RESULT=" + json.dumps(result, allow_nan=False))
        return 0
    except Exception as exc:
        # Exception text and raw output may contain secrets. Only validated,
        # bounded task reasons are eligible for model-facing diagnostics.
        failure = {"status": "FAIL", "error_type": type(exc).__name__}
        if isinstance(exc, _ActionFailure):
            failure.update(error_type="TaskExecutionError", diagnostic=exc.diagnostic)
        print("GEAK_ARENA_RESULT=" + json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
