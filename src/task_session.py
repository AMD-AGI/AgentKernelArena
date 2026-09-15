"""Framework-owned baseline snapshots and the two task validation phases.

The session lives outside the agent's working tree. It records command evidence
and exports a read-only-by-contract context for agent adapters. GPU execution
still happens through the task's own commands inside the selected runtime.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil

from .evaluator_utils import _is_unimplemented_target_stub
from .task_execution import ExecutedAction, TaskExecutionError, run_action
from .task_protocol import CaseManifest, baseline_correctness_accepted
from .task_spec import TaskConfigError, TaskSpec, resolve_task_path


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _source_digest(path: Path) -> str:
    if path.is_symlink():
        return "symlink:" + os.readlink(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(workspace: Path, destination: Path) -> dict[str, str]:
    """Copy source bytes, never hardlink mutable candidates to their baseline."""
    links = []
    for path in workspace.rglob("*"):
        if path.is_symlink():
            relative = path.relative_to(workspace).as_posix()
            resolved = resolve_task_path(workspace, relative, must_exist=True)
            links.append((relative, resolved.relative_to(workspace)))
    shutil.copytree(workspace, destination, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    for relative, target in links:
        copied = destination / relative
        if not copied.is_symlink():
            continue
        # An absolute link inside the original workspace must not keep pointing
        # at the candidate after the snapshot has moved to a separate root.
        copied.unlink()
        copied.symlink_to(os.path.relpath(destination / target, copied.parent))
    return {p.relative_to(destination).as_posix(): _source_digest(p)
            for p in destination.rglob("*") if p.is_file() or p.is_symlink()}


@dataclass(frozen=True)
class InitialValidation:
    accepted: bool
    baseline_numerical_status: str
    baseline_diagnostic: bool
    candidate_initial_state: str
    candidate_checks: str
    errors: tuple[str, ...]


class TaskSession:
    """One freshly materialized task. Existing session paths are never reset.

    Resume orchestration must retain an existing original snapshot, or create a
    new workspace from the task package. It cannot create a new baseline from
    an already optimized candidate. ``create`` is only for fresh workspaces.
    """

    def __init__(self, spec: TaskSpec, workspace: Path, state_directory: Path,
                 baseline_sources: dict[str, str], logger: logging.Logger | None = None):
        self.spec = spec
        self.workspace = workspace
        self.state_directory = state_directory
        self.baseline_workspace = state_directory / "baseline"
        self.baseline_sources = baseline_sources
        self.logger = logger or logging.getLogger(__name__)
        self.manifest: CaseManifest | None = None
        self.task_evidence: ExecutedAction | None = None
        self.initial_validation: InitialValidation | None = None
        self.results: dict[tuple[str, str, str], ExecutedAction] = {}
        self._invocations = 0
        self._candidate_identity: dict[str, str] | None = None

    @classmethod
    def create(cls, spec: TaskSpec, workspace: Path, state_directory: Path,
               logger: logging.Logger | None = None) -> "TaskSession":
        workspace = Path(workspace).resolve(strict=True)
        state_directory = Path(state_directory).resolve()
        if state_directory.is_relative_to(workspace) or workspace.is_relative_to(state_directory):
            raise TaskConfigError("Task session state and candidate workspace must be separate directories")
        state_directory.mkdir(parents=True, exist_ok=False)
        sources = _snapshot(workspace, state_directory / "baseline")
        _write_json(state_directory / "initial_sources.json", sources)
        _write_json(state_directory / "task_spec.json", spec.to_mapping())
        return cls(spec, workspace, state_directory, sources, logger)

    def verify_baseline_sources(self) -> None:
        for relative, digest in self.baseline_sources.items():
            path = self.baseline_workspace / relative
            if not (path.is_file() or path.is_symlink()) or _source_digest(path) != digest:
                raise TaskExecutionError(f"Frozen baseline source changed: {relative}")

    def _execute(self, role: str, action: str, phase: str) -> ExecutedAction:
        workspace = self.baseline_workspace if role in ("task", "baseline") else self.workspace
        self._invocations += 1
        record = self.state_directory / f"action-{self._invocations:04d}-{role}-{action}.json"
        self.verify_baseline_sources()
        try:
            executed = run_action(self.spec, workspace, role=role, action=action, phase=phase,
                                  manifest=None if role == "task" else self.manifest, logger=self.logger)
        except TaskExecutionError as exc:
            _write_json(record, {"role": role, "action": action, "phase": phase,
                                 "execution_error": str(exc),
                                 "commands": [asdict(c) for c in exc.commands]})
            raise
        _write_json(record, {"invocation_id": executed.invocation_id, "phase": phase,
                             "result": executed.result.to_mapping(),
                             "commands": [asdict(c) for c in executed.commands]})
        self.results[(phase, role, action)] = executed
        self.verify_baseline_sources()
        return executed

    def validate_initial(self) -> InitialValidation:
        if self.initial_validation is not None:
            raise TaskExecutionError("Initial task validation has already completed for this session")
        phase = "task_validation"
        numerical = "NOT_RUN"
        diagnostic = False
        candidate_checks = "NOT_RUN"
        errors = []
        try:
            task = self._execute("task", "validate-task", phase)
            self.task_evidence = task
            if not task.result.passed:
                raise TaskExecutionError(task.result.reason or "Task validation failed")
            states = {metadata.get("candidate_state") for metadata in
                      (task.result.metadata or {}).get("commands", [])
                      if isinstance(metadata, dict) and "candidate_state" in metadata}
            if states != {self.spec.candidate.initial_state}:
                raise TaskExecutionError("Task check must verify candidate_state against the actual initial files")
            self.manifest = CaseManifest.from_result(task.result)
            for action in ("compile", "correctness", "performance"):
                result = self._execute("baseline", action, phase).result
                accepted = result.passed
                if action == "correctness":
                    numerical = result.status
                    accepted = baseline_correctness_accepted(
                        result, baseline=self.spec.baseline, phase=phase, manifest=self.manifest)
                    diagnostic = accepted and not result.passed
                if not accepted:
                    raise TaskExecutionError(f"Baseline {action}: {result.reason or 'failed'}")
            if self.spec.candidate.initial_state == "unimplemented":
                candidate_checks = "candidate_unimplemented"
            elif self.spec.baseline.kind == "initial_candidate":
                candidate_checks = "verified_as_frozen_baseline"
            else:
                for action in ("compile", "correctness", "performance"):
                    result = self._execute("candidate", action, phase).result
                    if not result.passed:
                        raise TaskExecutionError(f"Initial candidate {action}: {result.reason or 'failed'}")
                candidate_checks = "PASS"
        except (TaskExecutionError, TaskConfigError, ValueError) as exc:
            errors.append(str(exc))
        self.initial_validation = InitialValidation(
            not errors, numerical, diagnostic, self.spec.candidate.initial_state, candidate_checks, tuple(errors))
        _write_json(self.state_directory / "initial_validation.json", asdict(self.initial_validation))
        if not errors:
            self._write_agent_context()
        return self.initial_validation

    def _write_agent_context(self) -> None:
        assert self.task_evidence is not None
        _write_json(self.state_directory / "agent_context.json", {
            "version": 1,
            "task_id": self.spec.task_id,
            "task_config": self.spec.to_mapping(),
            "workspace": str(self.workspace),
            "baseline_workspace": str(self.baseline_workspace),
            "manifest": self.task_evidence.result.to_mapping(),
        })

    @property
    def agent_context_path(self) -> Path:
        if self.initial_validation is None or not self.initial_validation.accepted:
            raise TaskExecutionError("Agent context requires accepted initial task validation")
        return self.state_directory / "agent_context.json"

    def _check_candidate_files(self) -> None:
        # The runner enforces language/interface semantics. This independent
        # structural check catches empty files, missing symbols, and recognized
        # unconditional Python stubs before a harness could fall back to baseline.
        for edit in self.spec.candidate.editable:
            path = resolve_task_path(self.workspace, edit.path, must_exist=True)
            if edit.scope == "tree":
                if not path.is_dir():
                    raise TaskExecutionError(f"Candidate subtree is not a directory: {edit.path}")
            elif not path.is_file() or not path.stat().st_size:
                raise TaskExecutionError(f"Candidate file is missing or empty: {edit.path}")
        for entry in self.spec.candidate.entrypoints:
            path = resolve_task_path(self.workspace, entry.file, must_exist=True)
            if not path.is_file():
                raise TaskExecutionError(f"Candidate entrypoint is not a file: {entry.file}")
            if path.suffix != ".py" or entry.kind == "executable":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=entry.file)
            except (SyntaxError, UnicodeError) as exc:
                raise TaskExecutionError(f"Cannot parse candidate entrypoint {entry.file}: {exc}") from exc
            definitions = {node.name: node for node in tree.body
                           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
            target = definitions.get(entry.symbol)
            if target is None:
                raise TaskExecutionError(f"Missing candidate entrypoint: {entry.file}:{entry.symbol}")
            if _is_unimplemented_target_stub(target):
                raise TaskExecutionError(f"Unimplemented candidate entrypoint: {entry.file}:{entry.symbol}")

    def candidate_action(self, action: str) -> ExecutedAction:
        if self.initial_validation is None or not self.initial_validation.accepted:
            raise TaskExecutionError("Candidate evaluation requires accepted initial task validation")
        self._check_candidate_files()
        phase = "candidate_evaluation"
        prerequisites = {"compile": (), "correctness": ("compile",), "performance": ("compile", "correctness")}
        if action not in prerequisites:
            raise TaskExecutionError(f"Unsupported candidate action: {action}")
        identity = self._candidate_sources()
        if action == "compile":
            # A new compile begins a new candidate attempt; previous checks
            # cannot authorize timing of a changed implementation.
            for key in list(self.results):
                if key[:2] == (phase, "candidate"):
                    del self.results[key]
            self._candidate_identity = identity
        elif identity != self._candidate_identity:
            raise TaskExecutionError("Candidate changed after compilation; recompile and recheck it")
        for required in prerequisites[action]:
            previous = self.results.get((phase, "candidate", required))
            if previous is None or not previous.result.passed:
                raise TaskExecutionError(f"Candidate {action} requires successful {required}")
        executed = self._execute("candidate", action, phase)
        if self._candidate_sources() != identity:
            raise TaskExecutionError("Evaluation command modified candidate source")
        return executed

    def _candidate_sources(self) -> dict[str, str]:
        sources = {}
        runtime_dirs = {"__pycache__", ".pytest_cache", ".git", "build"}
        for edit in self.spec.candidate.editable:
            path = resolve_task_path(self.workspace, edit.path, must_exist=True)
            paths = path.rglob("*") if edit.scope == "tree" else [path]
            for source in paths:
                if not source.is_file():
                    continue
                relative = source.relative_to(self.workspace)
                if set(relative.parts[:-1]) & runtime_dirs:
                    continue
                sources[relative.as_posix()] = _source_digest(source)
        return sources
