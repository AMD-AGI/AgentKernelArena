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
from .task_protocol import (
    CaseManifest, baseline_correctness_accepted, merge_command_results, parse_command_result,
)
from .task_execution import CommandEvidence
from .task_spec import TaskConfigError, TaskSpec, resolve_task_path
from .harness_guard import (
    WorkspaceSnapshot, describe_workspace_harness, snapshot_workspace_harness,
    verify_workspace_harness,
)


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
        self.harness: WorkspaceSnapshot | None = None

    @classmethod
    def create(cls, spec: TaskSpec, workspace: Path, state_directory: Path,
               logger: logging.Logger | None = None) -> "TaskSession":
        workspace = Path(workspace).resolve(strict=True)
        state_directory = Path(state_directory).resolve()
        if state_directory.is_relative_to(workspace) or workspace.is_relative_to(state_directory):
            raise TaskConfigError("Task session state and candidate workspace must be separate directories")
        state_directory.mkdir(parents=True, exist_ok=False)
        harness = snapshot_workspace_harness(workspace, task_spec=spec)
        sources = _snapshot(workspace, state_directory / "baseline")
        _write_json(state_directory / "initial_sources.json", sources)
        _write_json(state_directory / "task_spec.json", spec.to_mapping())
        _write_json(state_directory / "session.json", {
            "version": 1, "task_id": spec.task_id, "workspace": str(workspace),
        })
        _write_json(state_directory / "harness.json", {
            "digests": harness.digests,
            "initial_symbols": {key: sorted(names) for key, names in harness.initial_symbols.items()},
            "effective_guard": describe_workspace_harness(workspace, snapshot=harness),
        })
        session = cls(spec, workspace, state_directory, sources, logger)
        session.harness = harness
        return session

    @classmethod
    def load(cls, spec: TaskSpec, workspace: Path, state_directory: Path,
             logger: logging.Logger | None = None, *, read_only: bool = False) -> "TaskSession":
        """Validate saved state; read-only queries never clean up added harness files."""
        workspace = Path(workspace).resolve(strict=True)
        state_directory = Path(state_directory).resolve(strict=True)
        descriptor = json.loads((state_directory / "session.json").read_text())
        saved_spec = json.loads((state_directory / "task_spec.json").read_text())
        if (descriptor != {"version": 1, "task_id": spec.task_id, "workspace": str(workspace)}
                or saved_spec != spec.to_mapping()):
            raise TaskExecutionError("Resume task identity/configuration does not match original session")
        sources = json.loads((state_directory / "initial_sources.json").read_text())
        if not isinstance(sources, dict) or not sources:
            raise TaskExecutionError("Original baseline source evidence is missing")
        session = cls(spec, workspace, state_directory, sources, logger)
        saved_harness = json.loads((state_directory / "harness.json").read_text())
        session.harness = WorkspaceSnapshot(
            workspace, saved_harness["digests"], spec,
            {key: frozenset(names) for key, names in saved_harness["initial_symbols"].items()},
        )
        session.verify_baseline_sources()
        session.verify_candidate_harness(discard_added=not read_only)
        records = sorted(state_directory.glob("action-*.json"))
        for path in records:
            number = int(path.name.split("-", 2)[1])
            session._invocations = max(session._invocations, number)
            raw = json.loads(path.read_text())
            if "execution_error" in raw:
                key = (raw["phase"], raw["role"], raw["action"])
                session.results.pop(key, None)
                if key == ("task_validation", "task", "validate-task"):
                    session.task_evidence = None
                    session.manifest = None
                continue
            result = raw["result"]
            commands = tuple(CommandEvidence(tuple(item["argv"]), item["returncode"],
                                             item["stdout"], item["stderr"], item["elapsed_s"])
                             for item in raw["commands"])
            parsed = merge_command_results(parse_command_result(
                command.stdout, role=result["role"], action=result["action"], returncode=command.returncode,
            ) for command in commands)
            if parsed.to_mapping() != result:
                raise TaskExecutionError(f"Saved result contradicts its command evidence: {path.name}")
            key = (raw["phase"], parsed.role, parsed.action)
            session.results[key] = ExecutedAction(raw["invocation_id"], parsed, commands)
            if key == ("task_validation", "task", "validate-task"):
                session.task_evidence = session.results[key]
                if parsed.passed:
                    session.manifest = CaseManifest.from_result(parsed)
            elif session.manifest is not None:
                session.manifest.validate(parsed)
        report_path = state_directory / "initial_validation.json"
        if report_path.exists():
            raw = json.loads(report_path.read_text())
            raw["errors"] = tuple(raw["errors"])
            report = InitialValidation(**raw)
            if report.accepted:
                session._verify_saved_initial_validation(report)
            session.initial_validation = report
        return session

    def _verify_saved_initial_validation(self, report: InitialValidation) -> None:
        if self.manifest is None or self.task_evidence is None or not self.task_evidence.result.passed:
            raise TaskExecutionError("Accepted initial report has no valid task manifest evidence")
        self._verify_initial_state(self.task_evidence)
        for action in ("compile", "correctness", "performance"):
            evidence = self.results.get(("task_validation", "baseline", action))
            if evidence is None:
                raise TaskExecutionError(f"Accepted initial report lacks baseline {action} evidence")
            result = evidence.result
            self.manifest.validate(result)
            accepted = result.passed if action != "correctness" else baseline_correctness_accepted(
                result, baseline=self.spec.baseline, phase="task_validation", manifest=self.manifest)
            if not accepted:
                raise TaskExecutionError(f"Saved baseline {action} does not satisfy the task policy")
        correctness = self.results[("task_validation", "baseline", "correctness")].result
        expected_candidate = "candidate_unimplemented"
        if self.spec.candidate.initial_state == "implemented":
            expected_candidate = "verified_as_frozen_baseline"
            if self.spec.baseline.kind == "provided":
                expected_candidate = "PASS"
                for action in ("compile", "correctness", "performance"):
                    evidence = self.results.get(("task_validation", "candidate", action))
                    if evidence is None or not evidence.result.passed:
                        raise TaskExecutionError(f"Saved initial candidate lacks passing {action}")
                    self.manifest.validate(evidence.result)
        expected = InitialValidation(True, correctness.status, not correctness.passed,
                                     self.spec.candidate.initial_state, expected_candidate, ())
        if report != expected:
            raise TaskExecutionError("Saved lifecycle verdict contradicts original command evidence")

    def _verify_initial_state(self, task: ExecutedAction) -> None:
        states = [metadata["candidate_state"] for metadata in
                  (task.result.metadata or {}).get("commands", [])
                  if isinstance(metadata, dict) and "candidate_state" in metadata]
        # Task-owned JSON may contain lists/dicts here. Reject every invalid or
        # conflicting observation without hashing it or losing the failure report.
        if not states or any(not isinstance(state, str) or state != self.spec.candidate.initial_state
                             for state in states):
            raise TaskExecutionError("Task check must verify candidate_state against the actual initial files")

    def verify_baseline_sources(self) -> None:
        for relative, digest in self.baseline_sources.items():
            path = self.baseline_workspace / relative
            if not (path.is_file() or path.is_symlink()) or _source_digest(path) != digest:
                raise TaskExecutionError(f"Frozen baseline source changed: {relative}")

    def verify_candidate_harness(self, *, discard_added: bool = True) -> None:
        if self.harness is None:
            raise TaskExecutionError("Original harness evidence is missing")
        verify_workspace_harness(self.harness, logger=self.logger, discard_added=discard_added)

    def _execute(self, role: str, action: str, phase: str) -> ExecutedAction:
        # Initial checks always inspect the preserved starting package, even
        # when recovering a partially executed session whose candidate changed.
        workspace = self.baseline_workspace if phase == "task_validation" or role == "baseline" else self.workspace
        self._invocations += 1
        record = self.state_directory / f"action-{self._invocations:04d}-{role}-{action}.json"
        self.verify_baseline_sources()
        try:
            executed = run_action(self.spec, workspace, role=role, action=action, phase=phase,
                                  manifest=None if role == "task" else self.manifest, logger=self.logger)
        except TaskExecutionError as exc:
            self.results.pop((phase, role, action), None)
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
            self._verify_initial_state(task)
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
        _write_json(self.state_directory / "validation_context.json", self.validation_context())
        return self.initial_validation

    def validation_context(self) -> dict:
        """Trusted input to the validator, including failed initial executions."""
        if self.initial_validation is None:
            raise TaskExecutionError("Initial task validation has not completed")
        if self.harness is None:
            raise TaskExecutionError("Original harness snapshot is missing")
        actions = []
        for path in sorted(self.state_directory.glob("action-*.json")):
            record = json.loads(path.read_text())
            if record.get("phase") == "task_validation":
                actions.append(record)
        return {
            "version": 1,
            "task_id": self.spec.task_id,
            "task_config": self.spec.to_mapping(),
            "workspace": str(self.workspace),
            "baseline_workspace": str(self.baseline_workspace),
            "initial_validation": asdict(self.initial_validation),
            "actions": actions,
            "harness": describe_workspace_harness(self.workspace, snapshot=self.harness),
        }

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
        self.verify_candidate_harness()
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
        self.verify_candidate_harness()
        if self._candidate_sources() != identity:
            raise TaskExecutionError("Evaluation command modified candidate source")
        return executed

    def _candidate_sources(self, *, allow_missing: bool = False) -> dict[str, str]:
        sources = {}
        runtime_dirs = {"__pycache__", ".pytest_cache", ".git", "build"}
        for edit in self.spec.candidate.editable:
            path = resolve_task_path(self.workspace, edit.path, must_exist=not allow_missing)
            if not path.exists():
                sources[edit.path] = "missing"
                continue
            paths = path.rglob("*") if edit.scope == "tree" else [path]
            for source in paths:
                relative = source.relative_to(self.workspace)
                # A tree declaration does not exempt nested symlinks from the
                # same containment rule as an individually declared file.
                resolve_task_path(self.workspace, relative.as_posix())
                if not source.is_file():
                    continue
                if set(relative.parts[:-1]) & runtime_dirs:
                    continue
                sources[relative.as_posix()] = _source_digest(source)
        return sources

    def candidate_source_evidence(self) -> dict:
        """Retain a failed submission report even when its paths are invalid.

        An error is explicit and cannot identify an accepted candidate. Never
        follow an escaping path merely to produce report/completion metadata.
        """
        try:
            return {"sources": self._candidate_sources(allow_missing=True), "error": None}
        except (OSError, ValueError, RuntimeError) as exc:
            return {"sources": None, "error": f"{type(exc).__name__}: {exc}"}
