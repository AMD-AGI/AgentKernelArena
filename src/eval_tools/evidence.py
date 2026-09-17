"""Immutable submission evidence used by external evaluation tools.

The normal Arena workflow edits a task workspace in place.  Numerical tools such
as Triton FpSan need both the as-shipped submission and the optimized candidate,
while resume support must be able to prove that a cached tool report belongs to
the current sources.  This module captures the declared submission files outside
the mutable task workspace and produces deterministic content fingerprints.

V2 captures the candidate boundary plus protected task inputs. Directory roots
are enumerated again for every candidate fingerprint, including newly created
files. Legacy source/target declarations retain their original lookup rules.

"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from src.task_spec import TaskConfigError, relative_path, resolve_task_path

from .task_declarations import candidate_paths, is_v2_task, protected_paths


_MANIFEST_NAME = "manifest.json"


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError(f"expected a string or list of strings, got {type(value).__name__}")


def declared_submission_paths(task_config: dict[str, Any]) -> tuple[str, ...]:
    """Return normalized candidate and protected evidence paths.

    V2 explicit profile paths extend the candidate and protected declarations;
    they cannot replace them and hide candidate changes. Legacy fixtures retain
    the explicit-profile-or-source/target fallback.
    """

    profile = task_config.get("evaluation_profile") or {}
    if not isinstance(profile, dict):
        raise ValueError("evaluation_profile must be a mapping")
    explicit = profile.get("submission_paths")
    values: list[str] = []
    if is_v2_task(task_config):
        values.extend(candidate_paths(task_config))
        values.extend(protected_paths(task_config))
        values.extend(_command_input_paths(task_config))
        values.extend(_string_list(explicit))
    elif explicit is not None:
        values.extend(_string_list(explicit))
    else:
        values.extend(_string_list(task_config.get("source_file_path")))
        values.extend(_string_list(task_config.get("target_file_path")))

    return tuple(sorted({_relative_path(raw) for raw in values}))


def _command_input_paths(config: Mapping[str, Any]) -> Iterable[str]:
    """Include local adapter scripts/data, without executing or parsing a shell.

    Command arguments are not a dependency graph. The capture caller must also
    supply the harness/task-package protected paths, including indirect helpers.
    """
    argv_keys = {"runner", "command", "comparison_command", "oracle_command", "launcher"}

    def visit(value: Any, key: str = "") -> Iterable[str]:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                yield from visit(child, str(child_key))
        elif isinstance(value, (list, tuple)):
            if key in argv_keys:
                for index, argument in enumerate(value):
                    if not isinstance(argument, str) or (index and value[index - 1] in {"-c", "-m"}):
                        continue
                    argument = argument.split("=", 1)[-1] if argument.startswith("--") else argument
                    if argument and not argument.startswith("-") and not Path(argument).is_absolute():
                        # Interpreter names, literal parameters and flags aren't
                        # file declarations. Capture arguments that name paths.
                        if "/" in argument or Path(argument).suffix:
                            try:
                                yield _relative_path(argument)
                            except ValueError:
                                # Literal command arguments are not declarations
                                # (for example inline code or a URL). The shared
                                # protected-file manifest covers indirect inputs.
                                continue
            else:
                for child in value:
                    yield from visit(child, "command" if key in {"commands", "setup"} else key)
        elif key in {"capsule", "code_object"} and isinstance(value, str):
            yield value

    for key in ("evaluation", "evaluation_tools", "exports"):
        yield from visit(config.get(key, {}))
    yield from visit(config.get("workspace", {}).get("setup", ()), "setup")


def _repo_subdir(task_config: dict[str, Any]) -> str | None:
    configured = task_config.get("repo_subdir")
    if configured:
        return Path(str(configured)).name
    for key in ("image_repo_path", "repo_url"):
        value = task_config.get(key)
        if not value:
            continue
        name = str(value).rstrip("/")
        if name.endswith(".git"):
            name = name[:-4]
        return Path(name).name
    return None


def _candidate_locations(
    workspace: Path, relative: str, task_config: dict[str, Any]
) -> Iterable[Path]:
    repo_subdir = None if is_v2_task(task_config) else _repo_subdir(task_config)
    if repo_subdir:
        yield workspace / repo_subdir / relative
    yield workspace / relative


def resolve_submission_path(
    workspace: Path, relative: str, task_config: dict[str, Any]
) -> Path:
    """Resolve a declared path without allowing it to escape ``workspace``.

    Missing files resolve to the first canonical candidate so their absence can
    be fingerprinted (important for tasks where the agent creates ``kernel.py``).
    """

    _lexical, resolved = _submission_location(workspace, relative, task_config)
    return resolved


def _submission_location(
    workspace: Path, relative: str, task_config: dict[str, Any]
) -> tuple[Path, Path]:
    """Return both the declared lookup path and its contained resolved target."""

    relative = _relative_path(relative)
    workspace = workspace.resolve()
    if is_v2_task(task_config):
        try:
            return workspace / relative, resolve_task_path(workspace, relative)
        except TaskConfigError as error:
            raise ValueError(f"submission path escapes workspace or cannot be resolved: {relative!r}") from error
    candidates = list(_candidate_locations(workspace, relative, task_config))
    lexical = next((path for path in candidates if path.exists()), candidates[0])
    resolved = lexical.resolve(strict=False)
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"submission path escapes workspace: {relative!r}")
    return lexical, resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_path(value: str) -> str:
    try:
        return relative_path(value, "submission path")
    except TaskConfigError as error:
        raise ValueError(f"submission path must be normalized and workspace-relative: {value!r}") from error


def _collect_entries(workspace: Path, roots: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Walk lexical roots and recheck containment for every file and directory."""
    entries: list[dict[str, Any]] = []

    def visit(relative: str, declared: str, ancestors: frozenset[Path]) -> None:
        relative = _relative_path(relative)
        lexical = workspace / relative
        try:
            # Unlike initial resolution, a resume check must retain the captured
            # lexical root: resolving the root again could follow its replacement
            # symlink and redefine the containment boundary.
            path = lexical.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise ValueError(f"cyclic or invalid submission path: {relative}") from error
        if not path.is_relative_to(workspace):
            raise ValueError(f"candidate submission path escapes workspace: {relative!r}")
        if path in ancestors:
            raise ValueError(f"cyclic submission directory: {relative!r}")
        kind = "directory" if path.is_dir() else "file" if path.is_file() else "missing"
        if kind == "missing" and path.exists():
            raise ValueError(f"submission path is not a regular file or directory: {relative!r}")
        entries.append({
            "declared_path": declared,
            "workspace_relative_path": relative,
            "resolved_workspace_relative_path": path.relative_to(workspace).as_posix(),
            "symlink_target": os.readlink(lexical) if lexical.is_symlink() else None,
            "kind": kind,
            "exists": kind != "missing",
            "sha256": _sha256_file(path) if kind == "file" else None,
            "size": path.stat().st_size if kind == "file" else None,
        })
        if kind == "directory":
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                visit((Path(relative) / child.name).as_posix(), declared, ancestors | {path})

    for root in roots:
        visit(root["workspace_relative_path"], root["declared_path"], frozenset())
    return sorted(entries, key=lambda entry: (entry["workspace_relative_path"], entry["declared_path"]))


@dataclass(frozen=True)
class SubmissionEvidence:
    storage_dir: Path
    workspace: Path
    manifest: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return str(self.manifest["fingerprint"])

    @property
    def files_dir(self) -> Path:
        return self.storage_dir / "files"

    def verify(self) -> None:
        """Raise when the external evidence was changed after capture."""

        manifest_path = self.storage_dir / _MANIFEST_NAME
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if loaded != self.manifest:
            raise RuntimeError("submission evidence manifest changed after capture")
        body = {key: value for key, value in loaded.items() if key != "fingerprint"}
        if loaded.get("fingerprint") != _stable_hash(body):
            raise RuntimeError("submission evidence manifest fingerprint mismatch")
        for entry in loaded["entries"]:
            if not entry["exists"] or entry.get("kind") == "directory":
                continue
            stored = (self.files_dir / _relative_path(entry["workspace_relative_path"])).resolve()
            if not stored.is_relative_to(self.files_dir.resolve()):
                raise RuntimeError("submission evidence file escapes evidence storage")
            if not stored.is_file() or _sha256_file(stored) != entry["sha256"]:
                raise RuntimeError(
                    "submission evidence file changed after capture: "
                    + entry["workspace_relative_path"]
                )

    def candidate_fingerprint(self) -> str:
        """Fingerprint the current optimized candidate over the same path set."""

        # ``workspace`` is canonicalized when evidence is captured and recorded
        # in the manifest. Do not resolve it again here: the candidate could
        # replace the workspace path itself with a symlink after capture and
        # thereby redefine the containment root.
        workspace = self.workspace
        if not workspace.is_absolute():
            raise ValueError("submission evidence workspace must be absolute")
        if self.manifest["schema_version"] == 3:
            return _stable_hash(_collect_entries(workspace, self.manifest["roots"]))
        # Keep schema-2 snapshots loadable without changing their resume identity.
        entries: list[dict[str, Any]] = []
        for original in self.manifest["entries"]:
            relative_value = str(original["workspace_relative_path"])
            relative = Path(relative_value)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError(
                    "submission evidence path must be workspace-relative: "
                    f"{relative_value!r}"
                )
            # Resolve the current candidate again instead of trusting the path
            # captured in the manifest. The candidate may have replaced a file
            # or one of its parent directories with a symlink after capture.
            # Continue with the resolved target so retargeting the originally
            # declared symlink cannot redirect the subsequent file read.
            lexical = workspace / relative
            path = lexical.resolve(strict=False)
            if not path.is_relative_to(workspace):
                raise ValueError(
                    "candidate submission path escapes workspace: "
                    f"{relative_value!r}"
                )
            exists = path.is_file()
            entries.append(
                {
                    "declared_path": original["declared_path"],
                    "workspace_relative_path": original["workspace_relative_path"],
                    "resolved_workspace_relative_path": path.relative_to(
                        workspace
                    ).as_posix(),
                    "symlink_target": (
                        os.readlink(lexical) if lexical.is_symlink() else None
                    ),
                    "exists": exists,
                    "sha256": _sha256_file(path) if exists else None,
                    "size": path.stat().st_size if exists else None,
                }
            )
        return _stable_hash(entries)


def capture_submission_evidence(
    workspace: Path,
    task_config: dict[str, Any],
    storage_dir: Path,
    *,
    protected_paths: Iterable[str] = (),
    original_workspace: Path | None = None,
) -> SubmissionEvidence:
    """Capture candidate and protected files outside the mutable workspace.

    Pass the shared harness/task-package protected-file manifest through
    ``protected_paths``. Paths are already relative to the materialized workspace;
    this includes indirect helpers outside conventional scripts/tests locations.
    ``original_workspace`` may name the framework's frozen starting snapshot;
    current candidate fingerprints still use ``workspace``. This supports
    evaluating a retained session without recapturing modified code as original.
    """
    workspace = workspace.resolve()
    original = Path(original_workspace).resolve(strict=True) if original_workspace is not None else workspace
    storage_dir = storage_dir.resolve()
    if storage_dir.is_relative_to(workspace) or storage_dir.is_relative_to(original):
        raise ValueError("submission evidence storage must be outside the task workspace")
    if storage_dir.exists():
        raise FileExistsError(f"submission evidence already exists: {storage_dir}")
    roots: dict[str, dict[str, str]] = {}
    for declared in declared_submission_paths(task_config):
        lexical, _ = _submission_location(original, declared, task_config)
        roots[lexical.relative_to(original).as_posix()] = {
            "declared_path": declared,
            "workspace_relative_path": lexical.relative_to(original).as_posix(),
        }
    for declared in protected_paths:
        relative = _relative_path(declared)
        roots[relative] = {"declared_path": relative, "workspace_relative_path": relative}
    root_list = [roots[key] for key in sorted(roots)]
    entries = _collect_entries(original, root_list)
    files_dir = storage_dir / "files"
    files_dir.mkdir(parents=True)
    for entry in entries:
        relative = entry["workspace_relative_path"]
        destination = files_dir / relative
        if entry["kind"] == "directory":
            destination.mkdir(parents=True, exist_ok=True)
        elif entry["kind"] == "file":
            source = original / entry["resolved_workspace_relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    manifest_body = {
        "schema_version": 3,
        "workspace": str(workspace),
        "roots": root_list,
        "entries": entries,
    }
    manifest = {**manifest_body, "fingerprint": _stable_hash(manifest_body)}
    (storage_dir / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence = SubmissionEvidence(storage_dir, workspace, manifest)
    evidence.verify()
    return evidence


def load_submission_evidence(
    storage_dir: Path,
    *,
    task_config: dict[str, Any] | None = None,
    protected_paths: Iterable[str] = (),
) -> SubmissionEvidence:
    """Load a snapshot; pass the current task declaration when resuming v2.

    Old file-only snapshots remain readable for legacy runs, but cannot provide
    the candidate/directory/protected-input coverage required by a v2 task.
    """
    storage_dir = storage_dir.resolve()
    manifest = json.loads((storage_dir / _MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in {2, 3}:
        raise ValueError("unsupported submission evidence schema version")
    # Capture stores an already-canonical absolute workspace path. Retain that
    # lexical boundary instead of following a symlink that may have replaced the
    # workspace between capture and resume.
    workspace = Path(str(manifest["workspace"]))
    if not workspace.is_absolute():
        raise ValueError("submission evidence workspace must be absolute")
    evidence = SubmissionEvidence(storage_dir, workspace, manifest)
    evidence.verify()
    if task_config is not None and is_v2_task(task_config):
        if manifest["schema_version"] != 3:
            raise RuntimeError("v2 tasks require directory-aware schema-3 submission evidence")
        roots = [Path(_relative_path(item["workspace_relative_path"])) for item in manifest["roots"]]
        required = {*declared_submission_paths(task_config), *protected_paths}
        missing = [
            path for path in sorted(required)
            if not any(Path(_relative_path(path)).is_relative_to(root) for root in roots)
        ]
        if missing:
            raise RuntimeError(f"submission evidence does not cover current task paths: {missing}")
    return evidence
