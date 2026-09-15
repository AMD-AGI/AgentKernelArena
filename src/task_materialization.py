"""Deadline-bound, non-destructive materialization of normalized v2 tasks.

Security/reproducibility boundary: only declared image trees or pinned Git
commits are acquired; Git hooks and implicit submodule downloads are disabled.
Commands use argv and isolated process groups. Task sources are never written,
and no clone cache is created under tasks/. Evidence and process logs live
outside the candidate. This is not an OS sandbox for task-authored setup code.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid

from .runtime_env import PYTHON_ENV_VAR, build_subprocess_env
from .task_spec import TaskSpec, load_task_spec, resolve_task_path


class MaterializationError(RuntimeError):
    """Setup or resume could not establish a complete materialized task."""


class MaterializationTimeout(MaterializationError):
    """The shared acquisition/copy/setup deadline expired."""


_STALE_OUTPUTS = frozenset({
    "task_result.yaml", "task_result.json", "validation_report.yaml",
    "validation_report.json", ".validation_complete",
})
_TRANSIENT_NAMES = frozenset({".git", "__pycache__", ".pytest_cache"})
_RECORD_SCHEMA = 1


class _Deadline:
    def __init__(self, timeout_s: float):
        self.timeout_s = timeout_s
        self.end = time.monotonic() + timeout_s

    def remaining(self) -> float:
        remaining = self.end - time.monotonic()
        if remaining <= 0:
            raise MaterializationTimeout(f"Materialization exceeded its {self.timeout_s}s total deadline")
        return remaining


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_digest(path: Path, deadline: _Deadline) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            deadline.remaining()
            data = handle.read(1024 * 1024)
            if not data:
                break
            digest.update(data)
    deadline.remaining()
    return digest.hexdigest()


def materialization_state_directory(workspace: Path) -> Path:
    """Framework-side state location; never inside the candidate workspace."""
    workspace = Path(workspace).absolute()
    parent = workspace.parent.resolve(strict=True)
    directory = parent / ".task-materialization" / workspace.name
    if directory.is_symlink() or directory.parent.is_symlink() or not directory.resolve().is_relative_to(parent / ".task-materialization"):
        raise MaterializationError("Materialization state path escapes its framework directory")
    return directory


def _write_record(directory: Path, record: dict) -> None:
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    value = {**body, "record_sha256": _digest(body)}
    temporary = directory / f".record-{uuid.uuid4().hex}.tmp"
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, directory / "materialization.json")


def load_materialization_record(workspace: Path) -> dict:
    """Read a complete record; this does not itself validate current inputs."""
    path = materialization_state_directory(workspace) / "materialization.json"
    if path.is_symlink():
        raise MaterializationError("Materialization record must not be a symlink")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        body = {key: value for key, value in record.items() if key != "record_sha256"}
    except (OSError, ValueError, AttributeError) as exc:
        raise MaterializationError("Existing workspace has no valid materialization record") from exc
    if record.get("schema_version") != _RECORD_SCHEMA or record.get("record_sha256") != _digest(body):
        raise MaterializationError("Materialization record identity is invalid")
    if record.get("status") != "complete":
        raise MaterializationError(f"Workspace materialization is not complete: {record.get('status')}")
    return record


def verify_original_materialization(workspace: Path, *, timeout_s: float = 3600) -> dict:
    """Before creating a missing session, require the original materialized tree.

    An existing session must instead load its own frozen baseline. This check
    includes new helper files outside declared candidate paths, which could
    otherwise silently change the baseline captured from a resumed workspace.
    """
    workspace = Path(workspace).absolute()
    record = load_materialization_record(workspace)
    _verify_workspace_identity(workspace, record)
    _verify_original_tree(workspace, record, _Deadline(timeout_s))
    return record


def _verify_original_tree(workspace: Path, record: dict, deadline: _Deadline) -> None:
    if _digest(_tree_manifest(workspace, deadline)) != record.get("initial_workspace_sha256"):
        raise MaterializationError(
            "Workspace changed after materialization but has no frozen task session; "
            "cannot capture a new baseline from the changed candidate"
        )


def _verify_workspace_identity(workspace: Path, record: dict) -> None:
    if workspace.is_symlink() or not workspace.is_dir():
        raise MaterializationError("Existing workspace must be its original directory, not a symlink")
    inode = {"device": workspace.stat().st_dev, "inode": workspace.stat().st_ino}
    if record.get("workspace") != str(workspace.resolve()) or record.get("workspace_inode") != inode:
        raise MaterializationError("Existing workspace was replaced after materialization")


def _excluded(relative: str, names: frozenset[str], paths: tuple[str, ...]) -> bool:
    path = Path(relative)
    return any(part in names for part in path.parts) or any(
        path.is_relative_to(Path(excluded)) for excluded in paths
    )


def _tree_manifest(root: Path, deadline: _Deadline, *, names: frozenset[str] = _TRANSIENT_NAMES,
                   excludes: tuple[str, ...] = ()) -> dict[str, dict]:
    result = {}

    def visit(directory: Path) -> None:
        deadline.remaining()
        for path in sorted(directory.iterdir()):
            deadline.remaining()
            relative = path.relative_to(root).as_posix()
            if _excluded(relative, names, excludes):
                continue
            mode = path.lstat().st_mode
            item = {"mode": stat.S_IMODE(mode)}
            if stat.S_ISLNK(mode):
                item.update(kind="symlink", target=os.readlink(path))
            elif stat.S_ISDIR(mode):
                item.update(kind="directory")
            elif stat.S_ISREG(mode):
                item.update(kind="file", sha256=_file_digest(path, deadline), size=path.stat().st_size)
            else:
                raise MaterializationError(f"Unsupported source file type: {relative}")
            result[relative] = item
            if item["kind"] == "directory":
                visit(path)

    visit(root)
    return result


def _run(argv: list[str], cwd: Path, deadline: _Deadline, state: Path, label: str,
         env: dict[str, str] | None = None) -> None:
    """Bound and clean up the process group, including ordinary background jobs."""
    deadline.remaining()
    log_path = state / f"{label}-{uuid.uuid4().hex}.log"
    with log_path.open("xb") as output:
        try:
            with subprocess.Popen(argv, cwd=cwd, env=env or build_subprocess_env(),
                                  stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                  start_new_session=True) as process:
                try:
                    returncode = process.wait(timeout=deadline.remaining())
                except BaseException as exc:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    if isinstance(exc, subprocess.TimeoutExpired):
                        raise MaterializationTimeout("Materialization command exceeded the total deadline") from exc
                    raise
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    raise MaterializationError(f"{label} left background processes; see {log_path.name}")
        except OSError as exc:
            raise MaterializationError(f"Cannot start {label}; see {log_path.name}") from exc
    if returncode != 0:
        raise MaterializationError(f"{label} failed with exit code {returncode}; see {log_path.name}")
    deadline.remaining()


def _copy_tree(source: Path, destination: Path, manifest: dict[str, dict],
               deadline: _Deadline, state: Path) -> None:
    """Reuse cp's reflink mechanism, without legacy deletion/fallback behavior.

    Copy only enumerated entries, so nested exclusions and every .git directory
    are omitted before copying. A failed copy is retained for inspection.
    """
    cp = shutil.which("cp")
    batches: dict[Path, list[Path]] = {}
    for relative, item in manifest.items():
        deadline.remaining()
        target = destination / relative
        original = source / relative
        if item["kind"] == "directory":
            target.mkdir(parents=True, exist_ok=False)
        elif item["kind"] == "symlink":
            link = item["target"]
            if Path(link).is_absolute() and Path(link).is_relative_to(source):
                link = os.path.relpath(destination / Path(link).relative_to(source), target.parent)
            target.symlink_to(link)
        elif cp:
            batches.setdefault(target.parent, []).append(original)
        else:
            with original.open("rb") as reader, target.open("xb") as writer:
                while True:
                    deadline.remaining()
                    data = reader.read(1024 * 1024)
                    if not data:
                        break
                    writer.write(data)
            shutil.copystat(original, target, follow_symlinks=False)
    for target, files in batches.items():
        for offset in range(0, len(files), 128):
            _run([cp, "-a", "--reflink=auto", "--", *map(str, files[offset:offset + 128]), str(target)],
                 destination, deadline, state, "copy")
    for relative, item in reversed(list(manifest.items())):
        deadline.remaining()
        if item["kind"] == "directory":
            shutil.copystat(source / relative, destination / relative, follow_symlinks=False)


def _verify_copy(original: dict, copied: dict, source: Path, destination: Path) -> None:
    expected = {key: dict(value) for key, value in original.items()}
    for relative, item in expected.items():
        if item["kind"] == "symlink":
            target = Path(item["target"])
            if target.is_absolute() and target.is_relative_to(source):
                item["target"] = os.path.relpath(destination / target.relative_to(source),
                                                 (destination / relative).parent)
    if copied != expected:
        raise MaterializationError("Copied source does not match its recorded input identity")


def _validate_paths(spec: TaskSpec, workspace: Path, deadline: _Deadline) -> None:
    config = spec.to_mapping()
    protected_names = {"config.yaml", "README.md", *config.get("instructions", [])}
    required = ["config.yaml", *config.get("instructions", []), *spec.baseline.source_files]
    if "workloads" in config["evaluation"]:
        required.append(config["evaluation"]["workloads"])
        protected_names.add(config["evaluation"]["workloads"])
    for relative in required:
        deadline.remaining()
        if not resolve_task_path(workspace, relative, must_exist=True).is_file():
            raise MaterializationError(f"Immutable task input is not a file: {relative}")
    implemented = spec.candidate.initial_state == "implemented"
    for edit in spec.candidate.editable:
        deadline.remaining()
        path = resolve_task_path(workspace, edit.path, must_exist=implemented)
        if path.exists() and not (path.is_dir() if edit.scope == "tree" else path.is_file()):
            raise MaterializationError(f"Candidate path kind conflicts with its scope: {edit.path}")
        for relative in protected_names | set(spec.baseline.source_files):
            protected = resolve_task_path(workspace, relative)
            if (protected == path or (edit.scope == "tree" and protected.is_relative_to(path))) and (
                    edit.scope != "symbols" or relative in protected_names):
                raise MaterializationError(f"Candidate resolves to protected task input: {relative}")
    for entry in spec.candidate.entrypoints:
        deadline.remaining()
        path = resolve_task_path(workspace, entry.file, must_exist=implemented)
        if path.exists() and not path.is_file():
            raise MaterializationError(f"Candidate entrypoint is not a file: {entry.file}")
    _validate_symlinks(workspace, deadline, must_exist=True)


def _validate_symlinks(workspace: Path, deadline: _Deadline, *, must_exist: bool) -> None:
    for path in workspace.rglob("*"):
        deadline.remaining()
        if path.is_symlink():
            resolve_task_path(workspace, path.relative_to(workspace).as_posix(), must_exist=must_exist)


def _runtime_identity() -> dict:
    return {key: os.environ[key] for key in ("AKA_SCORING_IMAGE_RUNTIME_REF", "AKA_SCORING_IMAGE_REFERENCE")
            if key in os.environ}


def _git_environment() -> dict[str, str]:
    # Inherited GIT_DIR/WORK_TREE must never redirect acquisition into a user
    # repository. Global filters/templates must not add implicit code/downloads.
    env = {key: value for key, value in build_subprocess_env().items()
           if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_SYSTEM=os.devnull,
               GIT_CONFIG_GLOBAL=os.devnull, GIT_ATTR_NOSYSTEM="1",
               GIT_TERMINAL_PROMPT="0", GIT_ALLOW_PROTOCOL="file:git:http:https:ssh")
    return env


def _image_input(source: dict, deadline: _Deadline) -> tuple[Path, dict]:
    path = Path(source["image_path"]).resolve(strict=True)
    if not path.is_dir():
        raise MaterializationError("Declared image source is not a directory")
    manifest = _tree_manifest(path, deadline, names=frozenset({".git"}),
                              excludes=tuple(source.get("exclude", [])))
    return path, manifest


def _resume(spec: TaskSpec, workspace: Path, identity: dict, deadline: _Deadline) -> Path:
    if workspace.is_symlink() or not workspace.is_dir():
        raise MaterializationError("Existing workspace must be its original directory, not a symlink")
    record = load_materialization_record(workspace)
    if any(record.get(key) != value for key, value in identity.items()):
        raise MaterializationError("Original task configuration, package, or runtime identity changed")
    _verify_workspace_identity(workspace, record)
    for source in record["sources"]:
        if source["declaration"]["kind"] == "image":
            _, manifest = _image_input(source["declaration"], deadline)
            if _digest(manifest) != source["input_tree_sha256"]:
                raise MaterializationError("Declared image source changed since materialization")
    _validate_paths(spec, workspace, deadline)
    for relative, expected in record["protected_files"].items():
        deadline.remaining()
        path = resolve_task_path(workspace, relative, must_exist=True)
        if _file_digest(path, deadline) != expected:
            raise MaterializationError(f"Protected materialized task input changed: {relative}")
    session = workspace.parent / ".task-sessions" / workspace.name
    if not os.path.lexists(session):
        _verify_original_tree(workspace, record, deadline)
    elif session.is_symlink() or session.parent.is_symlink() or not session.is_dir():
        raise MaterializationError("Existing task session must be its framework-owned directory")
    # Session contents/initial gate belong to TaskSession.load, not this layer.
    deadline.remaining()
    return workspace


def materialize_task_workspace(spec: TaskSpec, config_path: Path, workspace: Path,
                               logger: logging.Logger | None = None) -> Path:
    """Create a fresh workspace or verify a completed one without replaying setup."""
    log = logger or logging.getLogger(__name__)
    config = spec.to_mapping()
    deadline = _Deadline(config.get("workspace", {}).get("timeout_s", 3600))
    config_path = Path(config_path).resolve(strict=True)
    task_root = config_path.parent
    workspace = Path(workspace).absolute()
    workspace = workspace.parent.resolve() / workspace.name
    if workspace.is_relative_to(task_root) or task_root.is_relative_to(workspace):
        raise MaterializationError("Task package and workspace must be separate directories")
    if config_path.name != "config.yaml":
        raise MaterializationError("A task package must use config.yaml")
    for source in config.get("workspace", {}).get("sources", []):
        if source["kind"] == "image":
            origin = Path(source["image_path"]).resolve()
            if workspace.is_relative_to(origin) or origin.is_relative_to(workspace):
                raise MaterializationError("Image source and workspace must be separate directories")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    task_manifest = _tree_manifest(task_root, deadline, excludes=tuple(_STALE_OUTPUTS))
    identity = {
        "task_id": spec.task_id, "workspace": str(workspace),
        "original_config_sha256": _file_digest(config_path, deadline),
        "task_spec_sha256": _digest(config), "task_package_sha256": _digest(task_manifest),
        "runtime": _runtime_identity(),
    }
    if os.path.lexists(workspace):
        return _resume(spec, workspace, identity, deadline)
    state = materialization_state_directory(workspace)
    state.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": _RECORD_SCHEMA, "status": "running", **identity,
              "sources": [], "setup_commands": config.get("workspace", {}).get("setup", [])}
    _write_record(state, record)
    try:
        workspace.mkdir(exist_ok=False)
        record["workspace_inode"] = {"device": workspace.stat().st_dev, "inode": workspace.stat().st_ino}
        _copy_tree(task_root, workspace, task_manifest, deadline, state)
        _verify_copy(task_manifest, _tree_manifest(workspace, deadline), task_root, workspace)
        acquired_files = set()
        for source in config.get("workspace", {}).get("sources", []):
            deadline.remaining()
            destination = resolve_task_path(workspace, source["destination"])
            if os.path.lexists(destination):
                raise MaterializationError(f"Source destination already exists; refusing to overwrite: {source['destination']}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            provenance = {"declaration": source}
            if source["kind"] == "image":
                origin, manifest = _image_input(source, deadline)
                if workspace.is_relative_to(origin) or origin.is_relative_to(workspace):
                    raise MaterializationError("Image source and workspace must be separate directories")
                destination.mkdir(exist_ok=False)
                _copy_tree(origin, destination, manifest, deadline, state)
                copied = _tree_manifest(destination, deadline, names=frozenset({".git"}))
                _verify_copy(manifest, copied, origin, destination)
                provenance.update(input_tree_sha256=_digest(manifest), copied_tree_sha256=_digest(copied))
            else:
                git = ["git", "-c", "core.hooksPath=/dev/null", "-c", "protocol.ext.allow=never",
                       "-c", "submodule.recurse=false", "-c", "init.templateDir="]
                git_env = _git_environment()
                _run([*git, "init", "--", str(destination)], workspace, deadline, state, "git-init", git_env)
                _run([*git, "-C", str(destination), "remote", "add", "--", "origin", source["url"]],
                     workspace, deadline, state, "git-remote", git_env)
                _run([*git, "-C", str(destination), "fetch", "--no-tags", "--depth=1", "--", "origin", source["revision"]],
                     workspace, deadline, state, "git-fetch", git_env)
                _run([*git, "-C", str(destination), "checkout", "--detach", source["revision"]],
                     workspace, deadline, state, "git-checkout", git_env)
                head = (destination / ".git" / "HEAD").read_text().strip()
                if head.lower() != source["revision"].lower():
                    raise MaterializationError("Git checkout did not resolve to the declared pinned revision")
                copied = _tree_manifest(destination, deadline)
                provenance.update(revision=head, copied_tree_sha256=_digest(copied))
            acquired_files.update((Path(source["destination"]) / relative).as_posix()
                                  for relative, item in copied.items() if item["kind"] != "directory")
            record["sources"].append(provenance)
        _validate_symlinks(workspace, deadline, must_exist=False)
        env = build_subprocess_env()
        for key in tuple(env):
            if key.startswith("ARENA_"):
                del env[key]
        for index, command in enumerate(record["setup_commands"]):
            argv = list(command)
            if argv[0] in {"python", "python3"}:
                argv[0] = env[PYTHON_ENV_VAR]
            elif argv[0] == "pytest":
                argv = [env[PYTHON_ENV_VAR], "-m", "pytest", *argv[1:]]
            _run(argv, workspace, deadline, state, f"setup-{index:03d}", env)
        _validate_paths(spec, workspace, deadline)
        helper_record = state / "perf_helpers.json"
        helper_code = (
            "import json, pathlib, sys; "
            "from src.perf_helper_materialization import materialize_perf_helpers_in_workspace; "
            "root=pathlib.Path(sys.argv[1]); "
            "files=materialize_perf_helpers_in_workspace(root); "
            "pathlib.Path(sys.argv[2]).write_text(json.dumps([str(p.relative_to(root)) for p in files]))"
        )
        _run([sys.executable, "-c", helper_code, str(workspace), str(helper_record)],
             Path(__file__).resolve().parents[1], deadline, state, "perf-helpers", env)
        _validate_paths(spec, workspace, deadline)
        if load_task_spec(workspace / "config.yaml", task_id=spec.task_id).to_mapping() != config:
            raise MaterializationError("Materialized task configuration differs from its parsed specification")
        if _file_digest(workspace / "config.yaml", deadline) != identity["original_config_sha256"]:
            raise MaterializationError("Setup modified the task configuration")
        if any(os.path.lexists(workspace / name) for name in _STALE_OUTPUTS):
            raise MaterializationError("Setup produced reserved completion reports; these are not fresh evaluation evidence")
        immutable = acquired_files | {path for path, item in task_manifest.items() if item["kind"] != "directory"}
        immutable.update(spec.baseline.source_files)
        immutable.update(config.get("instructions", []))
        immutable.update(json.loads(helper_record.read_text()))
        if "workloads" in config["evaluation"]:
            immutable.add(config["evaluation"]["workloads"])
        resolved_edits = [(resolve_task_path(workspace, edit.path), edit.scope)
                          for edit in spec.candidate.editable]
        protected = {}
        for relative in sorted(immutable):
            # Symbol-scoped files contain editable implementation bodies. Their
            # protected AST portions are checked by the harness guard, not by a
            # whole-file digest that would reject legitimate candidate edits.
            if any(edit.contains(relative) for edit in spec.candidate.editable):
                continue
            path = resolve_task_path(workspace, relative, must_exist=True)
            # A contained symbolic alias of a candidate names the same bytes;
            # retaining its digest would incorrectly reject legitimate resume.
            if any(path == edited or (scope == "tree" and path.is_relative_to(edited))
                   for edited, scope in resolved_edits):
                continue
            if path.is_file():
                protected[relative] = _file_digest(path, deadline)
        initial_manifest = _tree_manifest(workspace, deadline)
        candidate_roots = [(path.relative_to(workspace).as_posix(), scope) for path, scope in resolved_edits]
        candidate_manifest = {relative: item for relative, item in initial_manifest.items()
                              if any(edit.contains(relative) for edit in spec.candidate.editable) or
                              any(relative == root or (scope == "tree" and relative.startswith(root + "/"))
                                  for root, scope in candidate_roots)}
        for edit in spec.candidate.editable:
            candidate_manifest.setdefault(edit.path, {"kind": "missing"})
        record.update(status="complete", protected_files=protected,
                      original_candidate_sha256=_digest(candidate_manifest),
                      initial_workspace_sha256=_digest(initial_manifest))
        deadline.remaining()
        _write_record(state, record)
        deadline.remaining()
        log.info("Materialized task %s with %d declared sources", spec.task_id, len(record["sources"]))
        return workspace
    except BaseException as exc:
        record.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        _write_record(state, record)
        raise
