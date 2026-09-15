"""The shared, agent-independent task v2 configuration contract.

Only this module interprets task declarations. Task-local runners stay
self-contained: they implement the command protocol, not imports of TaskSpec.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

import yaml


class TaskConfigError(ValueError):
    """A task declaration is ambiguous, unsupported, or malformed."""


ACTIONS = (
    ("task", "validate-task"),
    ("baseline", "compile"), ("baseline", "correctness"), ("baseline", "performance"),
    ("candidate", "compile"), ("candidate", "correctness"), ("candidate", "performance"),
)


def _mapping(value: Any, where: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise TaskConfigError(f"{where} must be a mapping with string keys")
    unknown = set(value) - allowed
    if unknown:
        raise TaskConfigError(f"{where}: unknown fields {sorted(unknown)}")
    return deepcopy(value)


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise TaskConfigError(f"{where} must be a nonempty string without NUL")
    return value


def _choice(value: Any, choices: tuple[str, ...], where: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise TaskConfigError(f"{where} must be one of {choices}")
    return value


def _timeout(value: Any, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise TaskConfigError(f"{where} must be a positive integer")
    return value


def relative_path(value: Any, where: str = "path") -> str:
    """Validate a literal task-root-relative path, including on Linux hosts."""
    path = _text(value, where)
    if ("\\" in path or path.startswith("/") or re.match(r"^[A-Za-z]:", path)
            or any(part in ("", ".", "..") for part in path.split("/"))):
        raise TaskConfigError(f"{where} must be a normalized task-relative path: {path!r}")
    return path


def resolve_task_path(root: Path, path: str, *, must_exist: bool = False) -> Path:
    """Retain subdirectories and reject escapes through existing parent symlinks.

    Call again immediately before installation/export. This check is a
    reproducibility boundary, not an OS sandbox against concurrent attackers.
    """
    relative_path(path)
    root = Path(root).resolve(strict=True)
    try:
        result = (root / path).resolve(strict=must_exist)
        result.relative_to(root)
    except (ValueError, OSError, RuntimeError) as exc:
        raise TaskConfigError(f"Task path cannot be resolved within workspace: {path!r}") from exc
    return result


def _paths(value: Any, where: str) -> list[str]:
    if not isinstance(value, list):
        raise TaskConfigError(f"{where} must be a list")
    result = [relative_path(v, where) for v in value]
    if len(result) != len(set(result)):
        raise TaskConfigError(f"{where} contains duplicate paths")
    return result


def _argv(value: Any, where: str) -> tuple[str, ...]:
    if (not isinstance(value, list) or not value
            or any(not isinstance(v, str) or "\x00" in v for v in value)):
        raise TaskConfigError(f"{where} must be a nonempty argv list of strings")
    _text(value[0], where)
    return tuple(value)


def _commands(value: Any, where: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list) or not value:
        raise TaskConfigError(f"{where} must be a nonempty list of argv lists")
    return tuple(_argv(v, where) for v in value)


@dataclass(frozen=True)
class EditScope:
    path: str
    scope: str = "file"
    symbols: tuple[str, ...] = ()
    allow_new_helpers: bool = False

    def contains(self, path: str) -> bool:
        return path == self.path or (
            self.scope == "tree" and PurePosixPath(self.path) in PurePosixPath(path).parents
        )


@dataclass(frozen=True)
class EntryPoint:
    file: str
    kind: str
    symbol: str | None = None


@dataclass(frozen=True)
class CandidateSpec:
    language: str
    initial_state: str
    initial_language: str | None
    editable: tuple[EditScope, ...]
    entrypoints: tuple[EntryPoint, ...]


@dataclass(frozen=True)
class BaselineSpec:
    kind: str
    language: str | None
    source_files: tuple[str, ...]
    correctness_policy: str
    diagnostic_reason: str | None


@dataclass(frozen=True)
class ActionSpec:
    role: str
    action: str
    commands: tuple[tuple[str, ...], ...]
    timeout_s: int


def _candidate(value: Any) -> CandidateSpec:
    obj = _mapping(value, "candidate", {
        "language", "initial_state", "initial_language", "editable", "entrypoints",
    })
    language = _text(obj.get("language"), "candidate.language")
    state = _choice(obj.get("initial_state", "implemented"),
                    ("implemented", "unimplemented"), "candidate.initial_state")
    initial = obj.get("initial_language", language if state == "implemented" else None)
    if state == "unimplemented" and "initial_language" in obj:
        raise TaskConfigError("An unimplemented candidate must omit initial_language")
    if state == "implemented":
        _text(initial, "candidate.initial_language")
    if not isinstance(obj.get("editable"), list) or not obj["editable"]:
        raise TaskConfigError("candidate.editable must be a nonempty list")
    edits = []
    for raw in obj["editable"]:
        edit = {"path": raw} if isinstance(raw, str) else _mapping(
            raw, "candidate.editable[]", {"path", "scope", "symbols", "allow_new_helpers"})
        path = relative_path(edit.get("path"), "candidate.editable[].path")
        scope = _choice(edit.get("scope", "file"), ("file", "symbols", "tree"), "scope")
        symbols = edit.get("symbols", [])
        if not isinstance(symbols, list) or any(not isinstance(s, str) or not s for s in symbols):
            raise TaskConfigError("Editable symbols must be a list of nonempty names")
        helpers = edit.get("allow_new_helpers", False)
        if type(helpers) is not bool:
            raise TaskConfigError("allow_new_helpers must be a boolean")
        if scope == "symbols":
            if not symbols or len(symbols) != len(set(symbols)):
                raise TaskConfigError("Symbol scope requires nonempty unique symbols")
        elif "symbols" in edit or "allow_new_helpers" in edit:
            raise TaskConfigError("symbols/allow_new_helpers only apply to symbol scope")
        item = EditScope(path, scope, tuple(symbols), helpers)
        if any(e.contains(path) or item.contains(e.path) for e in edits):
            raise TaskConfigError(f"Overlapping editable declarations: {path}")
        edits.append(item)
    entries = obj.get("entrypoints", [])
    if not isinstance(entries, list):
        raise TaskConfigError("candidate.entrypoints must be a list")
    parsed_entries = []
    for raw in entries:
        entry = _mapping(raw, "candidate.entrypoints[]", {"file", "kind", "symbol"})
        path = relative_path(entry.get("file"), "entrypoint.file")
        kind = _choice(entry.get("kind"), ("function", "builder", "class", "executable"), "entrypoint.kind")
        symbol = entry.get("symbol")
        if kind == "executable":
            if "symbol" in entry:
                raise TaskConfigError("Executable entrypoints must omit symbol")
        else:
            _text(symbol, "entrypoint.symbol")
        if not any(e.contains(path) for e in edits):
            raise TaskConfigError(f"Entrypoint is outside candidate boundary: {path}")
        parsed = EntryPoint(path, kind, symbol)
        if parsed in parsed_entries:
            raise TaskConfigError("Duplicate entrypoint")
        parsed_entries.append(parsed)
    return CandidateSpec(language, state, initial, tuple(edits), tuple(parsed_entries))


def _baseline(value: Any, candidate: CandidateSpec) -> BaselineSpec:
    obj = _mapping(value, "baseline", {
        "kind", "language", "source_files", "correctness_policy", "diagnostic_reason",
    })
    default = "initial_candidate" if candidate.initial_state == "implemented" else "provided"
    kind = _choice(obj.get("kind", default), ("initial_candidate", "provided"), "baseline.kind")
    if kind == "initial_candidate" and candidate.initial_state == "unimplemented":
        raise TaskConfigError("An unimplemented candidate cannot be the baseline")
    language = obj.get("language")
    if language is not None:
        _text(language, "baseline.language")
    policy = _choice(obj.get("correctness_policy", "required"),
                     ("required", "diagnostic"), "baseline.correctness_policy")
    reason = obj.get("diagnostic_reason")
    if policy == "diagnostic":
        _text(reason, "baseline.diagnostic_reason")
    elif "diagnostic_reason" in obj:
        raise TaskConfigError("diagnostic_reason requires the diagnostic policy")
    sources = _paths(obj.get("source_files", []), "baseline.source_files")
    return BaselineSpec(kind, language, tuple(sources), policy, reason)


def _actions(value: Any) -> tuple[ActionSpec, ...]:
    obj = _mapping(value, "evaluation", {
        "runner", "workloads", "timeout_s", "task", "baseline", "candidate",
    })
    runner = _argv(obj["runner"], "evaluation.runner") if "runner" in obj else None
    default_timeout = _timeout(obj.get("timeout_s", 3600), "evaluation.timeout_s")
    if "workloads" in obj:
        relative_path(obj["workloads"], "evaluation.workloads")
    for role in ("baseline", "candidate"):
        if role in obj:
            _mapping(obj[role], f"evaluation.{role}", {"compile", "correctness", "performance"})
    actions = []
    for role, action in ACTIONS:
        raw = obj.get("task", {}) if role == "task" else obj.get(role, {}).get(action, {})
        override = _mapping(raw, f"evaluation.{role}.{action}", {"commands", "timeout_s"})
        timeout = _timeout(override.get("timeout_s", default_timeout), f"{role}.{action}.timeout_s")
        if "commands" in override:
            commands = _commands(override["commands"], f"{role}.{action}.commands")
        elif runner:
            suffix = (action,) if role == "task" else (role, action)
            commands = (runner + suffix,)
        else:
            raise TaskConfigError(f"Missing commands for {role}.{action}; no evaluation.runner")
        actions.append(ActionSpec(role, action, commands, timeout))
    return tuple(actions)


def _optional_fields(obj: dict) -> None:
    if "description" in obj:
        _text(obj["description"], "description")
    if "instructions" in obj:
        _paths(obj["instructions"], "instructions")
    if "kernel_identity" in obj:
        identity = _mapping(obj["kernel_identity"], "kernel_identity", {"logical_operator", "source_owner"})
        for key, value in identity.items():
            _text(value, f"kernel_identity.{key}")
    if "platform_support" in obj:
        platform = _mapping(obj["platform_support"], "platform_support", {"required_arch", "status", "skip_reason"})
        if "required_arch" in platform:
            _text(platform["required_arch"], "platform_support.required_arch")
        status = _choice(platform.get("status", "active"), ("active", "skip"), "platform_support.status")
        if status == "skip" or "skip_reason" in platform:
            _text(platform.get("skip_reason"), "platform_support.skip_reason")
    workspace = _mapping(obj.get("workspace", {}), "workspace", {"sources", "setup", "timeout_s"})
    _timeout(workspace.get("timeout_s", 3600), "workspace.timeout_s")
    if "setup" in workspace:
        _commands(workspace["setup"], "workspace.setup")
    sources = workspace.get("sources", [])
    if not isinstance(sources, list):
        raise TaskConfigError("workspace.sources must be a list")
    destinations = []
    for source in sources:
        source = _mapping(source, "workspace.sources[]", {
            "kind", "image_path", "destination", "exclude", "url", "revision",
        })
        kind = _choice(source.get("kind"), ("image", "git"), "source.kind")
        dest = relative_path(source.get("destination"), "source.destination")
        if any(d == dest or dest.startswith(d + "/") or d.startswith(dest + "/") for d in destinations):
            raise TaskConfigError("Source destinations overlap")
        destinations.append(dest)
        if kind == "image":
            path = _text(source.get("image_path"), "source.image_path")
            if not path.startswith("/") or ".." in path.split("/") or "\\" in path:
                raise TaskConfigError("image_path must be an absolute path inside the runtime image")
            if "url" in source or "revision" in source:
                raise TaskConfigError("Image source cannot declare Git fields")
            _paths(source.get("exclude", []), "source.exclude")
        else:
            _text(source.get("url"), "source.url")
            revision = source.get("revision")
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                raise TaskConfigError("Git source requires a full immutable commit SHA")
            if "image_path" in source or "exclude" in source:
                raise TaskConfigError("Git source cannot declare image fields")
    exports = obj.get("exports", [])
    if not isinstance(exports, list):
        raise TaskConfigError("exports must be a list")
    outputs = set()
    for export in exports:
        export = _mapping(export, "exports[]", {"format", "output", "command", "timeout_s"})
        _text(export.get("format"), "exports.format")
        output = relative_path(export.get("output"), "exports.output")
        if output in outputs:
            raise TaskConfigError("Duplicate export output")
        outputs.add(output)
        _argv(export.get("command"), "exports.command")
        _timeout(export.get("timeout_s", 60), "exports.timeout_s")
    # Tool plugins own their schemas; the evaluator performs capability and
    # option validation with the selected run policy before executing anything.
    for field in ("evaluation_profile", "evaluation_tools"):
        if field in obj and not isinstance(obj[field], dict):
            raise TaskConfigError(f"{field} must be a mapping")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    candidate: CandidateSpec
    baseline: BaselineSpec
    actions: tuple[ActionSpec, ...]
    _config: dict

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any], *, task_id: str) -> "TaskSpec":
        relative_path(task_id, "task_id")
        obj = _mapping(config, "task config", {
            "schema_version", "description", "instructions", "kernel_identity",
            "candidate", "baseline", "evaluation", "workspace", "platform_support",
            "evaluation_profile", "evaluation_tools", "exports",
        })
        if type(obj.get("schema_version")) is not int or obj["schema_version"] != 2:
            raise TaskConfigError("Expected schema_version: 2; migrate legacy configuration explicitly")
        candidate = _candidate(obj.get("candidate"))
        baseline = _baseline(obj.get("baseline", {}), candidate)
        actions = _actions(obj.get("evaluation"))
        _optional_fields(obj)
        protected = {"config.yaml", "README.md", *obj.get("instructions", [])}
        if "workloads" in obj["evaluation"]:
            protected.add(obj["evaluation"]["workloads"])
        # Read-only baseline material must not overlap editable candidate files.
        # initial_candidate has an independent frozen snapshot instead.
        for path in protected | set(baseline.source_files):
            if any(e.contains(path) and (e.scope != "symbols" or path in protected)
                   for e in candidate.editable):
                raise TaskConfigError(f"Editable declaration includes protected task input: {path}")
        return cls(task_id, candidate, baseline, actions, obj)

    def action(self, role: str, action: str) -> ActionSpec:
        for spec in self.actions:
            if (spec.role, spec.action) == (role, action):
                return spec
        raise TaskConfigError(f"Unsupported task action {role}.{action}")

    def to_mapping(self) -> dict:
        """Return an independent v2 declaration, with semantic defaults explicit."""
        obj = deepcopy(self._config)
        obj["candidate"]["initial_state"] = self.candidate.initial_state
        if self.candidate.initial_language is not None:
            obj["candidate"]["initial_language"] = self.candidate.initial_language
        baseline = obj.setdefault("baseline", {})
        baseline["kind"] = self.baseline.kind
        baseline["correctness_policy"] = self.baseline.correctness_policy
        obj["evaluation"].setdefault("timeout_s", 3600)
        return obj


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise TaskConfigError(f"Duplicate or non-string YAML key: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_task_spec(config_path: Path, *, task_id: str) -> TaskSpec:
    """The caller supplies the stable discovery ID, never a scratch folder name."""
    with Path(config_path).open(encoding="utf-8") as handle:
        try:
            config = yaml.load(handle, Loader=_UniqueKeyLoader)
        except yaml.YAMLError as exc:
            raise TaskConfigError(f"Invalid task YAML: {exc}") from exc
    return TaskSpec.from_mapping(config, task_id=task_id)
