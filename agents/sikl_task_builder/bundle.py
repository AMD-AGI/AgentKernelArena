"""Read SIKL bundles without importing or executing their solution code."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class ImportProblem(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def relative_path(raw: str) -> str:
    path = PurePosixPath(raw)
    if (not raw or path.is_absolute() or ".." in path.parts or "\\" in raw
            or str(path) != raw or raw == "."):
        raise ImportProblem("invalid_path", f"Expected a normalized relative path: {raw!r}")
    return raw


def parse_json(content: str, origin: str) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result
    try:
        return json.loads(content, object_pairs_hook=unique,
                          parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    except (OSError, ValueError) as exc:
        raise ImportProblem("invalid_json", f"{origin}: {exc}") from exc


def read_json(path: Path) -> Any:
    return parse_json(path.read_text(), str(path))


def source_files(solution: dict) -> dict[str, str]:
    spec = solution.get("spec", {})
    if spec.get("language") != "python" or spec.get("destination_passing_style") is not False:
        raise ImportProblem("unsupported_interface", "Require Python, return-value solutions")
    files = {}
    for source in solution.get("sources", []):
        path = relative_path(source.get("path", ""))
        content = source.get("content")
        if path in files or not isinstance(content, str) or not content.strip():
            raise ImportProblem("invalid_source", f"Duplicate or empty source: {path}")
        if not path.endswith(".py"):
            raise ImportProblem("unsupported_source", f"Only Python sources are supported: {path}")
        try:
            ast.parse(content, filename=path)
        except SyntaxError as exc:
            raise ImportProblem("invalid_source", str(exc)) from exc
        files[path] = content
    entry = spec.get("entry_point", "")
    if entry.count("::") != 1:
        raise ImportProblem("invalid_entry_point", f"Invalid entry point: {entry!r}")
    path, symbol = entry.split("::")
    if path not in files or not symbol.isidentifier():
        raise ImportProblem("invalid_entry_point", f"Missing source or invalid symbol: {entry}")
    # Namespace loading preserves package-relative imports. Absolute local
    # imports would collide between baseline and reference; never silently bind
    # the wrong helper. External package imports (torch, aiter, etc.) are retained.
    local_roots = {PurePosixPath(p).parts[0].removesuffix(".py") for p in files}
    for path, content in files.items():
        for node in ast.walk(ast.parse(content)):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.append(node.module)
            if any(n.split(".")[0] in local_roots for n in names):
                raise ImportProblem("unsupported_local_import", f"{path}: use package-relative local imports")
    return files


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    definition: dict
    rows: list[dict]
    baseline: dict
    reference: dict
    origins: dict[str, str]

    def contract(self) -> dict:
        return {"definition": self.definition, "rows": self.rows,
                "baseline_spec": self.baseline["spec"], "reference_spec": self.reference["spec"]}

    @property
    def digest(self) -> str:
        return fingerprint({**self.contract(), "baseline": self.baseline, "reference": self.reference})

    def summary(self) -> dict:
        return {"task_id": self.task_id, "op_type": self.definition["op_type"],
                "cases": len(self.rows), "digest": self.digest, "origins": self.origins,
                "target_hardware": self.baseline["spec"].get("target_hardware", [])}


def _validate_rows(definition: dict, rows: list[dict], origin: str) -> None:
    axes = definition.get("axes", {})
    if not axes or not definition.get("inputs") or not definition.get("outputs"):
        raise ImportProblem("invalid_definition", f"{origin}: axes, inputs and outputs are required")
    for name, axis in axes.items():
        if axis.get("type") not in {"var", "const"}:
            raise ImportProblem("unsupported_axis", f"{origin}: axis {name}")
        if axis["type"] == "const" and (type(axis.get("value")) is not int or axis["value"] <= 0):
            raise ImportProblem("invalid_axis", f"{origin}: axis {name} must be positive")
    variable = {k for k, v in axes.items() if v["type"] == "var"}
    for group in ("inputs", "outputs"):
        for name, spec in definition[group].items():
            shape = spec.get("shape")
            if shape is None and group == "inputs":
                continue
            if not isinstance(shape, list) or any(
                not ((isinstance(d, str) and d in axes) or (type(d) is int and d > 0)) for d in shape
            ):
                raise ImportProblem("unsupported_shape", f"{origin}: {name}: require literal dimensions or axis names")
    seen = set()
    for index, row in enumerate(rows, 1):
        location = f"{origin}:{index}"
        if row.get("definition") != definition["name"]:
            raise ImportProblem("mixed_definitions", f"{location}: one definition per JSONL is required")
        workload = row.get("workload", {})
        uuid = workload.get("uuid")
        if not isinstance(uuid, str) or not uuid or uuid in seen:
            raise ImportProblem("duplicate_case", f"{location}: missing or duplicate UUID {uuid!r}")
        seen.add(uuid)
        if set(workload.get("axes", {})) != variable:
            raise ImportProblem("invalid_axes", f"{location}: variable axes must be {sorted(variable)}")
        if any(type(v) is not int or v <= 0 for v in workload["axes"].values()):
            raise ImportProblem("invalid_axes", f"{location}: positive integer dimensions required")
        dimensions = {**{k: v["value"] for k, v in axes.items() if v["type"] == "const"}, **workload["axes"]}
        # Never eval source expressions. Initially support only comparisons of
        # dimensions and integer literals; reject everything else explicitly.
        import operator
        comparisons = {"<=": operator.le, "<": operator.lt, ">=": operator.ge,
                       ">": operator.gt, "==": operator.eq, "!=": operator.ne}
        for constraint in definition.get("constraints", []):
            match = re.fullmatch(r"\s*(\w+)\s*(<=|>=|==|!=|<|>)\s*(\w+)\s*", constraint)
            if not match:
                raise ImportProblem("unsupported_constraint", f"{location}: {constraint}")
            left, op, right = match.groups()
            if any(v not in dimensions and not v.isdecimal() for v in (left, right)):
                raise ImportProblem("unsupported_constraint", f"{location}: {constraint}")
            values = [dimensions[v] if v in dimensions else int(v) for v in (left, right)]
            if not comparisons[op](*values):
                raise ImportProblem("constraint_failed", f"{location}: {constraint}")
        if set(workload.get("inputs", {})) != set(definition["inputs"]):
            raise ImportProblem("invalid_inputs", f"{location}: input names differ from definition")
        for name, value in workload["inputs"].items():
            if value.get("type") not in {"random", "scalar"}:
                raise ImportProblem("unsupported_input", f"{location}: {name}: {value.get('type')}")
            allowed = {"type", "value"} if value["type"] == "scalar" else {"type"}
            if set(value) - allowed:
                raise ImportProblem("unsupported_input", f"{location}: {name}: unsupported descriptor fields")
            if value["type"] == "scalar" and type(value.get("value")) not in {int, float, bool}:
                raise ImportProblem("invalid_scalar", f"{location}: {name}")
            if (value["type"] == "scalar") != (definition["inputs"][name].get("shape") is None):
                raise ImportProblem("invalid_scalar", f"{location}: {name}: scalar/tensor descriptor mismatch")


def inspect_bundle(root: Path, selections: dict | None = None) -> list[TaskSpec]:
    root = root.resolve()
    selections = selections or {}
    if not root.is_dir():
        raise ImportProblem("missing_bundle", f"Not a directory: {root}")
    # Reject links rather than accidentally reading files outside the bundle.
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ImportProblem("symlink", "Bundle must contain regular files, not symbolic links")
    definitions = {}
    for path in sorted((root / "definitions").rglob("*.json")):
        data = read_json(path)
        name = data.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in definitions:
            raise ImportProblem("invalid_definition", f"Duplicate or invalid definition name in {path}")
        definitions[name] = (data, path)
    solutions: dict[tuple[str, str], list] = {}
    for role in ("baseline", "reference"):
        for path in sorted((root / "solutions" / role).rglob("*.json")):
            data = read_json(path)
            solutions.setdefault((role, data.get("definition")), []).append((data, path))
    tasks, seen_definitions = [], set()
    for path in sorted((root / "workloads").rglob("*.jsonl")):
        rows = []
        for index, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = parse_json(line, f"{path}:{index}")
                if not isinstance(row, dict):
                    raise ValueError("row is not an object")
                rows.append(row)
            except ValueError as exc:
                raise ImportProblem("invalid_jsonl", f"{path}:{index}: {exc}") from exc
        if not rows:
            raise ImportProblem("empty_workload", str(path))
        name = rows[0].get("definition")
        if name not in definitions or name in seen_definitions:
            raise ImportProblem("ambiguous_workload", f"{path}: missing definition or multiple workload files for {name}")
        seen_definitions.add(name)
        definition, definition_path = definitions[name]
        _validate_rows(definition, rows, str(path.relative_to(root)))
        chosen, origins = {}, {"definition": str(definition_path.relative_to(root)),
                               "workload": str(path.relative_to(root))}
        for role in ("baseline", "reference"):
            matches = solutions.get((role, name), [])
            selected = selections.get(name, {}).get(role)
            if selected:
                matches = [item for item in matches if item[0].get("name") == selected]
            if len(matches) != 1:
                raise ImportProblem("ambiguous_solution", f"{name}: choose exactly one {role}; found {len(matches)}")
            chosen[role], source = matches[0]
            source_files(chosen[role])
            origins[role] = str(source.relative_to(root))
        tasks.append(TaskSpec(name, definition, rows, chosen["baseline"], chosen["reference"], origins))
    if not tasks:
        raise ImportProblem("empty_bundle", "No workloads/**/*.jsonl files found")
    return tasks
