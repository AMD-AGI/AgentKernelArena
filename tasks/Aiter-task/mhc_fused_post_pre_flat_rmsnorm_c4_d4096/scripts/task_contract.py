"""Task-local declarations and case identities; no Arena or agent imports."""
from __future__ import annotations

import ast
import json
import math
import operator
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

AXES = {"streams", "hidden_size", "projection_size", "mixes", "scale_groups", "one"}
DTYPES = {"bfloat16", "float32", "int32"}
_ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub,
               ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}
_COMPARISONS = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}


def task_path(value: str, *, must_exist: bool = True) -> Path:
    if (not isinstance(value, str) or not value or value.startswith("/")
            or "\\" in value or ":" in value
            or any(p in ("", ".", "..") for p in value.split("/"))):
        raise ValueError(f"Expected a normalized task-relative path: {value!r}")
    path = (ROOT / value).resolve(strict=must_exist)
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Task path escapes the workspace: {value}")
    return path


def load_config() -> dict:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    if config.get("schema_version") != 2:
        raise ValueError("This runner requires task schema v2")
    return config


def load_workload(config: dict | None = None) -> dict:
    config = load_config() if config is None else config
    workload = json.loads(task_path(config["evaluation"]["workloads"]).read_text())
    case_manifest(workload)  # Validate before any candidate code is imported.
    return workload


def candidate_entry(config: dict | None = None) -> dict:
    config = load_config() if config is None else config
    entries = config["candidate"]["entrypoints"]
    if len(entries) != 1 or entries[0]["kind"] != "builder":
        raise ValueError("This task requires one declared builder entrypoint")
    entry = entries[0]
    if not entry["symbol"].isidentifier():
        raise ValueError("The builder symbol must be a Python identifier")
    task_path(entry["file"], must_exist=False)
    return entry


def _dimension(node: ast.AST, dims: dict) -> int:
    """Integer arithmetic over declared dimension names; source text is never evaluated."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name) and node.id in dims:
        return dims[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
        return _ARITHMETIC[type(node.op)](_dimension(node.left, dims), _dimension(node.right, dims))
    raise ValueError(f"Unsupported dimension expression: {ast.dump(node)}")


def constraint_holds(text: str, dims: dict) -> bool:
    tree = ast.parse(text, mode="eval").body
    if (not isinstance(tree, ast.Compare) or len(tree.ops) != 1
            or type(tree.ops[0]) not in _COMPARISONS):
        raise ValueError(f"Unsupported constraint: {text}")
    return _COMPARISONS[type(tree.ops[0])](_dimension(tree.left, dims),
                                           _dimension(tree.comparators[0], dims))


def _check_tensor_specs(specs: dict, dims: set, *, allow_scalar: bool) -> None:
    if not isinstance(specs, dict) or not specs:
        raise ValueError("Tensor specifications must be a nonempty mapping")
    for name, spec in specs.items():
        if (not isinstance(name, str) or not name.isidentifier() or not isinstance(spec, dict)
                or set(spec) != {"shape", "dtype"} or spec["dtype"] not in DTYPES):
            raise ValueError(f"Invalid tensor specification: {name}")
        shape = spec["shape"]
        if shape is None and allow_scalar:
            continue
        if not isinstance(shape, list) or not shape or any(d not in dims for d in shape):
            raise ValueError(f"{name}: shape must name declared dimensions")


def _check_scalars(scalars: dict, inputs: dict) -> None:
    declared = {name for name, spec in inputs.items() if spec["shape"] is None}
    if not isinstance(scalars, dict) or set(scalars) != declared:
        raise ValueError("Scalar values must cover exactly the declared scalar inputs")
    for name, value in scalars.items():
        if inputs[name]["dtype"] == "int32":
            valid = type(value) is int and -2**31 <= value < 2**31
        else:
            valid = type(value) is float and math.isfinite(value)
        if not valid:
            raise ValueError(f"Scalar {name} does not match its declared dtype")


def case_manifest(workload: dict) -> list[dict]:
    """Complete protected case list, independent of candidate execution/results."""
    if workload["op_type"] != "mhc":
        raise ValueError(f"Unsupported operator: {workload['op_type']}")
    axes, variable = workload["axes"], workload["variable_axis"]
    if set(axes) != AXES or any(type(x) is not int or x <= 0 for x in axes.values()):
        raise ValueError("Invalid operator axes")
    if not isinstance(variable, str) or not variable.isidentifier() or variable in axes:
        raise ValueError("Invalid variable axis")
    dims = set(axes) | {variable}
    _check_tensor_specs(workload["inputs"], dims, allow_scalar=True)
    _check_tensor_specs(workload["outputs"], dims, allow_scalar=False)
    _check_scalars(workload["scalars"], workload["inputs"])
    if type(workload["seed"]) is not int or not workload["definition"]:
        raise ValueError("Invalid definition or seed")
    bench = workload["bench"]
    for key in ("warmup", "repetition"):
        if type(bench[key]) is not int or bench[key] <= 0:
            raise ValueError(f"Invalid benchmark {key}")
    if (type(bench["target_ms"]) not in (float, int)
            or not math.isfinite(bench["target_ms"]) or bench["target_ms"] <= 0):
        raise ValueError("Invalid benchmark target_ms")
    records, ids, uuids = [], set(), set()
    for case in workload["cases"]:
        ident, uuid, size = case["case_id"], case["uuid"], case[variable]
        if (not isinstance(ident, str) or not ident or ident in ids
                or not isinstance(uuid, str) or not uuid or uuid in uuids
                or type(size) is not int or size <= 0 or set(case) != {"case_id", "uuid", variable}):
            raise ValueError("Invalid or duplicate workload case")
        ids.add(ident)
        uuids.add(uuid)
        values = {**axes, variable: size}
        for constraint in workload["constraints"]:
            if not constraint_holds(constraint, values):
                raise ValueError(f"Case {ident} violates constraint: {constraint}")
        params = {**values, **workload["scalars"], "seed": workload["seed"],
                  "definition": workload["definition"], "uuid": uuid}
        records.append({"test_case_id": ident,
                        "shape": [size, axes["streams"], axes["hidden_size"]],
                        "dtype": "bfloat16", "params": params, "status": "PASS",
                        "checks": ["correctness", "performance"]})
    if not records:
        raise ValueError("The task has no workload cases")
    return records


def _torch_aliases(tree: ast.AST) -> set[str]:
    """Local names bound to torch or one of its submodules."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "torch":
                    aliases.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch":
            aliases.update(alias.asname or alias.name for alias in node.names)
    return aliases


def _attribute_root(node: ast.Attribute) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def assert_source_independent(source: str) -> None:
    """Reject direct protected imports, including imported members and aliases.

    This is a static guard, not proof against reflection or arbitrary hostile
    Python. The task's dependency policy and numerical checks also apply.
    """
    tree = ast.parse(source)
    allowed = {"__future__", "typing", "collections", "dataclasses", "enum",
               "functools", "itertools", "math", "operator", "numbers",
               "abc", "types", "torch", "flydsl"}
    matrix = {"matmul", "mm", "bmm", "einsum", "linear", "addmm", "addbmm",
              "baddbmm", "tensordot"}
    # FlyDSL exposes math intrinsics such as rsqrt and exp under the same names,
    # so these are rejected only when reached through a torch binding.
    torch_compute = {"softmax", "log_softmax", "sigmoid", "rsqrt", "logsumexp",
                     "rms_norm", "layer_norm", "normalize", "silu"}
    torch_names = _torch_aliases(tree)
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise RuntimeError("candidate imports a protected task/local module")
            prefix = node.module or ""
            names = [prefix, *(f"{prefix}.{a.name}" for a in node.names)]
        for name in names:
            parts = name.split(".")
            root, leaf = parts[0], parts[-1]
            if root == "aiter":
                raise RuntimeError(f"candidate imports the framework under test: {name}")
            if root == "scripts" or leaf.startswith("task_") or root not in allowed:
                raise RuntimeError(f"candidate imports a protected task or unsupported module: {name}")
            if root == "torch" and any(p in matrix | torch_compute for p in parts):
                raise RuntimeError(f"candidate imports library operator computation: {name}")
        if ((isinstance(node, ast.BinOp) or isinstance(node, ast.AugAssign))
                and isinstance(node.op, ast.MatMult)):
            raise RuntimeError("candidate uses the library matrix multiplication operator")
        if isinstance(node, ast.Attribute):
            if node.attr in matrix:
                raise RuntimeError(f"candidate references library matrix computation: {node.attr}")
            if node.attr in torch_compute and _attribute_root(node) in torch_names:
                raise RuntimeError(f"candidate references torch operator computation: {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("__import__", "eval", "exec", "compile", "open"):
                raise RuntimeError(f"candidate uses unsupported dynamic access: {node.func.id}")


def json_safe(value):
    """Preserve nonfinite diagnostics as null; callers provide an explanation."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value
