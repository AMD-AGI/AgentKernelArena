"""Task-local declarations and case identities; no Arena or agent imports."""
from __future__ import annotations

import ast
import json
import math
import operator
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

AXES = {"plan_fields", "k", "page_size"}
VARIABLE_AXES = ["batch", "width", "plan_rows", "pages"]
DTYPES = {"float32", "int32", "int64"}
CASE_FIELDS = {"case_id", "uuid", *VARIABLE_AXES, "lengths"}
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


def _check_specs(specs: dict, dims: set, *, outputs: bool) -> None:
    if not isinstance(specs, dict) or not specs:
        raise ValueError("Tensor specifications must be a nonempty mapping")
    for name, spec in specs.items():
        fields = {"shape", "dtype", "param"} if outputs else {"shape", "dtype"}
        if (not isinstance(name, str) or not name.isidentifier() or not isinstance(spec, dict)
                or set(spec) != fields or spec["dtype"] not in DTYPES):
            raise ValueError(f"Invalid tensor specification: {name}")
        if outputs and spec["param"] != name:
            raise ValueError(f"{name}: destination parameter must carry the output name")
        shape = spec["shape"]
        if shape is None and not outputs:
            continue
        if not isinstance(shape, list) or not shape or any(d not in dims for d in shape):
            raise ValueError(f"{name}: shape must name declared dimensions")


def _check_scalars(scalars: dict, inputs: dict, axes: dict) -> None:
    declared = {name for name, spec in inputs.items() if spec["shape"] is None}
    if not isinstance(scalars, dict) or set(scalars) != declared:
        raise ValueError("Scalar values must cover exactly the declared scalar inputs")
    for name, value in scalars.items():
        if inputs[name]["dtype"] not in ("int32", "int64") or type(value) is not int:
            raise ValueError(f"Scalar {name} does not match its declared dtype")
    if scalars.get("page_size") != axes["page_size"]:
        raise ValueError("The page_size scalar must equal the page_size axis")


def capacity(case: dict, axes: dict) -> int:
    """Largest legal valid length: score width and page-table coverage both bound it."""
    return min(case["width"], case["pages"] * axes["page_size"])


def row_lengths(case: dict, workload: dict) -> list[int] | None:
    """Declared per-row valid lengths, or None for the initializer's own lengths."""
    lengths = case["lengths"]
    if lengths == "bundle":
        return None
    if set(lengths) == {"uniform"}:
        return [lengths["uniform"]] * case["batch"]
    values = workload["boundary_lengths"]
    rule = lengths["cycle"]
    return [values[(rule["stride"] * row + rule["offset"]) % len(values)] for row in range(case["batch"])]


def _check_lengths(case: dict, workload: dict, limit: int) -> None:
    lengths = case["lengths"]
    if lengths == "bundle":
        return
    if not isinstance(lengths, dict) or len(lengths) != 1:
        raise ValueError(f"Case {case['case_id']}: invalid length descriptor")
    if "uniform" in lengths:
        value = lengths["uniform"]
        if type(value) is not int or not 0 <= value <= limit:
            raise ValueError(f"Case {case['case_id']}: uniform length outside [0, {limit}]")
        return
    rule = lengths.get("cycle")
    count = len(workload["boundary_lengths"])
    if (not isinstance(rule, dict) or set(rule) != {"stride", "offset"}
            or any(type(rule[k]) is not int for k in rule)
            or not 0 <= rule["offset"] < count or math.gcd(rule["stride"], count) != 1):
        raise ValueError(f"Case {case['case_id']}: cycle must permute the boundary lengths")
    if any(value > limit for value in workload["boundary_lengths"]):
        raise ValueError(f"Case {case['case_id']}: a boundary length exceeds capacity {limit}")


def case_manifest(workload: dict) -> list[dict]:
    """Complete protected case list, independent of candidate execution/results."""
    if workload["op_type"] != "topk":
        raise ValueError(f"Unsupported operator: {workload['op_type']}")
    axes = workload["axes"]
    if set(axes) != AXES or any(type(x) is not int or x <= 0 for x in axes.values()):
        raise ValueError("Invalid operator axes")
    if workload["variable_axes"] != VARIABLE_AXES:
        raise ValueError("Invalid variable axes")
    dims = set(axes) | set(VARIABLE_AXES)
    _check_specs(workload["inputs"], dims, outputs=False)
    _check_specs(workload["outputs"], dims, outputs=True)
    _check_scalars(workload["scalars"], workload["inputs"], axes)
    boundary = workload["boundary_lengths"]
    if (not isinstance(boundary, list) or not boundary or len(set(boundary)) != len(boundary)
            or any(type(v) is not int or v < 0 for v in boundary)):
        raise ValueError("Invalid boundary lengths")
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
        if not isinstance(case, dict) or set(case) != CASE_FIELDS:
            raise ValueError("Invalid workload case fields")
        ident, uuid = case["case_id"], case["uuid"]
        if (not isinstance(ident, str) or not ident or ident in ids
                or not isinstance(uuid, str) or not uuid or uuid in uuids
                or any(type(case[a]) is not int or case[a] <= 0 for a in VARIABLE_AXES)):
            raise ValueError("Invalid or duplicate workload case")
        ids.add(ident)
        uuids.add(uuid)
        values = {**axes, **{a: case[a] for a in VARIABLE_AXES}}
        for constraint in workload["constraints"]:
            if not constraint_holds(constraint, values):
                raise ValueError(f"Case {ident} violates constraint: {constraint}")
        _check_lengths(case, workload, capacity(case, axes))
        params = {**values, **workload["scalars"], "lengths": case["lengths"],
                  "seed": workload["seed"], "definition": workload["definition"], "uuid": uuid}
        records.append({"test_case_id": ident, "shape": [case["batch"], case["width"]],
                        "dtype": "float32", "params": params, "status": "PASS",
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
    # Library selection is the operator itself. FlyDSL may expose unrelated
    # helpers under generic names, so these are rejected only through torch.
    torch_selection = {"topk", "sort", "argsort", "msort", "kthvalue", "searchsorted",
                       "unique", "bucketize"}
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
            if root in ("aiter", "sglang", "sgl_kernel"):
                raise RuntimeError(f"candidate imports the framework under test: {name}")
            if root == "scripts" or leaf.startswith("task_") or root not in allowed:
                raise RuntimeError(f"candidate imports a protected task or unsupported module: {name}")
            if root == "torch" and any(p in torch_selection for p in parts):
                raise RuntimeError(f"candidate imports library selection: {name}")
        if isinstance(node, ast.Attribute) and node.attr in torch_selection \
                and _attribute_root(node) in torch_names:
            raise RuntimeError(f"candidate references torch selection: {node.attr}")
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
