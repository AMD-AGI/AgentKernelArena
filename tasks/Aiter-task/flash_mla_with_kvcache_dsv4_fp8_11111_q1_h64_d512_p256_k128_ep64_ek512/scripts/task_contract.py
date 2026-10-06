"""Task-local declarations and case identities; no Arena or agent imports."""
from __future__ import annotations

import ast
import itertools
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

BASE_AXES = {"query_length", "heads", "head_dim", "page_size", "one", "packed_width", "topk"}
EXTRA_AXES = {"extra_page_size", "extra_topk"}
DTYPES = {"bfloat16", "float8_e4m3fn", "int32", "float32"}
# Each KV pool: its cache, index table, length vector, and the axis bounding its prefix.
POOLS = {
    "sparse": {"cache": "kv_cache", "indices": "sparse_indices", "lengths": "sparse_lens",
               "width": "topk", "pages": "pages", "page_size": "page_size"},
    "extra": {"cache": "extra_kv_cache", "indices": "extra_sparse_indices", "lengths": "extra_sparse_lens",
              "width": "extra_topk", "pages": "extra_pages", "page_size": "extra_page_size"},
}


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


def pools(workload: dict) -> list[str]:
    """The KV pools this definition declares, main sliding window first."""
    return ["sparse", "extra"] if "extra_topk" in workload["axes"] else ["sparse"]


def variable_axes(workload: dict) -> list[str]:
    return ["batch", "pages", "extra_pages"] if "extra_topk" in workload["axes"] else ["batch", "pages"]


def grid_combinations(workload: dict) -> list[tuple[int, ...]]:
    """All length combinations of the declared grid, in pool order."""
    return list(itertools.product(*(workload["length_grid"][pool] for pool in pools(workload))))


def row_lengths(case: dict, workload: dict) -> dict[str, list[int]] | None:
    """Declared per-row valid lengths per pool, or None for the initializer's own lengths."""
    lengths = case["lengths"]
    if lengths == "bundle":
        return None
    if "uniform" in lengths:
        return {pool: [lengths["uniform"][pool]] * case["batch"] for pool in pools(workload)}
    combos, rule = grid_combinations(workload), lengths["cycle"]
    rows = [combos[(rule["stride"] * row + rule["offset"]) % len(combos)] for row in range(case["batch"])]
    return {pool: [row[index] for row in rows] for index, pool in enumerate(pools(workload))}


def index_rule(case: dict) -> dict:
    """Index pattern of a case: -1 holes inside prefixes and legal tails beyond them."""
    lengths = case["lengths"]
    if not isinstance(lengths, dict):
        return {"holes": {}, "legal_tail": False}
    return {"holes": dict(lengths.get("holes", {})), "legal_tail": lengths.get("tail") == "legal"}


def _check_specs(specs: dict, dims: set, *, outputs: bool) -> None:
    if not isinstance(specs, dict) or not specs:
        raise ValueError("Tensor specifications must be a nonempty mapping")
    for name, spec in specs.items():
        if (not isinstance(name, str) or not name.isidentifier() or not isinstance(spec, dict)
                or set(spec) != {"shape", "dtype"} or spec["dtype"] not in DTYPES):
            raise ValueError(f"Invalid tensor specification: {name}")
        shape = spec["shape"]
        if shape is None and not outputs:
            continue
        if not isinstance(shape, list) or not shape or any(d not in dims for d in shape):
            raise ValueError(f"{name}: shape must name declared dimensions")


def _check_lengths(case: dict, workload: dict) -> None:
    lengths, names, axes = case["lengths"], pools(workload), workload["axes"]
    if lengths == "bundle":
        return
    if not isinstance(lengths, dict):
        raise ValueError(f"Case {case['case_id']}: invalid length descriptor")
    if set(lengths) == {"cycle"}:
        rule, count = lengths["cycle"], len(grid_combinations(workload))
        if (not isinstance(rule, dict) or set(rule) != {"stride", "offset"}
                or any(type(rule[k]) is not int for k in rule)
                or not 0 <= rule["offset"] < count or math.gcd(rule["stride"], count) != 1):
            raise ValueError(f"Case {case['case_id']}: cycle must permute the length grid")
        return
    if "uniform" not in lengths or not set(lengths) <= {"uniform", "holes", "tail"}:
        raise ValueError(f"Case {case['case_id']}: invalid length descriptor")
    uniform = lengths["uniform"]
    if not isinstance(uniform, dict) or set(uniform) != set(names):
        raise ValueError(f"Case {case['case_id']}: uniform lengths must name every pool")
    for pool, value in uniform.items():
        if type(value) is not int or not 0 <= value <= axes[POOLS[pool]["width"]]:
            raise ValueError(f"Case {case['case_id']}: {pool} length outside its index width")
    holes = lengths.get("holes", {})
    if (not isinstance(holes, dict) or not set(holes) <= set(names)
            or any(type(k) is not int or k < 1 for k in holes.values())
            or ("holes" in lengths and not holes)):
        raise ValueError(f"Case {case['case_id']}: holes must give a positive period per pool")
    if "tail" in lengths:
        if lengths["tail"] != "legal" or not any(
                0 < uniform[pool] < axes[POOLS[pool]["width"]] for pool in names):
            raise ValueError(f"Case {case['case_id']}: a legal tail needs a pool with a partial prefix")


def case_manifest(workload: dict) -> list[dict]:
    """Complete protected case list, independent of candidate execution/results."""
    if workload["op_type"] != "mla":
        raise ValueError(f"Unsupported operator: {workload['op_type']}")
    axes = workload["axes"]
    if set(axes) not in (BASE_AXES, BASE_AXES | EXTRA_AXES) or any(
            type(x) is not int or x <= 0 for x in axes.values()):
        raise ValueError("Invalid operator axes")
    if workload["variable_axes"] != variable_axes(workload):
        raise ValueError("Invalid variable axes")
    dims = set(axes) | set(workload["variable_axes"])
    _check_specs(workload["inputs"], dims, outputs=False)
    _check_specs(workload["outputs"], dims, outputs=True)
    if list(workload["outputs"]) != ["output", "lse"]:
        raise ValueError("Outputs must be output and lse")
    scalars = workload["scalars"]
    declared = {name for name, spec in workload["inputs"].items() if spec["shape"] is None}
    if (set(scalars) != declared or declared != {"sm_scale"} or type(scalars["sm_scale"]) is not float
            or not math.isfinite(scalars["sm_scale"]) or scalars["sm_scale"] <= 0):
        raise ValueError("The scalar input must be a positive finite sm_scale")
    grid = workload["length_grid"]
    if not isinstance(grid, dict) or set(grid) != set(pools(workload)):
        raise ValueError("The length grid must name every pool")
    for pool, values in grid.items():
        if (not isinstance(values, list) or not values or len(set(values)) != len(values)
                or any(type(v) is not int or not 0 <= v <= axes[POOLS[pool]["width"]] for v in values)):
            raise ValueError(f"Invalid {pool} length grid")
    if type(workload["seed"]) is not int or not workload["definition"]:
        raise ValueError("Invalid definition or seed")
    bench = workload["bench"]
    for key in ("warmup", "repetition"):
        if type(bench[key]) is not int or bench[key] <= 0:
            raise ValueError(f"Invalid benchmark {key}")
    if (type(bench["target_ms"]) not in (float, int)
            or not math.isfinite(bench["target_ms"]) or bench["target_ms"] <= 0):
        raise ValueError("Invalid benchmark target_ms")
    fields = {"case_id", "uuid", *workload["variable_axes"], "lengths"}
    records, ids, uuids = [], set(), set()
    for case in workload["cases"]:
        if not isinstance(case, dict) or set(case) != fields:
            raise ValueError("Invalid workload case fields")
        ident, uuid = case["case_id"], case["uuid"]
        if (not isinstance(ident, str) or not ident or ident in ids
                or not isinstance(uuid, str) or not uuid or uuid in uuids
                or any(type(case[a]) is not int or case[a] <= 0 for a in workload["variable_axes"])):
            raise ValueError("Invalid or duplicate workload case")
        ids.add(ident)
        uuids.add(uuid)
        for pool in pools(workload):
            spec = POOLS[pool]
            if case[spec["pages"]] * axes[spec["page_size"]] > 2**31:
                raise ValueError(f"Case {ident}: {pool} pool capacity exceeds int32 slots")
        _check_lengths(case, workload)
        values = {**axes, **{a: case[a] for a in workload["variable_axes"]}}
        params = {**values, **scalars, "lengths": case["lengths"], "seed": workload["seed"],
                  "definition": workload["definition"], "uuid": uuid}
        records.append({"test_case_id": ident,
                        "shape": [case["batch"], axes["query_length"], axes["heads"], axes["head_dim"]],
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
    # FlyDSL exposes math intrinsics such as exp under the same names, so these
    # are rejected only when reached through a torch binding.
    torch_compute = {"softmax", "log_softmax", "logsumexp", "exp", "exp2",
                     "scaled_dot_product_attention", "flash_attention"}
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
            if root in ("aiter", "sglang", "sgl_kernel", "tilelang", "flash_mla"):
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
