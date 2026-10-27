"""Task-local declarations and case identities; no Arena or agent imports."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


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


def case_manifest(workload: dict) -> list[dict]:
    """Complete protected case list, independent of candidate execution/results."""
    kind = workload["op_type"]
    if kind not in ("gemm", "moe"):
        raise ValueError(f"Unsupported operator: {kind}")
    axes = workload["axes"]
    required = ({"n", "k"} if kind == "gemm" else
                {"model_dim", "num_experts", "topk", "w1_rows", "w1_cols",
                 "w2_cols", "w1_scale_cols", "w2_scale_cols"})
    if set(axes) != required or any(type(x) is not int or x <= 0 for x in axes.values()):
        raise ValueError("Invalid operator axes")
    if type(workload["seed"]) is not int or not workload["definition"]:
        raise ValueError("Invalid definition or seed")
    bench = workload["bench"]
    for key in ("warmup", "repetition"):
        if type(bench[key]) is not int or bench[key] <= 0:
            raise ValueError(f"Invalid benchmark {key}")
    if (type(bench["target_ms"]) not in (float, int)
            or not math.isfinite(bench["target_ms"]) or bench["target_ms"] <= 0):
        raise ValueError("Invalid benchmark target_ms")
    if kind == "gemm" and workload["trans_b"] is not True:
        raise ValueError("This GEMM contract requires transposed B")
    if kind == "moe":
        d, i = axes["model_dim"], axes["w1_rows"] // 2
        if (axes["w1_rows"] % 2 or d % 32 or i % 32
                or axes["w1_cols"] != d // 2 or axes["w2_cols"] != i // 2
                or axes["w1_scale_cols"] != d // 32
                or axes["w2_scale_cols"] != i // 32
                or axes["topk"] > axes["num_experts"]):
            raise ValueError("Inconsistent packed MoE layout")
    records, ids, uuids = [], set(), set()
    variable = "m" if kind == "gemm" else "num_tokens"
    for case in workload["cases"]:
        ident, uuid, size = case["case_id"], case["uuid"], case[variable]
        if (not isinstance(ident, str) or not ident or ident in ids
                or not isinstance(uuid, str) or not uuid or uuid in uuids
                or type(size) is not int or size <= 0):
            raise ValueError("Invalid or duplicate workload case")
        ids.add(ident)
        uuids.add(uuid)
        params = {**axes, variable: size, "seed": workload["seed"],
                  "definition": workload["definition"], "uuid": uuid}
        if kind == "gemm":
            shape = [size, axes["n"], axes["k"]]
            params["trans_b"] = True
        else:
            shape = [size, axes["model_dim"], axes["w1_rows"] // 2,
                     axes["num_experts"], axes["topk"]]
            params.update(activation=0, doweight_stage1=False,
                          weight_dtype="float4_e2m1fn_x2", scale_dtype="uint8",
                          quant_group_size=32, preshuffled=True)
        records.append({"test_case_id": ident, "shape": shape, "dtype": "bfloat16",
                        "params": params, "status": "PASS",
                        "checks": ["correctness", "performance"]})
    if not records:
        raise ValueError("The task has no workload cases")
    return records


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
            root, leaf = name.split(".")[0], name.split(".")[-1]
            if root == "aiter":
                raise RuntimeError(f"candidate imports the framework under test: {name}")
            if root == "scripts" or leaf.startswith("task_") or root not in allowed:
                raise RuntimeError(f"candidate imports a protected task or unsupported module: {name}")
            if root == "torch" and any(p in matrix for p in name.split(".")):
                raise RuntimeError(f"candidate imports library matrix computation: {name}")
        if ((isinstance(node, ast.BinOp) or isinstance(node, ast.AugAssign))
                and isinstance(node.op, ast.MatMult)):
            raise RuntimeError("candidate uses the library matrix multiplication operator")
        if isinstance(node, ast.Attribute) and node.attr in matrix:
            raise RuntimeError(f"candidate references library matrix computation: {node.attr}")
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
