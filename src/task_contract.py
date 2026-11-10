"""Schema-v3 kernel tasks: a language-neutral operator contract.

A v3 task package declares the operator interface, a high-precision PyTorch
reference, a baseline implementation in any supported language and its
workloads. It never names the language a candidate must use: the run
configuration selects that target. This module validates ``task.yaml`` and
resolves it, for one target language, into the framework's execution
configuration, the evaluator's ``contract.json`` and the candidate's starting
``solution/`` directory.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Mapping

import yaml

from .task_spec import TaskConfigError, relative_path, required_gpu_arches

TASK_FILE = "task.yaml"
SCHEMA_VERSION = 3
TARGET_LANGUAGES = ("triton", "gluon", "hip", "flydsl", "asm")
BASELINE_LANGUAGES = TARGET_LANGUAGES + ("tilelang",)
PYTHON_LANGUAGES = ("triton", "gluon", "flydsl", "tilelang", "asm")
CATEGORIES = (
    "activation", "attention", "convolution", "elementwise", "embedding", "gemm", "kv_cache",
    "moe", "normalization", "positional", "quantization", "reduction", "sampling", "sorting",
    "state_space", "other",
)
DTYPES = (
    "bfloat16", "float16", "float32", "float64", "int8", "uint8", "int16", "int32", "int64", "bool",
    "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz", "float8_e8m0fnu",
    "float4_e2m1fn_x2",
)
FLOAT_DTYPES = ("bfloat16", "float16", "float32", "float64", "float8_e4m3fn", "float8_e4m3fnuz",
                "float8_e5m2", "float8_e5m2fnuz")
SCALAR_DTYPES = ("float32", "float64", "int32", "int64", "bool")
DISTRIBUTIONS = {
    "normal": {"mean", "std"}, "uniform": {"low", "high"}, "log_uniform": {"log2_low", "log2_high", "signed"},
    "randint": {"low", "high"}, "permutation": set(), "zeros": set(), "generator": set(),
}
BOUNDS = {
    "baseline_relative": {"k", "max_violation_fraction", "baseline_min_sqnr_db"},
    "exact": set(),
    "tolerance": {"atol", "rtol", "max_violation_fraction"},
    "sqnr": {"min_db"},
    "output_rounding": {"max_violation_fraction"},
}
MIN_PERFORMANCE_CASES = 4
MAX_PERFORMANCE_CASES = 16
MIN_SIZE_SPAN = 100
UNALIGNED_BLOCK = 16
TIMING = {"warmup": 10, "repetition": 100, "cache_flush_bytes": 1 << 30}
EVAL_DIR = "_aka_eval"
SOLUTION_DIR = "solution"
BASELINE_DIR = "baseline"
TEMPLATE_MARKER = "AKA-UNIMPLEMENTED-TEMPLATE"
FORBIDDEN_PACKAGE_NAMES = ("config.yaml", SOLUTION_DIR, EVAL_DIR)


def _mapping(value: Any, where: str, allowed: set[str], required: set[str] = frozenset()) -> dict:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise TaskConfigError(f"{where} must be a mapping with string keys")
    unknown = set(value) - allowed
    if unknown:
        raise TaskConfigError(f"{where}: unknown fields {sorted(unknown)}")
    missing = set(required) - set(value)
    if missing:
        raise TaskConfigError(f"{where}: missing fields {sorted(missing)}")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise TaskConfigError(f"{where} must be a nonempty string")
    return value


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.isidentifier() or value.startswith("_"):
        raise TaskConfigError(f"{where} must be a public Python identifier: {value!r}")
    return value


def _int(value: Any, where: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise TaskConfigError(f"{where} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))
    return value


def _number(value: Any, where: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise TaskConfigError(f"{where} must be a finite number")
    return float(value)


def _entry(value: Any, where: str) -> dict:
    entry = _mapping(value, where, {"file", "symbol"}, {"file", "symbol"})
    path = relative_path(entry["file"], f"{where}.file")
    if not path.endswith(".py") or "/" in path:
        raise TaskConfigError(f"{where}.file must be a .py file at the task root")
    return {"file": path, "symbol": _identifier(entry["symbol"], f"{where}.symbol")}


def _expression_names(expression: Any, where: str) -> set[str]:
    if type(expression) is int:
        if expression < 0:
            raise TaskConfigError(f"{where}: negative dimension")
        return set()
    if not isinstance(expression, str):
        raise TaskConfigError(f"{where}: dimensions are ints or expressions over axes")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise TaskConfigError(f"{where}: invalid expression {expression!r}") from exc
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ("min", "max", "ceil_div"):
                raise TaskConfigError(f"{where}: unsupported call in {expression!r}")
        elif not isinstance(node, (ast.Expression, ast.BinOp, ast.Constant, ast.Load, ast.Add, ast.Sub,
                                   ast.Mult, ast.FloorDiv, ast.Mod)):
            raise TaskConfigError(f"{where}: unsupported syntax in {expression!r}")
    return names - {"min", "max", "ceil_div"}


def _evaluate(expression: Any, symbols: dict[str, int]) -> int:
    if type(expression) is int:
        return expression
    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            return symbols[node.id]
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            op = type(node.op)
            if op in (ast.FloorDiv, ast.Mod) and right == 0:
                raise TaskConfigError(f"Division by zero in {expression!r}")
            return {ast.Add: left + right, ast.Sub: left - right, ast.Mult: left * right,
                    ast.FloorDiv: left // right if right else 0, ast.Mod: left % right if right else 0}[op]
        if isinstance(node, ast.Call):
            args = [visit(a) for a in node.args]
            return {"min": min, "max": max, "ceil_div": lambda a, b: -(-a // b)}[node.func.id](*args)
        raise TaskConfigError(f"Unsupported expression {expression!r}")

    value = visit(tree)
    if type(value) is not int or value < 0:
        raise TaskConfigError(f"Expression {expression!r} evaluated to {value!r}")
    return value


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def case_seeds(task_id: str, case_id: str) -> list[int]:
    """Two framework-derived seeds per case; task code never chooses them."""
    seeds = []
    for index in range(2):
        digest = hashlib.sha256(f"aka-v3|{task_id}|{case_id}|{index}".encode()).digest()
        seeds.append(int.from_bytes(digest[:4], "big") & 0x7FFFFFFF)
    return seeds


class TaskContract:
    """A validated v3 task declaration and its workloads."""

    def __init__(self, raw: dict, workloads: dict, task_id: str):
        self.raw = raw
        self.workloads = workloads
        self.task_id = task_id

    @property
    def baseline_language(self) -> str:
        return self.raw["baseline"]["language"]

    @property
    def baseline_entry(self) -> str:
        return self.raw["baseline"]["entry"]

    @property
    def task_directory(self) -> str:
        """Where a curated task lives under tasks/, without its scenario directory."""
        operator = self.raw["operator"]
        return f"{self.baseline_language}/{operator['category']}/{operator['name']}"

    def layout_problem(self) -> str | None:
        """Why this task's ID does not follow tasks/<baseline>/<category>/<operator>/<scenario>, if it does not."""
        parent, _, scenario = self.task_id.rpartition("/")
        if parent == self.task_directory and scenario:
            return None
        return (f"task {self.task_id} must be {self.task_directory}/<scenario>: "
                "baseline.language, operator.category and operator.name name its directories")

    def candidate_entry(self, language: str) -> str:
        return f"{SOLUTION_DIR}/kernel.hip" if language == "hip" else f"{SOLUTION_DIR}/kernel.py"

    @property
    def interface(self) -> dict:
        return self.raw["interface"]

    @property
    def checked_tensors(self) -> list[str]:
        return list(self.interface["outputs"]) + list(self.interface.get("mutable", []))

    @property
    def code_objects(self) -> dict[str, str]:
        """Vendored code objects of an asm baseline: baseline path -> SHA256."""
        return dict(self.raw["baseline"]["provenance"].get("code_objects", {}))

    def resolved_contract(self, target_language: str) -> dict:
        interface = self.interface
        cases = []
        for case in self.workloads["cases"]:
            cases.append({"id": case["id"], "axes": dict(case["axes"]), "scalars": dict(case.get("scalars", {})),
                          "checks": list(case["checks"]), "seeds": case_seeds(self.task_id, case["id"])})
        compare = {}
        for name in self.checked_tensors:
            declaration = dict(self.raw["compare"][name])
            if declaration["bound"] == "baseline_relative":
                declaration.setdefault("k", 2.0)
            compare[name] = declaration
        baseline_entry = self.baseline_entry
        return {
            "version": 1,
            "task_id": self.task_id,
            "axes": deepcopy(interface["axes"]),
            "derived_axes": deepcopy(interface.get("derived_axes", {})),
            "inputs": deepcopy(interface["inputs"]),
            "outputs": deepcopy(interface["outputs"]),
            "mutable": list(interface.get("mutable", [])),
            "compare": compare,
            "element_order": self.raw.get("element_order", "fixed"),
            "timing": dict(TIMING),
            "reference": dict(self.raw["reference"]),
            "generator": deepcopy(interface.get("generator")),
            "layout": deepcopy(interface.get("layout")),
            "baseline": {"language": self.baseline_language, "root": BASELINE_DIR,
                         "entry_file": baseline_entry, "symbol": "run"},
            # SHA256 -> baseline path of every vendored code object; only these load.
            "code_objects": {digest: path for path, digest in self.code_objects.items()},
            "candidate": {"language": target_language, "root": SOLUTION_DIR,
                          "entry_file": self.candidate_entry(target_language), "symbol": "run"},
            "cases": cases,
        }

    def initial_state(self, target_language: str) -> str:
        return "implemented" if target_language == self.baseline_language else "unimplemented"

    def execution_config(self, target_language: str, task_files: list[str]) -> dict:
        """The schema-v2 execution configuration the framework runs for this target."""
        if target_language not in TARGET_LANGUAGES:
            raise TaskConfigError(f"target_language must be one of {TARGET_LANGUAGES}")
        state = self.initial_state(target_language)
        candidate = {"language": target_language, "initial_state": state,
                     "editable": [{"path": SOLUTION_DIR, "scope": "tree"}],
                     "entrypoints": [{"file": self.candidate_entry(target_language), "kind": "function",
                                      "symbol": "run"}]}
        if state == "implemented":
            candidate["initial_language"] = target_language
        instructions = [f"{EVAL_DIR}/CANDIDATE.md"]
        if "README.md" in task_files:
            instructions.insert(0, "README.md")
        provenance = self.raw["baseline"]["provenance"]
        config = {
            "schema_version": 2,
            "description": self.raw["description"],
            "instructions": instructions,
            "kernel_identity": {"logical_operator": self.raw["operator"]["name"],
                                "source_owner": provenance["source"]},
            "candidate": candidate,
            "baseline": {"kind": "provided", "language": self.baseline_language,
                         "source_files": sorted(f for f in task_files
                                                if f not in ("README.md", TASK_FILE)),
                         "correctness_policy": "required"},
            "evaluation": {"runner": ["python3", "-m", f"{EVAL_DIR}.evaluate"],
                           "workloads": self.raw["workloads"],
                           "timeout_s": self.raw.get("timeout_s", 3600)},
            "contract": {"schema_version": SCHEMA_VERSION, "task_file": TASK_FILE,
                         "target_language": target_language, "baseline_language": self.baseline_language,
                         "operator": dict(self.raw["operator"])},
        }
        if "platform_support" in self.raw:
            config["platform_support"] = deepcopy(self.raw["platform_support"])
        return config


def _validate_interface(raw: dict) -> tuple[dict, dict, list[str]]:
    interface = _mapping(raw, "interface", {"axes", "derived_axes", "inputs", "outputs", "mutable",
                                            "generator", "layout"}, {"axes", "inputs", "outputs"})
    axes = _mapping(interface["axes"], "interface.axes", set(interface["axes"]))
    if not axes:
        raise TaskConfigError("interface.axes must declare at least one axis")
    var = []
    for name, axis in axes.items():
        _identifier(name, f"axis {name}")
        if not isinstance(axis, dict) or axis.get("kind") not in ("var", "const"):
            raise TaskConfigError(f"axis {name} must declare kind var or const")
        if axis["kind"] == "const":
            _mapping(axis, f"axis {name}", {"kind", "value", "description"}, {"kind", "value"})
            _int(axis["value"], f"axis {name}.value", 0)
        else:
            _mapping(axis, f"axis {name}", {"kind", "min", "max", "multiple_of", "description"},
                     {"kind", "min", "max"})
            low = _int(axis["min"], f"axis {name}.min", 1)
            high = _int(axis["max"], f"axis {name}.max", low)
            step = _int(axis.get("multiple_of", 1), f"axis {name}.multiple_of", 1)
            if low % step or high % step:
                raise TaskConfigError(f"axis {name} bounds must be multiples of multiple_of")
            var.append(name)
    if not var:
        raise TaskConfigError("interface.axes must declare at least one var axis")
    symbols = set(axes)
    derived = interface.get("derived_axes", {})
    _mapping(derived, "interface.derived_axes", set(derived))
    for name, expression in derived.items():
        _identifier(name, f"derived axis {name}")
        if name in symbols:
            raise TaskConfigError(f"derived axis {name} duplicates an axis")
        unknown = _expression_names(expression, f"derived axis {name}") - symbols
        if unknown:
            raise TaskConfigError(f"derived axis {name} reads unknown axes {sorted(unknown)}")
        symbols.add(name)
    inputs = _mapping(interface["inputs"], "interface.inputs", set(interface["inputs"]))
    outputs = _mapping(interface["outputs"], "interface.outputs", set(interface["outputs"]))
    if not outputs and not interface.get("mutable"):
        raise TaskConfigError("interface must declare at least one output or mutable input")
    if set(inputs) & set(outputs):
        raise TaskConfigError("Input and output names must be distinct")
    generator_needed = False
    for name, declaration in inputs.items():
        _identifier(name, f"input {name}")
        if not isinstance(declaration, dict):
            raise TaskConfigError(f"input {name} must be a mapping")
        if declaration.get("kind") == "scalar":
            _mapping(declaration, f"input {name}", {"kind", "dtype", "default", "description"},
                     {"kind", "dtype", "default"})
            if declaration["dtype"] not in SCALAR_DTYPES:
                raise TaskConfigError(f"input {name}: scalar dtype must be one of {SCALAR_DTYPES}")
            _scalar_value(declaration["dtype"], declaration["default"], f"input {name}.default")
            continue
        _mapping(declaration, f"input {name}", {"kind", "shape", "dtype", "distribution", "role", "description"},
                 {"kind", "shape", "dtype", "distribution"})
        if declaration["kind"] != "tensor":
            raise TaskConfigError(f"input {name}.kind must be tensor or scalar")
        _shape(declaration["shape"], symbols, f"input {name}")
        if declaration["dtype"] not in DTYPES:
            raise TaskConfigError(f"input {name}: unknown dtype {declaration['dtype']!r}")
        if declaration.get("role", "activation") not in ("activation", "weight", "state", "index", "metadata"):
            raise TaskConfigError(f"input {name}.role is invalid")
        distribution = declaration["distribution"]
        if not isinstance(distribution, dict) or distribution.get("kind") not in DISTRIBUTIONS:
            raise TaskConfigError(f"input {name}: distribution kind must be one of {sorted(DISTRIBUTIONS)}")
        _mapping(distribution, f"input {name}.distribution", {"kind"} | DISTRIBUTIONS[distribution["kind"]])
        kind = distribution["kind"]
        generator_needed |= kind == "generator"
        if kind in ("normal",) and "std" in distribution and _number(distribution["std"], "std") <= 0:
            raise TaskConfigError(f"input {name}: std must be positive")
        if kind == "uniform" and _number(distribution["low"], "low") >= _number(distribution["high"], "high"):
            raise TaskConfigError(f"input {name}: uniform low must be below high")
        if kind == "randint" and _int(distribution["low"], "low") >= _int(distribution["high"], "high"):
            raise TaskConfigError(f"input {name}: randint low must be below high")
        if kind in ("normal", "uniform", "log_uniform") and declaration["dtype"] not in FLOAT_DTYPES:
            raise TaskConfigError(f"input {name}: {kind} needs a floating dtype")
        if kind == "zeros" and declaration.get("role") != "state":
            raise TaskConfigError(f"input {name}: only state inputs may start as zeros")
    for name, declaration in outputs.items():
        _identifier(name, f"output {name}")
        declaration = _mapping(declaration, f"output {name}", {"shape", "dtype", "description"},
                               {"shape", "dtype"})
        _shape(declaration["shape"], symbols, f"output {name}")
        if declaration["dtype"] not in DTYPES:
            raise TaskConfigError(f"output {name}: unknown dtype {declaration['dtype']!r}")
    mutable = interface.get("mutable", [])
    if not isinstance(mutable, list) or len(set(mutable)) != len(mutable):
        raise TaskConfigError("interface.mutable must be a list of distinct input names")
    for name in mutable:
        if name not in inputs or inputs[name].get("kind") != "tensor":
            raise TaskConfigError(f"mutable {name!r} is not a tensor input")
    for key in ("generator", "layout"):
        if key in interface:
            interface[key] = _entry(interface[key], f"interface.{key}")
    if generator_needed and "generator" not in interface:
        raise TaskConfigError("Inputs with distribution generator need interface.generator")
    return axes, symbols, var


def _shape(shape: Any, symbols: set[str], where: str) -> None:
    if not isinstance(shape, list):
        raise TaskConfigError(f"{where}.shape must be a list")
    for entry in shape:
        unknown = _expression_names(entry, f"{where}.shape") - symbols
        if unknown:
            raise TaskConfigError(f"{where}.shape reads unknown axes {sorted(unknown)}")


def _scalar_value(dtype: str, value: Any, where: str) -> None:
    if dtype == "bool":
        if type(value) is not bool:
            raise TaskConfigError(f"{where} must be a boolean")
    elif dtype.startswith("int"):
        _int(value, where)
    else:
        _number(value, where)


def _validate_compare(raw: dict, checked: list[str], outputs: dict, inputs: dict) -> None:
    compare = _mapping(raw, "compare", set(checked), set(checked))
    for name, declaration in compare.items():
        if not isinstance(declaration, dict) or declaration.get("bound") not in BOUNDS:
            raise TaskConfigError(f"compare.{name}.bound must be one of {sorted(BOUNDS)}")
        _mapping(declaration, f"compare.{name}", {"bound"} | BOUNDS[declaration["bound"]])
        dtype = (outputs.get(name) or inputs[name])["dtype"]
        if dtype not in FLOAT_DTYPES and declaration["bound"] != "exact":
            raise TaskConfigError(f"compare.{name}: {dtype} outputs compare exactly")
        if "k" in declaration and not 1.0 <= _number(declaration["k"], f"compare.{name}.k") <= 8.0:
            raise TaskConfigError(f"compare.{name}.k must be between 1 and 8")
        if "max_violation_fraction" in declaration and not 0.0 <= _number(
                declaration["max_violation_fraction"], "max_violation_fraction") <= 0.01:
            raise TaskConfigError(f"compare.{name}.max_violation_fraction must be at most 0.01")
        if declaration["bound"] == "tolerance":
            for key in ("atol", "rtol"):
                if key not in declaration or _number(declaration[key], key) < 0:
                    raise TaskConfigError(f"compare.{name} tolerance needs nonnegative {key}")
        if declaration["bound"] == "sqnr" and _number(declaration.get("min_db"), "min_db") <= 0:
            raise TaskConfigError(f"compare.{name}.min_db must be positive")


def _validate_workloads(workloads: Any, axes: dict, var: list[str], inputs: dict) -> None:
    obj = _mapping(workloads, "workloads", {"cases", "description"}, {"cases"})
    cases = obj["cases"]
    if not isinstance(cases, list) or not cases:
        raise TaskConfigError("workloads.cases must be a nonempty list")
    ids, performance, boundaries = set(), [], {"minimum": set(), "non_power_of_two": set(), "unaligned": set()}
    scalar_names = {n for n, d in inputs.items() if d.get("kind") == "scalar"}
    seen_identities = set()
    for case in cases:
        case = _mapping(case, "workload case", {"id", "axes", "scalars", "checks", "source", "description"},
                        {"id", "axes", "checks"})
        case_id = _text(case["id"], "case id")
        if not all(c.isalnum() or c in "_-." for c in case_id) or case_id in ids:
            raise TaskConfigError(f"Case ids must be unique [A-Za-z0-9_.-]: {case_id!r}")
        ids.add(case_id)
        values = _mapping(case["axes"], f"case {case_id}.axes", set(var), set(var))
        for name, value in values.items():
            axis = axes[name]
            _int(value, f"case {case_id}.axes.{name}", axis["min"])
            if value > axis["max"] or value % axis.get("multiple_of", 1):
                raise TaskConfigError(f"case {case_id}: {name}={value} is outside its declared domain")
        scalars = _mapping(case.get("scalars", {}), f"case {case_id}.scalars", scalar_names)
        for name, value in scalars.items():
            _scalar_value(inputs[name]["dtype"], value, f"case {case_id}.scalars.{name}")
        checks = case["checks"]
        if checks not in (["correctness"], ["correctness", "performance"]):
            raise TaskConfigError(f"case {case_id}.checks must be [correctness] or [correctness, performance]")
        identity = json.dumps([values, scalars], sort_keys=True)
        if identity in seen_identities:
            raise TaskConfigError(f"case {case_id} duplicates another case's axes and scalars")
        seen_identities.add(identity)
        if "performance" in checks:
            performance.append(math.prod(values.values()))
        for name, value in values.items():
            step = axes[name].get("multiple_of", 1)
            units = value // step
            if value == axes[name]["min"]:
                boundaries["minimum"].add(case_id)
            if not _is_power_of_two(units):
                boundaries["non_power_of_two"].add(case_id)
            if units > UNALIGNED_BLOCK and units % UNALIGNED_BLOCK:
                boundaries["unaligned"].add(case_id)
    if not MIN_PERFORMANCE_CASES <= len(performance) <= MAX_PERFORMANCE_CASES:
        raise TaskConfigError(f"A task needs {MIN_PERFORMANCE_CASES}-{MAX_PERFORMANCE_CASES} performance "
                              f"cases; found {len(performance)}")
    if max(performance) < MIN_SIZE_SPAN * min(performance):
        raise TaskConfigError(f"Performance cases must span at least {MIN_SIZE_SPAN}x in var-axis size; "
                              f"found {min(performance)}..{max(performance)}")
    missing = [kind for kind, found in boundaries.items() if not found]
    if missing:
        raise TaskConfigError(f"Correctness cases lack boundary shapes: {missing} "
                              f"(minimum axis value, non-power-of-two, not a multiple of {UNALIGNED_BLOCK})")
    chosen = _distinct_representatives(boundaries)
    if chosen is None:
        raise TaskConfigError("The minimum, non-power-of-two and unaligned boundary shapes must be three "
                              "distinct cases")


def _distinct_representatives(groups: dict[str, set[str]]) -> dict | None:
    names = list(groups)

    def search(index: int, used: set[str]) -> dict | None:
        if index == len(names):
            return {}
        for case in sorted(groups[names[index]]):
            if case not in used:
                rest = search(index + 1, used | {case})
                if rest is not None:
                    return {names[index]: case, **rest}
        return None

    return search(0, set())


def validate_task_contract(raw: Any, workloads: Any, *, task_id: str) -> TaskContract:
    relative_path(task_id, "task_id")
    obj = _mapping(raw, TASK_FILE, {"schema_version", "description", "operator", "interface", "reference",
                                    "baseline", "compare", "element_order", "workloads", "platform_support",
                                    "timeout_s"},
                   {"schema_version", "description", "operator", "interface", "reference", "baseline",
                    "compare", "workloads"})
    if type(obj["schema_version"]) is not int or obj["schema_version"] != SCHEMA_VERSION:
        raise TaskConfigError(f"{TASK_FILE} requires schema_version: {SCHEMA_VERSION}")
    obj = deepcopy(obj)
    _text(obj["description"], "description")
    operator = _mapping(obj["operator"], "operator", {"name", "category"}, {"name", "category"})
    _identifier(operator["name"], "operator.name")
    if operator["category"] not in CATEGORIES:
        raise TaskConfigError(f"operator.category must be one of {CATEGORIES}")
    axes, symbols, var = _validate_interface(obj["interface"])
    interface = obj["interface"]
    obj["reference"] = _entry(obj["reference"], "reference")
    baseline = _mapping(obj["baseline"], "baseline", {"language", "entry", "provenance"},
                        {"language", "entry", "provenance"})
    if baseline["language"] not in BASELINE_LANGUAGES:
        raise TaskConfigError(f"baseline.language must be one of {BASELINE_LANGUAGES}")
    expected = f"{BASELINE_DIR}/kernel.hip" if baseline["language"] == "hip" else f"{BASELINE_DIR}/kernel.py"
    if baseline["entry"] != expected:
        raise TaskConfigError(f"baseline.entry must be {expected}")
    provenance = _mapping(baseline["provenance"], "baseline.provenance",
                          {"source", "revision", "path", "license", "notes", "modifications", "code_objects"},
                          {"source", "revision", "path", "license"})
    if "code_objects" in provenance:
        _validate_code_objects(provenance["code_objects"], baseline["language"])
    for key in ("source", "revision", "license"):
        _text(provenance[key], f"baseline.provenance.{key}")
    paths = provenance["path"] if isinstance(provenance["path"], list) else [provenance["path"]]
    if not paths:
        raise TaskConfigError("baseline.provenance.path must name the upstream source")
    for path in paths:
        _text(path, "baseline.provenance.path")
    for entry in (obj["reference"], interface.get("generator"), interface.get("layout")):
        if entry is not None and entry["file"].startswith((BASELINE_DIR + "/", SOLUTION_DIR + "/")):
            raise TaskConfigError("reference, generator and layout live at the task root")
    checked = list(interface["outputs"]) + list(interface.get("mutable", []))
    _validate_compare(obj["compare"], checked, interface["outputs"], interface["inputs"])
    if obj.get("element_order", "fixed") not in ("fixed", "any_in_full_rows"):
        raise TaskConfigError("element_order must be fixed or any_in_full_rows")
    if relative_path(obj["workloads"], "workloads") != "workloads.json":
        raise TaskConfigError("workloads must be workloads.json")
    if "platform_support" in obj:
        platform = _mapping(obj["platform_support"], "platform_support", {"required_arch", "status", "skip_reason"})
        required_gpu_arches(platform)
        if platform.get("status", "active") not in ("active", "skip"):
            raise TaskConfigError("platform_support.status must be active or skip")
        if platform.get("status") == "skip" or "skip_reason" in platform:
            _text(platform.get("skip_reason"), "platform_support.skip_reason")
    if "timeout_s" in obj:
        _int(obj["timeout_s"], "timeout_s", 60)
    _validate_workloads(workloads, axes, var, interface["inputs"])
    for case in workloads["cases"]:
        symbols_values = {name: (axis["value"] if axis["kind"] == "const" else case["axes"][name])
                          for name, axis in axes.items()}
        for name, expression in interface.get("derived_axes", {}).items():
            symbols_values[name] = _evaluate(expression, symbols_values)
        for name, declaration in [*interface["inputs"].items(), *interface["outputs"].items()]:
            for entry in declaration.get("shape", []):
                _evaluate(entry, symbols_values)
    return TaskContract(obj, deepcopy(workloads), task_id)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise TaskConfigError(f"Duplicate or non-string YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TaskConfigError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


CODE_OBJECT_SUFFIX = ".co"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _validate_code_objects(declared, language: str) -> None:
    """``baseline.provenance.code_objects``: vendored AMDGPU code objects, by path and SHA256."""
    if language != "asm":
        raise TaskConfigError("baseline.provenance.code_objects is only for asm baselines")
    if not isinstance(declared, dict) or not declared:
        raise TaskConfigError("baseline.provenance.code_objects maps baseline/<file>.co paths to SHA256 hashes")
    for path, digest in declared.items():
        if (not isinstance(path, str) or not path.startswith(BASELINE_DIR + "/") or not path.endswith(CODE_OBJECT_SUFFIX)
                or ".." in Path(path).parts):
            raise TaskConfigError(f"code object {path!r} must be a {CODE_OBJECT_SUFFIX} file under {BASELINE_DIR}/")
        if not isinstance(digest, str) or not _SHA256.match(digest):
            raise TaskConfigError(f"code object {path} needs a lowercase hex SHA256")


def verify_code_objects(contract: "TaskContract", root: Path) -> None:
    """Every vendored code object is declared, present and matches its SHA256."""
    declared = contract.code_objects
    present = {p.relative_to(root).as_posix() for p in (Path(root) / BASELINE_DIR).rglob(f"*{CODE_OBJECT_SUFFIX}")}
    if present - set(declared):
        raise TaskConfigError(f"undeclared code objects in {BASELINE_DIR}/: {sorted(present - set(declared))[:4]}; "
                              "declare each in baseline.provenance.code_objects")
    for path, digest in declared.items():
        file = Path(root) / path
        if not file.is_file() or file.is_symlink():
            raise TaskConfigError(f"declared code object {path} is missing")
        if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise TaskConfigError(f"code object {path} does not match its declared SHA256")


def load_task_contract(task_file: Path, *, task_id: str) -> TaskContract:
    task_file = Path(task_file)
    root = task_file.parent
    try:
        raw = yaml.load(task_file.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise TaskConfigError(f"Invalid {TASK_FILE}: {exc}") from exc
    workloads_name = (raw or {}).get("workloads", "workloads.json") if isinstance(raw, dict) else "workloads.json"
    try:
        workloads = json.loads((root / relative_path(workloads_name, "workloads")).read_text(encoding="utf-8"),
                               object_pairs_hook=_unique_json,
                               parse_constant=lambda value: (_ for _ in ()).throw(
                                   TaskConfigError(f"Invalid JSON constant {value}")))
    except (OSError, ValueError) as exc:
        raise TaskConfigError(f"Cannot read workloads: {exc}") from exc
    contract = validate_task_contract(raw, workloads, task_id=task_id)
    verify_code_objects(contract, root)
    return contract


def package_files(root: Path) -> list[str]:
    """Task-relative files of a v3 package; generated names are reserved."""
    files = []
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root)
        if any(part in ("__pycache__", ".pytest_cache", ".git") for part in relative.parts):
            continue
        if relative.parts[0] in FORBIDDEN_PACKAGE_NAMES:
            raise TaskConfigError(f"A v3 task package cannot contain {relative.parts[0]}")
        if path.is_symlink():
            raise TaskConfigError(f"Task packages cannot contain symlinks: {relative}")
        if path.is_file():
            files.append(relative.as_posix())
    return files


def execution_config(task_file: Path, *, task_id: str, target_language: str) -> dict:
    contract = load_task_contract(task_file, task_id=task_id)
    files = package_files(Path(task_file).parent)
    for required in (contract.baseline_entry, contract.raw["reference"]["file"], "workloads.json"):
        if required not in files:
            raise TaskConfigError(f"Task package lacks {required}")
    return contract.execution_config(target_language, files)


def _hip_type(declaration: dict, name: str, writes: dict) -> str:
    if declaration.get("kind") == "scalar":
        return {"float32": "float", "float64": "float", "int32": "int", "int64": "int", "bool": "bool"}[
            declaration["dtype"]]
    return f"Tensor({writes[name]}!)" if name in writes else "Tensor"


def hip_op_schema(contract: TaskContract) -> str:
    interface = contract.interface
    names = list(interface["inputs"]) + list(interface["outputs"])
    written = list(interface["outputs"]) + list(interface.get("mutable", []))
    letters = {name: chr(ord("a") + index) for index, name in enumerate(written)}
    declarations = {**interface["inputs"], **interface["outputs"]}
    arguments = ", ".join(f"{_hip_type(declarations[n], n, letters)} {n}" for n in names)
    return f"run({arguments}) -> ()"


def _signature(contract: TaskContract) -> str:
    names = list(contract.interface["inputs"]) + list(contract.interface["outputs"])
    return "def run(*, " + ", ".join(names) + ") -> None:"


def _template(contract: TaskContract, language: str) -> str:
    operator = contract.raw["operator"]["name"]
    if language == "hip":
        cpp_types = {"Tensor": "at::Tensor", "float": "double", "int": "int64_t", "bool": "bool"}
        interface = contract.interface
        declarations = {**interface["inputs"], **interface["outputs"]}
        parameters = []
        for name in list(interface["inputs"]) + list(interface["outputs"]):
            kind = _hip_type(declarations[name], name, {}).split("(")[0]
            parameters.append(f"{'const ' if kind == 'Tensor' and name not in contract.checked_tensors else ''}"
                              f"{cpp_types[kind]}{'&' if kind == 'Tensor' else ''} {name}")
        return (f"// HIP implementation of {operator}. {TEMPLATE_MARKER}\n"
                "// Replace this template: launch your __global__ kernels on\n"
                "// at::hip::getCurrentHIPStream() and write every output. See _aka_eval/CANDIDATE.md.\n"
                "#include <ATen/ATen.h>\n#include <ATen/hip/HIPContext.h>\n#include <hip/hip_runtime.h>\n"
                "#include <torch/library.h>\n\n"
                f"void run({', '.join(parameters)}) {{\n"
                f"  TORCH_CHECK(false, \"{TEMPLATE_MARKER}\");\n}}\n\n"
                "#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)\n"
                "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {\n"
                f"  m.def(\"{hip_op_schema(contract)}\", &run);\n}}\n")
    imports = {"triton": "import torch\nimport triton\nimport triton.language as tl\n",
               "gluon": "import torch\nimport triton\nfrom triton.experimental import gluon\n"
                        "from triton.experimental.gluon import language as gl\n",
               "flydsl": "import torch\nimport flydsl\n",
               "asm": "import torch\nimport aka_asm\n\n"
                      "# Load kernels from .s files in this directory at import time, for example\n"
                      "# KERNEL = aka_asm.load(\"kernel.s\", \"my_kernel\"), and launch them from run().\n"}[language]
    return (f'"""{language} implementation of {operator}. {TEMPLATE_MARKER}\n\n'
            "Replace this template with kernels written in the target language and launch them\n"
            "from run(). See _aka_eval/CANDIDATE.md for the interface and rules.\n"
            '"""\n'
            f"{imports}\n\n{_signature(contract)}\n"
            f"    raise NotImplementedError(\"{TEMPLATE_MARKER}\")\n")


def _candidate_document(contract: TaskContract, language: str) -> str:
    interface = contract.interface
    lines = [f"# Candidate contract: {contract.raw['operator']['name']} in {language}", "",
             contract.raw["description"].strip(), "",
             f"Write the implementation in `{SOLUTION_DIR}/`, the only editable directory. "
             f"The required implementation language is **{language}**. "
             f"The baseline to beat is `{contract.baseline_entry}` ({contract.baseline_language}); "
             "it, the reference, the inputs and every other task file are read-only.", "",
             "## Entry point", ""]
    if language == "hip":
        lines += [f"`{SOLUTION_DIR}/kernel.hip` registers exactly one op with this schema:", "",
                  "```cpp", "#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)",
                  "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {",
                  f"  m.def(\"{hip_op_schema(contract)}\", &run);", "}", "```", "",
                  "Other `.hip` files and headers in the directory are compiled with it. Launch on "
                  "`at::hip::getCurrentHIPStream()`; include only HIP, ATen/c10, `torch/library.h`, "
                  "standard headers and your own headers."]
    else:
        lines += [f"`{SOLUTION_DIR}/kernel.py` defines:", "", "```python", _signature(contract), "```", "",
                  "Other `.py` files in the directory form one flat package; import them relatively "
                  "(`from .helpers import ...`)."]
        if language == "asm":
            lines += ["", "Kernels are AMDGPU assembly in `.s` files in the same directory, assembled for "
                      "the current GPU with ROCm `clang` and `ld.lld`. `run` may only allocate, take views "
                      "and launch them through `aka_asm`:", "", "```python", "import aka_asm",
                      'KERNEL = aka_asm.load("kernel.s", "my_kernel")          # at import time',
                      "KERNEL.launch(grid=(gx, gy, gz), block=(256, 1, 1),     # in run()",
                      "              args=[a, out, aka_asm.u32(n)], shared_bytes=0)", "```", "",
                      "`grid` counts workgroups. Arguments are packed in order at their natural alignment: "
                      "tensors as 64-bit device pointers, scalars only through `aka_asm.u32`, `i32`, `u64`, "
                      "`i64` and `f32`. Each kernel's `.amdgpu_metadata` must declare the same arguments. "
                      "`.incbin` and `.include` are not allowed; each kernel's whole source is in its `.s` "
                      "file."]
    lines += ["", "## Interface", "", "| Name | Kind | Shape | Dtype |", "|---|---|---|---|"]
    for name, declaration in interface["inputs"].items():
        kind = "mutable input" if name in interface.get("mutable", []) else declaration["kind"] + " input"
        shape = "scalar" if declaration["kind"] == "scalar" else "[" + ", ".join(map(str, declaration["shape"])) + "]"
        lines.append(f"| `{name}` | {kind} | {shape} | {declaration['dtype']} |")
    for name, declaration in interface["outputs"].items():
        lines.append(f"| `{name}` | output | [{', '.join(map(str, declaration['shape']))}] | {declaration['dtype']} |")
    lines += ["", "Axes: " + ", ".join(
        f"`{n}` = {a['value']}" if a["kind"] == "const" else f"`{n}` in [{a['min']}, {a['max']}]"
        + (f", multiple of {a['multiple_of']}" if a.get("multiple_of", 1) > 1 else "")
        for n, a in interface["axes"].items()) + ".", "",
        "## Rules", "",
        "- `run` writes every output (and each mutable input) completely and returns `None`. Outputs are "
        "preallocated by the evaluator and poisoned before every call.",
        "- Inputs other than mutable inputs are read-only. Keep no state derived from inputs across calls.",
        f"- All computation runs in kernels you write in {language}. Host code may only allocate, take "
        "views or same-dtype copies and launch. PyTorch arithmetic, reductions, dtype conversions, library "
        "operators and other kernel languages are rejected at run time.",
        "- No synchronization, host round trips, I/O, randomness, reflection, dynamic imports or changes "
        "to imported modules. Do not import task, reference, baseline or evaluator modules.",
        "- Every call is timed by one CUDA-graph replay after a cache flush; launches must be capturable on "
        "the current stream.",
        "- Correctness: each output must stay within the declared error bound of the high-precision "
        "reference on every workload, for two input draws and a repeated call on the same buffers.",
        "", "## Commands", "", "```sh",
        f"python3 -m {EVAL_DIR}.evaluate candidate compile",
        f"python3 -m {EVAL_DIR}.evaluate candidate correctness",
        f"python3 -m {EVAL_DIR}.evaluate candidate performance",
        "```", ""]
    return "\n".join(lines)


def render_workspace(task_root: Path, workspace: Path, *, task_id: str, target_language: str) -> list[str]:
    """Write the evaluator, contract and candidate start into a copied v3 task.

    Returns the generated workspace-relative files. ``workspace`` already holds
    a verified copy of the task package; existing files are never overwritten.
    """
    task_root, workspace = Path(task_root), Path(workspace)
    contract = load_task_contract(task_root / TASK_FILE, task_id=task_id)
    files = package_files(task_root)
    config = contract.execution_config(target_language, files)
    generated = []

    def write(relative: str, text: str) -> None:
        path = workspace / relative
        if path.exists() or path.is_symlink():
            raise TaskConfigError(f"Refusing to overwrite {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        generated.append(relative)

    write("config.yaml", yaml.safe_dump(config, sort_keys=False))
    package = Path(__file__).resolve().parent / "tools" / "arena_eval"
    for source in sorted(package.glob("*.py")):
        write(f"{EVAL_DIR}/{source.name}", source.read_text(encoding="utf-8"))
    write(f"{EVAL_DIR}/_aka_benchmark.py",
          (Path(__file__).resolve().parent / "tools" / "perf" / "aka_benchmark.py").read_text(encoding="utf-8"))
    # Declaration order is part of the contract: a reference may return its outputs as a tuple, and derived
    # axes may build on earlier ones, so the mappings are written in task.yaml order, not sorted.
    write(f"{EVAL_DIR}/contract.json", json.dumps(contract.resolved_contract(target_language), indent=2) + "\n")
    write(f"{EVAL_DIR}/CANDIDATE.md", _candidate_document(contract, target_language))
    if contract.initial_state(target_language) == "implemented":
        source = task_root / BASELINE_DIR
        destination = workspace / SOLUTION_DIR
        if destination.exists():
            raise TaskConfigError("Refusing to overwrite solution/")
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.md", "LICENSE*"))
        generated.extend(sorted(p.relative_to(workspace).as_posix() for p in destination.rglob("*") if p.is_file()))
    else:
        write(contract.candidate_entry(target_language), _template(contract, target_language))
    return generated


def is_v3_task(path: Path) -> bool:
    return Path(path).name == TASK_FILE


def audit_target_language(task_file: Path, *, task_id: str) -> str:
    """The language a task audit materializes: the baseline's when it is a target.

    The candidate then starts as an exact copy of the baseline, so the audit
    also proves that the evaluator accepts a correct implementation through
    the candidate path. A TileLang baseline has no such positive control.
    """
    language = load_task_contract(Path(task_file), task_id=task_id).baseline_language
    return language if language in TARGET_LANGUAGES else "triton"


def target_language_for(eval_config: Mapping[str, Any] | None, task_name: str) -> str | None:
    """The run's target language for one task: a per-task override, then the default."""
    config = eval_config or {}
    overrides = config.get("_task_target_languages", {})
    if task_name in overrides:
        return overrides[task_name]
    return config.get("target_language")
