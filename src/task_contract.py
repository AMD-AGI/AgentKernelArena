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
import shutil
from typing import Any, Mapping

import yaml

from .task_spec import TaskConfigError, relative_path, required_gpu_arches

TASK_FILE = "task.yaml"
SCHEMA_VERSION = 3
TARGET_LANGUAGES = ("triton", "gluon", "hip", "flydsl")
BASELINE_LANGUAGES = TARGET_LANGUAGES + ("tilelang",)
PYTHON_LANGUAGES = ("triton", "gluon", "flydsl", "tilelang")
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
SPAN_POLICY = "production_range"
SPAN_EVIDENCE_METHODS = ("corpus_statistics", "live_capture")
SPAN_FREQUENCY_SUM = (0.9, 1.1)  # rounded production weights need not sum to exactly 1
UNALIGNED_BLOCK = 16
TIMING = {"warmup": 10, "repetition": 100, "cache_flush_bytes": 1 << 30}
EVAL_DIR = "_aka_eval"
SOLUTION_DIR = "solution"
BASELINE_DIR = "baseline"
TEMPLATE_MARKER = "AKA-UNIMPLEMENTED-TEMPLATE"
FORBIDDEN_PACKAGE_NAMES = ("config.yaml", SOLUTION_DIR, EVAL_DIR)
# Multi-GPU tasks: the framework appends these keyword inputs to every rank's call.
RESERVED_INPUTS = ("rank", "world_size", "peers", "symmetric")
RESERVED_SYMBOL = "world_size"
MIN_WORLD_SIZE, MAX_WORLD_SIZE = 2, 8
MAX_SYMMETRIC_BYTES = 1 << 32


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

    def __init__(self, raw: dict, workloads: dict, task_id: str, performance_span: dict | None = None):
        self.raw = raw
        self.workloads = workloads
        self.task_id = task_id
        self.performance_span = performance_span

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
    def distributed(self) -> dict | None:
        """The validated multi-GPU declaration, or None for a single-GPU task."""
        return self.raw.get("distributed")

    @property
    def world_size(self) -> int:
        return self.distributed["world_size"] if self.distributed else 1

    @property
    def call_names(self) -> list[str]:
        """Every keyword of one call: inputs, outputs, then the reserved multi-GPU inputs."""
        names = list(self.interface["inputs"]) + list(self.interface["outputs"])
        return names + list(RESERVED_INPUTS) if self.distributed else names

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
        resolved = {
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
            "candidate": {"language": target_language, "root": SOLUTION_DIR,
                          "entry_file": self.candidate_entry(target_language), "symbol": "run"},
            "cases": cases,
        }
        if self.distributed:
            distributed = deepcopy(self.distributed)
            distributed.setdefault("ranks_agree", [])
            # The rank launcher bounds itself by the action timeout, so it can stop every rank and
            # report before the framework kills the process group.
            distributed["timeout_s"] = self.raw.get("timeout_s", 3600)
            resolved["distributed"] = distributed
        if self.performance_span:
            # Held-out cases of an exempt task stay inside the evidenced production range.
            resolved["performance_span"] = deepcopy(self.performance_span)
        return resolved

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
        if self.distributed:
            # The scheduler places a multi-GPU task only on a worker whose GPU group has this size.
            config["contract"]["world_size"] = self.world_size
        if "platform_support" in self.raw:
            config["platform_support"] = deepcopy(self.raw["platform_support"])
        return config


def _validate_interface(raw: dict, reserved: frozenset[str] = frozenset()) -> tuple[dict, dict, list[str]]:
    """Validate the interface; ``reserved`` names framework shape symbols (``world_size``)."""
    interface = _mapping(raw, "interface", {"axes", "derived_axes", "inputs", "outputs", "mutable",
                                            "generator", "layout"}, {"axes", "inputs", "outputs"})
    axes = _mapping(interface["axes"], "interface.axes", set(interface["axes"]))
    if not axes:
        raise TaskConfigError("interface.axes must declare at least one axis")
    var = []
    for name, axis in axes.items():
        _identifier(name, f"axis {name}")
        if name in reserved:
            raise TaskConfigError(f"axis {name} is reserved: the framework sets it for multi-GPU tasks")
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
    symbols = set(axes) | set(reserved)
    derived = interface.get("derived_axes", {})
    _mapping(derived, "interface.derived_axes", set(derived))
    for name, expression in derived.items():
        _identifier(name, f"derived axis {name}")
        if name in reserved:
            raise TaskConfigError(f"derived axis {name} is reserved: the framework sets it for multi-GPU tasks")
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


def _validate_distributed(raw: Any, interface: dict, symbols: set[str]) -> dict:
    """The optional multi-GPU block: world size, symmetric buffer size and per-rank views."""
    obj = _mapping(raw, "distributed", {"world_size", "symmetric_bytes", "shard", "expected", "ranks_agree"},
                   {"world_size", "symmetric_bytes", "shard", "expected"})
    world_size = obj["world_size"]
    if type(world_size) is not int or not MIN_WORLD_SIZE <= world_size <= MAX_WORLD_SIZE:
        raise TaskConfigError(f"distributed.world_size must be an integer from {MIN_WORLD_SIZE} to "
                              f"{MAX_WORLD_SIZE} (the GPUs of one node)")
    unknown = _expression_names(obj["symmetric_bytes"], "distributed.symmetric_bytes") - symbols
    if unknown:
        raise TaskConfigError(f"distributed.symmetric_bytes reads unknown axes {sorted(unknown)}")
    for key in ("shard", "expected"):
        obj[key] = _entry(obj[key], f"distributed.{key}")
        if obj[key]["file"].startswith((BASELINE_DIR + "/", SOLUTION_DIR + "/")):
            raise TaskConfigError(f"distributed.{key} lives at the task root")
    for name in [*interface["inputs"], *interface["outputs"]]:
        if name in RESERVED_INPUTS:
            raise TaskConfigError(f"{name!r} is a reserved multi-GPU call input {RESERVED_INPUTS}; "
                                  "the framework passes it, so the task must not declare it")
    checked = list(interface["outputs"]) + list(interface.get("mutable", []))
    agree = obj.get("ranks_agree", [])
    if (not isinstance(agree, list) or any(not isinstance(n, str) for n in agree) or len(set(agree)) != len(agree)
            or any(n not in checked for n in agree)):
        raise TaskConfigError(f"distributed.ranks_agree must list distinct outputs or mutable inputs of {checked}")
    return obj


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


def _decode(value: Any, where: str) -> dict:
    """A packed output's decode hook: a task-root function and the precision its decoded values carry."""
    entry = _mapping(value, where, {"file", "symbol", "precision"}, {"file", "symbol"})
    decode = _entry({key: entry[key] for key in ("file", "symbol")}, where)
    if "precision" in entry:
        if entry["precision"] not in FLOAT_DTYPES:
            raise TaskConfigError(f"{where}.precision must be one of {FLOAT_DTYPES}")
        decode["precision"] = entry["precision"]
    return decode


def _validate_compare(raw: dict, checked: list[str], outputs: dict, inputs: dict) -> None:
    compare = _mapping(raw, "compare", set(checked), set(checked))
    for name, declaration in compare.items():
        if not isinstance(declaration, dict) or declaration.get("bound") not in BOUNDS:
            raise TaskConfigError(f"compare.{name}.bound must be one of {sorted(BOUNDS)}")
        _mapping(declaration, f"compare.{name}", {"bound", "decode"} | BOUNDS[declaration["bound"]])
        dtype = (outputs.get(name) or inputs[name])["dtype"]
        if "decode" in declaration:
            if declaration["bound"] == "exact":
                raise TaskConfigError(f"compare.{name}.decode needs a numeric bound; exact compares the bytes")
            if dtype in FLOAT_DTYPES:
                raise TaskConfigError(f"compare.{name}.decode is for integral or packed outputs; {dtype} "
                                      "values compare directly")
            declaration["decode"] = _decode(declaration["decode"], f"compare.{name}.decode")
        elif dtype not in FLOAT_DTYPES and declaration["bound"] != "exact":
            raise TaskConfigError(f"compare.{name}: {dtype} outputs compare exactly unless they declare decode")
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


def _span_declaration(workloads: Any) -> dict | None:
    """The validated ``workloads.performance_span`` declaration, or None when the task has none."""
    if not isinstance(workloads, dict) or "performance_span" not in workloads:
        return None
    declaration = _mapping(workloads["performance_span"], "workloads.performance_span", {"policy", "evidence"},
                           {"policy", "evidence"})
    if declaration["policy"] != SPAN_POLICY:
        raise TaskConfigError(f"workloads.performance_span.policy must be {SPAN_POLICY!r}")
    name = relative_path(declaration["evidence"], "workloads.performance_span.evidence")
    if not name.endswith(".json") or "/" in name or name in ("workloads.json", TASK_FILE):
        raise TaskConfigError("workloads.performance_span.evidence must name a .json file at the task root "
                              "other than workloads.json")
    return {"policy": SPAN_POLICY, "evidence": name}


def _validate_span_evidence(evidence: Any, declaration: dict, axes: dict, var: list[str],
                            performance: list[dict]) -> dict:
    """Check the production-range evidence that exempts a task from the 100x performance-span rule.

    The exemption is for model-specialized kernels whose >= 10 us production shapes span less than
    100x. The perf range (perf_min..perf_max) is the part of the observed production range whose
    baseline takes at least 10 us; the observed range is recorded for transparency and may itself
    span more. Returns the summary carried into the resolved contract.
    """
    where = declaration["evidence"]
    if evidence is None:
        raise TaskConfigError(f"workloads.performance_span names {where}, which the task package does not contain")
    obj = _mapping(evidence, where, {"source", "method", "axes", "fixed_dimensions", "notes"},
                   {"source", "method", "axes", "fixed_dimensions"})
    _text(obj["source"], f"{where}.source")
    _text(obj["fixed_dimensions"], f"{where}.fixed_dimensions")
    if "notes" in obj:
        _text(obj["notes"], f"{where}.notes")
    if obj["method"] not in SPAN_EVIDENCE_METHODS:
        raise TaskConfigError(f"{where}.method must be one of {SPAN_EVIDENCE_METHODS}")
    evidence_axes = _mapping(obj["axes"], f"{where}.axes", set(obj["axes"]) if isinstance(obj["axes"], dict) else set())
    if set(evidence_axes) != set(var):
        raise TaskConfigError(f"{where}.axes names {sorted(evidence_axes)}; it must name exactly the var axes "
                              f"{sorted(var)}")
    ranges = {}
    for name in var:
        label = f"{where}.axes.{name}"
        entry = _mapping(evidence_axes[name], label,
                         {"observed_min", "observed_max", "perf_min", "perf_max", "frequency"},
                         {"observed_min", "observed_max", "perf_min", "perf_max", "frequency"})
        values = {key: _int(entry[key], f"{label}.{key}", 1)
                  for key in ("observed_min", "perf_min", "perf_max", "observed_max")}
        if not values["observed_min"] <= values["perf_min"] <= values["perf_max"] <= values["observed_max"]:
            raise TaskConfigError(f"{label}: need observed_min <= perf_min <= perf_max <= observed_max; found "
                                  f"{values['observed_min']} <= {values['perf_min']} <= {values['perf_max']} <= "
                                  f"{values['observed_max']}")
        axis = axes[name]
        if values["observed_min"] < axis["min"] or values["observed_max"] > axis["max"]:
            raise TaskConfigError(f"{label}: the observed range {values['observed_min']}..{values['observed_max']} "
                                  f"must lie inside the axis domain {axis['min']}..{axis['max']}")
        step = axis.get("multiple_of", 1)
        if values["perf_min"] % step or values["perf_max"] % step:
            raise TaskConfigError(f"{label}: perf_min and perf_max must be multiples of {name}'s multiple_of {step}")
        frequency = entry["frequency"]
        if not isinstance(frequency, dict) or not frequency:
            raise TaskConfigError(f"{label}.frequency must map observed values to production weights")
        total = 0.0
        for key, weight in frequency.items():
            if not isinstance(key, str) or not key.isdigit() or \
                    not values["observed_min"] <= int(key) <= values["observed_max"]:
                raise TaskConfigError(f"{label}.frequency: {key!r} is not an observed value of {name} "
                                      f"in {values['observed_min']}..{values['observed_max']}")
            if _number(weight, f"{label}.frequency.{key}") < 0:
                raise TaskConfigError(f"{label}.frequency.{key} must be nonnegative")
            total += weight
        if not SPAN_FREQUENCY_SUM[0] <= total <= SPAN_FREQUENCY_SUM[1]:
            raise TaskConfigError(f"{label}.frequency weights sum to {total:.3g}; they are production fractions "
                                  "and must sum to about 1")
        ranges[name] = {**values, "perf_frequency": sum(w for k, w in frequency.items()
                                                       if values["perf_min"] <= int(k) <= values["perf_max"])}
    perf_span = math.prod(r["perf_max"] for r in ranges.values()) / math.prod(r["perf_min"] for r in ranges.values())
    observed_span = (math.prod(r["observed_max"] for r in ranges.values())
                     / math.prod(r["observed_min"] for r in ranges.values()))
    if perf_span >= MIN_SIZE_SPAN:
        raise TaskConfigError(f"{where}: the >= 10 us production range spans {perf_span:.3g}x in var-axis size, "
                              f"so the {MIN_SIZE_SPAN}x rule is reachable; drop workloads.performance_span and "
                              f"choose performance cases spanning at least {MIN_SIZE_SPAN}x")
    for name, bounds in ranges.items():
        for case in performance:
            value = case["axes"][name]
            if not bounds["perf_min"] <= value <= bounds["perf_max"]:
                raise TaskConfigError(f"performance case {case['id']}: {name}={value} lies outside the evidenced "
                                      f"perf range {bounds['perf_min']}..{bounds['perf_max']}; keep it "
                                      "correctness-only")
        for key in ("perf_min", "perf_max"):
            if not any(case["axes"][name] == bounds[key] for case in performance):
                raise TaskConfigError(f"No performance case has {name} at {key}={bounds[key]}: an exempt task "
                                      "times both ends of its evidenced perf range on every var axis")
    return {"policy": SPAN_POLICY, "evidence": where, "source": obj["source"], "method": obj["method"],
            "perf_span": perf_span, "observed_span": observed_span,
            "axes": {name: {key: bounds[key] for key in ("observed_min", "observed_max", "perf_min", "perf_max",
                                                          "perf_frequency")}
                     for name, bounds in ranges.items()}}


def _validate_workloads(workloads: Any, axes: dict, var: list[str], inputs: dict,
                        evidence: Any = None) -> dict | None:
    """Validate the cases; returns the production-range exemption summary when one is declared."""
    obj = _mapping(workloads, "workloads", {"cases", "description", "performance_span"}, {"cases"})
    declaration = _span_declaration(obj)
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
            performance.append({"id": case_id, "axes": values, "size": math.prod(values.values())})
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
    sizes = [case["size"] for case in performance]
    span = None
    if declaration is not None:
        span = _validate_span_evidence(evidence, declaration, axes, var, performance)
    elif max(sizes) < MIN_SIZE_SPAN * min(sizes):
        raise TaskConfigError(f"Performance cases must span at least {MIN_SIZE_SPAN}x in var-axis size; "
                              f"found {min(sizes)}..{max(sizes)} (a model-specialized kernel whose >= 10 us "
                              "production range spans less may declare workloads.performance_span; see "
                              "docs/how-to/add-task.md)")
    if not any(_unaligned_reachable(axes[name]) for name in var):
        # No var axis can hold a value above 16 units that is not a multiple of 16.
        del boundaries["unaligned"]
    missing = [kind for kind, found in boundaries.items() if not found]
    if missing:
        raise TaskConfigError(f"Correctness cases lack boundary shapes: {missing} "
                              f"(minimum axis value, non-power-of-two, not a multiple of {UNALIGNED_BLOCK} "
                              f"above {UNALIGNED_BLOCK} units when a var axis's domain reaches that far)")
    chosen = _distinct_representatives(boundaries)
    if chosen is None:
        raise TaskConfigError(f"The {', '.join(k.replace('_', '-') for k in boundaries)} boundary shapes must be "
                              f"{len(boundaries)} distinct cases")
    return span


def _unaligned_reachable(axis: dict) -> bool:
    """Whether the axis domain holds a value above 16 units (of multiple_of) that is not a multiple of 16."""
    step = axis.get("multiple_of", 1)
    low, high = max(-(-axis["min"] // step), UNALIGNED_BLOCK + 1), axis["max"] // step
    return low < high or (low == high and low % UNALIGNED_BLOCK != 0)


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


def validate_task_contract(raw: Any, workloads: Any, *, task_id: str, span_evidence: Any = None) -> TaskContract:
    """Validate a task; ``span_evidence`` is the parsed file a ``performance_span`` declaration names."""
    relative_path(task_id, "task_id")
    obj = _mapping(raw, TASK_FILE, {"schema_version", "description", "operator", "interface", "reference",
                                    "baseline", "compare", "element_order", "workloads", "platform_support",
                                    "timeout_s", "distributed"},
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
    reserved = frozenset({RESERVED_SYMBOL}) if "distributed" in obj else frozenset()
    axes, symbols, var = _validate_interface(obj["interface"], reserved)
    interface = obj["interface"]
    if "distributed" in obj:
        obj["distributed"] = _validate_distributed(obj["distributed"], interface, symbols)
    obj["reference"] = _entry(obj["reference"], "reference")
    baseline = _mapping(obj["baseline"], "baseline", {"language", "entry", "provenance"},
                        {"language", "entry", "provenance"})
    if baseline["language"] not in BASELINE_LANGUAGES:
        raise TaskConfigError(f"baseline.language must be one of {BASELINE_LANGUAGES}")
    expected = f"{BASELINE_DIR}/kernel.hip" if baseline["language"] == "hip" else f"{BASELINE_DIR}/kernel.py"
    if baseline["entry"] != expected:
        raise TaskConfigError(f"baseline.entry must be {expected}")
    provenance = _mapping(baseline["provenance"], "baseline.provenance",
                          {"source", "revision", "path", "license", "notes", "modifications"},
                          {"source", "revision", "path", "license"})
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
    span = _validate_workloads(workloads, axes, var, interface["inputs"], span_evidence)
    for case in workloads["cases"]:
        symbols_values = {name: (axis["value"] if axis["kind"] == "const" else case["axes"][name])
                          for name, axis in axes.items()}
        if "distributed" in obj:
            symbols_values[RESERVED_SYMBOL] = obj["distributed"]["world_size"]
        for name, expression in interface.get("derived_axes", {}).items():
            symbols_values[name] = _evaluate(expression, symbols_values)
        for name, declaration in [*interface["inputs"].items(), *interface["outputs"].items()]:
            for entry in declaration.get("shape", []):
                _evaluate(entry, symbols_values)
        if "distributed" in obj:
            nbytes = _evaluate(obj["distributed"]["symmetric_bytes"], symbols_values)
            if not 1 <= nbytes <= MAX_SYMMETRIC_BYTES:
                raise TaskConfigError(f"case {case['id']}: distributed.symmetric_bytes is {nbytes}; it must be "
                                      f"between 1 and {MAX_SYMMETRIC_BYTES} bytes")
    return TaskContract(obj, deepcopy(workloads), task_id, span)


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


def load_task_contract(task_file: Path, *, task_id: str) -> TaskContract:
    task_file = Path(task_file)
    root = task_file.parent
    try:
        raw = yaml.load(task_file.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise TaskConfigError(f"Invalid {TASK_FILE}: {exc}") from exc
    workloads_name = (raw or {}).get("workloads", "workloads.json") if isinstance(raw, dict) else "workloads.json"
    try:
        workloads = _read_json(root / relative_path(workloads_name, "workloads"))
    except (OSError, ValueError) as exc:
        raise TaskConfigError(f"Cannot read workloads: {exc}") from exc
    evidence = None
    declaration = _span_declaration(workloads)
    if declaration is not None:
        path = root / declaration["evidence"]
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise TaskConfigError(f"workloads.performance_span.evidence {declaration['evidence']} is not a file "
                                  "in the task package")
        try:
            evidence = _read_json(path)
        except (OSError, ValueError) as exc:
            raise TaskConfigError(f"Cannot read {declaration['evidence']}: {exc}") from exc
    contract = validate_task_contract(raw, workloads, task_id=task_id, span_evidence=evidence)
    for name, declaration in contract.raw["compare"].items():
        if "decode" in declaration:
            _check_task_function(root, declaration["decode"], f"compare.{name}.decode")
    return contract


def _check_task_function(root: Path, entry: dict, where: str) -> None:
    """A task-root function the evaluator imports: a file of the package that defines it at top level."""
    path = root / entry["file"]
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise TaskConfigError(f"{where}.file {entry['file']} is not a file in the task package")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=entry["file"])
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise TaskConfigError(f"Cannot parse {entry['file']}: {exc}") from exc
    if not any(isinstance(node, ast.FunctionDef) and node.name == entry["symbol"] for node in tree.body):
        raise TaskConfigError(f"{where}: {entry['file']} does not define function {entry['symbol']}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          TaskConfigError(f"Invalid JSON constant {value}")))


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
    required_files = [contract.baseline_entry, contract.raw["reference"]["file"], "workloads.json"]
    if contract.performance_span:
        required_files.append(contract.performance_span["evidence"])
    for required in required_files:
        if required not in files:
            raise TaskConfigError(f"Task package lacks {required}")
    return contract.execution_config(target_language, files)


def _hip_type(declaration: dict, name: str, writes: dict) -> str:
    if declaration.get("kind") == "scalar":
        return {"float32": "float", "float64": "float", "int32": "int", "int64": "int", "bool": "bool"}[
            declaration["dtype"]]
    return f"Tensor({writes[name]}!)" if name in writes else "Tensor"


def reserved_declarations(contract: TaskContract) -> dict:
    """The reserved multi-GPU call inputs, declared like task inputs (empty for a single-GPU task)."""
    if not contract.distributed:
        return {}
    return {"rank": {"kind": "scalar", "dtype": "int32"},
            "world_size": {"kind": "scalar", "dtype": "int32"},
            "peers": {"kind": "tensor", "shape": [RESERVED_SYMBOL], "dtype": "int64"},
            "symmetric": {"kind": "tensor", "shape": [contract.distributed["symmetric_bytes"]], "dtype": "uint8"}}


def _call_declarations(contract: TaskContract) -> dict:
    interface = contract.interface
    return {**interface["inputs"], **interface["outputs"], **reserved_declarations(contract)}


def _written(contract: TaskContract) -> list[str]:
    """Tensors the op writes: outputs, mutable inputs and, for multi-GPU tasks, the symmetric buffer."""
    return contract.checked_tensors + (["symmetric"] if contract.distributed else [])


def hip_op_schema(contract: TaskContract) -> str:
    letters = {name: chr(ord("a") + index) for index, name in enumerate(_written(contract))}
    declarations = _call_declarations(contract)
    arguments = ", ".join(f"{_hip_type(declarations[n], n, letters)} {n}" for n in contract.call_names)
    return f"run({arguments}) -> ()"


def _signature(contract: TaskContract) -> str:
    return "def run(*, " + ", ".join(contract.call_names) + ") -> None:"


def _template(contract: TaskContract, language: str) -> str:
    operator = contract.raw["operator"]["name"]
    if language == "hip":
        cpp_types = {"Tensor": "at::Tensor", "float": "double", "int": "int64_t", "bool": "bool"}
        declarations = _call_declarations(contract)
        written = _written(contract)
        parameters = []
        for name in contract.call_names:
            kind = _hip_type(declarations[name], name, {}).split("(")[0]
            parameters.append(f"{'const ' if kind == 'Tensor' and name not in written else ''}"
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
               "flydsl": "import torch\nimport flydsl\n"}[language]
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
    lines += ["", "## Interface", "", "| Name | Kind | Shape | Dtype |", "|---|---|---|---|"]
    for name, declaration in interface["inputs"].items():
        kind = "mutable input" if name in interface.get("mutable", []) else declaration["kind"] + " input"
        shape = "scalar" if declaration["kind"] == "scalar" else "[" + ", ".join(map(str, declaration["shape"])) + "]"
        lines.append(f"| `{name}` | {kind} | {shape} | {declaration['dtype']} |")
    for name, declaration in interface["outputs"].items():
        lines.append(f"| `{name}` | output | [{', '.join(map(str, declaration['shape']))}] | {declaration['dtype']} |")
    for name, declaration in reserved_declarations(contract).items():
        shape = "scalar" if declaration["kind"] == "scalar" else "[" + ", ".join(map(str, declaration["shape"])) + "]"
        lines.append(f"| `{name}` | reserved multi-GPU input | {shape} | {declaration['dtype']} |")
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
        *_multi_gpu_lines(contract),
        "", "## Commands", "", "```sh",
        f"python3 -m {EVAL_DIR}.evaluate candidate compile",
        f"python3 -m {EVAL_DIR}.evaluate candidate correctness",
        f"python3 -m {EVAL_DIR}.evaluate candidate performance",
        "```", ""]
    return "\n".join(lines)


def _multi_gpu_lines(contract: TaskContract) -> list[str]:
    if not contract.distributed:
        return []
    distributed = contract.distributed
    world = distributed["world_size"]
    agree = ", ".join(f"`{n}`" for n in distributed.get("ranks_agree", [])) or "none"
    return [
        "", "## Multi-GPU", "",
        f"This operator runs on **{world} GPUs** of one node: the evaluator starts {world} processes and rank "
        f"`r` calls `run` on `cuda:r`. The interface above is one rank's call; every rank has the same shapes "
        f"and `world_size` = {world} in shape expressions. Each rank receives its own shard of the inputs and "
        "is checked against its own slice of the reference.",
        "",
        "- `rank` and `world_size` are Python ints (int32 range).",
        "- `symmetric` is this rank's framework-owned peer buffer of "
        f"`{distributed['symmetric_bytes']}` bytes (uncached device memory). `peers[p]` is the base address "
        "of rank `p`'s buffer mapped into this process (`peers[rank]` is `symmetric`). Kernels may load from "
        "and store to peer memory through these addresses (system-scope atomics/stores for cross-GPU "
        "visibility).",
        "- The evaluator zeroes every rank's `symmetric` buffer before every call, outside timing, and starts "
        "no rank's call before every rank's buffer is zero, so a call never sees a previous call's mailbox "
        "words. Persistent-kernel state (epoch counters, scratch) is an ordinary input the evaluator resets "
        "before every call; keep no host-side state.",
        "- Host-side communication is forbidden: no `torch.distributed`, RCCL, HIP IPC, peer-access or "
        "memory-allocation APIs. The framework owns all host-side cross-rank work.",
        f"- Every rank's outputs are checked; outputs that every rank must produce bitwise identical: {agree}.",
        "- Timing: every sample releases all ranks together (a barrier, then a shared host deadline), each "
        "rank times its own graph replay with GPU events, and the sample's time is the **maximum over ranks**. "
        "The reported time is the median of those maxima.",
    ]


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
