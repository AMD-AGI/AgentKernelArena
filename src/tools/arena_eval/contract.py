# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""The resolved task contract, as the framework wrote it into the workspace.

The framework validates ``task.yaml`` and writes ``_aka_eval/contract.json``
during materialization. This module only reads that trusted file; it never
interprets task-authored YAML. It depends on the standard library and PyTorch.
"""
from __future__ import annotations

import ast
import json
import math
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parent
WORKSPACE = EVAL_ROOT.parent
CONTRACT_PATH = EVAL_ROOT / "contract.json"
CONTRACT_VERSION = 1

PYTHON_LANGUAGES = frozenset({"triton", "gluon", "flydsl", "tilelang", "python", "asm"})
TARGET_LANGUAGES = frozenset({"triton", "gluon", "hip", "flydsl", "asm"})


class ContractError(ValueError):
    """The resolved contract or a case is inconsistent."""


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_FUNCTIONS = {"min": min, "max": max, "ceil_div": lambda a, b: -(-a // b)}


def evaluate_dim(expression: Any, symbols: dict[str, int]) -> int:
    """Evaluate one shape entry: an int, an axis name or integer arithmetic."""
    if type(expression) is int:
        return expression
    if not isinstance(expression, str):
        raise ContractError(f"Invalid shape expression {expression!r}")
    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in symbols:
                raise ContractError(f"Unknown axis {node.id!r} in {expression!r}")
            return symbols[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](visit(node.left), visit(node.right))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCTIONS and not node.keywords):
            return _FUNCTIONS[node.func.id](*[visit(arg) for arg in node.args])
        raise ContractError(f"Unsupported shape expression {expression!r}")

    value = visit(tree)
    if type(value) is not int or value < 0:
        raise ContractError(f"Shape expression {expression!r} is not a nonnegative int")
    return value


@dataclass(frozen=True)
class Implementation:
    role: str
    language: str
    root: str
    entry_file: str
    symbol: str

    @property
    def is_python(self) -> bool:
        return self.language in PYTHON_LANGUAGES


@dataclass(frozen=True)
class Case:
    case_id: str
    axes: dict
    scalars: dict
    checks: tuple
    seeds: tuple

    def identity(self) -> dict:
        return {"test_case_id": self.case_id,
                "params": {"axes": dict(self.axes), "scalars": dict(self.scalars)}}


class Contract:
    def __init__(self, raw: dict):
        if raw.get("version") != CONTRACT_VERSION:
            raise ContractError("Unsupported resolved contract version")
        self.raw = raw
        self.task_id = raw["task_id"]
        self.axes = raw["axes"]
        self.derived_axes = raw.get("derived_axes", {})
        self.inputs = raw["inputs"]
        self.outputs = raw["outputs"]
        self.mutable = tuple(raw.get("mutable", ()))
        self.compare = raw["compare"]
        self.element_order = raw.get("element_order", "fixed")
        self.timing = raw["timing"]
        self.reference = raw["reference"]
        self.generator = raw.get("generator")
        self.layout = raw.get("layout")
        self.baseline = Implementation("baseline", **raw["baseline"])
        # SHA256 -> baseline path of the asm baseline's vendored code objects.
        self.code_objects = dict(raw.get("code_objects", {}))
        self.candidate = Implementation("candidate", **raw["candidate"])
        self.cases = tuple(Case(c["id"], dict(c["axes"]), dict(c.get("scalars", {})),
                                tuple(c["checks"]), tuple(c["seeds"])) for c in raw["cases"])
        names = list(self.inputs) + list(self.outputs)
        if len(names) != len(set(names)):
            raise ContractError("Input and output names must be distinct")
        for name in self.mutable:
            if name not in self.inputs or self.inputs[name]["kind"] != "tensor":
                raise ContractError(f"Mutable entry {name!r} is not a tensor input")

    @property
    def call_keys(self) -> tuple[str, ...]:
        return tuple(self.inputs) + tuple(self.outputs)

    @property
    def checked_tensors(self) -> tuple[str, ...]:
        """Every tensor the operator writes: outputs, then mutable inputs."""
        return tuple(self.outputs) + self.mutable

    def implementation(self, role: str) -> Implementation:
        if role == "baseline":
            return self.baseline
        if role == "candidate":
            return self.candidate
        raise ContractError(f"Unknown role {role!r}")

    def symbols(self, case: Case) -> dict[str, int]:
        symbols = {}
        for name, axis in self.axes.items():
            symbols[name] = axis["value"] if axis["kind"] == "const" else case.axes[name]
        for name, expression in self.derived_axes.items():
            symbols[name] = evaluate_dim(expression, symbols)
        return symbols

    def shape(self, declaration: dict, symbols: dict[str, int]) -> tuple[int, ...]:
        return tuple(evaluate_dim(entry, symbols) for entry in declaration["shape"])

    def scalars(self, case: Case) -> dict[str, Any]:
        values = {}
        for name, declaration in self.inputs.items():
            if declaration["kind"] == "scalar":
                values[name] = case.scalars.get(name, declaration.get("default"))
        return values

    def case(self, case_id: str) -> Case:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise ContractError(f"Unknown case {case_id!r}")


def load_contract(path: Path = CONTRACT_PATH) -> Contract:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError(f"Cannot read resolved contract: {exc}") from exc
    return Contract(raw)


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def finite_float(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)
