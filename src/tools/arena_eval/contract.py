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

PYTHON_LANGUAGES = frozenset({"triton", "gluon", "flydsl", "tilelang", "python"})
TARGET_LANGUAGES = frozenset({"triton", "gluon", "hip", "flydsl"})
# Keyword inputs the evaluator appends to every rank's call of a multi-GPU task.
RESERVED_INPUTS = ("rank", "world_size", "peers", "symmetric")


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
        self.candidate = Implementation("candidate", **raw["candidate"])
        self.cases = tuple(Case(c["id"], dict(c["axes"]), dict(c.get("scalars", {})),
                                tuple(c["checks"]), tuple(c["seeds"])) for c in raw["cases"])
        self.distributed = raw.get("distributed")
        names = list(self.inputs) + list(self.outputs)
        if len(names) != len(set(names)):
            raise ContractError("Input and output names must be distinct")
        for name in self.mutable:
            if name not in self.inputs or self.inputs[name]["kind"] != "tensor":
                raise ContractError(f"Mutable entry {name!r} is not a tensor input")
        if self.distributed is not None:
            world = self.distributed.get("world_size")
            if type(world) is not int or not 2 <= world <= 8:
                raise ContractError("distributed.world_size must be an integer from 2 to 8")
            if set(names) & set(RESERVED_INPUTS):
                raise ContractError(f"Reserved multi-GPU inputs {RESERVED_INPUTS} cannot be task names")
            if any(n not in self.checked_tensors for n in self.ranks_agree):
                raise ContractError("distributed.ranks_agree names a tensor the operator does not write")

    @property
    def world_size(self) -> int:
        return self.distributed["world_size"] if self.distributed else 1

    @property
    def ranks_agree(self) -> tuple[str, ...]:
        return tuple((self.distributed or {}).get("ranks_agree", ()))

    @property
    def call_keys(self) -> tuple[str, ...]:
        keys = tuple(self.inputs) + tuple(self.outputs)
        return keys + RESERVED_INPUTS if self.distributed else keys

    @property
    def checked_tensors(self) -> tuple[str, ...]:
        """Every tensor the operator writes: outputs, then mutable inputs."""
        return tuple(self.outputs) + self.mutable

    @property
    def written_tensors(self) -> tuple[str, ...]:
        """What a HIP op schema marks written: the checked tensors and the symmetric buffer."""
        return self.checked_tensors + (("symmetric",) if self.distributed else ())

    def symmetric_bytes(self, case: Case) -> int:
        """Bytes of the symmetric peer buffer one call of ``case`` receives."""
        value = evaluate_dim(self.distributed["symmetric_bytes"], self.symbols(case))
        if value < 1:
            raise ContractError(f"case {case.case_id}: symmetric_bytes must be positive")
        return value

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
        if self.distributed is not None:
            symbols["world_size"] = self.distributed["world_size"]
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
