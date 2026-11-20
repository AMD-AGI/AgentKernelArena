"""Concrete tensor shapes of one workload case, from the task's declared interface."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from ..task_contract import TaskContract, case_symbols, evaluate_dimension
from .dtypes import tensor_bytes


@dataclass(frozen=True)
class Tensor:
    name: str
    direction: str  # "input" or "output"
    role: str       # activation, weight, state, index, metadata; outputs are "output"
    dtype: str
    shape: tuple[int, ...]
    expressions: tuple[Any, ...]
    mutable: bool = False
    # Derived axes that round up (directly or through another derived axis), e.g. ceil_div chunk counts.
    rounded_axes: frozenset[str] = frozenset()

    @property
    def nbytes(self) -> int:
        return tensor_bytes(self.shape, self.dtype)

    @property
    def maybe_padded(self) -> bool:
        """Whether a declared extent is rounded up (``(m + 31) // 32 * 32``, ``max(...)``).

        Rounded-up rows or columns can be padding that no correct kernel has to touch.
        """
        return any(_rounds_up(expression) or bool(_names(expression) & self.rounded_axes)
                   for expression in self.expressions)

    @property
    def packed_values_per_element(self) -> int:
        """2 for a uint8 tensor whose innermost extent is ``<axis> // 2``: two fp4 values per byte."""
        if self.dtype == "float4_e2m1fn_x2":
            return 2
        if self.dtype != "uint8" or not self.expressions:
            return 1
        inner = self.expressions[-1]
        return 2 if isinstance(inner, str) and inner.replace(" ", "").endswith("//2") else 1


def _names(expression: Any) -> set[str]:
    if not isinstance(expression, str):
        return set()
    return {node.id for node in ast.walk(ast.parse(expression, mode="eval")) if isinstance(node, ast.Name)}


def _rounds_up(expression: Any) -> bool:
    if not isinstance(expression, str):
        return False
    for node in ast.walk(ast.parse(expression, mode="eval")):
        if isinstance(node, ast.Call) and node.func.id in ("max", "ceil_div"):
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            for side in (node.left, node.right):
                # (x + c) // c * c rounds up; a plain ratio such as (heads // groups) * dim does not.
                if (isinstance(side, ast.BinOp) and isinstance(side.op, ast.FloorDiv)
                        and isinstance(side.left, ast.BinOp) and isinstance(side.left.op, ast.Add)):
                    return True
                if isinstance(side, ast.Call) and side.func.id == "ceil_div":
                    return True
    return False


@dataclass(frozen=True)
class CaseShapes:
    task_id: str
    case_id: str
    operator: str
    category: str
    axes: dict[str, int]
    symbols: dict[str, int]
    tensors: tuple[Tensor, ...]
    scalars: dict[str, Any] = field(default_factory=dict)
    # The task's ``description``, on one line: scenario facts that are not axes (non-causal, no activation).
    description: str = ""

    def tensor(self, name: str) -> Tensor:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(f"{self.task_id} declares no tensor {name!r}")

    @property
    def inputs(self) -> tuple[Tensor, ...]:
        return tuple(t for t in self.tensors if t.direction == "input")

    @property
    def outputs(self) -> tuple[Tensor, ...]:
        return tuple(t for t in self.tensors if t.direction == "output")

    def value(self, expression: Any) -> int:
        return evaluate_dimension(expression, self.symbols)


def case_shapes(contract: TaskContract, case: dict) -> CaseShapes:
    interface = contract.interface
    symbols = case_symbols(interface, case["axes"])
    mutable = set(interface.get("mutable", []))
    rounded: set[str] = set()
    for name, expression in interface.get("derived_axes", {}).items():  # declaration order resolves chains
        if _rounds_up(expression) or _names(expression) & rounded:
            rounded.add(name)
    rounded_axes = frozenset(rounded)
    tensors = []
    for name, declaration in interface["inputs"].items():
        if declaration["kind"] != "tensor":
            continue  # scalars carry no bytes
        expressions = tuple(declaration["shape"])
        tensors.append(Tensor(name, "input", declaration.get("role", "activation"), declaration["dtype"],
                              tuple(evaluate_dimension(e, symbols) for e in expressions), expressions,
                              name in mutable, rounded_axes))
    for name, declaration in interface["outputs"].items():
        expressions = tuple(declaration["shape"])
        tensors.append(Tensor(name, "output", "output", declaration["dtype"],
                              tuple(evaluate_dimension(e, symbols) for e in expressions), expressions))
    scalars = {name: case.get("scalars", {}).get(name, declaration["default"])
               for name, declaration in interface["inputs"].items() if declaration["kind"] == "scalar"}
    operator = contract.raw["operator"]
    description = " ".join(str(contract.raw.get("description", "")).split())
    return CaseShapes(contract.task_id, case["id"], operator["name"], operator["category"],
                      dict(case["axes"]), symbols, tuple(tensors), scalars, description)


def performance_cases(contract: TaskContract) -> list[dict]:
    return [case for case in contract.workloads["cases"] if "performance" in case["checks"]]
