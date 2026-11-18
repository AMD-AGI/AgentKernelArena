"""Operator models: FLOPs (and optionally tighter bytes) per operator name.

Models are keyed by ``operator.name`` from ``task.yaml``, never by task ID, so every
scenario of an operator shares one model. A model must only count work that every correct
implementation has to do, so every floor is a lower bound; each one documents why its count is a lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..interface import CaseShapes


@dataclass
class Estimate:
    # Necessary matrix FLOPs, split by the datasheet peak key that applies to them.
    flops_matrix: dict[str, int] = field(default_factory=dict)
    flops_vector: int = 0
    # A tighter byte lower bound than the generic rule, with its reason in notes.
    bytes_override: int | None = None
    notes: list[str] = field(default_factory=list)

    def add_matrix(self, key: str, flops: int) -> None:
        self.flops_matrix[key] = self.flops_matrix.get(key, 0) + flops


@dataclass(frozen=True)
class OperatorModel:
    name: str
    version: int
    estimate: Callable[[CaseShapes], Estimate]

    @property
    def label(self) -> str:
        return f"{self.name}@v{self.version}"


REGISTRY: dict[str, OperatorModel] = {}


def register(operator: str, model: OperatorModel) -> None:
    if operator in REGISTRY:
        raise ValueError(f"operator {operator!r} already has a model")
    REGISTRY[operator] = model


def model_for(operator: str) -> OperatorModel | None:
    return REGISTRY.get(operator)


from . import attention, chunked, composite, conv, gemm, moe, partial, reduction  # noqa: E402,F401  (register models)
