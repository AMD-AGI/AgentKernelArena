"""Memory-bound reductions whose reference skips part of an input (byte overrides only)."""
from __future__ import annotations

from ..interface import CaseShapes
from . import Estimate, OperatorModel, register


def cross_entropy_label_smoothing(shapes: CaseShapes) -> Estimate:
    """Rows whose target equals ignore_index need not have their logits read, and from the shapes alone
    every row may be ignored, so no logits are counted here. ``floors --generate`` counts the rows that
    are not ignored exactly (``partial_data.cross_entropy_label_smoothing``)."""
    estimate = Estimate(notes=["bytes: logits not counted without generated targets (any row may be ignored)"])
    others = sum(t.nbytes for t in shapes.inputs if t.name != "x" and t.role != "weight")
    estimate.bytes_override = others + sum(t.nbytes for t in shapes.outputs)
    return estimate


register("cross_entropy_label_smoothing",
         OperatorModel("reduction/cross_entropy_label_smoothing", 2, cross_entropy_label_smoothing))
