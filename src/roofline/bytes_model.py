"""Generic lower bounds on the bytes a correct kernel must move to or from HBM.

G1 (every input read once, every output written once) applies when the operator has no
mutable, state or index tensors: the evaluator poisons outputs before each call and
compares them whole, so every output byte must be written. Inputs a reference reads only in
part are listed in ``operators/tiers.py:PARTIAL_INPUTS`` and counted at their used size.

G2 is used otherwise. Weight, state, index and mutable tensors may be touched only in part
(paged KV caches, routed experts, gathered rows, cache slots), so they count zero bytes.
Index and metadata inputs are also left out under G2: a block table or gather list can be
read only up to the entries a case uses, so counting it whole would make the bound too high.

Both rules exclude inputs whose declared extent is rounded up (``(m + 31) // 32 * 32``):
the padding need not be read, and the unpadded size cannot be recovered generically.
Outputs are always counted whole because the evaluator compares them whole.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .interface import CaseShapes

G2_ROLES = ("state", "index")
G2_UNCOUNTED_ROLES = ("weight", "state", "index")


@dataclass
class ByteCount:
    bytes: int
    rule: str
    notes: list[str] = field(default_factory=list)


def needs_g2(shapes: CaseShapes) -> bool:
    return any(t.mutable or t.role in G2_ROLES for t in shapes.inputs)


def _partial_bytes(shapes: CaseShapes, tensor, rule: str | float | None) -> int:
    if rule is None:
        return 0
    if isinstance(rule, float):
        return int(tensor.nbytes * rule)
    count = 1
    for extent in tensor.shape:
        count *= extent
    element = tensor.nbytes // max(1, count)
    return min(tensor.nbytes, shapes.value(rule) * element)


def generic_bytes(shapes: CaseShapes) -> ByteCount:
    from .operators.tiers import PARTIAL_INPUTS

    rule = "G2" if needs_g2(shapes) else "G1"
    partial = PARTIAL_INPUTS.get(shapes.operator, {})
    total, notes = 0, []
    for tensor in shapes.inputs:
        if rule == "G2" and (tensor.mutable or tensor.role in G2_UNCOUNTED_ROLES):
            continue
        if tensor.maybe_padded:
            notes.append(f"padded_input_excluded:{tensor.name}")
            continue
        if tensor.name in partial:
            notes.append(f"partial_input:{tensor.name}")
            total += _partial_bytes(shapes, tensor, partial[tensor.name])
            continue
        total += tensor.nbytes
    total += sum(t.nbytes for t in shapes.outputs)
    return ByteCount(total, rule, notes)


def g1_bytes(shapes: CaseShapes) -> int:
    """G1 regardless of applicability; used to check that G2 never exceeds it."""
    return sum(t.nbytes for t in shapes.inputs if not t.maybe_padded) + sum(t.nbytes for t in shapes.outputs)
