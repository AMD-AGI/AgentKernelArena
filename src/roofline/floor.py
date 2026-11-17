"""Combine byte and FLOP lower bounds with peaks into a per-case floor record."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..task_contract import TaskContract
from .arch import Peaks
from .bytes_model import generic_bytes
from .generated import DATA_MODELS, generated_inputs
from .interface import case_shapes, performance_cases
from .operators import model_for
from .operators.tiers import AUDIT_G1, MEMORY_BOUND
from .precision_policy import charge_fp32_products

GENERIC_MODEL = "generic@v1"
GPU_GENERATION_NOTE = "data_model_requires_gpu_generation"

# Model labels whose lower-bound argument a human has reviewed. Adding a label
# here is a reviewed change; join downgrades any model a measurement beats.
REVIEWED_MODELS: frozenset[str] = frozenset()


def case_floor(contract: TaskContract, case: dict, peaks: Peaks, inputs: dict | None = None,
               task_dir: Path | None = None, probe: dict | None = None) -> dict[str, Any]:
    """The floor of one case; ``inputs`` (generated tensors) enable the operator's data model, if any.

    ``probe`` is a loaded precision-probe file (``precision_policy.load_probe``); without one, float32
    products are charged at the float32 peak. ``task_dir`` is the package whose digest is compared with
    the probe's (default: tasks/<id>).
    """
    shapes = case_shapes(contract, case)
    counted = generic_bytes(shapes)
    notes = list(counted.notes)
    model = model_for(shapes.operator)
    data_model = DATA_MODELS.get(shapes.operator) if inputs is not None else None
    flops_matrix: dict[str, int] = {}
    flops_vector = 0
    bytes_min, bytes_rule = counted.bytes, counted.rule
    if data_model is not None:
        estimate = data_model[1](shapes, inputs)
        flops_matrix, flops_vector = estimate.flops_matrix, estimate.flops_vector
        notes.extend(estimate.notes)
        if estimate.bytes_override is not None:
            bytes_min, bytes_rule = estimate.bytes_override, "operator"
        label = data_model[0]
    elif model is not None:
        estimate = model.estimate(shapes)
        flops_matrix, flops_vector = estimate.flops_matrix, estimate.flops_vector
        notes.extend(estimate.notes)
        if estimate.bytes_override is not None:  # the model argues its own lower bound
            bytes_min, bytes_rule = estimate.bytes_override, "operator"
        label = model.label
    else:
        label = GENERIC_MODEL
    flops_matrix, policy_notes = charge_fp32_products(shapes.task_id, shapes.case_id, flops_matrix, task_dir,
                                                      probe)
    notes.extend(policy_notes)

    t_matrix = sum(flops / peaks.matrix_flops_per_s(key) for key, flops in flops_matrix.items())
    t_vector = flops_vector / peaks.vector_flops_per_s
    t_memory = bytes_min / peaks.bytes_per_s
    times = {"compute_matrix": t_matrix, "compute_vector": t_vector, "memory": t_memory}
    bound = max(times, key=times.get)

    if max(times.values()) == 0:
        tier = "none"
        notes.append("no_lower_bound: every tensor is excluded as padded, partial or mutable")
    elif model is None and data_model is None and shapes.operator not in MEMORY_BOUND:
        tier = "none"
        notes.append("no_work_model: a bytes-only floor is not meaningful for this operator")
    elif label in REVIEWED_MODELS and bytes_rule != "G2":
        tier = "verified"
    else:
        tier = "estimated"
    if model is None and shapes.operator in AUDIT_G1:
        notes.append("g1_audit_pending: may read only part of an input")
    unconfirmed = sorted(set(flops_matrix) & set(peaks.unconfirmed))
    if unconfirmed:
        notes.append("unconfirmed_peak:" + ",".join(unconfirmed))
        if tier == "verified":
            tier = "estimated"
    keys = sorted(flops_matrix)
    return {
        "task_id": shapes.task_id, "case_id": shapes.case_id,
        "operator": shapes.operator, "category": shapes.category, "axes": shapes.axes,
        "flops_matrix": sum(flops_matrix.values()), "flops_matrix_by_precision": flops_matrix,
        "flops_vector": flops_vector, "bytes_min": bytes_min, "bytes_rule": bytes_rule,
        "precision_key": keys[0] if len(keys) == 1 else (keys or None),
        "peaks": peaks.describe(),
        "t_compute_matrix_us": t_matrix * 1e6, "t_compute_vector_us": t_vector * 1e6,
        "t_memory_us": t_memory * 1e6, "t_floor_us": max(times.values()) * 1e6,
        "bound": bound, "tier": tier, "model": label, "notes": notes,
    }


def task_floors(contract: TaskContract, peaks: Peaks, task_dir: Path | None = None,
                generate: bool = False, device: str = "cuda", probe: dict | None = None) -> list[dict[str, Any]]:
    """Floors of every performance case. With ``generate``, operators that have a data model are
    counted from the case's generated inputs (both seeds; the smaller floor is kept).

    Only inputs drawn on the GPU reproduce the evaluator's timed inputs. A CPU draw is a different
    sample, and for high-variance routing or lengths it can exceed the timed work by tens of percent
    (the minimum over two seeds does not bound that), so on the CPU no data-model floor is emitted:
    the case keeps its shape-only floor, or tier ``none`` when the operator has no shape-only model.
    A case whose generation fails falls back the same way, with the error in its notes.
    """
    records = []
    use_data = generate and contract.raw["operator"]["name"] in DATA_MODELS and task_dir is not None
    for case in performance_cases(contract):
        if not use_data:
            records.append(case_floor(contract, case, peaks, task_dir=task_dir, probe=probe))
            continue
        if device != "cuda":
            records.append(_shape_only(contract, case, peaks, task_dir, GPU_GENERATION_NOTE, probe))
            continue
        try:
            candidates = [case_floor(contract, case, peaks,
                                     generated_inputs(task_dir, contract, case["id"], seed, device), task_dir,
                                     probe)
                          for seed in (0, 1)]
            record = min(candidates, key=lambda candidate: candidate["t_floor_us"])
            record["generated_on"] = device
            records.append(record)
        except Exception as exc:  # a generator that needs a GPU, or a missing length tensor
            records.append(_shape_only(contract, case, peaks, task_dir,
                                       f"data_model_failed: {type(exc).__name__}: {exc}", probe))
    return records


def _shape_only(contract: TaskContract, case: dict, peaks: Peaks, task_dir: Path | None, note: str,
                probe: dict | None = None) -> dict[str, Any]:
    """A data-model case without usable generated inputs: its shape-only floor, or tier ``none`` when the
    operator has no shape-only model."""
    record = case_floor(contract, case, peaks, task_dir=task_dir, probe=probe)
    if model_for(record["operator"]) is None:
        record["tier"] = "none"
    record["notes"].append(note)
    return record
