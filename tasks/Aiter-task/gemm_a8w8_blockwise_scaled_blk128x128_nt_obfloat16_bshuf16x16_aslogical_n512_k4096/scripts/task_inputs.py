# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Input construction for a blockwise-scaled FP8 GEMM workload.

The buffers are allocated here, in the definition's declared shapes and dtypes,
and filled by ``task_initialize``, which is the schema bundle's own
``initialize`` callback: standard-normal FP8 activations and weights, positive
nonuniform block scales, the AITER 16x16 weight shuffle, and the declared
activation-scale storage. Nothing about the distribution or the encodings is
decided in this file, because the acceptance run that verifies a result uses
that callback and a second implementation of it here would be a second operator.

Each case is built on its own, from a generator re-seeded to ``seed`` for that
case. The activations carry the case's m and are drawn first, so the weights
land at a different point in the stream for every case.

Every constant that varies between tasks in this family lives in the configured
workload JSON, so the helpers and runner stay byte-identical across the family.
Arena copies each task directory into its own workspace, so a task cannot import
from a sibling and every task has to carry its own copy of these modules.
"""

from __future__ import annotations

from typing import Any

import torch

import task_compare
import task_initialize

# The declared workload path is resolved only inside this task workspace.
import task_contract

WORKLOAD = task_contract.load_workload()

DEFINITION = str(WORKLOAD["definition"])
AXES: dict[str, int] = dict(WORKLOAD["axes"])
N, K = AXES["n"], AXES["k"]
INPUTS: dict[str, dict] = dict(WORKLOAD["inputs"])
OUTPUT_SPEC: dict = WORKLOAD["outputs"]["out"]
A_SCALE_STORAGE = str(WORKLOAD["a_scale_storage"])
SEED = int(WORKLOAD["seed"])

DTYPES = {"float8_e4m3fn": torch.float8_e4m3fn, "float32": torch.float32, "bfloat16": torch.bfloat16}

# The entrypoint is explicit task data; it is not derived from operator identity.
BUILDER_SYMBOL = task_contract.candidate_entry()["symbol"]

# Benchmark parameters used for both baseline and candidate. Timing must be
# CUDA-graph based: at the small-m cases this operator runs for microseconds
# and eager timing would be dominated by per-call host dispatch.
BENCH_WARMUP = int(WORKLOAD["bench"]["warmup"])
BENCH_REPETITION = int(WORKLOAD["bench"]["repetition"])
BENCH_TARGET_MS = float(WORKLOAD["bench"]["target_ms"])

CASES: tuple[dict[str, Any], ...] = tuple(WORKLOAD["cases"])
CASE_IDS: tuple[str, ...] = tuple(str(case["case_id"]) for case in CASES)

GATE_EXPLANATION = (
    "gate: scripts/task_compare.py, the schema bundle's own comparison callback. "
    "It admits a candidate when every element is within its tolerance, and it "
    "owns that tolerance -- nothing here sets or relaxes it."
)


def dimensions(case: dict[str, Any]) -> dict[str, int]:
    return {**AXES, "m": int(case["m"])}


def declared_shape(spec: dict, dims: dict[str, int]) -> tuple[int, ...]:
    return tuple(dims[name] for name in spec["shape"])


def build_case_inputs(case: dict[str, Any], device: str = "cuda") -> dict[str, Any]:
    """Allocate one case's declared buffers and let the bundle fill them.

    The bundle's callback writes preallocated buffers in place and validates
    their dtype, shape, contiguity and non-overlap before it writes anything, so
    allocating them is the whole of this task's share of input construction.
    """
    dims = dimensions(case)
    inputs = {name: torch.empty(declared_shape(spec, dims), dtype=DTYPES[spec["dtype"]], device=device)
              for name, spec in INPUTS.items()}
    return task_initialize.run(inputs, seed=SEED)


def refill_case_inputs(inputs: dict[str, Any], seed: int) -> dict[str, Any]:
    """Redraw a case's buffers in place, keeping their storage.

    The bundle's callback writes preallocated buffers rather than allocating
    them, so redrawing through it changes the values while leaving every
    property the operator depends on untouched. A CUDA graph captured over these
    buffers therefore reads the new draw on its next replay.
    """
    return task_initialize.run(inputs, seed=seed)


# The operands a production caller holds fixed while the activations change.
# ``b`` and ``b_scale`` are the quantized weight and its block scales: loaded
# once and reused for every batch. Holding them across the timed samples is what
# lets an implementation re-lay them out once without being charged for it.
PERSISTENT_INPUTS: tuple[str, ...] = ("b", "b_scale")


def redraw_call_varying_inputs(inputs: dict[str, Any], seed: int) -> dict[str, Any]:
    """Redraw only what changes between two calls on a live model.

    The draw itself stays the bundle's: the full callback runs, and the operands
    the caller owns across calls are then restored. Selecting a subset of the
    bundle's initializers instead would put a second copy of which distribution
    fills which buffer in this file, and that copy is what goes stale.
    """
    held = {name: inputs[name].detach().clone() for name in PERSISTENT_INPUTS}
    refill_case_inputs(inputs, seed=seed)
    for name, value in held.items():
        inputs[name].copy_(value)
    return inputs


def call_varying_draws(
    inputs: dict[str, Any], seeds: list[int]
) -> list[dict[str, torch.Tensor]]:
    """One snapshot of the call-varying operands per seed, drawn by the bundle.

    The buffers themselves end as they started, so drawing ahead of time does
    not change what the next call reads.
    """
    names = tuple(name for name, value in inputs.items()
                  if isinstance(value, torch.Tensor) and name not in PERSISTENT_INPUTS)
    current = {name: inputs[name].detach().clone() for name in names}
    draws = []
    for seed in seeds:
        redraw_call_varying_inputs(inputs, seed=seed)
        draws.append({name: inputs[name].detach().clone() for name in names})
    load_draw(inputs, current)
    return draws


def load_draw(inputs: dict[str, Any], draw: dict[str, torch.Tensor]) -> None:
    """Copy a snapshot into the live buffers, keeping their storage."""
    for name, value in draw.items():
        inputs[name].copy_(value)


def call_kwargs(inputs: dict[str, Any]) -> dict[str, Any]:
    """The operator's full argument set, in the schema's input order."""
    return {name: inputs[name] for name in INPUTS}


def call_args(inputs: dict[str, Any]) -> tuple[Any, ...]:
    """Positional launch arguments, in the schema's input order."""
    return tuple(inputs[name] for name in INPUTS)


def verdict(got: torch.Tensor, expected: torch.Tensor) -> tuple[bool, str]:
    """Apply the bundle's comparison callback and report what it decided.

    The callback raises AssertionError for a candidate that fails its contract
    or its tolerance, and ValueError for a reference it considers invalid. Only
    the first is a candidate verdict, so only the first is caught: an invalid
    reference is this task's bug and has to stop the run rather than be scored
    as a failed port.
    """
    try:
        task_compare.run(got, expected)
    except AssertionError as failure:
        return False, str(failure)
    return True, "within tolerance"


def assert_candidate_is_independent(source: str) -> None:
    """Apply the documented dependency policy before candidate import."""
    task_contract.assert_source_independent(source)
