# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Input construction and output naming for the fused mHC post->pre workload.

The buffers are allocated here, in the definition's declared shapes and dtypes,
and filled by ``task_initialize``, which is the schema bundle's own
``initialize`` callback: normal residual streams and layer output, bounded
incoming post gates, approximately doubly stochastic combination matrices,
fan-in scaled projection weights, unit mix scales, small mix biases and a
positive norm weight. Nothing about the distribution is decided in this file,
because the acceptance run that verifies a result uses that callback and a
second implementation of it here would be a second operator.

Each case is built on its own, from a generator re-seeded to ``seed`` for that
case, because the bundle initializes one workload point at a time. The residual
carries the case's token count and is drawn first, so the weights land at a
different point in the stream for every case.

Scalar inputs keep the literal values of the bundle's workload rows. Every
constant lives in the configured workload JSON, so these helpers carry no
operator dimensions of their own. Arena copies each task directory into its own
workspace, so a task cannot import from a sibling and every task has to carry
its own copy of these modules.
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
VARIABLE_AXIS = str(WORKLOAD["variable_axis"])
INPUTS: dict[str, dict] = dict(WORKLOAD["inputs"])
OUTPUTS: dict[str, dict] = dict(WORKLOAD["outputs"])
OUTPUT_NAMES: tuple[str, ...] = tuple(OUTPUTS)
SCALARS: dict[str, Any] = dict(WORKLOAD["scalars"])
STREAMS = AXES["streams"]
HIDDEN_SIZE = AXES["hidden_size"]
SEED = int(WORKLOAD["seed"])

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "int32": torch.int32}

# The entrypoint is explicit task data; it is not derived from operator identity.
BUILDER_SYMBOL = task_contract.candidate_entry()["symbol"]

# Benchmark parameters used for both baseline and candidate. Timing must be
# CUDA-graph based: at small token counts this operator runs for microseconds
# and eager timing would be dominated by per-call host dispatch.
BENCH_WARMUP = int(WORKLOAD["bench"]["warmup"])
BENCH_REPETITION = int(WORKLOAD["bench"]["repetition"])
BENCH_TARGET_MS = float(WORKLOAD["bench"]["target_ms"])

CASES: tuple[dict[str, Any], ...] = tuple(WORKLOAD["cases"])
CASE_IDS: tuple[str, ...] = tuple(str(case["case_id"]) for case in CASES)

GATE_EXPLANATION = (
    "gate: scripts/task_compare.py, the schema bundle's own comparison callback. "
    "It admits a candidate when every element of every named output is within "
    "its tolerance, and it owns that tolerance -- nothing here sets or relaxes it."
)


def dimensions(case: dict[str, Any]) -> dict[str, int]:
    return {**AXES, VARIABLE_AXIS: int(case[VARIABLE_AXIS])}


def declared_shape(spec: dict, dims: dict[str, int]) -> tuple[int, ...]:
    return tuple(dims[name] for name in spec["shape"])


def build_case_inputs(case: dict[str, Any], device: str = "cuda") -> dict[str, Any]:
    """Allocate one case's declared buffers and let the bundle fill them.

    The bundle's callback writes preallocated buffers in place and validates
    their dtype, shape, contiguity, non-overlap and scalar values before it
    writes anything, so allocating them in the declared input order is the whole
    of this task's share of input construction.
    """
    dims = dimensions(case)
    inputs = {
        name: (SCALARS[name] if spec["shape"] is None else
               torch.empty(declared_shape(spec, dims), dtype=DTYPES[spec["dtype"]], device=device))
        for name, spec in INPUTS.items()
    }
    return task_initialize.run(inputs, seed=SEED)


def refill_case_inputs(inputs: dict[str, Any], seed: int) -> dict[str, Any]:
    """Redraw a case's buffers in place, keeping their storage.

    The bundle's callback writes preallocated buffers rather than allocating
    them, so redrawing through it changes the values while leaving every
    property the operator depends on untouched. A CUDA graph captured over these
    buffers therefore reads the new draw on its next replay, which is what makes
    the timed invocation answerable for a result it cannot have precomputed.
    """
    return task_initialize.run(inputs, seed=seed)


# The operands a production caller holds fixed while the activations change.
# The projection weight, mix scale/bias and norm weight are layer parameters,
# loaded once; the layer output, residual streams and incoming mixes change on
# every call. Holding the parameters across the timed samples is what lets an
# implementation re-lay them out once without being charged for it.
PERSISTENT_INPUTS: tuple[str, ...] = ("proj_weight", "mix_scale", "mix_bias", "norm_weight")


def redraw_call_varying_inputs(inputs: dict[str, Any], seed: int) -> dict[str, Any]:
    """Redraw only what changes between two calls on a live model.

    A new draw for a timed invocation has to move the ground under it without
    invalidating work a real deployment would legitimately do once. Re-laying
    the projection weight out on the first call and reusing it is that kind of
    work, so a redraw that also replaced it would make an implementation which
    did it look like one that skipped the operator.

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

    Each snapshot is what ``redraw_call_varying_inputs`` would leave in the
    call-varying buffers for that seed. The buffers themselves end as they
    started, so drawing ahead of time does not change what the next call reads.
    """
    names = tuple(
        name
        for name, value in inputs.items()
        if isinstance(value, torch.Tensor) and name not in PERSISTENT_INPUTS
    )
    current = {name: inputs[name].detach().clone() for name in names}
    draws = []
    for seed in seeds:
        redraw_call_varying_inputs(inputs, seed=seed)
        draws.append({name: inputs[name].detach().clone() for name in names})
    load_draw(inputs, current)
    return draws


def load_draw(inputs: dict[str, Any], draw: dict[str, torch.Tensor]) -> None:
    """Copy a snapshot into the live buffers, keeping their storage.

    A captured graph reads these addresses on every replay, so the copy is what
    the next replay consumes; the copy is enqueued on the current stream.
    """
    for name, value in draw.items():
        inputs[name].copy_(value)


def call_kwargs(inputs: dict[str, Any]) -> dict[str, Any]:
    """The operator's full argument set, in the schema's input order."""
    return {name: inputs[name] for name in INPUTS}


def call_args(inputs: dict[str, Any]) -> tuple[Any, ...]:
    """Positional launch arguments, in the schema's input order."""
    return tuple(inputs[name] for name in INPUTS)


def named_outputs(value: Any) -> dict[str, Any]:
    """Map a returned tuple/list (declared order) or mapping onto output names.

    Raises AssertionError for a return value that does not carry exactly the
    declared outputs, which is a candidate contract failure.
    """
    if isinstance(value, dict):
        if set(value) != set(OUTPUT_NAMES):
            raise AssertionError(f"outputs must be named {list(OUTPUT_NAMES)}")
        return {name: value[name] for name in OUTPUT_NAMES}
    if not isinstance(value, (tuple, list)) or len(value) != len(OUTPUT_NAMES):
        raise AssertionError(f"operator must return {len(OUTPUT_NAMES)} outputs in declared order")
    return dict(zip(OUTPUT_NAMES, value))


def verdict(got: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, str]:
    """Apply the bundle's comparison callback and report what it decided.

    The callback raises AssertionError for a candidate that fails its contract
    or its tolerance, and ValueError for a reference it considers invalid. Only
    the first is a candidate verdict, so only the first is caught: an invalid
    reference is this task's bug and has to stop the run rather than be scored
    as a failed port.

    No metric is recomputed here. The callback's own message carries the output
    name and numbers it rejected on, and a second implementation of them would
    be the thing this whole file exists to avoid.
    """
    try:
        task_compare.run(got, expected)
    except AssertionError as failure:
        return False, str(failure)
    return True, "within tolerance"


def assert_candidate_is_independent(source: str) -> None:
    """Apply the documented dependency policy before candidate import."""
    task_contract.assert_source_independent(source)
