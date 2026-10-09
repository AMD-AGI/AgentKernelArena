# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Input construction, valid lengths and destination buffer for paged top-k.

The buffers are allocated here, in the definition's declared shapes and dtypes,
and filled by ``task_initialize``, which is the schema bundle's own
``initialize`` callback: tie-free scores, a random legal page permutation per
row, lengths chosen from 0, a quarter, half or all of the capacity, and the
conservative v2 routing plan. Nothing about the distribution is decided in this
file, because the acceptance run that verifies a result uses that callback and a
second implementation of it here would be a second operator.

The callback's lengths alone do not exercise the operator: the valid lengths
are values inside ``seq_lens``, and a decode step sees them change while every
buffer capacity stays fixed. Each case therefore names its lengths. A
``bundle`` case keeps the callback's own; every other case writes its declared
lengths into ``seq_lens`` after the callback has run. The routing plan the
callback writes routes no row to the cluster pool and its threshold exceeds
every legal length, so it matches every declared length; ``check_case_inputs``
verifies that, rather than assuming it, for every input that is checked or
timed.

Each case is built on its own, from a generator re-seeded to ``seed`` for that
case. Every constant lives in the configured workload JSON. Arena copies each
task directory into its own workspace, so a task cannot import from a sibling
and every task has to carry its own copy of these modules.
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
INPUTS: dict[str, dict] = dict(WORKLOAD["inputs"])
OUTPUTS: dict[str, dict] = dict(WORKLOAD["outputs"])
(OUTPUT_NAME,) = tuple(OUTPUTS)
SCALARS: dict[str, Any] = dict(WORKLOAD["scalars"])
K = AXES["k"]
PAGE_SIZE = AXES["page_size"]
SEED = int(WORKLOAD["seed"])

DTYPES = {"float32": torch.float32, "int32": torch.int32, "int64": torch.int64}

# Written into the destination before every call, outside timing. Legal outputs
# are -1 padding or nonnegative slots, so any element left unwritten fails.
OUTPUT_POISON = -2
# The v2 plan's threshold field is read as uint32 by the backend.
PLAN_UINT32_MAX = 2**32 - 1

# The entrypoint is explicit task data; it is not derived from operator identity.
BUILDER_SYMBOL = task_contract.candidate_entry()["symbol"]

# Benchmark parameters used for both baseline and candidate. Timing must be
# CUDA-graph based: short rows and small batches run for microseconds and
# eager timing would be dominated by per-call host dispatch.
BENCH_WARMUP = int(WORKLOAD["bench"]["warmup"])
BENCH_REPETITION = int(WORKLOAD["bench"]["repetition"])
BENCH_TARGET_MS = float(WORKLOAD["bench"]["target_ms"])

CASES: tuple[dict[str, Any], ...] = tuple(WORKLOAD["cases"])
CASE_IDS: tuple[str, ...] = tuple(str(case["case_id"]) for case in CASES)

GATE_EXPLANATION = (
    "gate: scripts/task_compare.py, the schema bundle's own comparison callback. "
    "It requires the exact selected set per row, -1 padding in the reference "
    "positions and the exact order of short rows -- nothing here relaxes it."
)


def dimensions(case: dict[str, Any]) -> dict[str, int]:
    return {**AXES, **{axis: int(case[axis]) for axis in task_contract.VARIABLE_AXES}}


def declared_shape(spec: dict, dims: dict[str, int]) -> tuple[int, ...]:
    return tuple(dims[name] for name in spec["shape"])


def initialize_case(inputs: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Fill existing buffers for a case: the bundle's draw, then the case lengths."""
    task_initialize.run(inputs, seed=SEED)
    lengths = task_contract.row_lengths(case, WORKLOAD)
    if lengths is not None:
        inputs["seq_lens"].copy_(torch.tensor(lengths, dtype=torch.int32))
    return inputs


def build_case_inputs(case: dict[str, Any], device: str = "cuda") -> dict[str, Any]:
    """Allocate one case's declared buffers and let the bundle fill them."""
    dims = dimensions(case)
    inputs = {
        name: (SCALARS[name] if spec["shape"] is None else
               torch.empty(declared_shape(spec, dims), dtype=DTYPES[spec["dtype"]], device=device))
        for name, spec in INPUTS.items()
    }
    return initialize_case(inputs, case)


def refill_case_inputs(inputs: dict[str, Any], seed: int) -> dict[str, Any]:
    """Redraw a case's buffers in place, keeping their storage.

    The bundle's callback writes preallocated buffers rather than allocating
    them, so redrawing through it changes the values while leaving every
    property the operator depends on untouched. A CUDA graph captured over these
    buffers therefore reads the new draw on its next replay.
    """
    return task_initialize.run(inputs, seed=seed)


# The operands a case holds fixed while the call-varying ones change: the valid
# lengths it declares and the routing plan that has to match them. Scores and
# the page table are new for every decode step.
PERSISTENT_INPUTS: tuple[str, ...] = ("seq_lens", "metadata")


def redraw_call_varying_inputs(inputs: dict[str, Any], seed: int) -> dict[str, Any]:
    """Redraw the scores and page table through the bundle, keeping the case lengths.

    The full callback runs, and the operands the case owns are then restored.
    Selecting a subset of the bundle's initializers instead would put a second
    copy of which distribution fills which buffer in this file.
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
    """Copy a snapshot into the live buffers, keeping their storage."""
    for name, value in draw.items():
        inputs[name].copy_(value)


def call_kwargs(inputs: dict[str, Any]) -> dict[str, Any]:
    """The reference's argument set, in the schema's input order."""
    return {name: inputs[name] for name in INPUTS}


def allocate_output(inputs: dict[str, Any]) -> torch.Tensor:
    """The destination buffer the caller owns, in the declared shape and dtype."""
    spec = OUTPUTS[OUTPUT_NAME]
    dims = {**AXES, "batch": inputs["scores"].shape[0]}
    return torch.empty(declared_shape(spec, dims), dtype=DTYPES[spec["dtype"]],
                       device=inputs["scores"].device)


def poison_output(out: torch.Tensor) -> None:
    out.fill_(OUTPUT_POISON)


def baseline_kwargs(inputs: dict[str, Any], out: torch.Tensor) -> dict[str, Any]:
    return {**call_kwargs(inputs), OUTPUT_NAME: out}


def launch_args(inputs: dict[str, Any], out: torch.Tensor) -> tuple[Any, ...]:
    """Positional candidate arguments: declared inputs in order, then the destination."""
    return (*(inputs[name] for name in INPUTS), out)


def plan_matches(seq_lens: torch.Tensor, metadata: torch.Tensor) -> bool:
    """Whether metadata is a valid v2 routing plan for these lengths.

    Row 0 holds (cluster_threshold, number of routed items); rows 1..N hold the
    (batch_id, seq_len) of exactly the rows longer than the threshold.
    """
    lengths = seq_lens.to(torch.int64).cpu()
    plan = metadata.to(torch.int64).cpu()
    threshold, count = int(plan[0, 0]) & PLAN_UINT32_MAX, int(plan[0, 1])
    routed = {(row, int(length)) for row, length in enumerate(lengths.tolist()) if length > threshold}
    items = {(int(row), int(length)) for row, length in plan[1:1 + count].tolist()}
    return 0 <= count <= lengths.numel() and len(items) == count and items == routed


def check_case_inputs(inputs: dict[str, Any], case: dict[str, Any], *, thorough: bool = False) -> dict:
    """Validate a case's inputs before they are checked or timed.

    Lengths must lie in [0, capacity] and equal the declared ones; the plan must
    match them; the page table must map every slot into signed int32. The
    thorough form also checks the valid score prefixes are tie-free and finite,
    which the definition requires of qualification inputs.
    """
    lengths = inputs["seq_lens"].to(torch.int64)
    limit = task_contract.capacity(case, AXES)
    if bool((lengths < 0).any()) or bool((lengths > limit).any()):
        raise RuntimeError(f"seq_lens outside [0, {limit}]")
    declared = task_contract.row_lengths(case, WORKLOAD)
    if declared is not None and lengths.cpu().tolist() != declared:
        raise RuntimeError("seq_lens differ from the case's declared lengths")
    if not plan_matches(inputs["seq_lens"], inputs["metadata"]):
        raise RuntimeError("metadata is not a matching v2 plan for seq_lens")
    tables = inputs["page_tables"].to(torch.int64)
    if bool((tables < 0).any()) or int(tables.max()) * PAGE_SIZE + PAGE_SIZE - 1 > 2**31 - 1:
        raise RuntimeError("page table maps outside signed int32 slots")
    evidence = {"lengths_min": int(lengths.min()), "lengths_max": int(lengths.max()),
                "distinct_lengths": sorted({int(v) for v in lengths.unique().tolist()})[:32],
                "plan_matches_lengths": True}
    if thorough:
        scores = inputs["scores"]
        valid = torch.arange(scores.shape[1], device=scores.device)[None, :] < inputs["seq_lens"][:, None]
        if bool((torch.isnan(scores) & valid).any()):
            raise RuntimeError("valid scores contain NaN")
        for row in range(scores.shape[0]):
            prefix = scores[row, :int(lengths[row])]
            if prefix.numel() > 1 and bool((prefix.sort().values.diff() == 0).any()):
                raise RuntimeError(f"row {row} has tied valid scores")
        evidence["valid_scores_tie_free"] = True
    return evidence


def verdict(got: torch.Tensor, expected: torch.Tensor) -> tuple[bool, str]:
    """Apply the bundle's comparison callback and report what it decided.

    The callback raises AssertionError for a candidate that fails its contract
    or its rule, and ValueError for a reference it considers invalid. Only the
    first is a candidate verdict, so only the first is caught.
    """
    try:
        task_compare.run(got, expected)
    except AssertionError as failure:
        return False, str(failure)
    return True, "matches the reference selection"


def assert_candidate_is_independent(source: str) -> None:
    """Apply the documented dependency policy before candidate import."""
    task_contract.assert_source_independent(source)
