# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Input construction for an a16w16 GEMM workload.

The buffers are allocated here and filled by ``task_initialize``, which is the
schema bundle's own ``initialize`` callback: both ``a`` and ``b`` are standard
normal. Nothing about the distribution is decided in this file, because the
acceptance run that verifies a result uses that callback and a second
implementation of it here would be a second operator.

That the distribution is not ours to pick is not an abstraction for its own
sake. An earlier revision of the bundle drew ``b`` at 1/sqrt(k), which put the
output at unit scale and therefore inside the comparison's absolute tolerance
everywhere -- a kernel that truncated its partial sums to bf16 matched on every
element under it and missed roughly a tenth of the output under this one. The
task followed the bundle in both directions without a line of policy here.

Each case is built on its own, from a generator re-seeded to ``seed`` for that
case, because the bundle initializes one workload point at a time. ``a`` carries
the case's m and is drawn first, so ``b`` lands at a different point in the
stream for every case -- the cases do not share a weight and cannot be built
from one pass.

Every constant that varies between tasks in this family lives in
the configured workload JSON, so the helpers and runner stay byte-identical
across the GEMM tasks. Arena copies each task
directory into its own workspace, so a task cannot import from a sibling and
every task has to carry its own copy of these modules.
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
N = int(WORKLOAD["axes"]["n"])
K = int(WORKLOAD["axes"]["k"])
TRANS_B = bool(WORKLOAD["trans_b"])
SEED = int(WORKLOAD["seed"])

# The draw used to re-arm a timed invocation. It only has to differ from SEED:
# the point is that the values a captured graph replays over are ones no earlier
# call in this process has seen, not that they come from a second distribution.
REFILL_SEED = SEED + 1

# The draws the timed samples rotate through, one per replay, and the draws the
# timed unit is replayed over once each after timing. The two sets are disjoint
# from each other and from SEED and REFILL_SEED, so an unseen draw is one the
# implementation cannot have encountered before its timed replay.
TIMED_DRAWS = 3
UNSEEN_DRAWS = 4
TIMED_DRAW_SEEDS: tuple[int, ...] = tuple(
    range(REFILL_SEED + 1, REFILL_SEED + 1 + TIMED_DRAWS)
)
UNSEEN_DRAW_SEEDS: tuple[int, ...] = tuple(
    range(TIMED_DRAW_SEEDS[-1] + 1, TIMED_DRAW_SEEDS[-1] + 1 + UNSEEN_DRAWS)
)

# How much slower than the reported mean a replay over an unseen draw may be.
# The timed samples rotate through a few draws, so an implementation that keeps
# results keyed on its inputs' values can still serve every sample from memory
# once it has seen them all; a draw it has never seen is a miss, and that miss
# is the operator's actual cost. An implementation that computes on every call
# runs an unseen draw in the time of any other replay.
UNSEEN_DRAW_MARGIN = 1.5

# The entrypoint is explicit task data; it is not derived from operator identity.
BUILDER_SYMBOL = task_contract.candidate_entry()["symbol"]

# Benchmark parameters used for both baseline and candidate. Timing
# must be CUDA-graph based: at the small-M cases this operator runs for tens of
# microseconds and eager timing would be dominated by per-call host dispatch.
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


def build_case_inputs(case: dict[str, Any], device: str = "cuda") -> dict[str, Any]:
    """Allocate one case's declared buffers and let the bundle fill them.

    The bundle's callback writes preallocated buffers in place and validates
    their dtype, shape, contiguity and non-overlap before it writes anything, so
    allocating them is the whole of this task's share of input construction.
    """
    inputs = {
        "a": torch.empty((int(case["m"]), K), dtype=torch.bfloat16, device=device),
        "b": torch.empty((N, K), dtype=torch.bfloat16, device=device),
    }
    return task_initialize.run(inputs, seed=SEED)


def refill_case_inputs(
    inputs: dict[str, Any], seed: int = REFILL_SEED
) -> dict[str, Any]:
    """Redraw a case's buffers in place, keeping their storage.

    The bundle's callback writes preallocated buffers rather than allocating
    them, so redrawing through it changes the values while leaving every
    property the operator depends on untouched. A CUDA graph captured over these
    buffers therefore reads the new draw on its next replay, which is what makes
    the timed invocation answerable for a result it cannot have precomputed.
    """
    return task_initialize.run(inputs, seed=seed)


# The operands a production caller holds fixed while the activations change.
# ``b`` is the weight: sglang loads it once and calls the operator per batch.
# Holding it across the timed samples is what lets an implementation pack it
# once, the way aiter preshuffles at load time, without being charged for it.
PERSISTENT_INPUTS: tuple[str, ...] = ("b",)


def redraw_call_varying_inputs(
    inputs: dict[str, Any], seed: int = REFILL_SEED
) -> dict[str, Any]:
    """Redraw only what changes between two calls on a live model.

    Re-arming a timed invocation has to move the ground under it without
    invalidating work a real deployment would legitimately do once. Packing the
    weight into a kernel's preferred layout on the first call and reusing it is
    that kind of work, so a redraw that also replaced the weight would make an
    implementation which did it look like one that skipped the operator.

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
    inputs: dict[str, Any], seeds: tuple[int, ...]
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
    return {"a": inputs["a"], "b": inputs["b"]}


def verdict(got: torch.Tensor, expected: torch.Tensor) -> tuple[bool, str]:
    """Apply the bundle's comparison callback and report what it decided.

    The callback raises AssertionError for a candidate that fails its contract
    or its tolerance, and ValueError for a reference it considers invalid. Only
    the first is a candidate verdict, so only the first is caught: an invalid
    reference is this task's bug and has to stop the run rather than be scored
    as a failed port.

    No metric is recomputed here. The callback's own message carries the numbers
    it rejected on, and a second implementation of them would be the thing this
    whole file exists to avoid.
    """
    try:
        task_compare.run(got, expected)
    except AssertionError as failure:
        return False, str(failure)
    return True, "within tolerance"


def assert_candidate_is_independent(source: str) -> None:
    """Apply the documented dependency policy before candidate import."""
    task_contract.assert_source_independent(source)
