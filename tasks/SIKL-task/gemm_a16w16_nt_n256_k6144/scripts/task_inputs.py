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


def refill_case_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Redraw a case's buffers in place, keeping their storage.

    The bundle's callback writes preallocated buffers rather than allocating
    them, so redrawing through it changes the values while leaving every
    property the operator depends on untouched. A CUDA graph captured over these
    buffers therefore reads the new draw on its next replay, which is what makes
    the timed invocation answerable for a result it cannot have precomputed.
    """
    return task_initialize.run(inputs, seed=REFILL_SEED)


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
