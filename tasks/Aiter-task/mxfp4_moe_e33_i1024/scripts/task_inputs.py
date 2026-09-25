# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Input construction for an MXFP4 fused-MoE workload.

The buffers are allocated here and filled by ``task_initialize``, which is the
schema bundle's own ``initialize`` callback: standard-normal activations, expert
weights drawn at 1/sqrt(their logical fan-in) and quantized to nearest-even E2M1
under upward-rounded E8M0 block scales, then the expert-local shuffle, and a
router that draws logits and takes their softmax top-k. Nothing about the
distribution or the quantization is decided in this file, because the acceptance
run that verifies a result uses that callback and a second implementation of it
here would be a second operator.

That also settles what "random" means for the stored MXFP4 layout, which uniform
random bytes cannot: a random e8m0 scale byte spans 2**-127 to 2**127, so the
dequantized weights would overflow bf16 accumulation and any comparison would be
meaningless. The callback quantizes real weights instead.

Each case is built on its own, from a generator re-seeded to ``seed`` for that
case, because the bundle initializes one workload point at a time. The
activation carries the case's num_tokens and is drawn first, so the weights land
at a different point in the stream for every case -- the cases do not share a
weight set and cannot be built from one pass.

Every constant that varies between tasks in this family lives in
the configured workload JSON, so the helpers and runner stay byte-identical
across the MoE tasks. Arena copies each task
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

# Axes from the schema definition. num_tokens is the only `var` axis.
MODEL_DIM = int(WORKLOAD["axes"]["model_dim"])
NUM_EXPERTS = int(WORKLOAD["axes"]["num_experts"])
TOPK = int(WORKLOAD["axes"]["topk"])
W1_ROWS = int(WORKLOAD["axes"]["w1_rows"])
W1_COLS = int(WORKLOAD["axes"]["w1_cols"])
W2_COLS = int(WORKLOAD["axes"]["w2_cols"])
W1_SCALE_COLS = int(WORKLOAD["axes"]["w1_scale_cols"])
W2_SCALE_COLS = int(WORKLOAD["axes"]["w2_scale_cols"])

# Derived: w1 rows hold [gate | up], and MXFP4 packs two values per stored byte.
INTER_DIM = W1_ROWS // 2
QUANT_GROUP_SIZE = 32

SEED = int(WORKLOAD["seed"])

# The draw used to re-arm a timed invocation. It only has to differ from SEED:
# the point is that the values a captured graph replays over are ones no earlier
# call in this process has seen, not that they come from a second distribution.
REFILL_SEED = SEED + 1
ACTIVATION = 0
DOWEIGHT_STAGE1 = False

# The entrypoint is explicit task data; it is not derived from operator identity.
BUILDER_SYMBOL = task_contract.candidate_entry()["symbol"]

# Benchmark parameters used for both baseline and candidate. Timing
# must be CUDA-graph based: eager timing of this operator is dominated by
# per-call host dispatch, not by the device work.
BENCH_WARMUP = int(WORKLOAD["bench"]["warmup"])
BENCH_REPETITION = int(WORKLOAD["bench"]["repetition"])
BENCH_TARGET_MS = float(WORKLOAD["bench"]["target_ms"])

CASES: tuple[dict[str, Any], ...] = tuple(WORKLOAD["cases"])
CASE_IDS: tuple[str, ...] = tuple(str(case["case_id"]) for case in CASES)

GATE_EXPLANATION = (
    "gate: scripts/task_compare.py, the schema bundle's own comparison callback. "
    "It owns the threshold -- nothing here sets or relaxes it."
)


def verdict(got: torch.Tensor, expected: torch.Tensor) -> tuple[bool, str]:
    """Apply the bundle's comparison callback and report what it decided.

    The callback raises AssertionError for a candidate that fails its contract
    or its threshold, and ValueError for a reference it considers invalid. Only
    the first is a candidate verdict, so only the first is caught: an invalid
    reference is this task's bug and has to stop the run rather than be scored
    as a failed port.

    No metric is recomputed here. The callback's own message carries the number
    it rejected on, and a second implementation of it would be the thing this
    whole file exists to avoid.
    """
    try:
        task_compare.run(got, expected)
    except AssertionError as failure:
        return False, str(failure)
    return True, "within threshold"


def build_case_inputs(case: dict[str, Any], device: str = "cuda") -> dict[str, Any]:
    """Allocate one case's declared buffers and let the bundle fill them.

    The bundle's callback writes preallocated buffers in place, and it validates
    their dtype, shape, contiguity and non-overlap before it writes anything, so
    allocating them in the layout the schema declares is the whole of this
    task's share of input construction. It quantizes the expert weights itself
    -- nearest-even E2M1 codes under upward-rounded E8M0 block scales, then the
    expert-local shuffle -- so no aiter quantizer is involved and the weights a
    candidate sees are the weights the acceptance run generates.

    The packed weights are native ``float4_e2m1fn_x2``: the callback requires
    that dtype and refuses a uint8 stand-in.
    """
    num_tokens = int(case["num_tokens"])
    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4 is None:
        raise RuntimeError(
            "this task needs native torch.float4_e2m1fn_x2; the schema's input "
            "callback refuses a uint8 stand-in for the packed expert weights"
        )
    empty = lambda shape, dtype: torch.empty(shape, dtype=dtype, device=device)
    inputs = {
        "hidden_states": empty((num_tokens, MODEL_DIM), torch.bfloat16),
        "w1": empty((NUM_EXPERTS, W1_ROWS, W1_COLS), fp4),
        "w2": empty((NUM_EXPERTS, MODEL_DIM, W2_COLS), fp4),
        "topk_weights": empty((num_tokens, TOPK), torch.float32),
        "topk_ids": empty((num_tokens, TOPK), torch.int32),
        "w1_scale": empty((NUM_EXPERTS, W1_ROWS, W1_SCALE_COLS), torch.uint8),
        "w2_scale": empty((NUM_EXPERTS, MODEL_DIM, W2_SCALE_COLS), torch.uint8),
        "activation": ACTIVATION,
        "doweight_stage1": DOWEIGHT_STAGE1,
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
    return {
        "hidden_states": inputs["hidden_states"],
        "w1": inputs["w1"],
        "w2": inputs["w2"],
        "topk_weights": inputs["topk_weights"],
        "topk_ids": inputs["topk_ids"],
        "w1_scale": inputs["w1_scale"],
        "w2_scale": inputs["w2_scale"],
        "activation": inputs["activation"],
        "doweight_stage1": inputs["doweight_stage1"],
    }


def assert_candidate_is_independent(source: str) -> None:
    """Apply the documented dependency policy before candidate import."""
    task_contract.assert_source_independent(source)
