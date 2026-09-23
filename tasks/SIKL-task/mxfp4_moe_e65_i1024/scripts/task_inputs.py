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
``workload.json``, so the whole ``scripts/`` tree plus the harness and the
driver stay byte-identical across the MoE tasks. Arena copies each task
directory into its own workspace, so a task cannot import from a sibling and
every task has to carry its own copy of these modules.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import torch

import task_compare
import task_initialize

# A candidate implementation may not import the framework under test. Importing
# aiter -- including its FlyDSL kernel factories under aiter/ops/flydsl/kernels/
# -- launches the implementation this task exists to replace, so the run would
# compare the baseline's kernels against themselves and report the removal of
# per-call host dispatch as a speedup.
#
# Unlike the GEMM tasks there is no ban on torch's matrix product: a MoE written
# as a Python loop over experts is orders of magnitude slower than the fused
# baseline, so it is not a way to tie the baseline without writing a kernel.
BANNED_CANDIDATE_IMPORT_ROOTS = ("aiter",)

# Candidates run with the task's scripts on sys.path. Importing one of those
# modules can reach the baseline indirectly (for example, task_baseline.run)
# while avoiding a direct aiter import.
BANNED_CANDIDATE_TASK_MODULES = frozenset(
    {"forge_driver", "test_kernel_harness"}
)


def _find_workload() -> Path:
    """Locate workload.json, whichever layout these modules were copied into.

    The task keeps them under ``scripts/`` next to the harness, while the
    rewrite launcher copies them and the driver side by side into a scratch
    workspace one level below the task. Both have to resolve, and a module that
    cannot find its workload fails every mode rather than silently running a
    different shape.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here.parent, here, here.parent.parent):
        path = candidate / "workload.json"
        if path.is_file():
            return path
    raise RuntimeError(
        f"workload.json not found above {here}; the task's numeric contract is "
        "unreadable, so no input can be built"
    )


_WORKLOAD_PATH = _find_workload()
WORKLOAD = json.loads(_WORKLOAD_PATH.read_text())

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

ACTIVATION = 0
DOWEIGHT_STAGE1 = False

# The FlyDSL factory the port must expose. KernelForge derives it from the
# task's logical operator and passes it to the driver in the environment, so the
# harness reads the generated value rather than deriving it a second way: a
# harness that looked for a different symbol than the pipeline asked the agent
# to write would find no factory and score the baseline as a port.
BUILDER_SYMBOL = str(WORKLOAD["builder_symbol"])

# Benchmark parameters, shared by the Arena harness and the rewrite driver so
# the score and the pipeline's own speedup are measured the same way. Timing
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


# The operands a production caller holds fixed while the activations and the
# routing change. The expert weights are loaded once; sglang calls the layer per
# batch with new tokens against the same experts.
PERSISTENT_INPUTS: tuple[str, ...] = ("w1", "w1_scale", "w2", "w2_scale")


def redraw_call_varying_inputs(
    inputs: dict[str, Any], seed: int = REFILL_SEED
) -> dict[str, Any]:
    """Redraw only what changes between two calls on a live model.

    Re-arming a timed invocation has to move the ground under it without
    invalidating work a real deployment would legitimately do once. Laying the
    expert weights out the way a kernel wants them on the first call and reusing
    them is that kind of work -- aiter preshuffles at load time for the same
    reason -- so a redraw that also replaced them would make an implementation
    which did it look like one that skipped the layer.

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


# How much further the timed path may sit from the eager one than the eager one
# sits from itself, and a floor under that measured spread. The spread is what
# an implementation's own nondeterminism costs it -- a split reduction over
# atomics does not repeat bit for bit, and the shipped aiter dispatch does not
# at several of these shapes -- so it is measured per case rather than assumed.
# The floor only has to clear the comparison's own arithmetic for an
# implementation that does repeat exactly.
TIMED_PATH_MARGIN = 4.0
TIMED_PATH_FLOOR = 1e-6

# How much slower than the reported mean a replay over an unseen draw may be.
# The timed samples rotate through a few draws, so an implementation that keeps
# results keyed on its inputs' values can still serve every sample from memory
# once it remembers them all; a draw it has never seen is a miss, and that miss
# is the operator's actual cost. An implementation that computes on every call
# runs an unseen draw in the time of any other replay.
UNSEEN_DRAW_MARGIN = 1.5


def result_distance(got: torch.Tensor, other: torch.Tensor) -> float:
    """Largest elementwise gap between two results, against the result's scale.

    Normalized by the whole tensor's magnitude rather than elementwise: a
    per-element ratio is unbounded wherever the reference is near zero, which
    says nothing about whether the two runs computed the same thing. Taking the
    maximum rather than a mean is deliberate -- error concentrated in a few
    elements is exactly what a path that skips most of the work produces, and
    for a routed layer that is also what a mistake on one expert looks like.
    """
    got_f32, other_f32 = got.float(), other.float()
    scale = other_f32.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
    return ((got_f32 - other_f32).abs().max() / scale).item()


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


def banned_candidate_imports(source: str) -> list[str]:
    """Return protected modules a candidate implementation must not import."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise RuntimeError(f"the candidate does not parse: {error}") from error

    found: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            root = name.split(".", 1)[0]
            if root in BANNED_CANDIDATE_IMPORT_ROOTS and name not in found:
                found.append(name)
            leaf = name.rsplit(".", 1)[-1]
            if (
                leaf.startswith("task_") or leaf in BANNED_CANDIDATE_TASK_MODULES
            ) and name not in found:
                found.append(name)
    return found


def assert_candidate_is_independent(source: str) -> None:
    """Raise when a candidate reuses the framework it is meant to replace."""
    banned = banned_candidate_imports(source)
    if banned:
        raise RuntimeError(
            f"the candidate imports the framework under test or imports a protected task "
            f"module: {banned}. "
            "Reusing its baseline path measures the baseline against itself; implement the "
            "operator in FlyDSL (import flydsl and torch only)."
        )
