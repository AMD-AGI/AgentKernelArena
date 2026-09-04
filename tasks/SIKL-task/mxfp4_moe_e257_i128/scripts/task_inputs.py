# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Deterministic input construction for an MXFP4 fused-MoE workload.

Extracted from the workload-schema definition named by ``workload.json``. Every
tensor input is ``type: random``, ``activation`` is the scalar ``0`` (SiLU) and
``doweight_stage1`` the scalar ``false``; the declared constraint is
``topk <= num_experts``.

Every constant that varies between tasks in this family lives in
``workload.json``, so the whole ``scripts/`` tree plus the harness and the
driver stay byte-identical across the MoE tasks. Arena copies each task
directory into its own workspace, so a task cannot import from a sibling and
every task has to carry its own copy of these modules.

Why "random" is not uniform random bytes
----------------------------------------
The schema declares the weights and scales in the *stored* MXFP4 layout that
aiter's ``fused_moe`` requires, and its reference un-shuffles and dequantizes
them. Filling those tensors with uniform random bytes is not a valid instance
of that layout: a random e8m0 scale byte spans 2**-127 .. 2**127, so the
dequantized weights overflow bf16 accumulation and any correctness comparison
becomes meaningless. Inputs are therefore produced the way a live model
produces them -- bf16 weights, ``dynamic_mxfp4_quant`` at group_size 32 with
e8m0 scales, then aiter's load-time preshuffle -- which yields exactly the
dtypes and shapes the schema declares.

Case ordering matters: the shared weights and then every case's activation are
drawn from one generator seeded once, in the order ``workload.json`` declares
the cases. That order is the workload schema's order, so the inputs are
reproducible from the schema alone.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

import torch

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

# How much further from the fp32 reference a candidate may sit than the
# production implementation does, and a floor under that measured distance.
# These are the correctness constants the task fixes; the distance they act on
# is measured at run time.
GATE_MULTIPLIER = float(WORKLOAD["gate_multiplier"])
GATE_FLOOR = float(WORKLOAD["gate_floor"])

# The second correctness gate, expressed in the L2 domain. The gate above acts
# on a mean relative error, an L1 statistic that averages away error
# concentrated in a few elements: a routing or per-expert mistake can leave most
# of the output untouched and still pass. SNR is a power ratio and penalizes
# exactly that concentration, so a case has to clear both.
#
# It introduces no new policy constants. It is the SAME policy as the error
# gate, restated for a different statistic: allowing the noise power to be
# GATE_MULTIPLIER times larger is a fixed offset in dB, and GATE_FLOOR's role --
# never demand more accuracy than this -- becomes a ceiling on the SNR that may
# be required. The mapping between an L1 relative error and an L2 power ratio is
# an analogue rather than an identity, which is why these bound a derived
# measurement instead of replacing it.
#
# A fixed floor was the wrong shape here. This operator's baseline is bimodal
# against the reference across the var axis -- the tuned stage-1 variants that
# fuse the output quantization into their epilogue sit an order of magnitude
# further out than the ones that do not -- and its worst cases fall below 30 dB.
# Any fixed threshold high enough to be meaningful would fail the production
# implementation itself, capping every run at the compile score.
SNR_MARGIN_DB = 10.0 * math.log10(GATE_MULTIPLIER)
SNR_CEILING_DB = -20.0 * math.log10(GATE_FLOOR)

CASES: tuple[dict[str, Any], ...] = tuple(WORKLOAD["cases"])
CASE_IDS: tuple[str, ...] = tuple(str(case["case_id"]) for case in CASES)

_WEIGHT_SCALE = 0.125
_ACT_SCALE = 0.25


def derive_gates(baseline: dict[str, list[float]]) -> dict[str, float]:
    """Return both correctness gates for this run, from the measured baseline.

    The gates are derived, never stored. What the task fixes is the policy -- a
    candidate may be at most ``GATE_MULTIPLIER`` times as far from the fp32
    reference as the production implementation itself is, but is never held to a
    distance tighter than ``GATE_FLOOR`` -- and the distance is measured against
    the same inputs, on the same device, with the same framework build that is
    about to score the candidate. A recorded number would silently go stale the
    moment any of those changed.

    The floor matters because the measured distance says as much about which
    reduction strategy the baseline happens to use as about what correctness
    requires. Where the tuned dispatch lands on a near-exact implementation at
    every case the measurement collapses to around 1e-6, and a gate derived from
    that alone would demand that a port reproduce the baseline's accumulation
    order rather than merely be correct.

    Both are deliberately ONE gate for the operator rather than one per case,
    for the same reason: the baseline selects a different kernel per bucket of
    the var axis and its own accuracy moves by orders of magnitude across them.

    The two act on different statistics on purpose. ``error`` bounds a mean, and
    ``snr_db`` bounds a power ratio, which is what catches error concentrated in
    a few elements rather than spread over the output.
    """
    errors, snrs = baseline["errors"], baseline["snrs"]
    if not errors or not snrs:
        raise RuntimeError("no baseline accuracy was measured, so no gate can be derived")
    return {
        "error": max(max(errors), GATE_FLOOR) * GATE_MULTIPLIER,
        "snr_db": min(min(snrs), SNR_CEILING_DB) - SNR_MARGIN_DB,
    }


def gate_explanation(baseline: dict[str, list[float]]) -> str:
    """One line naming both gates and what set each, for the harness and driver."""
    gates = derive_gates(baseline)
    worst_error, worst_snr = max(baseline["errors"]), min(baseline["snrs"])
    error_basis = "worst baseline error" if worst_error >= GATE_FLOOR else "floor"
    snr_basis = "worst baseline snr" if worst_snr <= SNR_CEILING_DB else "ceiling"
    return (
        f"gates: error {gates['error']:.8f} = max({worst_error:.8f}, "
        f"{GATE_FLOOR:g}) x {GATE_MULTIPLIER:g} set by the {error_basis}; "
        f"snr {gates['snr_db']:.2f} dB = min({worst_snr:.2f}, "
        f"{SNR_CEILING_DB:.2f}) - {SNR_MARGIN_DB:.2f} set by the {snr_basis}"
    )


def _assert_declared_layout(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        raise AssertionError(f"{name}: schema declares {shape}, built {tuple(tensor.shape)}")


def build_inputs(device: str = "cuda") -> dict[str, Any]:
    """Build one instance of every workload case in the declared layout.

    The expert weights depend only on the constant axes, so all cases share
    them; only the activation and the routing carry the num_tokens axis.
    """
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant, e8m0_shuffle

    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)

    w1_bf16 = _WEIGHT_SCALE * torch.randn(
        (NUM_EXPERTS, W1_ROWS, MODEL_DIM), device=device, dtype=torch.bfloat16,
        generator=generator,
    )
    w2_bf16 = _WEIGHT_SCALE * torch.randn(
        (NUM_EXPERTS, MODEL_DIM, INTER_DIM), device=device, dtype=torch.bfloat16,
        generator=generator,
    )

    w1_packed, w1_scale = dynamic_mxfp4_quant(w1_bf16.reshape(-1, MODEL_DIM))
    w1_packed = w1_packed.view(NUM_EXPERTS, W1_ROWS, -1)
    w1_scale = w1_scale.view(NUM_EXPERTS, W1_ROWS, -1)
    w2_packed, w2_scale = dynamic_mxfp4_quant(w2_bf16.reshape(-1, INTER_DIM))
    w2_packed = w2_packed.view(NUM_EXPERTS, MODEL_DIM, -1)
    w2_scale = w2_scale.view(NUM_EXPERTS, MODEL_DIM, -1)
    del w1_bf16, w2_bf16

    # aiter's load-time preshuffle: e8m0_shuffle on a 2D view of the scales,
    # shuffle_weight((16, 16)) on the packed weights.
    experts, rows, _ = w1_scale.shape
    w1_scale = e8m0_shuffle(w1_scale.reshape(experts * rows, -1)).view(experts, rows, -1)
    experts, rows, _ = w2_scale.shape
    w2_scale = e8m0_shuffle(w2_scale.reshape(experts * rows, -1)).view(experts, rows, -1)
    w1 = shuffle_weight(w1_packed.contiguous(), (16, 16))
    w2 = shuffle_weight(w2_packed.contiguous(), (16, 16))

    _assert_declared_layout("w1", w1, (NUM_EXPERTS, W1_ROWS, W1_COLS))
    _assert_declared_layout("w1_scale", w1_scale, (NUM_EXPERTS, W1_ROWS, W1_SCALE_COLS))
    _assert_declared_layout("w2", w2, (NUM_EXPERTS, MODEL_DIM, W2_COLS))
    _assert_declared_layout("w2_scale", w2_scale, (NUM_EXPERTS, MODEL_DIM, W2_SCALE_COLS))

    cases = []
    for case in CASES:
        num_tokens = int(case["num_tokens"])
        hidden_states = _ACT_SCALE * torch.randn(
            (num_tokens, MODEL_DIM), device=device, dtype=torch.bfloat16,
            generator=generator,
        )
        # topk distinct experts per token; weights normalized over the selection.
        scores = torch.rand((num_tokens, NUM_EXPERTS), device=device, generator=generator)
        topk_ids = scores.topk(TOPK, dim=-1).indices.to(torch.int32).contiguous()
        topk_weight = torch.softmax(
            scores.gather(1, topk_ids.long()).float(), dim=-1
        ).contiguous()
        cases.append(
            {
                "case_id": str(case["case_id"]),
                "num_tokens": num_tokens,
                "hidden_states": hidden_states,
                "topk_weight": topk_weight,
                "topk_ids": topk_ids,
            }
        )

    return {"w1": w1, "w2": w2, "w1_scale": w1_scale, "w2_scale": w2_scale, "cases": cases}


def call_kwargs(inputs: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """The operator's full argument set, in the schema's input order."""
    return {
        "hidden_states": case["hidden_states"],
        "w1": inputs["w1"],
        "w2": inputs["w2"],
        "topk_weights": case["topk_weight"],
        "topk_ids": case["topk_ids"],
        "w1_scale": inputs["w1_scale"],
        "w2_scale": inputs["w2_scale"],
        "activation": ACTIVATION,
        "doweight_stage1": DOWEIGHT_STAGE1,
    }


def banned_candidate_imports(source: str) -> list[str]:
    """Return the framework modules a candidate implementation must not import."""
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
    return found


def assert_candidate_is_independent(source: str) -> None:
    """Raise when a candidate reuses the framework it is meant to replace."""
    banned = banned_candidate_imports(source)
    if banned:
        raise RuntimeError(
            f"the candidate imports the framework under test: {banned}. Reusing "
            "its kernels measures the baseline against itself; implement the "
            "operator in FlyDSL (import flydsl and torch only)."
        )


def relative_error(got: torch.Tensor, expected: torch.Tensor) -> float:
    """Mean relative error against the reference, in fp32."""
    got_f32 = got.float()
    expected_f32 = expected.float()
    denominator = expected_f32.abs().mean().clamp_min(torch.finfo(torch.float32).tiny)
    return ((got_f32 - expected_f32).abs().mean() / denominator).item()
