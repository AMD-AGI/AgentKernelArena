# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Deterministic input construction for an a16w16 GEMM workload.

Extracted from the workload-schema definition named by ``workload.json``. Both
tensor inputs are ``type: random`` and the declared layout is ``a [m, k]``,
``b [n, k]`` with ``trans_b`` -- the operator computes ``a @ b.T``.

Every constant that varies between tasks in this family lives in
``workload.json``, so the whole ``scripts/`` tree plus the harness and the
driver stay byte-identical across the GEMM tasks. Arena copies each task
directory into its own workspace, so a task cannot import from a sibling and
every task has to carry its own copy of these modules.

Case ordering matters: the shared ``b`` and then every case's ``a`` are drawn
from one generator seeded once, in the order ``workload.json`` declares the
cases. That order is the workload schema's order, so the inputs are
reproducible from the schema alone.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

import torch

# A candidate implementation may not import the framework under test. aiter's
# tuned a16w16 dispatch resolves to aiter's own FlyDSL kernels for most of the
# small-M cases, so importing it would measure the baseline against itself.
BANNED_CANDIDATE_IMPORT_ROOTS = ("aiter",)

# A GEMM has a second cheat the MoE tasks do not: torch's own matmul is a real
# hipBLASLt call, and it is literally the baseline this task dispatches to at
# the larger M cases. A candidate that writes ``a @ b.T`` would tie the baseline
# exactly while implementing no kernel at all.
BANNED_CANDIDATE_CALL_ATTRS = frozenset(
    {
        "matmul",
        "mm",
        "bmm",
        "einsum",
        "linear",
        "addmm",
        "addbmm",
        "baddbmm",
        "tensordot",
    }
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
N = int(WORKLOAD["axes"]["n"])
K = int(WORKLOAD["axes"]["k"])
TRANS_B = bool(WORKLOAD["trans_b"])
SEED = int(WORKLOAD["seed"])

# The FlyDSL factory the port must expose. KernelForge derives it from the
# task's logical operator and passes it to the driver in the environment, so the
# harness reads the generated value rather than deriving it a second way: a
# harness that looked for a different symbol than the pipeline asked the agent
# to write would find no factory and score the aiter baseline as a port.
BUILDER_SYMBOL = str(WORKLOAD["builder_symbol"])

# Benchmark parameters, shared by the Arena harness and the rewrite driver so
# the score and the pipeline's own speedup are measured the same way. Timing
# must be CUDA-graph based: at the small-M cases this operator runs for tens of
# microseconds and eager timing would be dominated by per-call host dispatch.
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
# concentrated in a few elements: a candidate can sit well inside it while being
# badly wrong on part of the output. SNR is a power ratio and penalizes exactly
# that concentration, so a case has to clear both.
#
# It introduces no new policy constants. It is the SAME policy as the error
# gate, restated for a different statistic: allowing the noise power to be
# GATE_MULTIPLIER times larger is a fixed offset in dB, and GATE_FLOOR's role --
# never demand more accuracy than this -- becomes a ceiling on the SNR that may
# be required. The mapping between an L1 relative error and an L2 power ratio is
# an analogue rather than an identity, which is why these bound a derived
# measurement instead of replacing it.
#
# A fixed floor was the wrong shape here. The MoE baseline's own worst case sits
# below 30 dB against the fp32 reference, so any fixed threshold high enough to
# be meaningful for this GEMM family would fail the production implementation
# itself on the MoE tasks, capping every run at the compile score.
SNR_MARGIN_DB = 10.0 * math.log10(GATE_MULTIPLIER)
SNR_CEILING_DB = -20.0 * math.log10(GATE_FLOOR)

CASES: tuple[dict[str, Any], ...] = tuple(WORKLOAD["cases"])
CASE_IDS: tuple[str, ...] = tuple(str(case["case_id"]) for case in CASES)


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


def build_inputs(device: str = "cuda") -> dict[str, Any]:
    """Build one instance of every workload case in the declared layout.

    ``b`` depends only on the constant axes, so all cases share it; only ``a``
    carries the ``m`` axis. That is what keeps a whole-family task affordable:
    one 75 MB weight plus roughly 100 MB of activations covers all 13 cases.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)

    b = torch.randn((N, K), device=device, dtype=torch.bfloat16, generator=generator)

    cases = []
    for case in CASES:
        m = int(case["m"])
        a = torch.randn(
            (m, K), device=device, dtype=torch.bfloat16, generator=generator
        )
        cases.append({"case_id": str(case["case_id"]), "m": m, "a": a})

    return {"b": b, "cases": cases}


def call_kwargs(inputs: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """The operator's full argument set, in the schema's input order."""
    return {"a": case["a"], "b": inputs["b"]}


def _banned_candidate_findings(source: str) -> list[str]:
    """Return every rule violation found in a candidate implementation."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise RuntimeError(f"the candidate does not parse: {error}") from error

    findings: list[str] = []

    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            if name.split(".", 1)[0] in BANNED_CANDIDATE_IMPORT_ROOTS:
                finding = f"imports the framework under test: {name}"
                if finding not in findings:
                    findings.append(finding)

    for node in ast.walk(tree):
        # ``ast.MatMult`` only ever means the binary operator, so a decorator's
        # ``@`` cannot be mistaken for one.
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            finding = "uses the `@` matrix-multiply operator"
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.MatMult):
            finding = "uses the `@=` matrix-multiply operator"
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in BANNED_CANDIDATE_CALL_ATTRS
        ):
            finding = f"calls the library matrix product `{node.func.attr}`"
        else:
            continue
        if finding not in findings:
            findings.append(finding)

    return findings


def assert_candidate_is_independent(source: str) -> None:
    """Raise when a candidate reuses an implementation it is meant to replace."""
    findings = _banned_candidate_findings(source)
    if findings:
        joined = "; ".join(findings)
        raise RuntimeError(
            f"the candidate {joined}. Both defeat the rewrite: aiter's tuned "
            "a16w16 path dispatches to aiter's own FlyDSL kernels at the small-M "
            "cases, and torch's matmul IS the baseline at the larger ones. "
            "Implement the GEMM in FlyDSL (import flydsl and torch only, and use "
            "torch for tensor plumbing rather than for the product)."
        )


def relative_error(got: torch.Tensor, expected: torch.Tensor) -> float:
    """Mean relative error against the reference, in fp32."""
    got_f32 = got.float()
    expected_f32 = expected.float()
    denominator = expected_f32.abs().mean().clamp_min(torch.finfo(torch.float32).tiny)
    return ((got_f32 - expected_f32).abs().mean() / denominator).item()
