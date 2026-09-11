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
from pathlib import Path
from typing import Any

import torch

# A candidate implementation may not import the framework under test. aiter's
# tuned a16w16 dispatch resolves to aiter's own FlyDSL kernels for most of the
# small-M cases, so importing it would measure the baseline against itself.
BANNED_CANDIDATE_IMPORT_ROOTS = ("aiter",)

# Candidates run with the task's scripts on sys.path. Importing one of those
# modules can reach the baseline indirectly (for example, task_baseline.run)
# while avoiding a direct aiter import.
BANNED_CANDIDATE_TASK_MODULES = frozenset(
    {"forge_driver", "test_kernel_harness"}
)

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

# The correctness gate: every element of the candidate's output must land within
# ``ATOL + RTOL * |reference|`` of the fp32 reference. This is the acceptance
# criterion the workload bundle's own benchmark applies, and it is read from
# workload.json so the two cannot drift into two different definitions of
# correct.
#
# It replaces a gate derived from the production implementation's own distance
# to the reference. That gate could not see what this one is for. A kernel that
# truncates its split-k or K-warp partial sums to bf16 before reducing them
# lands about 0.0017 in mean relative error and around 51 dB SNR -- inside any
# aggregate bound worth setting -- while missing this elementwise tolerance on
# 6-11% of the output. Aggregate statistics average that tail away; the
# elementwise form is what sees it.
#
# The tolerance is not tight for a kernel that accumulates in fp32: measured at
# n=k=6144, fp32 partial sums match the reference on every element at split
# counts 2, 4, 8 and 16, with SNR around 113 dB. So the gate admits any
# reduction order and rejects a truncated accumulator, which is the distinction
# it exists to make.
ATOL = float(WORKLOAD["atol"])
RTOL = float(WORKLOAD["rtol"])

CASES: tuple[dict[str, Any], ...] = tuple(WORKLOAD["cases"])
CASE_IDS: tuple[str, ...] = tuple(str(case["case_id"]) for case in CASES)


def matched_ratio(got: torch.Tensor, expected: torch.Tensor) -> float:
    """Fraction of elements within the acceptance tolerance of the reference."""
    got_f32, expected_f32 = got.float(), expected.float()
    within = (got_f32 - expected_f32).abs() <= ATOL + RTOL * expected_f32.abs()
    return within.double().mean().item()


def gate_explanation() -> str:
    """One line naming the gate, for the harness and the driver."""
    return (
        f"gate: every element within atol {ATOL:g} + rtol {RTOL:g} x |reference| "
        "(matched_ratio 1.0); the production implementation's own accuracy is "
        "reported for context but does not set the bar"
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
            leaf = name.rsplit(".", 1)[-1]
            if leaf.startswith("task_") or leaf in BANNED_CANDIDATE_TASK_MODULES:
                finding = f"imports a protected task module: {name}"
                if finding not in findings:
                    findings.append(finding)

        # Also catch aliases such as ``from torch import matmul as product``.
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".", 1)[0] == "torch"
        ):
            for alias in node.names:
                if alias.name in BANNED_CANDIDATE_CALL_ATTRS:
                    finding = f"imports the library matrix product `{alias.name}`"
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
            isinstance(node, ast.Attribute)
            and node.attr in BANNED_CANDIDATE_CALL_ATTRS
        ):
            finding = f"references the library matrix product `{node.attr}`"
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
            f"the candidate {joined}. These defeat the rewrite: protected task "
            "modules can call the baseline indirectly, aiter's tuned "
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
