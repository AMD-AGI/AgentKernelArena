# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Input construction for an a16w16 GEMM workload.

The buffers are allocated here and filled by ``task_initialize``, which is the
schema bundle's own ``initialize`` callback: ``a`` is standard normal and ``b``
is drawn at 1/sqrt(k), its logical fan-in. Nothing about the distribution is
decided in this file, because the acceptance run that verifies a result uses
that callback and a second implementation of it here would be a second operator.

Each case is built on its own, from a generator re-seeded to ``seed`` for that
case, because the bundle initializes one workload point at a time. ``a`` carries
the case's m and is drawn first, so ``b`` lands at a different point in the
stream for every case -- the cases do not share a weight and cannot be built
from one pass.

Every constant that varies between tasks in this family lives in
``workload.json``, so the whole ``scripts/`` tree plus the harness and the
driver stay byte-identical across the GEMM tasks. Arena copies each task
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


def call_kwargs(inputs: dict[str, Any]) -> dict[str, Any]:
    """The operator's full argument set, in the schema's input order."""
    return {"a": inputs["a"], "b": inputs["b"]}


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
