# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""The QA auditor's rubric: every check, its severity and what fixes it.

A task passes only when no blocking finding remains. Warnings never block and
always carry their reason and a suggested fix. Blocking findings come from
deterministic framework checks, or from a model-review counterexample that the
framework itself reproduced twice; an unreproduced review opinion is a warning.
"""
from __future__ import annotations

from dataclasses import dataclass

RUBRIC_VERSION = "2026-11-05.2"
BLOCKING = "blocking"
WARNING = "warning"
MIN_BASELINE_CASE_MS = 0.010


@dataclass(frozen=True)
class Check:
    check_id: str
    severity: str
    title: str
    fix: str


CHECKS = {check.check_id: check for check in (
    Check("QA-AUDIT", BLOCKING, "The audit itself ran to completion",
          "Fix the reported framework or environment failure and rerun the audit."),
    Check("QA-SPEC", BLOCKING, "Task contract and workloads are valid",
          "Fix task.yaml/workloads.json until the framework loads them (see docs/how-to/add-task.md)."),
    Check("QA-PACKAGE", BLOCKING, "Task files depend only on the task and allowed libraries",
          "Vendor the code into the task; remove imports of production packages, image paths, "
          "process/network/dynamic-import use and absolute paths."),
    Check("QA-REFERENCE", BLOCKING, "The reference is independent high-precision PyTorch",
          "Write reference.py with PyTorch in float32/float64 only; do not call the baseline or a kernel "
          "language."),
    Check("QA-PROVENANCE", BLOCKING, "The baseline records its upstream source and license",
          "Fill baseline.provenance with the real source, revision, path and license; add baseline/LICENSE "
          "for a license other than Apache-2.0."),
    Check("QA-INITIAL", BLOCKING, "Framework initial validation accepted the task",
          "Make validate-task and baseline compile/correctness/performance pass; see the captured "
          "action evidence."),
    Check("QA-POSITIVE", BLOCKING, "A copy of the baseline passes as a candidate",
          "Make the baseline satisfy the candidate rules of its language (kernels only, no PyTorch "
          "computation, capturable launches) so the candidate path accepts a correct implementation."),
    Check("QA-TIMING-METHOD", BLOCKING, "Every performance case is a cold-cache CUDA-graph timing",
          "Make the baseline graph-capturable on the current stream."),
    Check("QA-TIMING-FLOOR", BLOCKING, "Every performance case takes the baseline at least 10 us",
          "Use larger production shapes for performance cases; keep small shapes as correctness-only."),
    Check("QA-CONTROLS", BLOCKING, "Correctness rejects every known-wrong control implementation",
          "Strengthen inputs, cases or the error bound so the surviving control fails."),
    Check("QA-CONTROLS-RUN", BLOCKING, "The control suite ran to completion",
          "Fix the inputs, reference or baseline failure reported by the control run."),
    Check("QA-INPUTS", BLOCKING, "Input draws are not degenerate",
          "Draw inputs with a distribution whose values vary and are not exactly representable in a "
          "lower precision than declared."),
    Check("QA-COUNTEREXAMPLE", BLOCKING, "No reproduced wrong implementation passes the task",
          "Add the counterexample's input domain to the workloads or tighten the bound."),
    Check("QA-POSITIVE-UNAVAILABLE", WARNING, "No candidate-path positive control for this baseline language",
          "A TileLang baseline cannot be copied as a candidate; consider a Triton baseline."),
    Check("QA-BOUND-OVERRIDE", WARNING, "The error bound departs from the default",
          "Justify the override in README.md or return to baseline_relative with k=2."),
    Check("QA-BOUND-CALIBRATION", WARNING, "A float32 evaluation of the reference fails the error bound",
          "A correct implementation with float32 accumulation may be rejected. Unless the baseline is "
          "known to be more accurate than float32 accumulation, loosen the bound (sqnr, or k and "
          "max_violation_fraction) and justify it in README.md."),
    Check("QA-NONDETERMINISTIC", WARNING, "The baseline is not bitwise deterministic",
          "Informational: baseline_relative compares against one baseline run per draw."),
    Check("QA-README", WARNING, "The task has no README.md",
          "Describe the operator, its production source and the workload origin in README.md."),
    Check("QA-UNVERIFIED-ARCH", WARNING, "The task claims architectures this audit did not run on",
          "Run the audit on each claimed architecture or document it as unverified."),
    Check("QA-REVIEW", WARNING, "Model review finding without a reproduced counterexample",
          "Review the finding; supply a counterexample solution and input to make it blocking."),
    Check("QA-REVIEW-UNAVAILABLE", WARNING, "Model review did not complete",
          "Rerun the audit with a working review backend."),
)}


def check(check_id: str) -> Check:
    return CHECKS[check_id]
