#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Arena harness for the a16w16 GEMM rewrite task.

Scores whichever implementation the workspace currently holds:

  * ``kernel.py`` still on its stub (no factory defined) -> the operator's own
    baseline, ``aiter.tuned_gemm.gemm_a16w16`` (task_baseline.py). This is what
    Arena measures before the agent runs.
  * ``kernel.py`` carrying a FlyDSL port -> that port. This is what Arena
    measures after the agent runs, so the reported speedup is the ported FlyDSL
    kernel against the production dispatch.

Correctness always compares against task_reference.py, never against the
baseline, so a port cannot pass by reproducing the baseline's accumulation
order.

How anything is measured lives in scripts/task_measure.py, which the rewrite
driver imports too: the score and the number KernelForge keeps a candidate on
have to come from one implementation, not two that agree today.

Every mode covers all of the workload's cases. The evaluator matches baseline
and optimized results by ``test_case_id`` and refuses to score a partial match,
so a port that only works at some shapes scores no performance at all.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import torch

import task_inputs
import task_measure

CANDIDATE_FILE = ROOT / "kernel.py"
BUILDER_SYMBOL = task_inputs.BUILDER_SYMBOL
REPORT_PATH = ROOT / "build" / "performance_report.json"
BASELINE_NAME = "aiter_gemm_a16w16_baseline"
PORT_NAME = "flydsl_port"


def _candidate_builder():
    """Return the FlyDSL builder from kernel.py, or None when absent.

    The independence check runs here as well as in the rewrite driver: a
    candidate the driver would reject during PORT must not be scoreable through
    Arena's own path either.
    """
    if not CANDIDATE_FILE.is_file():
        return None
    task_inputs.assert_candidate_is_independent(CANDIDATE_FILE.read_text())
    spec = importlib.util.spec_from_file_location("flydsl_candidate", CANDIDATE_FILE)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, BUILDER_SYMBOL, None)


def _resolve_calls(inputs):
    """Return (name, calls) for the implementation the workspace holds."""
    launches = task_measure.build_launches(_candidate_builder(), inputs)
    if launches is None:
        return BASELINE_NAME, task_measure.baseline_calls(inputs)
    return PORT_NAME, task_measure.candidate_calls(inputs, launches)


def _require_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this task requires a ROCm device")


def run_compile() -> int:
    """Import every task module and build each case without launching it."""
    _require_gpu()
    inputs = {"cases": task_inputs.CASES}
    if task_measure.build_launches(_candidate_builder(), inputs) is None:
        print(f"compile ok: kernel.py has no usable {BUILDER_SYMBOL} yet (stub state)")
        return 0
    print(f"compile ok: FlyDSL candidate built for {len(task_inputs.CASES)} cases")
    return 0


def run_correctness() -> int:
    _require_gpu()
    inputs = task_inputs.build_inputs()
    expected, baseline_errors, gate = task_measure.reference_and_gate(inputs)
    name, calls = _resolve_calls(inputs)
    print(f"implementation: {name}")
    print(
        f"gate {gate:.8f} = worst baseline error {max(baseline_errors):.8f} "
        f"x {task_inputs.GATE_MULTIPLIER:g}"
    )

    failed = []
    for record in task_measure.compare_cases(calls, expected):
        case_id = record["case_id"]
        if record["shape_mismatch"] is not None:
            got_shape, expected_shape = record["shape_mismatch"]
            print(f"case {case_id}: fail, shape {got_shape} != {expected_shape}")
        else:
            print(f"case {case_id}: mean relative error {record['error']:.8f}")
        if not task_measure.passes(record, gate):
            failed.append(case_id)

    if failed:
        print(f"correctness: fail ({len(failed)}/{len(calls)} cases: {', '.join(failed)})")
        return 1
    print(f"correctness: pass ({len(calls)} cases)")
    return 0


def run_full_benchmark() -> int:
    _require_gpu()
    inputs = task_inputs.build_inputs()
    name, calls = _resolve_calls(inputs)
    print(f"implementation: {name}")

    rows = []
    for sample in task_measure.time_cases(calls):
        row = {
            "test_case_id": sample["case_id"],
            "execution_time_ms": sample["execution_time_ms"],
            "params": {
                "m": sample["m"],
                "n": task_inputs.N,
                "k": task_inputs.K,
                "trans_b": task_inputs.TRANS_B,
                "dtype": "bfloat16",
                "seed": task_inputs.SEED,
            },
            "metadata": {"implementation": name},
        }
        row.update(sample["metadata"])
        rows.append(row)
        print(
            f"case {sample['case_id']}: {sample['execution_time_ms']:.6f} ms "
            f"({sample['metadata'].get('benchmark_method')})"
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(rows, indent=2))
    print(f"wrote {REPORT_PATH}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--compile", action="store_true")
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    if args.compile:
        return run_compile()
    if args.correctness:
        return run_correctness()
    return run_full_benchmark()


if __name__ == "__main__":
    raise SystemExit(main())
