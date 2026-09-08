#!/usr/bin/env python3
"""Protected runner for functional SIKL tasks; no candidate fallback."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.task_api import (assert_outputs, assert_unmodified, clone_inputs,
                              load_solution, outputs, poison_outputs, validate_inputs)
from scripts.task_inputs import make_inputs


class TimedRun:
    def _bind(self, rerun, outputs=None):
        self.rerun = rerun
        self.outputs = outputs


def candidate():
    path = ROOT / "source" / "kernel.py"
    spec = importlib.util.spec_from_file_location("aka_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("compile", "correctness", "performance", "source-check"), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("This task requires a compatible ROCm GPU")
    contract = json.loads((ROOT / "scripts" / "workload.json").read_text())
    definition, policy = contract["definition"], contract["policy"]
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    launch = candidate()
    reference = load_solution(ROOT / "scripts" / "reference", contract["reference_spec"]["entry_point"])
    baseline = load_solution(ROOT / "scripts" / "baseline", contract["baseline_spec"]["entry_point"])
    samples = []
    report_path = ROOT / "build" / "performance_report.json"
    if args.mode == "performance":
        report_path.unlink(missing_ok=True)
    for row in contract["rows"]:
        values = make_inputs(definition, row, policy)
        validate_inputs(values, definition, row, "cuda")
        pristine = clone_inputs(values)
        if args.mode == "source-check":
            expected = reference(**clone_inputs(values))
            got = baseline(**values)
            assert_outputs(got, expected, definition, row, policy, "cuda")
        elif args.mode == "correctness":
            expected = reference(**clone_inputs(values))
            got = launch(**values)
            assert_outputs(got, expected, definition, row, policy, "cuda")
        else:
            # Launching triggers lazy compilation; an import-only check would
            # not establish that the declared candidate builds on this GPU.
            got = launch(**values)
            outputs(got, definition, row, "cuda")
        assert_unmodified(pristine, values)
        torch.cuda.synchronize()
        if args.mode == "performance":
            from _aka_benchmark import benchmark_cuda_graph_or_events
            expected = reference(**clone_inputs(values))
            replay = TimedRun()
            elapsed, metadata = benchmark_cuda_graph_or_events(
                lambda: launch(**values), warmup=policy["warmup"],
                repetition=policy["repetition"], target_ms=policy["target_ms"],
                timed_run=replay,
            )
            # A stale allocation from capture/warmup cannot satisfy replay
            # validation: the measured graph must actually write the outputs.
            poison_outputs(replay.outputs, expected, values, definition, row, "cuda")
            assert_outputs(replay.rerun(), expected, definition, row, policy, "cuda")
            assert_unmodified(pristine, values)
            samples.append({"test_case_id": row["workload"]["uuid"],
                            "execution_time_ms": elapsed, **metadata,
                            "exact_graph_replay_validated": True,
                            "params": row["workload"]["axes"]})
        print(f"{args.mode}: {row['workload']['uuid']} PASS", flush=True)
        del values, pristine, got
    if args.mode == "performance":
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(json.dumps(samples, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
