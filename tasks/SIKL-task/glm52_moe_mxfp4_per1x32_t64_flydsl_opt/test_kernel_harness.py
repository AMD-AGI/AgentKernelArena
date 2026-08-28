#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Arena harness for optimizing GLM-5.2's MXFP4 MoE FlyDSL kernels in place.

The implementation under test is the aiter copy seeded into the workspace, so
the same command measures the shipped kernels before the agent runs and the
agent's edited kernels after. Correctness always compares against
scripts/task_reference.py, and the operator is invoked exactly as the workload
schema's baseline invokes it, so this task's numbers are directly comparable to
the from-scratch rewrite task next door.

Every mode fails closed if the import resolves past the workspace copy: an
agent's edits would be invisible and every number here would describe the
original kernel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import task_inputs

# Must precede the first `import aiter` anywhere in this process.
task_inputs.use_workspace_aiter()

import torch

from _aka_benchmark import benchmark_cuda_graph_or_events

import task_baseline
import task_reference

CASE_ID = "num_tokens_64"
REPORT_PATH = ROOT / "build" / "performance_report.json"
IMPLEMENTATION = "aiter_fused_moe_workspace"


def _prepare():
    if not torch.cuda.is_available():
        raise RuntimeError("this task requires a ROCm device")
    task_inputs.assert_aiter_is_workspace_copy()
    inputs = task_inputs.build_inputs()
    return inputs, task_inputs.call_kwargs(inputs)


def run_compile() -> int:
    """Dispatch the operator once so the edited FlyDSL kernels are compiled."""
    _, kwargs = _prepare()
    out = task_baseline.run(**kwargs)
    torch.cuda.synchronize()
    if tuple(out.shape) != (task_inputs.NUM_TOKENS, task_inputs.MODEL_DIM):
        print(f"compile failed: unexpected output shape {tuple(out.shape)}")
        return 1
    import aiter

    print(f"compile ok: dispatched through {Path(aiter.__file__).resolve()}")
    return 0


def run_correctness() -> int:
    _, kwargs = _prepare()
    got = task_baseline.run(**kwargs)
    torch.cuda.synchronize()
    expected = task_reference.run(**kwargs)
    torch.cuda.synchronize()

    if got.shape != expected.shape:
        print(f"correctness: fail shape {tuple(got.shape)} != {tuple(expected.shape)}")
        return 1
    if not torch.isfinite(got.float()).all().item():
        print("correctness: fail non-finite output")
        return 1

    relative_error = task_inputs.relative_error(got, expected)
    print(f"implementation: {IMPLEMENTATION}")
    print(f"case {CASE_ID}: mean relative error {relative_error:.6f} "
          f"(gate {task_inputs.MAX_RELERR})")
    if relative_error > task_inputs.MAX_RELERR:
        print("correctness: fail")
        return 1
    print("correctness: pass")
    return 0


def run_full_benchmark() -> int:
    _, kwargs = _prepare()
    execution_time_ms, metadata = benchmark_cuda_graph_or_events(
        lambda: task_baseline.run(**kwargs),
        warmup=task_inputs.BENCH_WARMUP,
        repetition=task_inputs.BENCH_REPETITION,
        target_ms=task_inputs.BENCH_TARGET_MS,
    )

    row = {
        "test_case_id": CASE_ID,
        "execution_time_ms": execution_time_ms,
        "params": {
            "num_tokens": task_inputs.NUM_TOKENS,
            "model_dim": task_inputs.MODEL_DIM,
            "inter_dim": task_inputs.INTER_DIM,
            "num_experts": task_inputs.NUM_EXPERTS,
            "topk": task_inputs.TOPK,
            "quant_type": "per_1x32",
            "seed": task_inputs.SEED,
        },
        "metadata": {"implementation": IMPLEMENTATION},
    }
    row.update(metadata)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps([row], indent=2))
    print(f"implementation: {IMPLEMENTATION}")
    print(f"case {CASE_ID}: {execution_time_ms:.6f} ms "
          f"({metadata.get('benchmark_method')})")
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
