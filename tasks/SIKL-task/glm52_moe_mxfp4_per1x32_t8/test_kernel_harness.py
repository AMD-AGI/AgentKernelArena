#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Arena harness for the GLM-5.2 MXFP4 MoE rewrite task.

Scores whichever implementation the workspace currently holds:

  * ``kernel.py`` still on its stub (the builder raises ``NotImplementedError``)
    -> the operator's own baseline, ``aiter.fused_moe`` (task_baseline.py). This
    is what Arena measures before the agent runs.
  * ``kernel.py`` carrying a FlyDSL port -> that port. This is what Arena
    measures after the agent runs, so the reported speedup is the ported FlyDSL
    kernel against the production implementation.

Correctness always compares against task_reference.py, never against the
baseline, so a port cannot pass by reproducing the baseline's quantization
idiosyncrasies.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from _aka_benchmark import benchmark_cuda_graph_or_events

import task_baseline
import task_inputs
import task_reference

CASE_ID = task_inputs.CASE_ID
CANDIDATE_FILE = ROOT / "kernel.py"
BUILDER_SYMBOL = task_inputs.BUILDER_SYMBOL
REPORT_PATH = ROOT / "build" / "performance_report.json"


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


def _build_candidate_launch():
    """Build the ported kernel, or return None while kernel.py is a stub.

    Only ``NotImplementedError`` counts as "not ported yet"; every other failure
    propagates, so a broken port fails the task instead of silently scoring the
    baseline a second time.
    """
    builder = _candidate_builder()
    if builder is None:
        return None
    try:
        return builder(
            num_tokens=task_inputs.NUM_TOKENS,
            model_dim=task_inputs.MODEL_DIM,
            inter_dim=task_inputs.INTER_DIM,
            num_experts=task_inputs.NUM_EXPERTS,
            topk=task_inputs.TOPK,
        )
    except NotImplementedError:
        return None


def _resolve_implementation(inputs: dict):
    """Return (name, zero-arg callable) for the implementation under test."""
    launch = _build_candidate_launch()
    if launch is None:
        kwargs = task_inputs.call_kwargs(inputs)
        return "aiter_fused_moe_baseline", lambda: task_baseline.run(**kwargs)

    def call_candidate():
        return launch(
            inputs["hidden_states"],
            inputs["w1"],
            inputs["w2"],
            inputs["topk_weight"],
            inputs["topk_ids"],
            inputs["w1_scale"],
            inputs["w2_scale"],
            inputs["activation"],
            inputs["doweight_stage1"],
        )

    return "flydsl_port", call_candidate


def _require_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this task requires a ROCm device")


def run_compile() -> int:
    """Import every task module and build the candidate without launching it."""
    _require_gpu()
    builder = _candidate_builder()
    if builder is None:
        print(f"compile ok: kernel.py has no {BUILDER_SYMBOL} yet (stub state)")
        return 0
    try:
        builder(
            num_tokens=task_inputs.NUM_TOKENS,
            model_dim=task_inputs.MODEL_DIM,
            inter_dim=task_inputs.INTER_DIM,
            num_experts=task_inputs.NUM_EXPERTS,
            topk=task_inputs.TOPK,
        )
    except NotImplementedError:
        print("compile ok: kernel.py builder is still the stub")
        return 0
    print("compile ok: FlyDSL candidate built")
    return 0


def run_correctness() -> int:
    _require_gpu()
    inputs = task_inputs.build_inputs()
    name, call = _resolve_implementation(inputs)
    got = call()
    torch.cuda.synchronize()
    expected = task_reference.run(**task_inputs.call_kwargs(inputs))
    torch.cuda.synchronize()

    if got.shape != expected.shape:
        print(f"correctness: fail ({name}) shape {tuple(got.shape)} != "
              f"{tuple(expected.shape)}")
        return 1
    if not torch.isfinite(got.float()).all().item():
        print(f"correctness: fail ({name}) non-finite output")
        return 1

    relative_error = task_inputs.relative_error(got, expected)
    print(f"implementation: {name}")
    print(f"case {CASE_ID}: mean relative error {relative_error:.6f} "
          f"(gate {task_inputs.MAX_RELERR})")
    if relative_error > task_inputs.MAX_RELERR:
        print("correctness: fail")
        return 1
    print("correctness: pass")
    return 0


def run_full_benchmark() -> int:
    _require_gpu()
    inputs = task_inputs.build_inputs()
    name, call = _resolve_implementation(inputs)
    execution_time_ms, metadata = benchmark_cuda_graph_or_events(
        call,
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
        "metadata": {"implementation": name},
    }
    row.update(metadata)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps([row], indent=2))
    print(f"implementation: {name}")
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
