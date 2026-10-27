# SPDX-License-Identifier: MIT
"""Schema-v2 evaluation of one MiniMax dense MLP rank-local GEMM site."""
from __future__ import annotations

import ast
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
import traceback

from contract import ROOT, load_contract
from inputs import check_inputs_unchanged, make_inputs, snapshot_inputs
from reference import check_output, reference
from runtime import verify_runtime


def candidate():
    path = ROOT / "source/kernel.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == "mlp_down" for node in tree.body):
        raise ValueError("Implemented candidate requires source/kernel.py::mlp_down")
    for source in (ROOT / "source").rglob("*.py"):
        compile(source.read_text(), str(source), "exec")
    sys.path.insert(0, str(ROOT))
    module = importlib.import_module("source.kernel")
    if not callable(module.mlp_down):
        raise ValueError("Candidate entrypoint is not callable")
    return module.mlp_down


def run_action(role, action, data):
    import torch

    metadata = verify_runtime()
    if action == "validate-task":
        candidate()
        from aiter.tuned_gemm import gemm_a16w16

        if not callable(gemm_a16w16):
            raise RuntimeError("Pinned upstream GEMM is unavailable")
        metadata.update(candidate_state="implemented", frequency_kind="config_derived_fixed_batching_scenario")
        return data["cases"], metadata
    if role == "baseline":
        from aiter.tuned_gemm import gemm_a16w16 as operation
    else:
        operation = candidate()
    rows = []
    for case in data["cases"]:
        values = make_inputs(case, data["input_generation"]["seeds"][0])
        frozen = snapshot_inputs(values)
        expected = reference(*values)
        launch = lambda: operation(*values)
        row = {key: value for key, value in case.items() if key != "checks"}
        row["status"] = "PASS"
        row["metrics"] = check_output(launch(), expected, data["tolerance"])
        torch.cuda.synchronize()
        replay = launch
        if action == "performance":
            from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events_samples

            timed = TimedRun()
            timed.after_sample = lambda output: check_output(output, expected, data["tolerance"])
            samples, timing = benchmark_cuda_graph_or_events_samples(
                launch, **data["timing"], use_cuda_graph=True, timed_run=timed,
            )
            if timing["benchmark_method"] != "cuda_graph":
                raise RuntimeError("Timing must use the declared CUDA/HIP graph method")
            check_output(timed.outputs, expected, data["tolerance"])
            check_output(timed.rerun(), expected, data["tolerance"])
            elapsed = statistics.median(samples)
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError("Device timing is not finite and positive")
            row.update(execution_time_ms=elapsed, benchmark_method="cuda_graph", metadata=timing)
            replay = timed.rerun
        check_inputs_unchanged(frozen)
        del frozen
        # The measured callable/graph must use current tensor values. Keep
        # object identities and layouts while replacing BOTH public operands.
        # This also rejects weight-value caches outside the timed invocation.
        alternate = make_inputs(case, data["input_generation"]["seeds"][1])
        for current, fresh in zip(values, alternate):
            current.copy_(fresh)
        del alternate
        expected = reference(*values)
        frozen = snapshot_inputs(values)
        row.setdefault("metrics", {})["current_value_replay"] = check_output(replay(), expected, data["tolerance"])
        check_inputs_unchanged(frozen)
        if role == "baseline":
            from aiter.tuned_gemm import get_GEMM_A16W16_config

            m, n, k = case["shape"]
            dispatch = get_GEMM_A16W16_config(
                M=m, N=n, K=k, bias=False, dtype="torch.bfloat16", otype="torch.bfloat16",
                scaleAB=False, bpreshuffle=False,
            )
            # Pandas table rows can contain NumPy scalars or NaN in an unused
            # backend field; keep diagnostic metadata valid strict JSON.
            clean = {}
            for key, value in dispatch.items():
                value = value.item() if hasattr(value, "item") else value
                clean[key] = None if isinstance(value, float) and not math.isfinite(value) else value
            row.setdefault("metadata", {})["upstream_dispatch"] = clean
        rows.append(row)
        del values, frozen, expected
    return ([] if action == "compile" else rows), metadata


def invoke(argv):
    role, action = "task", "validate-task"
    if len(argv) == 2:
        role, action = argv
    result = {"protocol": "arena-eval-v1", "role": role, "action": action,
              "status": "FAIL", "cases": []}
    try:
        if argv != ["validate-task"] and not (
            len(argv) == 2 and role in ("baseline", "candidate")
            and action in ("compile", "correctness", "performance")
        ):
            raise ValueError("Use validate-task or {baseline,candidate} {compile,correctness,performance}")
        data = load_contract()
        rows, metadata = run_action(role, action, data)
        result.update(status="PASS", cases=rows, metadata=metadata)
        json.dumps(result, allow_nan=False)
        return result, 0
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        result.update(reason=f"{type(error).__name__}: {error}", failure_kind="evaluation_error")
        return result, 1


if __name__ == "__main__":
    report, exit_code = invoke(sys.argv[1:])
    print("ARENA_EVAL_RESULT=" + json.dumps(report, allow_nan=False), flush=True)
    raise SystemExit(exit_code)
